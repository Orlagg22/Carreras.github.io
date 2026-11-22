import * as THREE from "three";
import type { CarTelemetry } from "@/types/f1";

/**
 * Live instruments of the "Modern Cartoon Sports Car" (scripts/cars/rebuild_cartoon_interior.py): the GLB only carries
 * two still pictures, a cluster (speedometer, rev counter, gear / speed readout) and a centre screen, both with their
 * needles at rest. This puts a canvas on each and redraws it from the game's telemetry:
 *
 * - the speedometer and rev counter needles follow `speedKmh` and `rpm` (the dials are drawn at the GLB's own sizes);
 * - the screen between the dials shows the gear (R / N / 1..) and the speed, and the trip distance driven;
 * - the centre screen is a navigation map that draws the circuit as the car drives it: every part of the lap the car
 *   has covered (placed by `trackProgress`) is traced, the car's arrow moves along it and the lap and its progress are
 *   shown. Nothing of the track data is needed: the map is the car's own path, like a satnav discovering the road.
 *
 * The rig shares the template's materials between cars, so each rig gets its own copy; `dispose` puts them back. The
 * canvases are only redrawn while the screens are in shot (three calls `onBeforeRender` for the meshes it draws).
 */

const CLUSTER_MATERIAL = "int_cluster";
const SCREEN_MATERIAL = "int_screen";

const REDRAW_INTERVAL_MS = 1000 / 24;
/** The screens count as in shot for this long after the last frame three drew them */
const SEEN_FOR_MS = 400;
const FONT = "Arial, Helvetica, sans-serif";
const TRACE_BINS = 360;
/** A recorded point this far from where the car is now means another circuit (or a teleport): the map starts over */
const MAP_RESET_DISTANCE_M = 150;
/** The trace of one lap is not joined across more than this many missing bins */
const MAX_TRACE_GAP_BINS = 3;

export interface CartoonDashSpec {
  /** Last number on the speedometer (km/h); default 320 */
  maxSpeedKmh?: number;
  /** Last number on the rev counter (rpm); default 9000, or the next 1000 above 110 % of the highest rpm seen */
  maxRpm?: number;
  /** Where the rev counter's red zone starts (rpm); default 85 % of its scale */
  redlineRpm?: number;
}

interface DialScale {
  vmax: number;
  step: number;
  minor: number;
  redFrom: number | null;
  numberDiv: number;
}

const CLUSTER = { w: 1024, h: 320 };
const SCREEN = { w: 512, h: 320 };
const SPEED_DIAL = { cx: 220, cy: 165, r: 138 };
const RPM_DIAL = { cx: 804, cy: 165, r: 138 };
const SWEEP_FROM = 225;
const SWEEP_TO = -45;

function polar(cx: number, cy: number, r: number, deg: number): [number, number] {
  const a = (deg * Math.PI) / 180;
  return [cx + r * Math.cos(a), cy - r * Math.sin(a)];
}

function makeCanvas(w: number, h: number): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

/** The dial's face: bezel, ticks and numerals (everything but the needle), drawn when its scale changes */
function drawDialFace(ctx: CanvasRenderingContext2D, d: typeof SPEED_DIAL, s: DialScale, unit: string): void {
  const { cx, cy, r } = d;
  ctx.beginPath();
  ctx.arc(cx, cy, r + 10, 0, Math.PI * 2);
  ctx.fillStyle = "rgb(26,27,30)";
  ctx.fill();
  ctx.lineWidth = 5;
  ctx.strokeStyle = "rgb(150,152,158)";
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fillStyle = "rgb(7,8,10)";
  ctx.fill();
  const minors = Math.round(s.vmax / s.minor);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let k = 0; k <= minors; k++) {
    const v = k * s.minor;
    const deg = SWEEP_FROM + ((SWEEP_TO - SWEEP_FROM) * v) / s.vmax;
    const ratio = v / s.step;
    const major = Math.abs(ratio - Math.round(ratio)) < 1e-6;
    const red = s.redFrom !== null && v >= s.redFrom;
    const col = red ? "rgb(235,40,40)" : "rgb(235,236,240)";
    const [x0, y0] = polar(cx, cy, r - (major ? 30 : 18), deg);
    const [x1, y1] = polar(cx, cy, r - 6, deg);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.lineWidth = major ? 6 : 3;
    ctx.strokeStyle = col;
    ctx.stroke();
    if (major) {
      const [tx, ty] = polar(cx, cy, r - 56, deg);
      ctx.font = `bold 26px ${FONT}`;
      ctx.fillStyle = col;
      ctx.fillText(String(Math.round(v / s.numberDiv)), tx, ty);
    }
  }
  ctx.font = `bold 22px ${FONT}`;
  ctx.fillStyle = "rgb(150,154,162)";
  ctx.fillText(unit, cx, cy + r * 0.52);
}

function drawNeedle(ctx: CanvasRenderingContext2D, d: typeof SPEED_DIAL, fraction: number): void {
  const { cx, cy, r } = d;
  const deg = SWEEP_FROM + (SWEEP_TO - SWEEP_FROM) * Math.min(1, Math.max(0, fraction));
  const [nx, ny] = polar(cx, cy, r - 24, deg);
  const [tx, ty] = polar(cx, cy, 22, deg + 180);
  ctx.beginPath();
  ctx.moveTo(tx, ty);
  ctx.lineTo(nx, ny);
  ctx.lineWidth = 7;
  ctx.lineCap = "round";
  ctx.strokeStyle = "rgb(255,90,20)";
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(cx, cy, 14, 0, Math.PI * 2);
  ctx.fillStyle = "rgb(40,42,46)";
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgb(200,202,206)";
  ctx.stroke();
}

function roundedRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function gearLabel(gear: number): string {
  if (gear < 0) return "R";
  if (gear === 0) return "N";
  return String(gear);
}

interface ScreenPair {
  mesh: THREE.Mesh;
  original: THREE.Material | THREE.Material[];
  material: THREE.MeshStandardMaterial;
  texture: THREE.CanvasTexture;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
}

export class CartoonDash {
  readonly found: boolean;
  private cluster: ScreenPair | null = null;
  private screen: ScreenPair | null = null;
  private face: HTMLCanvasElement | null = null;
  private faceKey = "";
  private streets: HTMLCanvasElement | null = null;
  private peakRpm = 0;
  private lastDrawMs = -Infinity;
  private lastSeenMs = -Infinity;
  private lastNowMs = 0;
  private tripM = 0;
  private telemetry: CarTelemetry | null = null;
  // The map: the car's world x / z where it was at each fraction of the lap
  private readonly traceX = new Float32Array(TRACE_BINS);
  private readonly traceZ = new Float32Array(TRACE_BINS);
  private readonly traced = new Uint8Array(TRACE_BINS);
  private tracedCount = 0;
  private readonly heading = new THREE.Vector3();

  /** The cluster and screen of the body under `body`, once its GLB has loaded (null for a model without them) */
  static from(root: THREE.Object3D, body: THREE.Object3D, spec?: CartoonDashSpec): CartoonDash | null {
    const dash = new CartoonDash(root, body, spec);
    return dash.found ? dash : null;
  }

  private constructor(
    private readonly root: THREE.Object3D,
    body: THREE.Object3D,
    private readonly spec: CartoonDashSpec = {}
  ) {
    body.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh || Array.isArray(mesh.material)) return;
      const name = mesh.material.name;
      if (name === CLUSTER_MATERIAL && !this.cluster) this.cluster = this.claim(mesh, CLUSTER);
      else if (name === SCREEN_MATERIAL && !this.screen) this.screen = this.claim(mesh, SCREEN);
    });
    this.found = this.cluster !== null || this.screen !== null;
    if (this.found) this.redraw(0);
  }

  private claim(mesh: THREE.Mesh, size: { w: number; h: number }): ScreenPair | null {
    const base = mesh.material as THREE.MeshStandardMaterial;
    if (!base.isMeshStandardMaterial) return null;
    const canvas = makeCanvas(size.w, size.h);
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    const texture = new THREE.CanvasTexture(canvas);
    // glTF textures have their origin at the top-left
    texture.flipY = false;
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = 8;
    const material = base.clone();
    material.map = texture;
    // A backlit display: lit by its own picture, not by the cabin's light
    material.emissive.set(0xffffff);
    material.emissiveMap = texture;
    material.emissiveIntensity = 0.85;
    const original = mesh.material;
    mesh.material = material;
    // three only calls this for a mesh it is drawing: a dash nobody is looking at is not redrawn
    mesh.onBeforeRender = () => {
      this.lastSeenMs = performance.now();
    };
    return { mesh, original, material, texture, canvas, ctx };
  }

  private scales(): { speed: DialScale; rpm: DialScale; maxRpm: number } {
    const maxSpeed = this.spec.maxSpeedKmh ?? 320;
    const speed: DialScale = {
      vmax: maxSpeed,
      step: maxSpeed > 400 ? 50 : maxSpeed > 200 ? 40 : 20,
      minor: maxSpeed > 400 ? 25 : maxSpeed > 200 ? 20 : 10,
      redFrom: null,
      numberDiv: 1,
    };
    const maxRpm = this.spec.maxRpm ?? Math.max(9000, Math.ceil((this.peakRpm * 1.1) / 1000) * 1000);
    const wide = maxRpm > 10000;
    const redFrom = this.spec.redlineRpm ?? Math.round((maxRpm * 0.85) / 500) * 500;
    const rpm: DialScale = {
      vmax: maxRpm,
      step: wide ? 2000 : 1000,
      minor: wide ? 1000 : 500,
      redFrom,
      numberDiv: 1000,
    };
    return { speed, rpm, maxRpm };
  }

  /** Every frame: remembers where the car is on the lap, and redraws the screens at their own pace while in shot */
  /** `nowMs` is `performance.now()` */
  update(t: CarTelemetry, nowMs: number): void {
    if (!this.found) return;
    this.telemetry = t;
    const dt = Math.min(0.25, Math.max(0, (nowMs - this.lastNowMs) / 1000));
    this.lastNowMs = nowMs;
    if (Number.isFinite(t.speedMs)) this.tripM += Math.max(0, t.speedMs) * dt;
    if (Number.isFinite(t.rpm) && t.rpm > this.peakRpm) this.peakRpm = t.rpm;
    this.record(t);
    const seen = nowMs - this.lastSeenMs < SEEN_FOR_MS;
    if (!seen || nowMs - this.lastDrawMs < REDRAW_INTERVAL_MS) return;
    this.lastDrawMs = nowMs;
    this.redraw(nowMs);
  }

  private record(t: CarTelemetry): void {
    const progress = t.trackProgress;
    if (!Number.isFinite(progress) || t.offTrack) return;
    const bin = Math.min(TRACE_BINS - 1, Math.max(0, Math.floor((((progress % 1) + 1) % 1) * TRACE_BINS)));
    const x = this.root.position.x;
    const z = this.root.position.z;
    if (this.traced[bin]) {
      // Another circuit, or the car put somewhere else: the old map is not this road
      if (Math.hypot(this.traceX[bin] - x, this.traceZ[bin] - z) > MAP_RESET_DISTANCE_M) {
        this.traced.fill(0);
        this.tracedCount = 0;
      } else return;
    }
    this.traceX[bin] = x;
    this.traceZ[bin] = z;
    this.traced[bin] = 1;
    this.tracedCount++;
  }

  // ------------------------------------------------------------------------------------------------ the cluster
  private ensureFace(speed: DialScale, rpm: DialScale): HTMLCanvasElement {
    const key = `${speed.vmax}|${rpm.vmax}|${rpm.redFrom}`;
    if (this.face && key === this.faceKey) return this.face;
    const face = makeCanvas(CLUSTER.w, CLUSTER.h);
    const ctx = face.getContext("2d")!;
    ctx.fillStyle = "rgb(8,9,11)";
    ctx.fillRect(0, 0, CLUSTER.w, CLUSTER.h);
    drawDialFace(ctx, SPEED_DIAL, speed, "km/h");
    drawDialFace(ctx, RPM_DIAL, rpm, "x1000 rpm");
    this.face = face;
    this.faceKey = key;
    return face;
  }

  private drawCluster(t: CarTelemetry | null): void {
    const c = this.cluster;
    if (!c) return;
    const { speed, rpm } = this.scales();
    const ctx = c.ctx;
    ctx.drawImage(this.ensureFace(speed, rpm), 0, 0);
    const kmh = t && Number.isFinite(t.speedKmh) ? Math.max(0, t.speedKmh) : 0;
    const revs = t && Number.isFinite(t.rpm) ? Math.max(0, t.rpm) : 0;
    drawNeedle(ctx, SPEED_DIAL, kmh / speed.vmax);
    drawNeedle(ctx, RPM_DIAL, revs / rpm.vmax);

    // The trip screen between the dials: gear, speed, trip distance
    roundedRect(ctx, 392, 52, 240, 218, 18);
    ctx.fillStyle = "rgb(10,20,30)";
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgb(70,90,110)";
    ctx.stroke();
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const gear = t ? gearLabel(t.gear) : "N";
    ctx.font = `bold 60px ${FONT}`;
    ctx.fillStyle = gear === "R" ? "rgb(255,90,60)" : "rgb(200,255,0)";
    ctx.fillText(gear, 512, 100);
    ctx.font = `bold 70px ${FONT}`;
    ctx.fillStyle = "rgb(235,240,245)";
    ctx.fillText(String(Math.round(kmh)), 512, 160);
    ctx.font = `bold 22px ${FONT}`;
    ctx.fillStyle = "rgb(150,160,170)";
    ctx.fillText("km/h", 512, 205);
    ctx.beginPath();
    ctx.moveTo(420, 232);
    ctx.lineTo(604, 232);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgb(60,80,100)";
    ctx.stroke();
    ctx.font = `bold 18px ${FONT}`;
    ctx.fillStyle = "rgb(150,160,170)";
    ctx.fillText(`TRIP ${(this.tripM / 1000).toFixed(2)} km`, 512, 252);
    c.texture.needsUpdate = true;
  }

  // ------------------------------------------------------------------------------------------------ the map
  private ensureStreets(): HTMLCanvasElement {
    if (this.streets) return this.streets;
    const s = makeCanvas(SCREEN.w, SCREEN.h);
    const ctx = s.getContext("2d")!;
    ctx.fillStyle = "rgb(12,16,22)";
    ctx.fillRect(0, 0, SCREEN.w, SCREEN.h);
    ctx.strokeStyle = "rgb(24,32,42)";
    ctx.lineWidth = 2;
    for (let x = 0; x <= SCREEN.w; x += 48) {
      ctx.beginPath();
      ctx.moveTo(x, 36);
      ctx.lineTo(x, SCREEN.h);
      ctx.stroke();
    }
    for (let y = 36; y <= SCREEN.h; y += 48) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(SCREEN.w, y);
      ctx.stroke();
    }
    this.streets = s;
    return s;
  }

  private drawScreen(t: CarTelemetry | null): void {
    const s = this.screen;
    if (!s) return;
    const ctx = s.ctx;
    ctx.drawImage(this.ensureStreets(), 0, 0);

    // The map's frame: the traced points' bounds, scaled to fit with the aspect kept
    const area = { x: 20, y: 48, w: SCREEN.w - 40, h: SCREEN.h - 48 - 34 };
    let minX = Infinity;
    let maxX = -Infinity;
    let minZ = Infinity;
    let maxZ = -Infinity;
    for (let i = 0; i < TRACE_BINS; i++) {
      if (!this.traced[i]) continue;
      minX = Math.min(minX, this.traceX[i]);
      maxX = Math.max(maxX, this.traceX[i]);
      minZ = Math.min(minZ, this.traceZ[i]);
      maxZ = Math.max(maxZ, this.traceZ[i]);
    }
    const carX = this.root.position.x;
    const carZ = this.root.position.z;
    // the car is on the map even before its first bin is traced
    minX = Math.min(minX, carX);
    maxX = Math.max(maxX, carX);
    minZ = Math.min(minZ, carZ);
    maxZ = Math.max(maxZ, carZ);
    const span = Math.max(60, maxX - minX, maxZ - minZ);
    const scale = Math.min(area.w, area.h) / span;
    const cx = (minX + maxX) / 2;
    const cz = (minZ + maxZ) / 2;
    const toX = (x: number) => area.x + area.w / 2 + (x - cx) * scale;
    const toY = (z: number) => area.y + area.h / 2 + (z - cz) * scale;

    // The circuit so far
    if (this.tracedCount > 1) {
      let first = -1;
      const pts: [number, number, boolean][] = []; // x, y, begins a new stroke
      let prev = -1;
      for (let i = 0; i < TRACE_BINS; i++) {
        if (!this.traced[i]) continue;
        if (first < 0) first = i;
        pts.push([toX(this.traceX[i]), toY(this.traceZ[i]), !(prev >= 0 && i - prev <= MAX_TRACE_GAP_BINS)]);
        prev = i;
      }
      // A lap that is all drawn is a closed loop
      if (this.tracedCount >= TRACE_BINS * 0.97 && first >= 0 && prev >= 0) pts.push([toX(this.traceX[first]), toY(this.traceZ[first]), false]);
      const stroke = (width: number, color: string) => {
        ctx.beginPath();
        for (const [x, y, begin] of pts) {
          if (begin) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.lineWidth = width;
        ctx.strokeStyle = color;
        ctx.stroke();
      };
      stroke(15, "rgb(16,52,78)");
      stroke(7, "rgb(0,190,255)");
      if (first >= 0) {
        ctx.beginPath();
        ctx.arc(toX(this.traceX[first]), toY(this.traceZ[first]), 9, 0, Math.PI * 2);
        ctx.fillStyle = "rgb(200,255,0)";
        ctx.fill();
        ctx.lineWidth = 3;
        ctx.strokeStyle = "rgb(255,255,255)";
        ctx.stroke();
      }
    } else {
      ctx.font = `bold 20px ${FONT}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "rgb(110,125,140)";
      ctx.fillText("DRIVE TO MAP THE CIRCUIT", SCREEN.w / 2, area.y + area.h / 2 + 40);
    }

    // The car: an arrow along its heading (the model's +Z is forward; the map's x / y are the world's x / z)
    this.heading.set(0, 0, 1).applyQuaternion(this.root.quaternion);
    const ang = Math.atan2(this.heading.z, this.heading.x);
    const px = toX(carX);
    const py = toY(carZ);
    ctx.save();
    ctx.translate(px, py);
    ctx.rotate(ang);
    ctx.beginPath();
    ctx.moveTo(15, 0);
    ctx.lineTo(-10, 9);
    ctx.lineTo(-5, 0);
    ctx.lineTo(-10, -9);
    ctx.closePath();
    ctx.fillStyle = "rgb(255,255,255)";
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgb(10,16,22)";
    ctx.stroke();
    ctx.restore();

    // Header: name, lap and the clock
    ctx.fillStyle = "rgb(20,26,34)";
    ctx.fillRect(0, 0, SCREEN.w, 36);
    ctx.textBaseline = "middle";
    ctx.font = `bold 18px ${FONT}`;
    ctx.textAlign = "left";
    ctx.fillStyle = "rgb(200,255,0)";
    ctx.fillText("NORYX NAV", 14, 18);
    const lap = t && Number.isFinite(t.lapCount) ? Math.max(1, t.lapCount) : 1;
    ctx.textAlign = "center";
    ctx.fillStyle = "rgb(235,240,245)";
    ctx.fillText(`LAP ${lap}`, SCREEN.w / 2, 18);
    const now = new Date();
    ctx.textAlign = "right";
    ctx.fillStyle = "rgb(220,225,232)";
    ctx.fillText(`${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`, SCREEN.w - 14, 18);

    // Footer: progress of the lap
    const progress = t && Number.isFinite(t.trackProgress) ? Math.min(1, Math.max(0, t.trackProgress)) : 0;
    ctx.fillStyle = "rgb(20,26,34)";
    ctx.fillRect(0, SCREEN.h - 30, SCREEN.w, 30);
    ctx.fillStyle = "rgb(36,48,62)";
    ctx.fillRect(14, SCREEN.h - 20, SCREEN.w - 130, 10);
    ctx.fillStyle = "rgb(0,190,255)";
    ctx.fillRect(14, SCREEN.h - 20, (SCREEN.w - 130) * progress, 10);
    ctx.textAlign = "right";
    ctx.font = `bold 16px ${FONT}`;
    ctx.fillStyle = "rgb(235,240,245)";
    ctx.fillText(`${Math.round(progress * 100)} %  ·  ${Math.round(t && Number.isFinite(t.speedKmh) ? Math.max(0, t.speedKmh) : 0)} km/h`, SCREEN.w - 14, SCREEN.h - 15);
    s.texture.needsUpdate = true;
  }

  private redraw(_nowMs: number): void {
    this.drawCluster(this.telemetry);
    this.drawScreen(this.telemetry);
  }

  dispose(): void {
    for (const pair of [this.cluster, this.screen]) {
      if (!pair) continue;
      pair.mesh.material = pair.original;
      pair.mesh.onBeforeRender = () => {};
      pair.texture.dispose();
      pair.material.dispose();
    }
    this.cluster = null;
    this.screen = null;
    this.face = null;
    this.streets = null;
  }
}
