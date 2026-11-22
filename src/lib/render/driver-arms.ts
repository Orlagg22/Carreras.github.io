import * as THREE from "three";
import { moveToDriverHeadLayer } from "@/lib/render/layers";

/**
 * The game's driver body for a ported car: the driver's hands and forearms on the wheel and, with a `body` entry, the
 * rest of him (the F1 car's driver, `car-builder-2026.ts`, has hands, forearms, torso and helmet too).
 *
 * A ported GLB opts in with a `driver` entry in its `CAR` extras (`PortedDriverSpec`): where the hands grip the rim, where
 * the elbows are, in the model's axes (+X left, +Y up, +Z forward). The hands are children of the `STEER` node, so they
 * turn with it; the forearms are in its parent and are re-aimed at the wrists after each turn (`update`).
 *
 * The hand is built to a real hand's size (a 19 cm palm-and-finger span, 85 mm across the knuckles): a palm, four fingers
 * of three phalanges each wrapped round the rim's tube from behind, over its front, a two-jointed thumb along its inner
 * face, a wrist and a shirt cuff, and a jacket sleeve tapering to the elbow. Gloves are an option (`glove`).
 *
 * When the wheel is turned further than a driver would take his hands round (`HOLD_DEG`) they slide back along the rim
 * the way a driver shuffles, so the arms stay clear of the view instead of sweeping across it.
 */
export interface PortedDriverBody {
  /** Centre of the torso (the chest; the shoulders are 0.21 m either side of it, near its top) */
  torso: [number, number, number];
  /** Centre of the helmet */
  head: [number, number, number];
  /** Racing-suit and helmet colours (CSS); the sleeve colour and the F1 helmet's scarlet by default */
  suit?: string;
  helmet?: string;
}

export interface PortedDriverSpec {
  /** Distance from the wheel's centre to the rim's centre line, each side (m): the wheel's radius at 9 and 3 o'clock */
  gripX: number;
  /** Height of the grip on the wheel (m, in the wheel's own axes; negative: below its centre) */
  gripY?: number;
  /** The rim's tube radius (m); default 0.021 */
  rimTube?: number;
  /** Size of the hands relative to a real hand; default 1 */
  handScale?: number;
  /** The elbows, in the steering node's parent's axes; `left` is on +X */
  elbows: { left: [number, number, number]; right: [number, number, number] };
  /** Bare hands' skin colour, the sleeve's, the shirt cuff's (CSS) */
  skin?: string;
  sleeve?: string;
  cuff?: string;
  /** Driving gloves instead of bare hands: their colour (CSS) and the knuckle pad's */
  glove?: string;
  pad?: string;
  /** The torso, helmet and upper arms; absent: hands and forearms only */
  body?: PortedDriverBody;
}

export interface DriverArms {
  /** Re-aims the forearms at the wrists; call after the steering node turns */
  update: () => void;
  dispose: () => void;
}

/** The steering angle up to which the hands turn with the rim; past it they slide back along it */
const HOLD_DEG = 70;
/** The furthest they slide back (deg) */
const SLIDE_MAX_DEG = 120;

const DEG = Math.PI / 180;

export function createDriverArms(spec: PortedDriverSpec, steer: THREE.Object3D, parent: THREE.Object3D): DriverArms {
  const scale = spec.handScale ?? 1;
  const tube = spec.rimTube ?? 0.021;
  const y0 = spec.gripY ?? 0;
  const skinMat = new THREE.MeshStandardMaterial({ color: spec.glove ?? spec.skin ?? "#d4a37e", roughness: spec.glove ? 0.62 : 0.7, metalness: 0.0 });
  const padMat = new THREE.MeshStandardMaterial({ color: spec.pad ?? spec.glove ?? "#d4a37e", roughness: 0.6, metalness: 0.02 });
  const sleeveMat = new THREE.MeshStandardMaterial({ color: spec.sleeve ?? "#1f2937", roughness: 0.88, metalness: 0.0 });
  const cuffMat = new THREE.MeshStandardMaterial({ color: spec.cuff ?? "#e5e7eb", roughness: 0.8, metalness: 0.0 });
  const materials: THREE.Material[] = [skinMat, padMat, sleeveMat, cuffMat];

  const geometries: THREE.BufferGeometry[] = [];
  const geo = <G extends THREE.BufferGeometry>(g: G): G => {
    geometries.push(g);
    return g;
  };
  // Unit pieces, scaled per use
  const tubeG = geo(new THREE.CylinderGeometry(1, 1, 1, 12)); // radius 1, height 1, along Y
  const ballG = geo(new THREE.SphereGeometry(1, 14, 10));
  const sleeveG = geo(new THREE.CylinderGeometry(0.043, 0.03, 1, 18)); // +Y end (the elbow) wider than the wrist end
  const cuffG = geo(new THREE.CylinderGeometry(0.032, 0.031, 1, 18));

  const Y = new THREE.Vector3(0, 1, 0);

  /** A capsule-like limb between two points: a tube and a ball at its far joint */
  const limb = (group: THREE.Object3D, a: THREE.Vector3, b: THREE.Vector3, r: number, mat: THREE.Material): void => {
    const d = new THREE.Vector3().subVectors(b, a);
    const len = d.length();
    const seg = new THREE.Mesh(tubeG, mat);
    seg.scale.set(r, len, r);
    seg.position.copy(a).addScaledVector(d, 0.5);
    seg.quaternion.setFromUnitVectors(Y, d.normalize());
    group.add(seg);
    const joint = new THREE.Mesh(ballG, mat);
    joint.scale.setScalar(r);
    joint.position.copy(b);
    group.add(joint);
  };
  const ball = (group: THREE.Object3D, p: THREE.Vector3, sx: number, sy: number, sz: number, mat: THREE.Material): THREE.Mesh => {
    const m = new THREE.Mesh(ballG, mat);
    m.scale.set(sx, sy, sz);
    m.position.copy(p);
    group.add(m);
    return m;
  };

  interface HandRig {
    /** Turns the hand about the wheel's centre (the slide back along the rim) */
    pivot: THREE.Group;
    hand: THREE.Group;
    /** The wrist, in the hand's own axes */
    wrist: THREE.Vector3;
    forearm: THREE.Group;
    sleeve: THREE.Mesh;
    elbow: THREE.Vector3;
  }
  const rigs: HandRig[] = [];

  for (const side of [-1, 1]) {
    // A point round the rim's tube, in the wheel's axes: phi 0 outward, 90 deg forward (+Z, over the front of the rim),
    // 180 inward, -90 behind it (the driver's side); at radius rho from the tube's centre line and height y
    const at = (phiDeg: number, rho: number, y: number) => {
      const phi = phiDeg * DEG;
      return new THREE.Vector3(side * (spec.gripX + rho * Math.cos(phi)), y, rho * Math.sin(phi));
    };
    const pivot = new THREE.Group();
    pivot.name = side > 0 ? "DRIVER_HAND_PIVOT_L" : "DRIVER_HAND_PIVOT_R";
    const hand = new THREE.Group();
    hand.name = side > 0 ? "DRIVER_HAND_L" : "DRIVER_HAND_R";
    hand.scale.setScalar(scale);
    // The hand is built about the rim's centre line, then scaled about the grip
    hand.position.set(side * spec.gripX * (1 - scale), 0, 0);
    pivot.add(hand);

    // Palm: a flattened ellipsoid on the back-outer side of the tube, its long axis along the rim's tangent there
    const palmPhi = -48;
    const palmCentre = at(palmPhi, tube + 0.013, y0);
    const palm = ball(hand, palmCentre, 0.052, 0.044, 0.016, skinMat);
    const tangent = new THREE.Vector3(-side * Math.sin(palmPhi * DEG), 0, Math.cos(palmPhi * DEG));
    const basis = new THREE.Matrix4().makeBasis(tangent, Y, new THREE.Vector3().crossVectors(tangent, Y));
    palm.quaternion.setFromRotationMatrix(basis);
    // The heel of the hand
    ball(hand, at(-78, tube + 0.016, y0), 0.026, 0.04, 0.02, skinMat).quaternion.copy(palm.quaternion);

    // Four fingers, index on top (+Y), three phalanges each wrapped over the front and round to the inner face
    const fingers = [
      { dy: 0.0275, scale: 1.0 },
      { dy: 0.0095, scale: 1.06 },
      { dy: -0.0085, scale: 1.0 },
      { dy: -0.0255, scale: 0.8 },
    ];
    for (const f of fingers) {
      const rho = tube + 0.0088;
      const r = 0.0083 * (f.scale > 0.9 ? 1 : 0.88);
      const phis = [-34, -34 + 66 * f.scale, -34 + 66 * f.scale + 46 * f.scale, -34 + 66 * f.scale + 46 * f.scale + 34 * f.scale];
      const pts = phis.map((p, i) => at(p, rho + (i === 0 ? 0.0035 : 0), y0 + f.dy));
      for (let i = 0; i < 3; i++) limb(hand, pts[i], pts[i + 1], r * (1 - i * 0.07), skinMat);
      // the knuckle
      ball(hand, pts[0], r * 1.12, r * 1.1, r * 1.1, padMat);
    }

    // Thumb: from the base of the palm, up the rim's inner-back face
    const thumb = [at(-82, tube + 0.02, y0 + 0.036), at(-112, tube + 0.015, y0 + 0.064), at(-142, tube + 0.0125, y0 + 0.088)];
    limb(hand, thumb[0], thumb[1], 0.0112, skinMat);
    limb(hand, thumb[1], thumb[2], 0.0098, skinMat);
    ball(hand, thumb[0], 0.014, 0.014, 0.014, padMat);

    // Wrist: behind the rim, where the forearm meets the hand
    const wristPoint = at(-100, tube + 0.034, y0 - 0.002);
    limb(hand, at(-72, tube + 0.02, y0 - 0.002), wristPoint, 0.0285, skinMat);

    rigs.push({
      pivot,
      hand,
      wrist: wristPoint,
      forearm: new THREE.Group(),
      sleeve: new THREE.Mesh(sleeveG, sleeveMat),
      elbow: new THREE.Vector3(...(side > 0 ? spec.elbows.left : spec.elbows.right)),
    });
    const rig = rigs[rigs.length - 1];
    rig.forearm.name = side > 0 ? "DRIVER_FOREARM_L" : "DRIVER_FOREARM_R";
    const cuff = new THREE.Mesh(cuffG, cuffMat);
    cuff.scale.set(1, 0.04, 1);
    cuff.position.y = 0.01;
    rig.forearm.add(cuff, rig.sleeve);
    steer.add(pivot);
    parent.add(rig.forearm);
  }

  // The rest of the driver
  let bodyGroup: THREE.Group | null = null;
  if (spec.body) {
    const b = spec.body;
    const suitMat = new THREE.MeshStandardMaterial({ color: b.suit ?? spec.sleeve ?? "#1f2937", roughness: 0.85, metalness: 0.02 });
    const shellMat = new THREE.MeshPhysicalMaterial({ color: b.helmet ?? "#dc2626", roughness: 0.3, metalness: 0.1, clearcoat: 1.0 });
    const visorMat = new THREE.MeshPhysicalMaterial({
      color: 0x0284c7,
      emissive: 0x0369a1,
      emissiveIntensity: 0.18,
      metalness: 0.92,
      roughness: 0.04,
      clearcoat: 1.0,
    });
    materials.push(suitMat, shellMat, visorMat);
    bodyGroup = new THREE.Group();
    bodyGroup.name = "DRIVER_BODY";
    const [tx, ty, tz] = b.torso;

    // Head and trunk: on the driver-head layer, so the driver's-eye view leaves them out
    const head = new THREE.Group();
    head.name = "DRIVER_HEAD";
    const torso = new THREE.Mesh(geo(new THREE.CapsuleGeometry(0.15, 0.22, 10, 16)), suitMat);
    torso.scale.set(1.35, 1, 0.8);
    torso.position.set(tx, ty, tz);
    head.add(torso);
    for (const side of [-1, 1]) {
      const shoulder = new THREE.Mesh(ballG, suitMat);
      shoulder.scale.setScalar(0.065);
      shoulder.position.set(tx + side * 0.21, ty + 0.2, tz);
      head.add(shoulder);
    }
    const neck = new THREE.Mesh(geo(new THREE.CylinderGeometry(0.055, 0.065, 0.14, 14)), suitMat);
    neck.position.set(b.head[0], (ty + 0.33 + b.head[1] - 0.1) / 2, (tz + b.head[2]) / 2);
    head.add(neck);
    const helmet = new THREE.Group();
    helmet.position.set(...b.head);
    const shell = new THREE.Mesh(geo(new THREE.SphereGeometry(0.125, 24, 18)), shellMat);
    shell.scale.set(0.95, 1.0, 1.08);
    helmet.add(shell);
    // As the F1 helmet's visor: a band of a slightly larger sphere, centred on +Z
    const visor = new THREE.Mesh(
      geo(new THREE.SphereGeometry(0.1285, 24, 14, Math.PI * 0.14, Math.PI * 0.72, Math.PI * 0.28, Math.PI * 0.25)),
      visorMat
    );
    visor.scale.set(0.95, 1.0, 1.08);
    helmet.add(visor);
    head.add(helmet);
    head.traverse((o: THREE.Object3D) => {
      if ((o as THREE.Mesh).isMesh) o.castShadow = true;
    });
    moveToDriverHeadLayer(head);
    bodyGroup.add(head);

    // Upper arms, shoulder to elbow (fixed): they stay in the driver's-eye view, at its lower corners
    const upperG = geo(new THREE.CylinderGeometry(0.04, 0.034, 1.0, 12));
    for (const side of [-1, 1]) {
      const from = new THREE.Vector3(tx + side * 0.21, ty + 0.2, tz);
      const to = rigs[side > 0 ? 1 : 0].elbow;
      const arm = new THREE.Mesh(upperG, suitMat);
      const d = new THREE.Vector3().subVectors(to, from);
      arm.scale.set(1, d.length(), 1);
      arm.position.copy(from).addScaledVector(d, 0.5);
      arm.quaternion.setFromUnitVectors(Y, d.normalize());
      arm.castShadow = true;
      bodyGroup.add(arm);
    }
    parent.add(bodyGroup);
  }

  // ---- each frame
  const rest = steer.quaternion.clone();
  const rel = new THREE.Quaternion();
  const wrist = new THREE.Vector3();
  const dir = new THREE.Vector3();
  const update = () => {
    // The wheel's turn about its column, from its rest pose
    rel.copy(rest).invert().multiply(steer.quaternion);
    let theta = 2 * Math.atan2(rel.z, rel.w);
    if (theta > Math.PI) theta -= 2 * Math.PI;
    if (theta < -Math.PI) theta += 2 * Math.PI;
    // The hands go round with the rim up to HOLD_DEG, then slide back along it
    const held = THREE.MathUtils.clamp(theta, -HOLD_DEG * DEG, HOLD_DEG * DEG);
    const slide = THREE.MathUtils.clamp(held - theta, -SLIDE_MAX_DEG * DEG, SLIDE_MAX_DEG * DEG);
    steer.updateMatrix();
    for (const rig of rigs) {
      rig.pivot.rotation.z = slide;
      rig.pivot.updateMatrix();
      rig.hand.updateMatrix();
      wrist.copy(rig.wrist).applyMatrix4(rig.hand.matrix).applyMatrix4(rig.pivot.matrix).applyMatrix4(steer.matrix);
      dir.subVectors(rig.elbow, wrist);
      const len = Math.max(0.06, dir.length());
      rig.forearm.position.copy(wrist);
      rig.forearm.quaternion.setFromUnitVectors(Y, dir.normalize());
      // The cuff covers the first 4 cm; the sleeve runs on to the elbow
      rig.sleeve.scale.set(1, Math.max(0.02, len - 0.035), 1);
      rig.sleeve.position.y = 0.035 + (len - 0.035) / 2;
    }
  };
  update();

  return {
    update,
    dispose: () => {
      for (const r of rigs) {
        r.pivot.removeFromParent();
        r.forearm.removeFromParent();
      }
      bodyGroup?.removeFromParent();
      for (const g of geometries) g.dispose();
      for (const m of materials) m.dispose();
    },
  };
}
