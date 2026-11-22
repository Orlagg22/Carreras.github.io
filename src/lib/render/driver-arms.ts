import * as THREE from "three";

/**
 * The game's driver body for a ported car: the gloved hands and forearms of the F1 car's driver
 * (`car-builder-2026.ts`, "Contoured 3D Driver Gloves ... Elbow-to-Wrist Forearms"), built from the same pieces: a
 * sculpted glove (dorsal shell, knuckle pad, four curled fingers, thumb, cuff) on each side of the wheel, and a forearm
 * from a fixed elbow in the cockpit to the wrist that follows the wheel as it turns.
 *
 * A ported GLB opts in with a `driver` entry in its `CAR` extras (`PortedDriverSpec`): where the hands grip the rim and
 * where the elbows are, in the model's axes (+X left, +Y up, +Z forward). The hands are children of the `STEER` node,
 * so they turn with it; the forearms are in its parent and are re-aimed after each turn (`update`).
 */
export interface PortedDriverSpec {
  /** Distance from the wheel's centre to the rim's grip, each side (m) */
  gripX: number;
  /** Height of the grip on the wheel (m, in the wheel's own axes; negative: below its centre) */
  gripY?: number;
  /** Size of the hands, 1 being the F1 driver's (a thicker rim wants bigger hands) */
  handScale?: number;
  /** The elbows, in the steering node's parent's axes; `left` is on +X */
  elbows: { left: [number, number, number]; right: [number, number, number] };
  /** Glove, knuckle-pad and sleeve colours (CSS); the F1 driver's glove yellow and a dark pad by default */
  glove?: string;
  pad?: string;
  sleeve?: string;
}

export interface DriverArms {
  /** Re-aims the forearms at the wrists; call after the steering node turns */
  update: () => void;
  dispose: () => void;
}

const GRIP_Z = -0.006;

export function createDriverArms(spec: PortedDriverSpec, steer: THREE.Object3D, parent: THREE.Object3D): DriverArms {
  const s = spec.handScale ?? 1;
  const gripY = spec.gripY ?? -0.014;
  const gloveMat = new THREE.MeshStandardMaterial({ color: spec.glove ?? "#eab308", roughness: 0.48, metalness: 0.05 });
  const padMat = new THREE.MeshStandardMaterial({ color: spec.pad ?? "#18181b", roughness: 0.68, metalness: 0.06 });
  const sleeveMat = new THREE.MeshStandardMaterial({ color: spec.sleeve ?? "#1d3a2f", roughness: 0.85, metalness: 0.02 });

  const geometries: THREE.BufferGeometry[] = [];
  const geo = <G extends THREE.BufferGeometry>(g: G): G => {
    geometries.push(g);
    return g;
  };
  // One set of pieces for both hands
  const dorsalG = geo(new THREE.SphereGeometry(0.021, 18, 16));
  const knuckleG = geo(new THREE.SphereGeometry(0.0185, 14, 12));
  const fingerG = geo(new THREE.CapsuleGeometry(0.0055, 0.018, 8, 12));
  const fingerTipG = geo(new THREE.CapsuleGeometry(0.0048, 0.013, 8, 12));
  const thumbG = geo(new THREE.CapsuleGeometry(0.0072, 0.02, 8, 12));
  const thumbTipG = geo(new THREE.CapsuleGeometry(0.0062, 0.014, 8, 12));
  const cuffG = geo(new THREE.CapsuleGeometry(0.017, 0.026, 10, 14));
  const bandG = geo(new THREE.CylinderGeometry(0.0178, 0.0182, 0.012, 16));
  const forearmG = geo(new THREE.CylinderGeometry(0.0175 * s, 0.023 * s, 1.0, 14));

  const hands: THREE.Group[] = [];
  const wrists: THREE.Object3D[] = [];
  const forearms: THREE.Mesh[] = [];
  const elbows: THREE.Vector3[] = [];

  for (const side of [-1, 1]) {
    const hand = new THREE.Group();
    hand.name = side > 0 ? "DRIVER_HAND_L" : "DRIVER_HAND_R";
    hand.position.set(side * spec.gripX, gripY, GRIP_Z);
    hand.scale.setScalar(s);

    const dorsal = new THREE.Mesh(dorsalG, gloveMat);
    dorsal.scale.set(0.82, 1.25, 0.92);
    dorsal.position.set(side * 0.004, -0.002, -0.005);
    hand.add(dorsal);

    const knuckle = new THREE.Mesh(knuckleG, padMat);
    knuckle.scale.set(0.75, 1.02, 0.82);
    knuckle.position.set(side * 0.006, 0.002, -0.01);
    hand.add(knuckle);

    // Four curled fingers wrapping the front (+Z) of the rim
    for (let f = 0; f < 4; f++) {
      const fy = 0.016 - f * 0.011;
      const proximal = new THREE.Mesh(fingerG, gloveMat);
      proximal.rotation.x = Math.PI * 0.48;
      proximal.rotation.y = side * -0.25;
      proximal.position.set(side * 0.003, fy, 0.011);
      hand.add(proximal);
      const tip = new THREE.Mesh(fingerTipG, padMat);
      tip.rotation.z = Math.PI * 0.5;
      tip.position.set(-side * 0.007, fy, 0.018);
      hand.add(tip);
    }

    const thumb = new THREE.Mesh(thumbG, gloveMat);
    thumb.position.set(-side * 0.013, 0.019, -0.004);
    thumb.rotation.z = side * 0.62;
    thumb.rotation.x = 0.28;
    hand.add(thumb);
    const thumbTip = new THREE.Mesh(thumbTipG, padMat);
    thumbTip.position.set(-side * 0.023, 0.026, 0.004);
    thumbTip.rotation.x = Math.PI * 0.38;
    hand.add(thumbTip);

    const cuff = new THREE.Mesh(cuffG, gloveMat);
    cuff.position.set(0, -0.026, -0.01);
    cuff.rotation.x = 0.42;
    hand.add(cuff);
    const band = new THREE.Mesh(bandG, sleeveMat);
    band.position.set(0, -0.032, -0.013);
    band.rotation.x = 0.42;
    hand.add(band);

    const wrist = new THREE.Object3D();
    wrist.position.set(0, -0.036, -0.015);
    hand.add(wrist);

    steer.add(hand);
    hands.push(hand);
    wrists.push(wrist);

    const forearm = new THREE.Mesh(forearmG, sleeveMat);
    forearm.name = side > 0 ? "DRIVER_FOREARM_L" : "DRIVER_FOREARM_R";
    parent.add(forearm);
    forearms.push(forearm);
    elbows.push(new THREE.Vector3(...(side > 0 ? spec.elbows.left : spec.elbows.right)));
  }

  const up = new THREE.Vector3(0, 1, 0);
  const wristPos = new THREE.Vector3();
  const dir = new THREE.Vector3();
  const update = () => {
    steer.updateMatrix();
    for (let i = 0; i < 2; i++) {
      wristPos.copy(wrists[i].position).multiplyScalar(s).add(hands[i].position).applyMatrix4(steer.matrix);
      dir.subVectors(wristPos, elbows[i]);
      const len = Math.max(0.05, dir.length());
      const forearm = forearms[i];
      forearm.scale.set(1, len, 1);
      forearm.position.copy(elbows[i]).addScaledVector(dir, 0.5);
      forearm.quaternion.setFromUnitVectors(up, dir.normalize());
    }
  };
  update();

  return {
    update,
    dispose: () => {
      for (const h of hands) h.removeFromParent();
      for (const f of forearms) f.removeFromParent();
      for (const g of geometries) g.dispose();
      gloveMat.dispose();
      padMat.dispose();
      sleeveMat.dispose();
    },
  };
}
