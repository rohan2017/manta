// Shared Three.js helpers for the homepage example renderers.
//
// Frame convention matches manta's WorldFrame: x forward, y left, z up,
// metres. Quaternions are [w, x, y, z]; Three wants [x, y, z, w].

import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";
export { THREE };

// Site palette (kept in sync with index.html).
export const C = {
  bg: 0xd0e0e3, ground: 0xc3d6da, grid: 0xa9c2c8,
  navy: 0x1c4587, blue: 0x3d85c6, ink: 0x0d1417,
  frame: 0x9fb2b8, surface: 0x3d85c6, accent: 0x1c4587,
  thrust: 0x1c4587, lift: 0x3d85c6, drag: 0xc23b3b, gravity: 0x6b7d85,
};

export const D2R = Math.PI / 180;
export const lerp = (a, b, k) => a + (b - a) * k;

// manta [w,x,y,z] at offset `off` → THREE.Quaternion(x,y,z,w).
export function quatWXYZ(arr, off = 0) {
  return new THREE.Quaternion(arr[off + 1], arr[off + 2], arr[off + 3], arr[off]);
}

// Yaw (heading about world z) from a THREE.Quaternion.
export function yawOf(q) {
  return Math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
}

// A box given FULL extent (sx,sy,sz) centred at `center` in its parent frame.
export function box(parent, size, center, color, opts = {}) {
  const m = new THREE.Mesh(
    new THREE.BoxGeometry(size[0], size[1], size[2]),
    new THREE.MeshStandardMaterial({ color, metalness: 0.05, roughness: 0.7, ...opts }));
  m.position.set(center[0], center[1], center[2]);
  m.castShadow = !opts.transparent; m.receiveShadow = true;
  parent.add(m);
  return m;
}

// A wireframe box (used for the skeleton/drag-volume overlay).
export function wireBox(parent, size, center, color, opts = {}) {
  const g = new THREE.BoxGeometry(size[0], size[1], size[2]);
  const m = new THREE.LineSegments(
    new THREE.EdgesGeometry(g),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: opts.opacity ?? 0.9 }));
  m.position.set(center[0], center[1], center[2]);
  parent.add(m);
  return m;
}

// A force/vector arrow rooted at `origin` (parent frame), pointing `vec`.
// Returned object has `.set(origin, vec)` to re-aim it each frame.
export function arrow(parent, color, opts = {}) {
  const a = new THREE.ArrowHelper(
    new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 1, color,
    opts.head ?? 0.16, opts.headW ?? 0.1);
  a.line.material.linewidth = 2;
  a.cone.material.transparent = a.line.material.transparent = true;
  parent.add(a);
  const scale = opts.scale ?? 1;
  return {
    obj: a,
    set(origin, vec) {
      const v = new THREE.Vector3(vec[0], vec[1], vec[2]).multiplyScalar(scale);
      const len = v.length();
      a.visible = len > 1e-3;
      if (!a.visible) return;
      a.position.set(origin[0], origin[1], origin[2]);
      a.setDirection(v.clone().normalize());
      a.setLength(len, Math.min(0.18, len * 0.32), Math.min(0.12, len * 0.22));
    },
  };
}

export function makeScene(canvas, opts = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(opts.fov ?? 50, 1, 0.05, 4000);
  camera.up.set(0, 0, 1);

  scene.add(new THREE.HemisphereLight(0xffffff, 0x9fb2b8, 1.05));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  sun.position.set(30, 50, 90);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.near = 1; sun.shadow.camera.far = 600;
  const sc = sun.shadow.camera;
  sc.left = sc.bottom = -(opts.shadowSpan ?? 80);
  sc.right = sc.top = (opts.shadowSpan ?? 80);
  scene.add(sun);
  return { renderer, scene, camera, sun };
}

// A large ground plane with a faint grid for motion reference.
export function ground(scene, { z = 0, size = 4000, grid = 400, cells = 80 } = {}) {
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(size, size),
    new THREE.MeshStandardMaterial({ color: C.ground, roughness: 1 }));
  plane.position.z = z; plane.receiveShadow = true; scene.add(plane);
  const g = new THREE.GridHelper(grid, cells, C.grid, C.grid);
  g.rotation.x = Math.PI / 2; g.position.z = z + 0.01;
  g.material.transparent = true; g.material.opacity = 0.5;
  scene.add(g);
  return { plane, grid: g };
}

// Keep the drawing buffer matched to the canvas's CSS box.
export function fitToCanvas(renderer, camera, canvas) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (w === 0 || h === 0) return;
  if (canvas.width !== w || canvas.height !== h) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }
}
