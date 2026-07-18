// The navball — a KSP-style artificial horizon at the bottom of the drive
// view. The ball is textured in SCENE axes (horizon, pitch ladder, compass
// letters) and rotated by conj(q_scene_from_craft), so it shows the world
// as seen from inside the hull: nose into the screen, screen-up = hull-up.
// A chevron marker rides the sphere at the orientation setpoint's forward
// direction — fly the ball onto the chevron.

import { qmul, qconj, qrot, qfromRotmat } from "./control.js";

// craft basis → navball view: +x (fwd) into the screen, +y (left) to
// screen-left, +z (up) to screen-up.
const Q_VIEW = qfromRotmat([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]);

function paintBallTexture() {
  const W = 1024, H = 512;
  const cv = document.createElement("canvas");
  cv.width = W; cv.height = H;
  const g = cv.getContext("2d");

  // upper (sky/water) and lower (ground) hemispheres
  const sky = g.createLinearGradient(0, 0, 0, H / 2);
  sky.addColorStop(0, "#7fb4d6"); sky.addColorStop(1, "#4f88b0");
  g.fillStyle = sky; g.fillRect(0, 0, W, H / 2);
  const gnd = g.createLinearGradient(0, H / 2, 0, H);
  gnd.addColorStop(0, "#8a6a45"); gnd.addColorStop(1, "#4c3a26");
  g.fillStyle = gnd; g.fillRect(0, H / 2, W, H / 2);

  // horizon
  g.fillStyle = "#f5f5f0"; g.fillRect(0, H / 2 - 2, W, 4);

  // pitch ladder every 15° (texture v = latitude from the top pole)
  g.textAlign = "center"; g.textBaseline = "middle";
  for (let p = -75; p <= 75; p += 15) {
    if (p === 0) continue;
    const y = H / 2 - (p / 180) * H;
    g.strokeStyle = "rgba(245,245,240,.75)";
    g.lineWidth = 2;
    for (let u = 0; u < 8; u++) {              // dashes around the band
      const x0 = (u / 8) * W + W / 64;
      g.beginPath(); g.moveTo(x0, y); g.lineTo(x0 + W / 16, y); g.stroke();
    }
    g.fillStyle = "rgba(245,245,240,.9)";
    g.font = "22px system-ui";
    for (let u = 0; u < 4; u++)
      g.fillText(`${Math.abs(p)}`, (u / 4) * W + W / 8, y);
  }

  // compass letters + yaw ticks on the horizon band. The +0.5 u-shift puts
  // the u=0 letter at the texture's centre column — the one facing the
  // camera at yaw 0 with the ball's seam rotation.
  g.font = "bold 40px system-ui";
  const letters = [["N", 0], ["W", 0.25], ["S", 0.5], ["E", 0.75]];
  for (const [ch, u] of letters) {
    g.fillStyle = "#ffe08a";
    g.fillText(ch, ((u + 0.5) % 1) * W, H / 2 - 26);
  }
  g.strokeStyle = "rgba(245,245,240,.8)";
  for (let i = 0; i < 24; i++) {
    const x = (i / 24) * W;
    g.lineWidth = i % 6 === 0 ? 4 : 2;
    g.beginPath();
    g.moveTo(x, H / 2 - (i % 6 === 0 ? 16 : 9));
    g.lineTo(x, H / 2 + (i % 6 === 0 ? 16 : 9));
    g.stroke();
  }
  return cv;
}

export function createNavball(THREE, canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true,
                                             alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 10);
  camera.position.set(0, 0, 3.1);
  scene.add(new THREE.AmbientLight(0xffffff, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 1.4);
  key.position.set(0.6, 1.2, 2.0);
  scene.add(key);

  const holder = new THREE.Group();            // gets the attitude quat
  scene.add(holder);
  const tex = new THREE.CanvasTexture(paintBallTexture());
  tex.anisotropy = 4;
  const ball = new THREE.Mesh(
    new THREE.SphereGeometry(1, 48, 32),
    new THREE.MeshStandardMaterial({ map: tex, roughness: 0.75 }));
  // sphere poles are along geometry +y; stand it up so +y_geo = scene +z
  // and the texture's u=0 seam lands on scene +x (north).
  ball.rotation.set(Math.PI / 2, 0, 0);
  holder.add(ball);

  // setpoint chevron riding the sphere surface (cone nose outward: +z
  // after the rotateX so lookAt() aims it)
  const markerGeo = new THREE.ConeGeometry(0.09, 0.16, 4);
  markerGeo.rotateX(Math.PI / 2);
  const marker = new THREE.Mesh(
    markerGeo,
    new THREE.MeshBasicMaterial({ color: 0xff4fa0 }));
  holder.add(marker);
  const anti = new THREE.Mesh(                 // dim marker for "behind you"
    new THREE.SphereGeometry(0.05, 8, 6),
    new THREE.MeshBasicMaterial({ color: 0xff4fa0, transparent: true,
                                  opacity: 0.35 }));
  holder.add(anti);

  // fixed bezel furniture (view space): center reticle + top roll caret
  const ret = new THREE.Group();
  const retMat = new THREE.MeshBasicMaterial({ color: 0xffc23e });
  const bar = (w, h, x, y) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), retMat);
    m.position.set(x, y, 1.05);
    ret.add(m);
  };
  bar(0.34, 0.035, -0.32, 0); bar(0.34, 0.035, 0.32, 0);
  bar(0.035, 0.12, -0.15, -0.045); bar(0.035, 0.12, 0.15, -0.045);
  bar(0.08, 0.08, 0, 0);
  const caret = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.1, 3), retMat);
  caret.position.set(0, 1.06, 0.4); caret.rotation.z = Math.PI;
  ret.add(caret);
  scene.add(ret);

  const toQ = (q) => new THREE.Quaternion(q[1], q[2], q[3], q[0]);

  return {
    /** qSC = scene-from-craft quat; qSp = setpoint quat (or null to hide
     *  the marker). */
    update(qSC, qSp) {
      const qBall = qmul(Q_VIEW, qconj(qSC));
      holder.quaternion.copy(toQ(qBall));
      // counteract the parent so the marker can be placed in HOLDER space
      // directly at the setpoint's forward direction (scene frame).
      if (qSp) {
        marker.visible = anti.visible = true;
        const f = qrot(qSp, [1, 0, 0]);
        marker.position.set(f[0] * 1.02, f[1] * 1.02, f[2] * 1.02);
        marker.lookAt(holder.localToWorld(new THREE.Vector3(
          f[0] * 2, f[1] * 2, f[2] * 2)));
        anti.position.set(-f[0] * 1.01, -f[1] * 1.01, -f[2] * 1.01);
        // hide whichever is on the far side
        const fv = qrot(qBall, f);
        marker.visible = fv[2] > -0.15;
        anti.visible = !marker.visible;
      } else {
        marker.visible = anti.visible = false;
      }
      const s = Math.min(canvas.clientWidth, canvas.clientHeight) || 150;
      if (canvas.width !== s * renderer.getPixelRatio())
        renderer.setSize(s, s, false);
      renderer.render(scene, camera);
    },
    dispose() { renderer.dispose(); tex.dispose(); },
  };
}
