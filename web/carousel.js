// Example carousel for the manta homepage.
//
// Owns all carousel UI — code blurb, render frame, arrows, dots, the
// skeleton toggle and click-to-fly — and delegates the live render to a
// per-example module under `module:`, loaded lazily when its slide is shown
// and in view. Each renderer exports:
//
//     start(canvas, ctx) -> controller
//       ctx = { setHud(text), setSkeleton?(bool) }   // helpers from the page
//       controller = {
//         dispose(),                 // tear down (GL + listeners)
//         setSkeleton(on),           // toggle the overlay
//         setInteractive(on),        // hand control to / from the keyboard
//       }
//
// A renderer that throws (no WASM yet, no WebGL2) drops the slide to a clean
// fallback message — the rest of the page is unaffected.

const EX = [
  {
    id: "quad",
    name: "Quadcopter",
    file: "quad.py",
    title: "quadcopter",
    module: "/quad/demo.js",
    keys: [["W A S D", "move"], ["space / shift", "up · down"]],
    code:
`<span class="c-kw">from</span> <span class="c-mod">manta</span> <span class="c-kw">import</span> Craft, EKF, Sim, TargetWasm, World
<span class="c-kw">from</span> <span class="c-mod">manta.fields</span> <span class="c-kw">import</span> GravityField
<span class="c-kw">from</span> <span class="c-mod">manta.parts</span> <span class="c-kw">import</span> IMU, DragSurface, Mass, PositionSensor, Thruster

<span class="c-com"># X-frame: each rotor lifts along +z; counter-rotating</span>
<span class="c-com"># pairs let differential throttle yaw the airframe.</span>
ARM, MAXT, KYAW = <span class="c-num">0.25</span>, <span class="c-num">6.0</span>, <span class="c-num">0.05</span>
ROTORS = {  <span class="c-com"># mount point + spin sign (CCW = +1)</span>
    <span class="c-str">"front"</span>: ((<span class="c-num">+ARM</span>, <span class="c-num">0</span>, <span class="c-num">0</span>), <span class="c-num">+1</span>), <span class="c-str">"right"</span>: ((<span class="c-num">0</span>, <span class="c-num">+ARM</span>, <span class="c-num">0</span>), <span class="c-num">-1</span>),
    <span class="c-str">"back"</span>:  ((<span class="c-num">-ARM</span>, <span class="c-num">0</span>, <span class="c-num">0</span>), <span class="c-num">+1</span>), <span class="c-str">"left"</span>:  ((<span class="c-num">0</span>, <span class="c-num">-ARM</span>, <span class="c-num">0</span>), <span class="c-num">-1</span>),
}

quad = <span class="c-cls">Craft</span>(<span class="c-str">"quad"</span>)
quad.<span class="c-cls">add</span>(<span class="c-cls">Mass</span>(<span class="c-str">"body"</span>, mass=<span class="c-num">1.0</span>, moi=(<span class="c-num">.01</span>, <span class="c-num">.01</span>, <span class="c-num">.02</span>)))
<span class="c-kw">for</span> name, (mount, spin) <span class="c-kw">in</span> ROTORS.items():
    quad.<span class="c-cls">add</span>(<span class="c-cls">Thruster</span>(name, force=(<span class="c-num">0</span>, <span class="c-num">0</span>, MAXT),
                      torque=(<span class="c-num">0</span>, <span class="c-num">0</span>, spin * KYAW * MAXT),
                      transform=mount))
quad.<span class="c-cls">add</span>(<span class="c-cls">DragSurface</span>(<span class="c-str">"drag"</span>, force=(<span class="c-num">-.2</span>, <span class="c-num">-.2</span>, <span class="c-num">-.3</span>)))
quad.<span class="c-cls">add</span>(<span class="c-cls">IMU</span>(<span class="c-str">"imu"</span>, gyro_noise_sigma=<span class="c-num">.01</span>))
quad.<span class="c-cls">add</span>(<span class="c-cls">PositionSensor</span>(<span class="c-str">"gps"</span>, position_noise_sigma=<span class="c-num">.04</span>))

world = <span class="c-cls">World</span>().<span class="c-cls">add_field</span>(<span class="c-cls">GravityField</span>(g=(<span class="c-num">0</span>, <span class="c-num">0</span>, <span class="c-num">-9.81</span>)))
world.<span class="c-cls">add_craft</span>(quad, position=(<span class="c-num">0</span>, <span class="c-num">0</span>, <span class="c-num">2</span>))

<span class="c-com"># lower to WebAssembly — this exact model runs in your browser</span>
sim = <span class="c-cls">TargetWasm</span>(<span class="c-cls">Sim</span>(world))     <span class="c-com"># truth</span>
ekf = <span class="c-cls">TargetWasm</span>(<span class="c-cls">EKF</span>(world))     <span class="c-com"># state estimate, same model</span></span>`,
  },
  {
    id: "airplane",
    name: "Cessna 172",
    file: "airplane.py",
    title: "cessna 172",
    module: "/airplane/demo.js",
    keys: [["W S", "pitch"], ["A D", "rudder"], ["Q E", "roll"], ["X Z", "throttle"]],
    code:
`<span class="c-kw">from</span> <span class="c-mod">manta</span> <span class="c-kw">import</span> Craft, EKF, Sim, TargetWasm, World
<span class="c-kw">from</span> <span class="c-mod">manta.fields</span> <span class="c-kw">import</span> CollisionField, FluidField, GravityField
<span class="c-kw">from</span> <span class="c-mod">manta.parts</span> <span class="c-kw">import</span> Collider, ControlSurface, DragSurface, Mass, Thruster, naca

<span class="c-com"># A Cessna 172 from a re-measured airframe survey: a NACA 2412</span>
<span class="c-com"># wing, every control a deflectable flap, statically stable on its own.</span>
plane = <span class="c-cls">Craft</span>(<span class="c-str">"plane"</span>)
plane.<span class="c-cls">add</span>(<span class="c-cls">Mass</span>(<span class="c-str">"body"</span>, mass=<span class="c-num">1043</span>, moi=(<span class="c-num">1742</span>, <span class="c-num">2475</span>, <span class="c-num">3616</span>), transform=(<span class="c-num">-2.45</span>, <span class="c-num">0</span>, <span class="c-num">-.2</span>)))

<span class="c-com"># Main wing — split into panels that tile the span at +1.5° incidence.</span>
plane.<span class="c-cls">add</span>(<span class="c-cls">naca</span>(<span class="c-str">"2412"</span>, <span class="c-str">"wing_l"</span>, area=<span class="c-num">5.6</span>, chord=<span class="c-num">1.49</span>, transform=(<span class="c-num">-2.44</span>, <span class="c-num">+1.5</span>, <span class="c-num">.85</span>)))
plane.<span class="c-cls">add</span>(<span class="c-cls">naca</span>(<span class="c-str">"2412"</span>, <span class="c-str">"wing_r"</span>, area=<span class="c-num">5.6</span>, chord=<span class="c-num">1.49</span>, transform=(<span class="c-num">-2.44</span>, <span class="c-num">-1.5</span>, <span class="c-num">.85</span>)))

<span class="c-com"># Ailerons, elevator, rudder — each a wing section with a trailing-edge flap.</span>
<span class="c-kw">for</span> name, sy <span class="c-kw">in</span> ((<span class="c-str">"ail_l"</span>, <span class="c-num">+1</span>), (<span class="c-str">"ail_r"</span>, <span class="c-num">-1</span>)):
    plane.<span class="c-cls">add</span>(<span class="c-cls">ControlSurface</span>(name, area=<span class="c-num">2.7</span>, chord=<span class="c-num">1.49</span>, flap_chord_fraction=<span class="c-num">.31</span>,
                              transform=(<span class="c-num">-2.44</span>, sy * <span class="c-num">3.9</span>, <span class="c-num">.85</span>)))
plane.<span class="c-cls">add</span>(<span class="c-cls">ControlSurface</span>(<span class="c-str">"elev"</span>, area=<span class="c-num">2.0</span>, chord=<span class="c-num">.6</span>,  flap_chord_fraction=<span class="c-num">.45</span>, transform=(<span class="c-num">-7.05</span>, <span class="c-num">0</span>, <span class="c-num">.25</span>)))
plane.<span class="c-cls">add</span>(<span class="c-cls">ControlSurface</span>(<span class="c-str">"rud"</span>,  area=<span class="c-num">1.1</span>, chord=<span class="c-num">.95</span>, flap_chord_fraction=<span class="c-num">.53</span>, transform=(<span class="c-num">-6.79</span>, <span class="c-num">0</span>, <span class="c-num">.9</span>)))

plane.<span class="c-cls">add</span>(<span class="c-cls">DragSurface</span>(<span class="c-str">"fuselage"</span>, force=(<span class="c-num">-3.2</span>, <span class="c-num">-46</span>, <span class="c-num">-58</span>), transform=(<span class="c-num">-2.3</span>, <span class="c-num">0</span>, <span class="c-num">0</span>)))
plane.<span class="c-cls">add</span>(<span class="c-cls">Thruster</span>(<span class="c-str">"prop"</span>, force=(<span class="c-num">1300</span>, <span class="c-num">0</span>, <span class="c-num">0</span>), torque=(<span class="c-num">-410</span>, <span class="c-num">0</span>, <span class="c-num">0</span>)))
<span class="c-kw">for</span> name, pos <span class="c-kw">in</span> GEAR.items():
    plane.<span class="c-cls">add</span>(<span class="c-cls">Collider</span>(name, stiffness=<span class="c-num">1.5e5</span>, friction=(<span class="c-num">0</span>, <span class="c-num">6000</span>, <span class="c-num">0</span>), transform=pos))

world = (<span class="c-cls">World</span>()
    .<span class="c-cls">add_field</span>(<span class="c-cls">GravityField</span>().<span class="c-cls">add_uniform</span>((<span class="c-num">0</span>, <span class="c-num">0</span>, <span class="c-num">-9.81</span>)))
    .<span class="c-cls">add_field</span>(<span class="c-cls">FluidField</span>().<span class="c-cls">add_uniform</span>(density=<span class="c-num">1.225</span>))
    .<span class="c-cls">add_field</span>(<span class="c-cls">CollisionField</span>().<span class="c-cls">add_half_space</span>(normal=(<span class="c-num">0</span>, <span class="c-num">0</span>, <span class="c-num">1</span>))))
world.<span class="c-cls">add_craft</span>(plane, position=(<span class="c-num">0</span>, <span class="c-num">0</span>, <span class="c-num">1.23</span>))

<span class="c-com"># lower to WebAssembly — this exact model runs in your browser</span>
sim = <span class="c-cls">TargetWasm</span>(<span class="c-cls">Sim</span>(world))     <span class="c-com"># truth</span>
ekf = <span class="c-cls">TargetWasm</span>(<span class="c-cls">EKF</span>(world))     <span class="c-com"># state estimate, same model</span></span>`,
  },
];

// --- DOM refs ---------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const els = {
  stage: $("renderStage"), canvas: $("renderCanvas"), hud: $("renderHud"),
  keys: $("renderKeys"), code: $("codeBlock"), codeFile: $("codeFile"),
  renderTitle: $("renderTitle"), exName: $("exName"), dots: $("dots"),
  fallback: $("renderFallback"), skel: $("skelToggle"), clickFly: $("clickFly"),
  prev: $("prevBtn"), next: $("nextBtn"),
};

let idx = 0;
let ctrl = null;       // active renderer controller
let booting = null;    // in-flight boot promise (guards double-boot)
let visible = false;   // carousel in viewport
let skeleton = false;

function setHud(text) { els.hud.textContent = text; }

function renderStatic(i) {
  const ex = EX[i];
  els.code.innerHTML = ex.code;
  els.codeFile.textContent = ex.file;
  els.renderTitle.textContent = ex.title;
  els.exName.innerHTML = `<b>${ex.name}</b> · ${i + 1}/${EX.length}`;
  els.keys.innerHTML = ex.keys
    .map(([k, v]) => `<span><kbd>${k}</kbd> ${v}</span>`).join("");
  [...els.dots.children].forEach((d, j) =>
    d.setAttribute("aria-current", j === i ? "true" : "false"));
}

async function bootRenderer(i) {
  const ex = EX[i];
  els.stage.classList.remove("failed", "flying");
  setHud("booting…");
  if (!window.WebGL2RenderingContext) return fail("live render needs WebGL2");
  booting = (async () => {
    try {
      const mod = await import(ex.module);
      // Renderer owns the loop; only boot when still the active, visible slide.
      if (i !== idx || !visible) return;
      ctrl = await mod.start(els.canvas, { setHud });
      ctrl.setSkeleton?.(skeleton);
    } catch (err) {
      console.error("[carousel]", ex.id, err);
      fail("live render building — check back soon");
    }
  })();
  return booting;
}

function fail(msg) { els.stage.classList.add("failed"); els.fallback.textContent = msg; }

function teardown() {
  try { ctrl?.dispose?.(); } catch (e) { /* ignore */ }
  ctrl = null; booting = null;
  els.stage.classList.remove("flying");
  els.clickFly && (els.clickFly.parentElement.style.opacity = "");
}

function go(i) {
  i = (i + EX.length) % EX.length;
  if (i === idx && ctrl) return;
  teardown();
  idx = i;
  renderStatic(i);
  if (visible) bootRenderer(i);
}

// --- wiring -----------------------------------------------------------------
EX.forEach((_, j) => {
  const b = document.createElement("button");
  b.setAttribute("aria-label", `Show example ${j + 1}`);
  b.addEventListener("click", () => go(j));
  els.dots.appendChild(b);
});
els.prev.addEventListener("click", () => go(idx - 1));
els.next.addEventListener("click", () => go(idx + 1));

els.skel.addEventListener("click", () => {
  skeleton = !skeleton;
  els.skel.setAttribute("aria-pressed", String(skeleton));
  els.skel.textContent = skeleton ? "hide skeleton" : "show skeleton";
  ctrl?.setSkeleton?.(skeleton);
});

// Click the stage → hand control to the keyboard.
els.stage.addEventListener("pointerdown", () => {
  if (!ctrl) return;
  els.stage.classList.add("flying");
  ctrl.setInteractive?.(true);
});

// Boot/suspend with viewport visibility so off-screen slides don't burn CPU.
if ("IntersectionObserver" in window) {
  const io = new IntersectionObserver((es) => {
    const now = es.some((e) => e.isIntersecting);
    if (now && !visible) { visible = true; if (!ctrl && !booting) bootRenderer(idx); }
    else if (!now && visible) { visible = false; teardown(); }
  }, { rootMargin: "120px 0px" });
  io.observe(els.stage);
} else { visible = true; bootRenderer(idx); }

renderStatic(0);
