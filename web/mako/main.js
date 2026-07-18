// Page glue: editor ↔ spawn API ↔ drive view.

import { createEditor } from "./editor.js";
import { startViewer } from "./viewer.js";

const $ = (id) => document.getElementById(id);

const editor = createEditor({
  svg: $("editor-svg"),
  paletteGroups: $("palette-groups"),
  search: $("palette-search"),
  config: $("config-panel"),
  stats: $("stats"),
});

let viewer = null;
let bundle = null;   // {base, meta} of the spawned bundle driving the viewer

// --- drive-view chrome -------------------------------------------------------

const MODE_LABEL = { manual: "manual", ori: "ori hold", auto: "auto" };
const deg = (r) => (r * 180 / Math.PI).toFixed(0);

const fmtLL = (v, pos, neg) =>
  `${Math.abs(v).toFixed(5)}°${v >= 0 ? pos : neg}`;

function renderHud(d) {
  const sp = d.rpySp;
  $("readouts").textContent =
    `depth ${d.depth.toFixed(1)} m\n` +
    `pos   ${fmtLL(d.lat, "N", "S")} ${fmtLL(d.lon, "E", "W")}\n` +
    `speed ${d.spd.toFixed(2)} m/s\n` +
    `hdg   ${Math.round(d.hdg) % 360}°\n` +
    `r/p   ${deg(d.rpy.roll)}° ${deg(-d.rpy.pitch)}°` +
    (sp ? `\nsp    r${deg(sp.roll)}° p${deg(-sp.pitch)}° ` +
          `y${deg(sp.yaw)}°` : "") +
    (d.light !== null ? `\nlight ${d.light ? "ON" : "off"}` : "");
  $("mode-btn").textContent = `mode: ${MODE_LABEL[d.mode]}`;
}

let warnTimer = 0;
function keyWarn(text) {
  const el = $("key-warn");
  el.textContent = text;
  el.classList.add("show");
  clearTimeout(warnTimer);
  warnTimer = setTimeout(() => el.classList.remove("show"), 1000);
}

const viewerOpts = () => ({
  setHud: renderHud,
  navballCanvas: $("navball-canvas"),
  onWarn: keyWarn,
});

$("mode-btn").addEventListener("click", () => viewer?.cycleMode());
$("keys-btn").addEventListener("click", () =>
  $("keys-pop").classList.toggle("open"));
$("vis-slider").addEventListener("input", (e) =>
  viewer?.setVisibility(Number(e.target.value) / 100));

// keep the form ↔ design.spawn in sync
const form = $("spawn-form");
for (const [k, v] of Object.entries(editor.design.spawn)) {
  if (form.elements[k]) form.elements[k].value = v;
}

async function spawn() {
  const design = editor.design;
  for (const inp of form.querySelectorAll("input"))
    design.spawn[inp.name] = Number(inp.value) || 0;

  const btn = $("spawn-btn");
  btn.disabled = true;
  btn.textContent = "Compiling…";
  editor.toast("Generating the manta model and compiling it to WebAssembly…");
  try {
    const res = await fetch("/api/mako/spawn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(design),
    });
    if (!res.ok && res.status !== 200) {
      throw new Error(res.status === 404
        ? "spawn API not found — run the compile server: `python -m server.mako`"
        : `spawn API error (HTTP ${res.status})`);
    }
    const out = await res.json();
    if (!out.ok) throw new Error(out.errors.join(" · "));

    viewer = await startViewer($("viewer-canvas"), {
      design, base: out.base, meta: out.meta, ...viewerOpts(),
    });
    bundle = { base: out.base, meta: out.meta };
    document.body.dataset.mode = "drive";
    // visibility is a build-form setting but frontend-only (not part of
    // the compiled bundle): seed the drive-view slider from the form
    const vis = Math.min(100, Math.max(0,
      Number(design.spawn.visibility ?? 55)));
    $("vis-slider").value = vis;
    viewer.setVisibility(vis / 100);
    for (const w of out.meta.warnings || []) editor.toast(w);
  } catch (err) {
    editor.toast(String(err.message || err), true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Spawn ▸";
  }
}

form.addEventListener("submit", (e) => { e.preventDefault(); spawn(); });

$("back-btn").addEventListener("click", () => {
  viewer?.dispose();
  viewer = null;
  document.body.dataset.mode = "design";
  editor.rerender();
});

// Dev/share path: /mako/?load=<hash> boots the drive view straight from an
// already-compiled bundle (meta.json embeds the design for the meshes).
const loadHash = new URLSearchParams(location.search).get("load");
if (loadHash) {
  (async () => {
    try {
      const base = `/mako/builds/${loadHash}/`;
      const meta = await (await fetch(base + "meta.json")).json();
      viewer = await startViewer($("viewer-canvas"), {
        design: meta.design, base, meta, ...viewerOpts(),
      });
      bundle = { base, meta };
      document.body.dataset.mode = "drive";
      viewer.setVisibility(Number($("vis-slider").value) / 100);
    } catch (err) {
      editor.toast(`load failed: ${err.message || err}`, true);
    }
  })();
}
