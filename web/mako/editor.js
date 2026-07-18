// The 2D craft editor: a side-view (x right, z up) SVG of the design.
//
// Interaction model: palette items and already-mounted modules are dragged
// with pointer events (no HTML5 DnD — works for touch and lets us draw the
// snap ghost ourselves). While dragging, every VALID empty target is marked;
// within the snap threshold a faded ghost of the module appears in its final
// pose, and releasing attaches it. Hull sections drag onto spine junctions
// (including between two existing modules). Clicking a module opens the
// config panel with its option sliders.
//
// The design object matches SPEC.md's design.json and is persisted to
// localStorage on every mutation.

import {
  GROUPS, HULL_R, MODULES, cleanOptions, layout, massBudget, starterDesign,
} from "./catalog.js";

const STORE_KEY = "mako-design-v2";
const SNAP = 0.45;        // snap threshold, metres (design space)
const SVG_NS = "http://www.w3.org/2000/svg";

const el = (tag, attrs = {}) => {
  const n = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
};

export function loadDesign() {
  try {
    const d = JSON.parse(localStorage.getItem(STORE_KEY));
    // Sanity: known types with the right kinds in the right places
    // (nose … spine modules … rear) — else start fresh.
    const kind = (m) => MODULES[m.type]?.kind;
    if (d && d.spine && d.spine.length >= 2
        && kind(d.spine[0]) === "nose"
        && kind(d.spine[d.spine.length - 1]) === "rear"
        && d.spine.slice(1, -1).every((m) => kind(m) === "spine"))
      return d;
  } catch { /* fall through */ }
  return starterDesign();
}

export function createEditor({ svg, paletteGroups, search, config, stats, onChange }) {
  let design = loadDesign();
  let selected = null;         // spine index | null
  let drag = null;             // live drag state
  let scale = 120, cx = 0, cy = 0;

  // ---------------------------------------------------------------- helpers
  const px = (x) => cx + x * scale;
  const pz = (z) => cy - z * scale;
  const toDesign = (ev) => {
    const r = svg.getBoundingClientRect();
    return { x: (ev.clientX - r.left - cx) / scale,
             z: -(ev.clientY - r.top - cy) / scale };
  };

  function persist() {
    localStorage.setItem(STORE_KEY, JSON.stringify(design));
    onChange?.(design);
  }

  let toastTimer = 0;
  function toast(msg, isError = false) {
    const t = document.getElementById("toast");
    t.textContent = msg;
    t.className = `toast show${isError ? " error" : ""}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove("show"), isError ? 5200 : 2600);
  }

  // ------------------------------------------------------------------ stats
  function renderStats() {
    const { total } = layout(design);
    const { mass } = massBudget(design);
    stats.innerHTML =
      `<span>length <b>${total.toFixed(2)} m</b></span>` +
      `<span>mass <b>≈${mass.toFixed(1)} kg</b> (neutral)</span>` +
      `<span>modules <b>${design.spine.length}</b></span>`;
  }

  // ---------------------------------------------------------------- palette
  function renderPalette(filter = "") {
    paletteGroups.innerHTML = "";
    const q = filter.trim().toLowerCase();
    for (const group of GROUPS) {
      const entries = Object.entries(MODULES).filter(([id, m]) =>
        m.group === group &&
        (!q || m.label.toLowerCase().includes(q) || id.includes(q)
            || m.blurb.toLowerCase().includes(q)));
      if (!entries.length) continue;
      const g = document.createElement("div");
      g.className = "pgroup";
      g.innerHTML = `<h3>${group}</h3>`;
      for (const [id, m] of entries) {
        const item = document.createElement("div");
        item.className = "pitem";
        item.dataset.type = id;
        const defs = cleanOptions(id, {});
        item.innerHTML =
          `<div class="icon">${iconSvg(id, defs)}</div>` +
          `<div><b>${m.label}</b><span>${m.blurb}</span></div>`;
        item.addEventListener("pointerdown", (ev) => startDrag(ev, {
          type: id, originEl: item,
        }));
        g.appendChild(item);
      }
      paletteGroups.appendChild(g);
    }
  }

  function iconSvg(type, opts) {
    const m = MODULES[type];
    // Fit the silhouette into the icon box; shapes are centred on their origin.
    const span = m.length / 2 + 0.04;
    const vs = Math.max(span * 0.75, 0.16);
    return `<svg viewBox="${-span} ${-vs} ${2 * span} ${2 * vs}"
              xmlns="${SVG_NS}">${m.shape2d(opts, {})}</svg>`;
  }

  // ----------------------------------------------------------------- render
  function render() {
    const rect = svg.getBoundingClientRect();
    const lay = layout(design);
    scale = Math.min(300, Math.max(60, (rect.width * 0.72) / Math.max(lay.total, 1)));
    cx = rect.width / 2;
    cy = rect.height * 0.48;
    svg.innerHTML = "";

    // backdrop: faint metre grid + centreline
    const grid = el("g");
    const gspan = Math.ceil(rect.width / scale / 2) + 1;
    for (let i = -gspan; i <= gspan; i++) {
      grid.appendChild(el("line", {
        x1: px(i), y1: 0, x2: px(i), y2: rect.height,
        stroke: "rgba(28,69,135,.06)", "stroke-width": 1 }));
    }
    for (let j = -3; j <= 3; j++) {
      grid.appendChild(el("line", {
        x1: 0, y1: pz(j), x2: rect.width, y2: pz(j),
        stroke: j === 0 ? "rgba(28,69,135,.14)" : "rgba(28,69,135,.06)",
        "stroke-width": 1 }));
    }
    svg.appendChild(grid);

    design.spine.forEach((m, i) => {
      const opts = cleanOptions(m.type, m.options);
      svg.appendChild(moduleNode(m.type, opts, lay.centers[i], 0, i));
    });

    // drop targets while dragging: insertion slots for spine modules,
    // a replace highlight over the matching end cap for nose/rear
    if (drag) {
      const kind = MODULES[drag.item.type].kind;
      if (kind === "spine") {
        for (const j of spineJunctions(lay)) {
          svg.appendChild(el("rect", {
            x: px(j.x) - 0.018 * scale, y: pz(HULL_R + 0.12),
            width: 0.036 * scale, height: (2 * HULL_R + 0.24) * scale,
            rx: 0.012 * scale, class: "spine-slot",
            style: `stroke-width:${0.01 * scale}` }));
        }
      } else {
        const i = kind === "nose" ? 0 : design.spine.length - 1;
        const L = lay.lengths[i];
        svg.appendChild(el("rect", {
          x: px(lay.centers[i] - L / 2) - 4, y: pz(HULL_R + 0.1),
          width: L * scale + 8, height: (2 * HULL_R + 0.2) * scale,
          rx: 0.02 * scale, class: "spine-slot",
          style: `stroke-width:${0.01 * scale}` }));
      }
    }

    // ghost preview
    if (drag && drag.ghost) {
      const { type, opts, x, z, ok } = drag.ghost;
      const node = moduleNode(type, opts, x, z, null);
      node.setAttribute("class", ok ? "mk-ghost" : "mk-ghost-bad");
      svg.appendChild(node);
    }
  }

  /** One module silhouette placed at design coords; idx != null wires
   *  selection. */
  function moduleNode(type, opts, x, z, idx) {
    const g = el("g", {
      transform: `translate(${px(x)},${pz(z)}) scale(${scale})`,
    });
    g.innerHTML = MODULES[type].shape2d(opts, {});
    if (idx !== null) {
      g.classList.add("mk-module");
      if (selected === idx) g.classList.add("selected");
      g.addEventListener("pointerdown", (ev) => {
        ev.stopPropagation();
        select(idx);
      });
    }
    return g;
  }

  /** x-positions where a spine module can be inserted (between any two
   *  modules — never before the nose or after the stern). */
  function spineJunctions(lay) {
    const out = [];
    for (let i = 1; i < design.spine.length; i++) {
      out.push({ index: i, x: lay.centers[i - 1] - lay.lengths[i - 1] / 2 });
    }
    return out;
  }

  // ------------------------------------------------------------ drag + drop
  function startDrag(ev, item) {
    ev.preventDefault();
    drag = { item, ghost: null, moved: false,
             startX: ev.clientX, startY: ev.clientY };
    item.originEl?.classList.add("dragging");

    const move = (e) => {
      if (Math.hypot(e.clientX - drag.startX, e.clientY - drag.startY) > 4)
        drag.moved = true;
      if (!drag.moved) return;
      const pt = toDesign(e);
      drag.ghost = ghostAt(pt);
      render();
    };
    const up = () => {
      removeEventListener("pointermove", move);
      removeEventListener("pointerup", up);
      item.originEl?.classList.remove("dragging");
      const d = drag; drag = null;
      if (d.moved && d.ghost?.ok) applyDrop(d);
      else render();
    };
    addEventListener("pointermove", move);
    addEventListener("pointerup", up);
  }

  /** Resolve the drop target for the cursor at design coords `pt`: spine
   *  modules snap to junctions (insert-between); nose/rear snap onto the
   *  matching end cap and REPLACE it. */
  function ghostAt(pt) {
    const item = drag.item;
    const opts = cleanOptions(item.type, item.options);
    const mod = MODULES[item.type];
    const lay = layout(design);
    const miss = { type: item.type, opts, ok: false, x: pt.x, z: pt.z };
    if (Math.abs(pt.z) > 0.9) return miss;

    if (mod.kind === "spine") {
      let best = null;
      for (const j of spineJunctions(lay)) {
        const dist = Math.abs(pt.x - j.x);
        if (dist < (best?.dist ?? SNAP)) best = { ...j, dist };
      }
      if (best) return { type: item.type, opts, ok: true,
                         x: best.x - mod.length / 2, z: 0,
                         insert: best.index };
      return miss;
    }

    // end cap → replace the existing one of the same kind
    const i = mod.kind === "nose" ? 0 : design.spine.length - 1;
    if (Math.abs(pt.x - lay.centers[i]) < lay.lengths[i] / 2 + SNAP)
      return { type: item.type, opts, ok: true,
               x: endCapX(lay, i, mod.length), z: 0, replace: i };
    return miss;
  }

  /** Centre x a replacement end cap will land at (its outer edge stays
   *  where the old cap's inner junction is). */
  function endCapX(lay, i, newLen) {
    const inner = i === 0
      ? lay.centers[0] - lay.lengths[0] / 2         // nose: aft junction
      : lay.centers[i] + lay.lengths[i] / 2;        // rear: fore junction
    return i === 0 ? inner + newLen / 2 : inner - newLen / 2;
  }

  function applyDrop(d) {
    const g = d.ghost, item = d.item;
    if (g.insert !== undefined) {
      design.spine.splice(g.insert, 0,
                          { type: item.type, options: {} });
      select(g.insert);
      toast(`${MODULES[item.type].label} inserted.`);
    } else {
      design.spine[g.replace] = { type: item.type, options: {} };
      select(g.replace);
      toast(`${MODULES[item.type].label} fitted.`);
    }
    persist();
    renderStats();
    render();
  }

  // -------------------------------------------------------------- selection
  function select(idx) {
    selected = idx;
    render();
    renderConfig();
  }

  function renderConfig() {
    const m = selected !== null ? design.spine[selected] : null;
    if (!m) { config.classList.remove("open"); return; }
    const def = MODULES[m.type];
    // end caps are replace-only (drop another nose/stern on them)
    const removable = def.kind === "spine";

    config.innerHTML =
      `<h2>${def.label}</h2><div class="sub">spine #${selected} — ${def.blurb}</div>` +
      (def.options.length ? "" :
        `<div class="sub">No options — mass and buoyancy are set by the
         module's geometry (neutrally buoyant).</div>`) +
      def.options.map((o) => {
        const opts = cleanOptions(m.type, m.options);
        return `
        <div class="row">
          <label>${o.label}<output id="out-${o.key}">${fmt(opts[o.key])} ${o.unit}</output></label>
          <input type="range" data-key="${o.key}" min="${o.min}" max="${o.max}"
                 step="${o.step}" value="${opts[o.key]}">
        </div>`;
      }).join("") +
      `<div class="actions">
         ${removable ? `<button class="remove">remove</button>` : "<span></span>"}
         <button class="close">close</button>
       </div>`;
    config.classList.add("open");

    config.querySelectorAll("input[type=range]").forEach((inp) => {
      inp.addEventListener("input", () => {
        const key = inp.dataset.key;
        m.options = { ...m.options, [key]: Number(inp.value) };
        config.querySelector(`#out-${key}`).textContent =
          `${fmt(Number(inp.value))} ${def.options.find((o) => o.key === key).unit}`;
        persist(); renderStats(); render();
      });
    });
    config.querySelector(".close").addEventListener("click", () => select(null));
    config.querySelector(".remove")?.addEventListener("click", removeSelected);
  }

  const fmt = (v) => Math.abs(v) >= 100 ? v.toFixed(0)
    : Math.abs(v) >= 1 ? +v.toFixed(2) + "" : +v.toPrecision(2) + "";

  function removeSelected() {
    if (selected === null) return;
    if (MODULES[design.spine[selected].type].kind !== "spine") return;
    design.spine.splice(selected, 1);
    select(null);
    persist(); renderStats(); render();
  }

  // ------------------------------------------------------------------ wire-up
  svg.addEventListener("pointerdown", () => select(null)); // click-away
  addEventListener("keydown", (e) => {
    if (document.body.dataset.mode !== "design") return;
    if ((e.key === "Delete" || e.key === "Backspace") && selected !== null
        && document.activeElement?.tagName !== "INPUT") {
      e.preventDefault(); removeSelected();
    }
    if (e.key === "Escape") select(null);
  });
  search.addEventListener("input", () => renderPalette(search.value));
  new ResizeObserver(() => render()).observe(svg);

  renderPalette();
  renderStats();
  render();
  onChange?.(design);

  return {
    get design() { return design; },
    toast,
    reset() { design = starterDesign(); select(null); persist(); renderStats(); render(); },
    rerender: render,
  };
}
