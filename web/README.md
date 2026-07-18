# manta homepage (`web/`)

The static site served at `mantapilot.org`. One assets-only Cloudflare Worker
(`wrangler.jsonc`) serves the homepage at `/` and the built MkDocs site under
`/docs/`.

| Path | What |
| --- | --- |
| `index.html` | the homepage (hand-written, self-contained CSS) |
| `carousel.js` | the example-carousel controller (code blurbs + render boot) |
| `lib/render.js` | shared Three.js helpers (scene, boxes, arrows, camera) |
| `quad/`, `airplane/` | per-example live renders (WASM bundle + `demo.js`) |
| `mako/` | the "build a Mako" submarine builder (see below) |
| `manta.svg`, `manta-logo.svg` | brand logo (full / background-stripped) |
| `demos.py` | the carousel's vehicle models (quad here, airplane reuses the example) |
| `build_demos.py` | regenerate + compile the WASM bundles |
| `docs/` | MkDocs build output (git-ignored; `mkdocs build`) |

## The example carousel

Each slide on the left shows a manta model verbatim; on the right that exact
model flies, lowered to WebAssembly by `TargetWasm` and stepped live. The quad
is open-loop unstable, so it's flown by manta's own LQR (the constant gain is
baked into `quad/demo.js`). The Cessna is statically stable and flies on trim.
Click a render to take the controls; the **show skeleton** toggle overlays the
manta primitives and force vectors.

To add an example: add a `build_*` to `demos.py`, a `<dir>/demo.js` renderer,
and an entry to the `EX` array in `carousel.js`, then rebuild (below).

## Regenerate the WASM bundles

From the repo root, with the [Emscripten SDK](https://emscripten.org) active:

```sh
source ~/emsdk/emsdk_env.sh        # put emcc on PATH
.venv/bin/python web/build_demos.py
```

This emits the C ABI + kernels + JS runtime + descriptor for each vehicle, then
compiles each to `<name>.mjs` + `<name>.wasm`. The `*_kernels.c` / `*_abi.c`
build inputs are git-ignored (re-emitted); the runtime artifacts are tracked.

## Build a Mako (`/mako/`)

A drag-and-drop submarine builder: assemble modules (hull sections, thrusters,
fins, sensors, ballast) in a 2D side-view editor, then **Spawn** — the compile
server builds the manta `World` from the design, lowers `Sim` + `EKF` + `LQR`
to WebAssembly with `TargetWasm`, and the browser drives the craft live with
keyboard setpoints through the baked LQR. The EKF toggle shows a ghost craft
estimating the state from the design's own sensors (GPS is water-gated, so it
dead-reckons at depth and re-fixes on surfacing, like
`examples/vehicles/submarine.py`).

- Contract between client and server: `mako/SPEC.md`.
- Client: `mako/catalog.js` (modules, icons, placement, 3D meshes),
  `mako/editor.js`, `mako/viewer.js`, `mako/main.js`.
- Server: `server/mako/` (stdlib-only). Run it for local dev — it serves
  `web/` **and** the `POST /api/mako/spawn` endpoint:

```sh
source ~/emsdk/emsdk_env.sh          # emcc, needed to compile spawns
.venv/bin/python -m server.mako      # http://localhost:8077/mako/
```

Compiled bundles land in `mako/builds/<design-hash>/` (git-ignored cache).
In production, the same server runs behind the static host and only
`/api/mako/spawn` needs routing to it. The 3D meshes are procedural stand-ins —
each module's mesh builder in `catalog.js` can be swapped for a glTF load
without touching placement.

## Local preview

```sh
python3 -m http.server -d web 8099   # then open http://localhost:8099/
```

(The homepage carousel works with any static server; the Mako builder's spawn
button needs `server.mako` above.)

## Known follow-ups

- **Hero showcase reel** — `index.html` has a placeholder where the looping
  vehicle video goes (`.hero-media`).
- **Rich 3D models** — the renders currently draw the manta skeleton only; drop
  glTF models into `quad/` / `airplane/` and load them in `demo.js`.
- **True aero force vectors** — the skeleton overlay draws thrust (exact). Lift
  and drag arrows wait on exposing `CraftTrace.own_wrench` (per-part body-frame
  wrench, already computed in `manta/tick/world_tick.py`) as a `Role.OUTPUT`
  port on `Sim.module`'s `step` entry; `demo.js` already has the arrow hooks.
