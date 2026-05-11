# ex1 — Lunar-orbit manoeuvring craft

A 15-tonne Apollo-LM-scale craft in a circular orbit ~100 m above the
Moon's surface. Six bipolar thrusters (two per body axis, each offset
3 m from the centerline) give 6-DOF control: equal throttle on a pair
translates along its axis; opposite throttles rotate about the
perpendicular axis.

Three processes talk over Zenoh:

- **The sim** publishes craft state on `manta/ex1/state` and
  subscribes to one cmd topic per thruster.
- **The viewer** subscribes to state and renders the Moon + craft +
  thruster force arrows in Rerun.
- **The controller** reads keys from the terminal and publishes
  per-thruster commands.

## Running

Build (one-time, double precision for orbital stability):

```
cmake -S . -B build -DMANTA_DOUBLE_PRECISION=ON
cmake --build build -j
```

Then in three terminals at the repo root:

```
./build/examples/ex1                                 # sim
.venv/bin/python examples/ex1_orbit/viewer.py        # Rerun viewer
.venv/bin/python examples/ex1_orbit/controller.py    # keyboard controller
```

The controller terminal must have keyboard focus for inputs to register.

## Keyboard controls (bang-bang)

Each thruster command is full ±1.0 or 0. The controller's mixer sums
the active DOF contributions and clamps to a sign.

| Key       | Action                                       |
| --------- | -------------------------------------------- |
| `W` / `S` | ±Y translation                               |
| `A` / `D` | ∓X / ±X translation                          |
| `Q` / `E` | ∓Z / ±Z translation                          |
| `J` / `L` | Roll  about the A/D (body X) axis            |
| `I` / `K` | Pitch about the W/S (body Y) axis            |
| `U` / `O` | Yaw   about the Q/E (body Z) axis            |
| `SPACE`   | Zero all thrusters                           |
| `Ctrl-C`  | Quit                                         |

## Regenerating after a config edit

If you change `config.py` (geometry, mass, thruster placement, …),
regenerate the C++ sources and rebuild:

```
.venv/bin/python -m manta_codegen.cli examples/ex1_orbit/config.py --workflow binary
cmake --build build --target ex1 -j
```

## Tuning the viewer

Three knobs at the top of `viewer.py`:

- `SCALE` — rerun units per metre of world geometry (Moon).
- `ALT_EXAGGERATE` — multiplier on (orbit altitude above the surface)
  before mapping to rerun units. Without exaggeration, 100 m of
  altitude against a 1.7 Mm-radius Moon is sub-pixel.
- `CRAFT_SCALE` — rerun units per metre of body geometry. Smaller =
  the LM model and force arrows shrink relative to the Moon.

The craft model is `Apollo_LM.gltf` in this directory; replace it (or
the `rr.Asset3D(...)` log call) to swap in a different mesh.
