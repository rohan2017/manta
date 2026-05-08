"""StateSpec-based codegen for EKF targets.

Emits the new `EKFGeneric<StateSpec, MeasDim, NoiseSlots>` shape — see
`include/manta/estimation/generic_ekf.hpp` and the reference
implementation at `examples/ex10_cpp_ekf/main.cpp`. Replaces
the legacy `EKF<NumCrafts, MeasDim, ...>` + per-sensor functor + R-init
+ `add_update` block flow with:

  * `auto state = make_state().track(craft0)...build();`
  * `EKFGeneric<decltype(state), MeasDim, NoiseSlots> ekf{state};`
  * `ekf.measure<Dim>(&est_part.field, reading_from<Dim>(sim_part.field));`
  * `ekf.predict(dt, Q); ekf.run_pending_updates();`

Gated by `EKF.use_state_spec=True` on the Python descriptor. Falls back
to legacy emit otherwise.

Scope: Pattern A (sim+est in one process) only. Same-name convention
matches sim and est parts. Pattern C (real-data Zenoh subscribers) and
multi-craft EKFs are migrated in subsequent passes.
"""

from __future__ import annotations

from .._format import cpp_float as _f, cpp_mfloat as _mf
from ._util import GENERATED_BANNER, CPP_INCLUDE_GUARD, class_name_for_craft


# ---- Per-part-type measurement field map -----------------------------
#
# Each entry: cpp_class_template → list of (member_name, dim).
# The codegen iterates these for each registered measurement part to
# emit one `ekf.measure(&est.<part>().<member>, reading_from(sim...))`
# call per field.
_MEAS_FIELDS: dict[str, list[tuple[str, int]]] = {
    "manta::parts::IMUT":          [("accel", 3), ("gyro", 3)],
    "manta::parts::DVLT":          [("velocity", 3)],
    "manta::parts::MagnetometerT": [("b", 3)],
}


# ---- Per-part-type RW bias channel map ------------------------------
#
# Each entry: cpp_class_template → list of (accessor, dim, sigma_attr).
# When a part's `<sigma_attr>` is ≥ 0, the codegen emits a
# `state.track(part.<accessor>())` call to grow the StateSpec by a
# `BiasRandomWalk<dim>` slice.
_BIAS_CHANNELS: dict[str, list[tuple[str, int, str]]] = {
    "manta::parts::IMUT": [
        ("accel_bias", 3, "accel_bias_sigma"),
        ("gyro_bias",  3, "gyro_bias_sigma"),
    ],
}


def _bias_channels_for(part) -> list[tuple[str, int, str]]:
    cct = getattr(type(part), "cpp_class_template", "")
    if cct not in _BIAS_CHANNELS:
        return []
    out: list[tuple[str, int, str]] = []
    for accessor, dim, sigma_attr in _BIAS_CHANNELS[cct]:
        sigma = getattr(part, sigma_attr, -1.0)
        if sigma is not None and float(sigma) >= 0.0:
            out.append((accessor, dim, sigma_attr))
    return out


def _meas_fields_for(part) -> list[tuple[str, int]]:
    cct = getattr(type(part), "cpp_class_template", "")
    if cct not in _MEAS_FIELDS:
        raise NotImplementedError(
            f"emit_filter_state_spec: no measurement-field map for "
            f"{type(part).__name__} ({cct!r}). Add an entry to "
            f"_MEAS_FIELDS in ekf_state_spec.py.")
    return _MEAS_FIELDS[cct]


def _world_unique_crafts(world):
    """Stable de-dup of crafts in the world (same logic as legacy emit)."""
    seen = set()
    out = []
    for entry in world.crafts:
        c = entry.craft
        if id(c) in seen:
            continue
        seen.add(id(c))
        out.append(c)
    return out


def _world_noise_slot_count(world) -> int:
    """Sum of all enabled noise channel driver-dims in the world's parts."""
    n = 0
    for entry in world.crafts:
        for part in entry.craft.all_parts():
            for ch in part.noise_channels():
                if ch.is_enabled():
                    n += ch.driver_dim()
    return n


def _find_sim_source_part(target, est_part):
    """Find a sim-side part with the same name as `est_part` in another
    craft in the same Target (Pattern A — sim+est in one process). The
    sim craft is identified as any non-EKF drive in the target whose
    crafts contain a part of the same name.

    Returns (sim_craft, sim_part) or (None, None) if no source is found.
    """
    from ..estimation.ekf import EKF as EKFDesc
    from ..estimation.ukf import UKF as UKFDesc
    from ..core import World as WorldDesc

    est_part_name = est_part.name

    # Walk every drive that's a World (not the EKF/UKF itself).
    for drive in target.drives:
        if isinstance(drive, (EKFDesc, UKFDesc)):
            continue
        if not isinstance(drive, WorldDesc):
            continue
        for entry in drive.crafts:
            sim_craft = entry.craft
            for sim_part in sim_craft.all_parts():
                if sim_part.name == est_part_name and \
                        type(sim_part) is type(est_part):
                    return sim_craft, sim_part
    return None, None


def emit_filter_hpp(target, filter_obj) -> str:
    """Header for the StateSpec-based EKF target."""
    from ..estimation.ekf import EKF as EKFDesc  # noqa: F401
    world = filter_obj.world
    name  = world.name

    crafts = _world_unique_crafts(world)
    num = len(crafts)
    if num == 0:
        raise RuntimeError("use_state_spec emit: filter has no crafts")

    def craft_var_for(idx: int) -> str:
        return "craft" if num == 1 else f"craft_{idx}"

    # Each craft's templated value-side type. EKFGeneric requires <double>.
    cls_real = [f"{class_name_for_craft(c.name)}T<double>" for c in crafts]
    # Spec: one RigidBody slice per craft, plus one BiasRandomWalk<dim>
    # slice for every enabled bias channel on every part.
    slice_args: list[str] = []
    bias_slices: list[tuple[int, str, int]] = []  # (craft_idx, part_name, dim) per bias slice
    for ci, c in enumerate(crafts):
        slice_args.append("manta::manifold::RigidBody")
    for ci, c in enumerate(crafts):
        for p in c.all_parts():
            for accessor, dim, _ in _bias_channels_for(p):
                slice_args.append(f"manta::manifold::BiasRandomWalk<{dim}>")
                bias_slices.append((ci, p.name, accessor, dim))
    spec_args = ", ".join(slice_args)
    noise_slots = _world_noise_slot_count(world)
    meas_dim = 0   # unused in MeasureDim-based legacy; bindings are dynamic.

    filter_var = filter_obj.cpp_var_name()

    lines: list[str] = [
        GENERATED_BANNER, CPP_INCLUDE_GUARD, "",
        '#include "manta/core/harness.hpp"',
        '#include "manta/core/scene.hpp"',
        '#include "manta/core/world.hpp"',
        '#include "manta/estimation/craft_view.hpp"',
        '#include "manta/estimation/generic_ekf.hpp"',
        '#include "manta/estimation/manifold.hpp"',
        '#include "manta/estimation/measurement.hpp"',
        '#include "manta/estimation/reading.hpp"',
        '#include "manta/estimation/state_spec.hpp"',
    ]
    seen: set[str] = set()
    for f in world.fields:
        if f.cpp_header not in seen:
            seen.add(f.cpp_header)
            lines.append(f'#include "{f.cpp_header}"')
    # The est craft hpp(s) — one per unique craft (deduped by name).
    seen_craft_hpp: set[str] = set()
    for c in crafts:
        if c.name not in seen_craft_hpp:
            seen_craft_hpp.add(c.name)
            lines.append(f'#include "{c.name}_craft.hpp"')
    lines += [
        "",
        f"namespace manta_gen::{name} {{",
        "",
        f"// Sim-tick parameters, frozen at codegen time.",
        f"inline constexpr float DT             = {_f(target.dt)};",
        f"inline constexpr float SIM_RATE_MULT  = {_f(target.sim_rate_mult)};",
        "",
        f"// State spec — one RigidBody slice per tracked craft.",
        f"using Spec = manta::estimation::StateSpec<{spec_args}>;",
        f"// Total noise slots = sum of every part's enabled noise driver dims.",
        f"using EkfT = manta::estimation::EKFGeneric<Spec, /*MeasDim=*/{meas_dim}, /*NoiseSlots=*/{noise_slots}>;",
        f"using JetType = EkfT::Jet;",
        "",
        f"extern manta::WorldT<double>          w;",
        f"extern manta::SceneT<double>*         scene;",
    ]
    if world.fields:
        for i, f in enumerate(world.fields):
            lines.append(f"extern {f.cpp_class} field_{i};")

    # Sim worlds live in their own emitted modules (e.g. ex5_sim.hpp);
    # the est .cpp pulls them in by include. Only the est-side world,
    # crafts, EKF, and CraftViews are exposed at this header level.
    for i, c in enumerate(crafts):
        lines.append(f"extern {cls_real[i]} {craft_var_for(i)};")
    lines += [
        "",
        f"extern EkfT {filter_var};",
    ]
    for i, _c in enumerate(crafts):
        lines.append(
            f"extern manta::estimation::CraftView<EkfT, {i}> view_{i};")
    lines += [
        "",
        "void setup();",
        "void tick();",
        "void shutdown();",
        "",
        "struct Harness : public manta::Harness {",
        "    void setup()    override;",
        "    void tick()     override;",
        "    void shutdown() override;",
        "};",
        "extern Harness harness;",
        "",
        f"}}  // namespace manta_gen::{name}",
        "",
    ]
    return "\n".join(lines)


def emit_filter_cpp(target, filter_obj) -> str:
    """Implementation for the StateSpec-based EKF target."""
    from ..core import World as WorldDesc
    from ..estimation.ekf import EKF as EKFDesc
    from ..estimation.ukf import UKF as UKFDesc

    world = filter_obj.world
    name  = world.name
    crafts = _world_unique_crafts(world)
    num = len(crafts)
    cls_real = [f"{class_name_for_craft(c.name)}T<double>" for c in crafts]
    filter_var = filter_obj.cpp_var_name()

    def craft_var_for(idx: int) -> str:
        return "craft" if num == 1 else f"craft_{idx}"

    bias_slices: list[tuple[int, str, str, int]] = []
    for ci, c in enumerate(crafts):
        for p in c.all_parts():
            for accessor, dim, _ in _bias_channels_for(p):
                bias_slices.append((ci, p.name, accessor, dim))

    # Identify sim-side worlds and crafts in the target. Each is its
    # own emitted module with namespace `manta_gen::<world.name>` that
    # already exports a `craft` (single) or `craft_0/1/...` (multi)
    # variable. We pull those in by include and reference them directly.
    sim_worlds: list[tuple[str, object]] = []   # (sim_world_name, world_obj)
    for drive in target.drives:
        if isinstance(drive, (EKFDesc, UKFDesc)) or not isinstance(drive, WorldDesc):
            continue
        sim_worlds.append((drive.name, drive))

    def sim_craft_var(sim_world, idx: int) -> str:
        # Single-craft sim worlds export a `craft` global; multi-craft
        # worlds export `craft_0`, `craft_1`, ... Match the convention
        # from `emit_filter_collect`.
        if len(sim_world.crafts) <= 1:
            return "craft"
        return f"craft_{idx}"

    # ---- Header ----
    # Map (part_id) → list of (field_name, dim, buffer_var, fresh_var).
    # Populated when the user has called `ekf.read_topic(part, topic)`.
    raw_reading_topics = list(getattr(filter_obj, "reading_topics", []) or [])

    # Resolve the est-craft-index for each tracked reading. Buffer names
    # encode the craft index too so that multi-craft worlds with
    # same-named parts (drone_0.imu, drone_1.imu) don't collide.
    def _est_craft_idx_for(part) -> int:
        for i, c in enumerate(crafts):
            for p in c.all_parts():
                if id(p) == id(part):
                    return i
        return -1

    reading_topics: list[tuple[int, object, str]] = []  # (craft_idx, part, topic)
    for part, topic in raw_reading_topics:
        ci = _est_craft_idx_for(part)
        if ci < 0:
            raise RuntimeError(
                f"emit_filter_state_spec: read_topic part {part.name!r} "
                f"not found in any tracked craft.")
        reading_topics.append((ci, part, topic))

    def reading_buf_var(craft_idx: int, part_name: str, field_name: str) -> str:
        return f"reading_c{craft_idx}_{part_name}_{field_name}_buf"
    def reading_fresh_var(craft_idx: int, part_name: str, field_name: str) -> str:
        return f"reading_c{craft_idx}_{part_name}_{field_name}_fresh"
    def reading_sub_var(craft_idx: int, part_name: str) -> str:
        return f"reading_c{craft_idx}_{part_name}_sub"

    lines: list[str] = [
        GENERATED_BANNER, "",
        f'#include "{name}.hpp"',
    ]
    # Pull in the sim modules so we can reference their craft globals.
    for sw_name, _sw in sim_worlds:
        lines.append(f'#include "{sw_name}.hpp"')
    lines += [
        "",
        "#include <atomic>",
        "#include <cstdio>",
        "#include <cstdlib>",
        "#include <optional>",
        "#include <string>",
        "#include <string_view>",
        "#include <vector>",
        "",
        "#include <Eigen/Core>",
        "#include <Eigen/Geometry>",
    ]
    if reading_topics:
        lines.append("#include <zenoh.hxx>")
    lines += [
        "",
    ]

    # ---- Public namespace: definitions ----
    lines += [
        f"namespace manta_gen::{name} {{",
        "",
        "manta::WorldT<double>  w{};",
        "manta::SceneT<double>* scene = nullptr;",
    ]
    for i, f in enumerate(world.fields):
        lines.append(f.emit_construction(f"field_{i}"))
    # Est-craft definitions. (Sim world + crafts live in their own
    # generated module — referenced via the include above.)
    for i, c in enumerate(crafts):
        lines.append(f"{cls_real[i]} {craft_var_for(i)}{{}};")
    # EKF construction — needs the StateSpec built from a track() chain.
    # Use a helper function to keep the template-deduced expression clean.
    # EkfT construction — one make_state().track(...) call per craft,
    # then one .track(part.<bias_accessor>()) per enabled RW bias.
    track_calls = [f"track({craft_var_for(i)})" for i in range(num)]
    for ci, part_name, accessor, _dim in bias_slices:
        track_calls.append(
            f"track({craft_var_for(ci)}.{part_name}().{accessor}())")
    track_chain = ".".join(track_calls)
    lines += [
        "",
        f"EkfT {filter_var}{{ manta::estimation::make_state().{track_chain}.build() }};",
    ]
    for i in range(num):
        lines.append(
            f"manta::estimation::CraftView<EkfT, {i}> view_{i}{{{filter_var}}};")
    lines += [
        "",
        f"}}  // namespace manta_gen::{name}",
        "",
    ]

    # ---- Anonymous namespace: file-private state ----
    lines += [
        "namespace {",
        "",
        f"using JetType = manta_gen::{name}::JetType;",
        f"using EkfT    = manta_gen::{name}::EkfT;",
        "",
        "manta::WorldT<JetType>  w_jet{};",
        "manta::SceneT<JetType>* scene_jet = nullptr;",
    ]
    for i, c in enumerate(crafts):
        lines.append(
            f"{class_name_for_craft(c.name)}T<JetType> {craft_var_for(i)}_jet{{}};")
    lines += [
        "",
        "EkfT::StateCov g_Q = EkfT::StateCov::Zero();",
        "",
    ]

    # Pattern C reading buffers + atomics + Zenoh handles.
    if reading_topics:
        lines.append("// ---- Pattern C reading sources (Zenoh-fed buffers) ----")
        lines.append("std::optional<zenoh::Session> g_reading_session;")
        for ci, part, _topic in reading_topics:
            for field_name, dim in _meas_fields_for(part):
                lines.append(
                    f"Eigen::Matrix<double, {dim}, 1> "
                    f"{reading_buf_var(ci, part.name, field_name)}{{}};")
                lines.append(
                    f"std::atomic<bool> "
                    f"{reading_fresh_var(ci, part.name, field_name)}{{false}};")
            lines.append(
                f"std::optional<zenoh::Subscriber<void>> "
                f"{reading_sub_var(ci, part.name)};")
        lines.append("")

    # Tiny float-array decoder for JSON-encoded payloads.
    if reading_topics:
        lines += [
            "static bool _parse_float_array(std::string_view s,",
            "                                std::vector<double>& out) {",
            "    out.clear();",
            "    bool in_num = false;",
            "    std::string buf;",
            "    for (char c : s) {",
            "        if ((c >= '0' && c <= '9') || c == '-' || c == '+'",
            "                || c == '.' || c == 'e' || c == 'E') {",
            "            buf.push_back(c); in_num = true;",
            "        } else if (in_num) {",
            "            try { out.push_back(std::stod(buf)); }",
            "            catch (...) { return false; }",
            "            buf.clear(); in_num = false;",
            "        }",
            "    }",
            "    if (in_num) {",
            "        try { out.push_back(std::stod(buf)); }",
            "        catch (...) { return false; }",
            "    }",
            "    return !out.empty();",
            "}",
            "",
        ]

    lines += [
        "}  // namespace",
        "",
    ]

    # ---- setup() ----
    lines += [
        f"namespace manta_gen::{name} {{",
        "",
        "void setup() {",
        f"    w.clock().set_dt(DT);",
        f"    scene = &w.create_scene();",
    ]
    # Register fields on the est world.
    for i, f in enumerate(world.fields):
        lines.append(f"    w.register_field(field_{i});")
    for i in range(num):
        lines.append(f"    scene->add_craft({craft_var_for(i)});")

    # Sim worlds: their own modules' setup() builds them. Nothing for
    # us to do here — the orchestrator (<target>.cpp) will call each
    # drive's setup() in order.

    # Jet world setup.
    lines += [
        "",
        "    w_jet.clock().set_dt(DT);",
        "    scene_jet = &w_jet.create_scene();",
    ]
    for i, f in enumerate(world.fields):
        lines.append(f"    w_jet.register_field(field_{i});")
    for i in range(num):
        lines.append(f"    scene_jet->add_craft({craft_var_for(i)}_jet);")

    jet_array = ", ".join(
        f"static_cast<void*>(&{craft_var_for(i)}_jet)" for i in range(num))
    lines += [
        "",
        f"    {filter_var}.bind(w_jet, {{ {jet_array} }});",
        "",
    ]

    # Pattern C subscribers — one Zenoh sub per part with reading_topics.
    # Each subscriber's lambda decodes the float payload and writes to
    # this part's per-field buffers + sets the per-field fresh atomics.
    if reading_topics:
        lines.append("    g_reading_session.emplace("
                     "zenoh::Session::open(zenoh::Config::create_default()));")
        for ci, part, topic in reading_topics:
            fields = _meas_fields_for(part)
            total_dim = sum(d for _, d in fields)
            lines.append(
                f"    {reading_sub_var(ci, part.name)}.emplace("
                f"g_reading_session->declare_subscriber(")
            lines.append(f'        zenoh::KeyExpr("{topic}"),')
            lines.append(f"        [](const zenoh::Sample& s) {{")
            lines.append(
                f"            std::vector<double> v;")
            lines.append(
                f"            std::string payload(s.get_payload().as_string());")
            lines.append(
                f"            if (!_parse_float_array(payload, v)) return;")
            lines.append(
                f"            if (v.size() < {total_dim}) return;")
            offset = 0
            for field_name, dim in fields:
                buf = reading_buf_var(ci, part.name, field_name)
                fresh = reading_fresh_var(ci, part.name, field_name)
                for j in range(dim):
                    lines.append(f"            {buf}({j}) = v[{offset + j}];")
                lines.append(f"            {fresh}.store(true);")
                offset += dim
            lines.append(f"        }},")
            lines.append(f"        zenoh::closures::none));")
        lines.append("")

    # Initial state via CraftView — per-craft. The EKF descriptor's
    # `initial_*_var` accept scalar (broadcast), list (positional), or
    # dict (by name); for now we resolve to a single broadcast value
    # and apply it to every craft. Per-craft variations land when the
    # codegen migrates the per-craft init resolution.
    def _resolve_var(field_name: str) -> float:
        v = getattr(filter_obj, field_name)
        if v is None:
            return float(filter_obj.initial_covariance)
        if isinstance(v, (list, tuple, dict)):
            # TODO: per-craft resolution (not exercised by single-craft
            # examples). For ex9, the user passes per-craft variants;
            # broadcast the first scalar we can find as a placeholder.
            if isinstance(v, dict):
                vals = [float(x) for x in v.values()]
                return vals[0] if vals else float(filter_obj.initial_covariance)
            if v:
                return float(v[0])
            return float(filter_obj.initial_covariance)
        return float(v)

    pos_var    = _resolve_var("initial_position_var")
    att_var    = _resolve_var("initial_attitude_var")
    vel_var    = _resolve_var("initial_velocity_var")
    angvel_var = _resolve_var("initial_angular_velocity_var")
    lines.append("    // Initial state.")
    for i in range(num):
        lines.append(f"    view_{i}.reset_to_rest();")
        lines.append(
            f"    view_{i}.set_state_covariance("
            f"{_f(pos_var)}, {_f(att_var)}, "
            f"{_f(vel_var)}, {_f(angvel_var)});")

    # Measurement registrations — for each measurement part, look up
    # the source via the same-name convention on the sim worlds. Each
    # sim world's namespace exports `craft` (single) or `craft_0/1/...`
    # (multi); we resolve the right one by matching the part's owning
    # craft within that world.
    lines.append("")
    lines.append("    // Measurement registrations.")
    # Set of measurement parts whose readings come from a Zenoh topic.
    reading_part_ids = {id(p) for _ci, p, _t in reading_topics}

    # Map each measurement to (est_craft_idx, sim_source_var | None).
    for m in filter_obj.measurements:
        # Find which est craft owns this measurement.
        est_idx = -1
        for i, c in enumerate(crafts):
            for p in c.all_parts():
                if id(p) == id(m):
                    est_idx = i
                    break
            if est_idx >= 0:
                break
        if est_idx < 0:
            raise RuntimeError(
                f"emit_filter_state_spec: measurement {m.name!r} not "
                f"found on any tracked craft.")

        est_var = craft_var_for(est_idx)

        # Search sim worlds for a matching part on the corresponding
        # craft slot (positional match across sim and est).
        sim_world_obj = None
        sim_craft_idx = -1
        sim_part = None
        for sw_name, sw in sim_worlds:
            sw_idx = est_idx if est_idx < len(sw.crafts) else 0
            sw_entry = sw.crafts[sw_idx]
            for cand in sw_entry.craft.all_parts():
                if cand.name == m.name and type(cand) is type(m):
                    sim_world_obj = sw
                    sim_craft_idx = sw_idx
                    sim_part = cand
                    break
            if sim_part is not None:
                break

        if id(m) in reading_part_ids:
            # Pattern C: per-field reading_from_buffer pointing at the
            # codegen-allocated buffers fed by the Zenoh subscriber.
            for field_name, dim in _meas_fields_for(m):
                lines.append(
                    f"    {filter_var}.measure<{dim}>("
                    f"&{est_var}.{m.name}().{field_name}, "
                    f"manta::reading_from_buffer<{dim}>("
                    f"&{reading_buf_var(est_idx, m.name, field_name)}, "
                    f"&{reading_fresh_var(est_idx, m.name, field_name)}));")
        elif sim_part is not None:
            source_var = (f"manta_gen::{sim_world_obj.name}::"
                          f"{sim_craft_var(sim_world_obj, sim_craft_idx)}"
                          f".{sim_part.name}()")
            for field_name, dim in _meas_fields_for(m):
                lines.append(
                    f"    {filter_var}.measure<{dim}>("
                    f"&{est_var}.{m.name}().{field_name}, "
                    f"manta::reading_from<{dim}>({source_var}.{field_name}));")
        else:
            # No source declared at all — emit a stub comment.
            for field_name, dim in _meas_fields_for(m):
                lines.append(
                    f"    // (no reading source for {m.name}.{field_name} — "
                    f"call ekf.read_topic({m.name}, '...') in Python config)")
    lines += ["}", ""]

    # ---- tick() ----
    #
    # Sim worlds step in their own modules' tick(); the per-target
    # orchestrator calls them before invoking ours. Here we just mirror
    # actuator state (so the predict's process model sees the sim's
    # commanded inputs) and run predict + measurement updates.
    lines += [
        "void tick() {",
    ]
    for ci, c in enumerate(crafts):
        for est_part in c.all_parts():
            actuator_pairs = getattr(type(est_part), "actuator_state", None) or []
            if not actuator_pairs:
                continue
            # Find sim source: same craft slot, same part name + type.
            found_sim_var = None
            found_sim_part = None
            for sw_name, sw in sim_worlds:
                sw_idx = ci if ci < len(sw.crafts) else 0
                sw_entry = sw.crafts[sw_idx]
                for cand in sw_entry.craft.all_parts():
                    if cand.name == est_part.name and \
                            type(cand) is type(est_part):
                        found_sim_var = (
                            f"manta_gen::{sw.name}::"
                            f"{sim_craft_var(sw, sw_idx)}")
                        found_sim_part = cand
                        break
                if found_sim_part is not None:
                    break
            if found_sim_part is None:
                continue
            est_var = craft_var_for(ci)
            jet_var = f"{est_var}_jet"
            for setter, getter in actuator_pairs:
                lines.append(
                    f"    {jet_var}.{est_part.name}().{setter}("
                    f"JetType({found_sim_var}.{found_sim_part.name}().{getter}()));")
                lines.append(
                    f"    {est_var}.{est_part.name}().{setter}("
                    f"{found_sim_var}.{found_sim_part.name}().{getter}());")

    lines += [
        f"    {filter_var}.predict(DT, g_Q);",
        f"    {filter_var}.run_pending_updates();",
        "}",
        "",
        "void shutdown() {}",
        "",
        f"Harness harness{{}};",
        f"void Harness::setup()    {{ manta_gen::{name}::setup(); }}",
        f"void Harness::tick()     {{ manta_gen::{name}::tick(); }}",
        f"void Harness::shutdown() {{ manta_gen::{name}::shutdown(); }}",
        "",
        f"}}  // namespace manta_gen::{name}",
        "",
    ]

    return "\n".join(lines)
