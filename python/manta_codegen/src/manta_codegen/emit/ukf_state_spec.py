"""StateSpec-based codegen for UKF targets.

Mirrors `ekf_state_spec.py` but produces `UKFGeneric<StateSpec, MeasDim>`
output. Differences from the EKF flavor:

  * No Jet shadow world — UKF runs sigma-point propagation through the
    value world directly.
  * `bind(w_real)` takes only the value world.
  * No bias-as-tracked-slice yet (the analytical L gain that EKFGeneric
    uses lives in `extract_F_L_and_ref`, which UKF doesn't have — the
    UKF gets bias-state diffusion via the registry-driven Q diagonal).

Gated by `UKF.use_state_spec=True`.
"""

from __future__ import annotations

from .._format import cpp_float as _f
from ._util import GENERATED_BANNER, CPP_INCLUDE_GUARD, class_name_for_craft
from .ekf_state_spec import (
    _MEAS_FIELDS, _meas_fields_for, _world_unique_crafts,
    _world_noise_slot_count,
)


def emit_filter_hpp(target, filter_obj) -> str:
    from ..core import World as WorldDesc
    from ..estimation.ekf import EKF as EKFDesc
    from ..estimation.ukf import UKF as UKFDesc

    world = filter_obj.world
    name  = world.name
    crafts = _world_unique_crafts(world)
    num = len(crafts)
    cls_real = [f"{class_name_for_craft(c.name)}T<double>" for c in crafts]
    spec_args = ", ".join(["manta::manifold::RigidBody"] * num)
    meas_dim = 0
    filter_var = filter_obj.cpp_var_name()

    def craft_var_for(idx: int) -> str:
        return "craft" if num == 1 else f"craft_{idx}"

    lines: list[str] = [
        GENERATED_BANNER, CPP_INCLUDE_GUARD, "",
        '#include "manta/core/harness.hpp"',
        '#include "manta/core/scene.hpp"',
        '#include "manta/core/world.hpp"',
        '#include "manta/estimation/craft_view.hpp"',
        '#include "manta/estimation/generic_ukf.hpp"',
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
    seen_craft_hpp: set[str] = set()
    for c in crafts:
        if c.name not in seen_craft_hpp:
            seen_craft_hpp.add(c.name)
            lines.append(f'#include "{c.name}_craft.hpp"')

    lines += [
        "",
        f"namespace manta_gen::{name} {{",
        "",
        f"inline constexpr float DT             = {_f(target.dt)};",
        f"inline constexpr float SIM_RATE_MULT  = {_f(target.sim_rate_mult)};",
        "",
        f"using Spec = manta::estimation::StateSpec<{spec_args}>;",
        f"using UkfT = manta::estimation::UKFGeneric<Spec, /*MeasDim=*/{meas_dim}>;",
        "",
        "extern manta::WorldT<double>          w;",
        "extern manta::SceneT<double>*         scene;",
    ]
    if world.fields:
        for i, f in enumerate(world.fields):
            lines.append(f"extern {f.cpp_class} field_{i};")
    for i, c in enumerate(crafts):
        lines.append(f"extern {cls_real[i]} {craft_var_for(i)};")
    lines.append("")
    lines.append(f"extern UkfT {filter_var};")
    for i, _c in enumerate(crafts):
        lines.append(
            f"extern manta::estimation::CraftView<UkfT, {i}> view_{i};")
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

    sim_worlds: list[tuple[str, object]] = []
    for drive in target.drives:
        if isinstance(drive, (EKFDesc, UKFDesc)) or not isinstance(drive, WorldDesc):
            continue
        sim_worlds.append((drive.name, drive))

    def sim_craft_var(sim_world, idx: int) -> str:
        if len(sim_world.crafts) <= 1:
            return "craft"
        return f"craft_{idx}"

    reading_topics = list(getattr(filter_obj, "reading_topics", []) or [])

    def reading_buf_var(part_name: str, field_name: str) -> str:
        return f"reading_{part_name}_{field_name}_buf"
    def reading_fresh_var(part_name: str, field_name: str) -> str:
        return f"reading_{part_name}_{field_name}_fresh"
    def reading_sub_var(part_name: str) -> str:
        return f"reading_{part_name}_sub"

    lines: list[str] = [
        GENERATED_BANNER, "",
        f'#include "{name}.hpp"',
    ]
    for sw_name, _sw in sim_worlds:
        lines.append(f'#include "{sw_name}.hpp"')
    lines += [
        "",
        "#include <atomic>",
        "#include <optional>",
        "#include <string>",
        "#include <string_view>",
        "#include <vector>",
        "",
        "#include <Eigen/Core>",
    ]
    if reading_topics:
        lines.append("#include <zenoh.hxx>")
    lines += [
        "",
        f"namespace manta_gen::{name} {{",
        "",
        "manta::WorldT<double>  w{};",
        "manta::SceneT<double>* scene = nullptr;",
    ]
    for i, f in enumerate(world.fields):
        lines.append(f.emit_construction(f"field_{i}"))
    for i, c in enumerate(crafts):
        lines.append(f"{cls_real[i]} {craft_var_for(i)}{{}};")

    track_chain = ".".join(f"track({craft_var_for(i)})" for i in range(num))
    lines += [
        "",
        f"UkfT {filter_var}{{ "
        f"manta::estimation::make_state().{track_chain}.build(), "
        f"{_f(filter_obj.alpha)}, {_f(filter_obj.beta)}, {_f(filter_obj.kappa)} "
        f"}};",
    ]
    for i in range(num):
        lines.append(
            f"manta::estimation::CraftView<UkfT, {i}> view_{i}{{{filter_var}}};")
    lines += [
        "",
        f"}}  // namespace manta_gen::{name}",
        "",
    ]

    # Anonymous-namespace state — Q jitter for the kernel's LLT.
    lines += [
        "namespace {",
        "",
        f"using UkfT  = manta_gen::{name}::UkfT;",
        f"UkfT::StateCov g_Q = UkfT::StateCov::Identity() * "
        f"{_f(filter_obj.q_jitter)};",
        "",
    ]

    if reading_topics:
        lines.append("// ---- Pattern C reading sources (Zenoh-fed buffers) ----")
        lines.append("std::optional<zenoh::Session> g_reading_session;")
        for part, _topic in reading_topics:
            for field_name, dim in _meas_fields_for(part):
                lines.append(
                    f"Eigen::Matrix<double, {dim}, 1> "
                    f"{reading_buf_var(part.name, field_name)}{{}};")
                lines.append(
                    f"std::atomic<bool> "
                    f"{reading_fresh_var(part.name, field_name)}{{false}};")
            lines.append(
                f"std::optional<zenoh::Subscriber<void>> "
                f"{reading_sub_var(part.name)};")
        lines += [
            "",
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

    # setup
    lines += [
        f"namespace manta_gen::{name} {{",
        "",
        "void setup() {",
        "    w.clock().set_dt(DT);",
        "    scene = &w.create_scene();",
    ]
    for i, f in enumerate(world.fields):
        lines.append(f"    w.register_field(field_{i});")
    for i in range(num):
        lines.append(f"    scene->add_craft({craft_var_for(i)});")
    lines += [
        "",
        f"    {filter_var}.bind(w);",
        "",
    ]

    # Pattern C subscribers.
    if reading_topics:
        lines.append("    g_reading_session.emplace("
                     "zenoh::Session::open(zenoh::Config::create_default()));")
        for part, topic in reading_topics:
            fields = _meas_fields_for(part)
            total_dim = sum(d for _, d in fields)
            lines.append(
                f"    {reading_sub_var(part.name)}.emplace("
                f"g_reading_session->declare_subscriber(")
            lines.append(f'        zenoh::KeyExpr("{topic}"),')
            lines.append(f"        [](const zenoh::Sample& s) {{")
            lines.append(f"            std::vector<double> v;")
            lines.append(f"            std::string payload(s.get_payload().as_string());")
            lines.append(f"            if (!_parse_float_array(payload, v)) return;")
            lines.append(f"            if (v.size() < {total_dim}) return;")
            offset = 0
            for field_name, dim in fields:
                buf = reading_buf_var(part.name, field_name)
                fresh = reading_fresh_var(part.name, field_name)
                for j in range(dim):
                    lines.append(f"            {buf}({j}) = v[{offset + j}];")
                lines.append(f"            {fresh}.store(true);")
                offset += dim
            lines.append(f"        }},")
            lines.append(f"        zenoh::closures::none));")
        lines.append("")

    # Initial state via CraftView.
    pos_var    = float(filter_obj.initial_position_var or filter_obj.initial_covariance)
    att_var    = float(filter_obj.initial_attitude_var or filter_obj.initial_covariance)
    vel_var    = float(filter_obj.initial_velocity_var or filter_obj.initial_covariance)
    angvel_var = float(filter_obj.initial_angular_velocity_var or filter_obj.initial_covariance)
    lines.append("    // Initial state.")
    for i in range(num):
        lines.append(f"    view_{i}.reset_to_rest();")
        lines.append(
            f"    view_{i}.set_state_covariance("
            f"{_f(pos_var)}, {_f(att_var)}, "
            f"{_f(vel_var)}, {_f(angvel_var)});")

    # Measurement registrations.
    reading_part_ids = {id(p) for p, _t in reading_topics}
    lines.append("")
    lines.append("    // Measurement registrations.")
    for m in filter_obj.measurements:
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
                f"emit_filter_state_spec (UKF): measurement {m.name!r} not "
                f"found on any tracked craft.")
        est_var = craft_var_for(est_idx)
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
            for field_name, dim in _meas_fields_for(m):
                lines.append(
                    f"    {filter_var}.measure<{dim}>("
                    f"&{est_var}.{m.name}().{field_name}, "
                    f"manta::reading_from_buffer<{dim}>("
                    f"&{reading_buf_var(m.name, field_name)}, "
                    f"&{reading_fresh_var(m.name, field_name)}));")
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
            for field_name, dim in _meas_fields_for(m):
                lines.append(
                    f"    // (no reading source for {m.name}.{field_name} — "
                    f"call ukf.read_topic({m.name}, '...') in Python config)")
    lines += ["}", ""]

    # tick
    lines += [
        "void tick() {",
    ]
    for ci, c in enumerate(crafts):
        for est_part in c.all_parts():
            actuator_pairs = getattr(type(est_part), "actuator_state", None) or []
            if not actuator_pairs:
                continue
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
            for setter, getter in actuator_pairs:
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
