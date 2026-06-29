"""The runtime descriptor — everything the JS side needs to drive the shims.

A pure-data dict (serialized to `<basename>.descriptor.json` and inlined into
the emitted JS) describing the Module's shape: its state-slot layout and init
vector, the control / noise / parameter / output field names, and — per entry
point — the flat in/out buffer layout from `layout.py`. The JS runtime reads
this to pack typed values into the WASM heap and unpack results, with no
manta-side knowledge baked into the JavaScript itself.

Built straight off the Module + the resolved `EntryLayout`s; dispatches only
on Port `Role` and State-field kind, exactly like every other backend.
"""

from __future__ import annotations

from ...ir.module import Role


def _fields(port):
    if port is None:
        return []
    return [{"name": f.name, "dim": f.dim, "default": _num(f.default),
             "sigma": f.sigma, "rate": f.rate} for f in port.fields]


def _num(v):
    """A JSON-safe scalar/vector default (numpy → list/float)."""
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return [float(x) for x in v.ravel()]
    except ImportError:                       # pragma: no cover
        pass
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    return float(v)


def _slot(s):
    d = {"name": s.name, "kind": s.kind, "off": s.off, "size": s.size}
    if s.writes_state:
        d["writesState"] = True
    return d


def build_descriptor(module, layouts, *, class_name: str,
                     basename: str) -> dict:
    """The JSON-able runtime descriptor for `module`."""
    spec = module.spec
    u = module.sole_port(Role.CONTROL)
    noise = module.sole_port(Role.NOISE)
    params = module.sole_port(Role.PARAMETER)

    state_slots = []
    if spec is not None:
        for slot in spec.slots:
            state_slots.append({"name": slot.name,
                                "off": slot.ambient_offset,
                                "dim": slot.ambient_dim})

    # Held state fields (e.g. the filter's `x` and covariance `P`) with their
    # init, so a stateful JS view can seed itself; kind is normalized to match
    # the entry-slot kinds ("manifold" -> "state"). Empty for stateless modules.
    state_fields = []
    if module.state is not None:
        for f in module.state.fields:
            state_fields.append({"name": f.name,
                                 "kind": f.kind.replace("manifold", "state"),
                                 "shape": list(f.shape),
                                 "init": _num(f.init)})

    # STATE-role ports (e.g. the regulator's live `x` and held `x_ref`) with
    # their init operating point. Empty for sim/filter modules.
    state_refs = [{"name": p.name, "dim": p.size, "init": _num(p.init)}
                  for p in module.ports_by_role(Role.STATE)]

    outputs = {}
    for p in module.ports_by_role(Role.OUTPUT):
        outputs[p.name] = {"dim": sum(f.dim for f in p.fields),
                           "fields": _fields(p)}
    measurements = {}
    for p in module.ports_by_role(Role.MEASUREMENT):
        measurements[p.name] = {"dim": p.size, "rate": p.rate}

    return {
        "name": module.name,
        "className": class_name,
        "basename": basename,
        "hosting": module.hosting.value,
        "ambientDim": spec.ambient_dim if spec else 0,
        "tangentDim": spec.tangent_dim if spec else 0,
        "stateSlots": state_slots,
        "stateFields": state_fields,
        "stateRefs": state_refs,
        "initialState": (_num(module.initial_x)
                         if module.initial_x is not None else []),
        "inputs": _fields(u),
        "noise": _fields(noise),
        "params": _fields(params),
        "measurements": measurements,
        "outputs": outputs,
        "entryPoints": [
            {"method": lay.method, "shim": lay.shim,
             "inSize": lay.in_size, "outSize": lay.out_size,
             "in": [_slot(s) for s in lay.inputs],
             "out": [_slot(s) for s in lay.outputs]}
            for lay in layouts],
    }
