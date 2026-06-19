"""The flat-buffer layout of one entry point — the single source of truth.

Both the C-ABI shim (`abi.py`) and the JS-facing descriptor (`descriptor.py`)
must agree, byte-for-byte, on where each argument and result lives inside the
two flat `double` buffers an entry point reads and writes. Computing that
layout in one place is what keeps them from drifting.

A backend dispatches only on the IR's types — but it never needs to *guess*
a buffer width: the densified `ca.Function` already knows the exact flat size
of every input and output (`numel_in`/`numel_out`), and `ep.args` /
`ep.writes + ep.returns` map 1:1 onto them in order. So the layout reads the
widths straight off the kernel and only consults the Module for the *meaning*
(role / state kind / public name) of each slot.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...ir.module import PortRef, StateRef


@dataclass(frozen=True)
class Slot:
    """One contiguous span of a flat `double` buffer.

    `name`  — the public name (the State field or Port name) JS marshals by.
    `kind`  — `"state"` / `"matrix"` for a StateRef, else the Port `Role`'s
              value (`"control"`, `"measurement"`, `"timestep"`, …).
    `off`   — element offset into the buffer.
    `size`  — element count (`numel` of the matching kernel input/output).
    `writes_state` — output slots only: this span is a written manifold state.
    """
    name: str
    kind: str
    off: int
    size: int
    writes_state: bool = False


@dataclass(frozen=True)
class EntryLayout:
    """The full in/out flat layout of one entry point."""
    method: str
    kernel: str            # the densified ca.Function's C symbol
    shim: str              # the exported C-ABI function name
    inputs: tuple[Slot, ...]
    outputs: tuple[Slot, ...]
    in_size: int
    out_size: int


def _arg_meaning(module, a) -> tuple[str, str]:
    """`(public name, kind)` for one kernel argument (a StateRef or PortRef)."""
    if isinstance(a, StateRef):
        return a.name, module.state.field(a.name).kind.replace("manifold",
                                                                "state")
    if isinstance(a, PortRef):
        return a.name, module.port(a.name).role.value
    raise TypeError(f"entry argument is neither StateRef nor PortRef: {a!r}")


def _out_meaning(module, name: str, writes: frozenset) -> tuple[str, str, bool]:
    """`(public name, kind, writes_state)` for one kernel output — a written
    State field (in `writes`) or a returned Port."""
    if name in writes:
        kind = module.state.field(name).kind.replace("manifold", "state")
        return name, kind, kind == "state"
    return name, module.port(name).role.value, False


def entry_layout(module, ep, kfn, *, kernel: str, shim: str) -> EntryLayout:
    """Resolve the flat in/out layout of `ep`, reading exact widths from the
    densified kernel `kfn` and meanings from the Module."""
    inputs: list[Slot] = []
    off = 0
    for i, a in enumerate(ep.args):
        name, kind = _arg_meaning(module, a)
        n = kfn.numel_in(i)
        inputs.append(Slot(name=name, kind=kind, off=off, size=n))
        off += n
    in_size = off

    outputs: list[Slot] = []
    off = 0
    out_names = list(ep.writes) + list(ep.returns)
    writes = frozenset(ep.writes)
    for j, nm in enumerate(out_names):
        name, kind, ws = _out_meaning(module, nm, writes)
        n = kfn.numel_out(j)
        outputs.append(Slot(name=name, kind=kind, off=off, size=n,
                            writes_state=ws))
        off += n
    out_size = off

    return EntryLayout(method=ep.method, kernel=kernel, shim=shim,
                       inputs=tuple(inputs), outputs=tuple(outputs),
                       in_size=in_size, out_size=out_size)
