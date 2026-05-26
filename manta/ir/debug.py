"""Graph introspection.

`Graph.summary()` dumps inputs, outputs, and a compact view of the
underlying CasADi function. We don't expose the full CasADi expression
graph here — that's better viewed with CasADi's own `print(function)`
or `function.print_options()`. The summary is meant for "did I wire up
the right inputs and outputs with the right frames?" debugging at
authoring time.
"""

from __future__ import annotations

from .frames import _format_frame
from .types import Mat3, Quat, Scalar, Vec3


def _spec_label(value) -> str:
    """One-line label for an IR value: type + frames + shape."""
    cls = type(value).__name__
    if isinstance(value, Scalar):
        return f"Scalar  shape={tuple(value._mx.shape)}"
    if isinstance(value, Vec3):
        return (f"Vec3[{_format_frame(value._frame)}]  "
                f"shape={tuple(value._mx.shape)}")
    if isinstance(value, Mat3):
        return (f"Mat3[{_format_frame(value._from_frame)}, "
                f"{_format_frame(value._to_frame)}]  "
                f"shape={tuple(value._mx.shape)}")
    if isinstance(value, Quat):
        return (f"Quat[{_format_frame(value._from_frame)}, "
                f"{_format_frame(value._to_frame)}]  "
                f"shape={tuple(value._mx.shape)}")
    # Fallback (e.g., _RawMatrix from Jacobian outputs).
    shape = getattr(value, "shape", "?")
    return f"{cls}  shape={tuple(shape) if shape != '?' else '?'}"


def render_summary(graph) -> str:
    """Render `Graph.summary()`."""
    lines: list[str] = []
    lines.append(f"Graph '{graph._name}':")

    lines.append(f"  Inputs ({len(graph._inputs)}):")
    if graph._inputs:
        name_w = max(len(n) for n in graph._inputs)
        for name, value in graph._inputs.items():
            lines.append(f"    {name.ljust(name_w)} : {_spec_label(value)}")
    else:
        lines.append("    (none)")

    lines.append(f"  Outputs ({len(graph._outputs)}):")
    if graph._outputs:
        name_w = max(len(n) for n in graph._outputs)
        for name, value in graph._outputs.items():
            lines.append(f"    {name.ljust(name_w)} : {_spec_label(value)}")
    else:
        lines.append("    (none)")

    return "\n".join(lines)
