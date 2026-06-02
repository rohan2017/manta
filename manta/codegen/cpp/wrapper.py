"""Back-compat shim for the Sim C++ wrapper.

The Sim wrapper now goes through the generic evaluator emitter
(`evaluator_wrapper.emit_evaluator_wrapper`) like LQR and the recurrence
blocks — Sim is just another evaluator block. `emit_wrapper(funcs, world,
…)` builds the Sim `EvaluatorSpec` from an already-extracted
`WorldFunctions` and renders it, so callers/tests that import `emit_wrapper`
directly keep working. New code should call the generic emitter.
"""

from __future__ import annotations

from pathlib import Path

from .evaluator_spec import sim_spec_from_funcs
from .evaluator_wrapper import emit_evaluator_wrapper


def emit_wrapper(funcs, world, out_dir: str | Path, *, class_name: str,
                 basename: str | None = None,
                 namespace: str = "manta_gen") -> dict[str, Path]:
    """Emit the Sim typed C++ wrapper from a `WorldFunctions` bundle."""
    espec = sim_spec_from_funcs(funcs, world)
    return emit_evaluator_wrapper(espec, out_dir, class_name=class_name,
                                  basename=basename, namespace=namespace)
