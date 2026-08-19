"""Thread-local active-Graph registry shared by graph construction and values."""

from __future__ import annotations

import threading
from typing import Any

_local = threading.local()


def current_graph() -> Any:
    graph = getattr(_local, "graph", None)
    if graph is None:
        raise RuntimeError(
            "No active Graph. Wrap IR-building code in 'with manta.ir.Graph() "
            "as g:' before calling .input() or constructing constants.")
    return graph


def get_active_graph() -> Any:
    return getattr(_local, "graph", None)


def set_active_graph(graph: Any) -> None:
    _local.graph = graph
