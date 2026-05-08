"""TetherEndpoint — one end of a spring-damper tether between two crafts.

Note: a TetherEndpoint requires a `manta::coupling::Tether` instance shared
between its two endpoints. The Python side currently doesn't model the
shared Tether object — descriptors hand out `cpp_extra_construction` snippets
so the codegen can emit the necessary tether instantiation. This is a
placeholder API; full multi-craft tether support in codegen will land
alongside multi-craft binary emission.
"""

from __future__ import annotations

from ...core import PartDescriptor


class TetherEndpoint(PartDescriptor):
    """One end of a Tether (spring-damper line connecting two parts, possibly
    on different crafts). Requires a Tether object to be constructed and
    paired up by the user's C++ glue (codegen-side wiring TBD).

    Required fields: none.
    Telemetry: none (yet).
    """

    cpp_class_template = "manta::parts::TetherEndpointT"
    cpp_header         = "manta/parts/coupling/tether_endpoint.hpp"

    def __init__(self, name: str, tether_var: str, is_a: bool, **kwargs) -> None:
        """`tether_var` is the C++ variable name of the
        `manta::coupling::TetherT<Scalar>` (or its `Tether` MFloat-alias) in scope
        at the construction site. The caller's glue code is responsible for
        ensuring it exists with the matching Scalar. `is_a=True` registers as
        the first endpoint, `False` as the second."""
        super().__init__(name=name, **kwargs)
        self.tether_var = tether_var
        self.is_a = bool(is_a)

    def emit_constructor_args(self, scalar: str = "manta::MFloat") -> str:
        return f'"{self.name}", {self.tether_var}, {"true" if self.is_a else "false"}'
