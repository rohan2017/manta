"""Name resolution — the one suffix-matching rule, shared everywhere.

`resolve_suffix` is the single home of manta's "exact name, else unique
`.suffix`, else raise" convention (`StateSpec.slot`, the numpy Sim/EKF
sensor lookup, and every transform that takes user-supplied
input/sensor/parameter names all route through it). It lives here, in the
IR layer, so foundational modules (`state_spec`) need not reach up into
`linearization`.
"""

from __future__ import annotations


def resolve_suffix(key: str, candidates, *, label: str, who: str) -> str:
    """Resolve one user-supplied name against `candidates`.

    Accepts an exact match, else a unique `.<suffix>` match (craft-relative
    shorthand like ``"t.throttle"`` for ``"drone.t.throttle"``). Raises
    `KeyError` on an unknown or ambiguous key.
    """
    cands = list(candidates)
    if key in cands:
        return key
    matches = [n for n in cands if n.endswith("." + key)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(
            f"{who}: ambiguous {label} name {key!r}; matches {matches}. "
            f"Use the fully-qualified form.")
    raise KeyError(
        f"{who}: unknown {label} name {key!r}. Available: {sorted(cands)}")
