"""Live dictionaries backed by the simulator's packed state after each step.

Explicit dictionary edits are staged until step validation; in-place array edits
write the current vector. Every successful step binds new array views, so saved
slot arrays go stale while saved owner dictionaries remain live.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, Self

import numpy as np

from ...ir.state_spec import StateSlot, StateSpec, flatten_nested


class _EditingDict(dict[str, Any]):
    """Route every ordinary dict mutation through one invalidation boundary."""

    def __init__(self, changed: Callable[[], None]) -> None:
        super().__init__()
        self._changed = changed

    def __setitem__(self, key: str, value: Any) -> None:
        dict.__setitem__(self, key, value)
        self._changed()

    def __delitem__(self, key: str) -> None:
        dict.__delitem__(self, key)
        self._changed()

    def clear(self) -> None:
        dict.clear(self)
        self._changed()

    def update(self, *args: Any, **kwargs: Any) -> None:
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, key: str, *default: Any) -> Any:
        value = dict.pop(self, key, *default)
        self._changed()
        return value

    def popitem(self) -> tuple[str, Any]:
        value = dict.popitem(self)
        self._changed()
        return value

    def __ior__(self, other: Any) -> Self:
        self.update(other)
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        result = {}
        memo[id(self)] = result
        for key, value in self.items():
            result[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return result


class PackedSimState(_EditingDict):
    """Packed state authority with the existing live nested dictionary surface."""

    def __init__(
        self, spec: StateSpec, check_keys: Callable[[dict], None], value: dict
    ) -> None:
        super().__init__(self._invalidate)
        self._spec = spec
        self._check_keys = check_keys
        self._dirty = True
        self._flat: dict[str, Any] = {}
        self._locations: list[tuple[StateSlot, dict, str]] = []
        self.update(value)

    def _invalidate(self) -> None:
        self._dirty = True

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, dict) and dict.get(self, key) is not value:
            owner = _EditingDict(self._invalidate)
            owner.update(value)
            value = owner
        super().__setitem__(key, value)

    def prepare(self, current: np.ndarray) -> tuple[dict, np.ndarray]:
        if self._dirty:
            flat = flatten_nested(self)
            self._check_keys(flat)
            packed = self._spec.pack_projected(flat)
            locations = {}
            for owner, values in self.items():
                if isinstance(values, dict):
                    locations.update(
                        (f"{owner}.{key}", (values, key)) for key in values
                    )
                else:
                    locations[owner] = (self, owner)
            self._locations = [
                (slot, *locations[slot.name]) for slot in self._spec.slots
            ]
            self._flat = flat
            return flat, packed
        if not np.isfinite(current).all():
            raise ValueError("simulator state contains non-finite values")
        return self._flat, current

    def bind(self, packed: np.ndarray) -> None:
        for slot, owner, key in self._locations:
            start = slot.ambient_offset
            value = (
                float(packed[start])
                if slot.ambient_dim == 1
                else packed[start : start + slot.ambient_dim]
            )
            dict.__setitem__(owner, key, value)
            self._flat[slot.name] = value
        self._dirty = False
