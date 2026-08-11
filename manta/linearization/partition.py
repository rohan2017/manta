"""Structural analyses over Jacobian sparsity patterns.

Both functions work at SLOT granularity on the tangent space: the
dependency closure decides which slots a filter must co-estimate, and
the partition finds tangent blocks whose covariance never couples.
"""

from __future__ import annotations

import numpy as np


def dependency_closure(F_pattern, sot, seed_slots) -> set[str]:
    """Expand `seed_slots` to the set closed under the dynamics:
    backward reachability on F's structural slot-dependency graph.

    `F_pattern[i, j] != 0` means tangent row i (next state) reads tangent
    column j (current state); `sot` maps tangent indices to slot names.
    A filter tracking slot A whose row depends on slot B must track B
    too, or B's frozen error silently biases A's predict.
    """
    rows, cols = np.nonzero(F_pattern)
    deps: dict[str, set[str]] = {}
    for i, j in zip(rows.tolist(), cols.tolist()):
        a, b = sot[i], sot[j]
        if a != b:
            deps.setdefault(a, set()).add(b)
    kept = set(seed_slots)
    frontier = list(seed_slots)
    while frontier:
        s = frontier.pop()
        for dep in deps.get(s, ()):
            if dep not in kept:
                kept.add(dep)
                frontier.append(dep)
    return kept


def partition_blocks(n_tangent, F_pattern, L_pattern, h_supports) -> list:
    """Partition the tangent state into independent subsystems — tangent
    blocks whose covariance never couples: under this partition a
    block-diagonal P stays block-diagonal through every predict and
    update. This is a *structural diagnostic* (surfaced as `EKF.n_blocks`
    and consumed by the tests as ground truth for cross-block zeros); the
    filters' kernels are dense — nothing exploits the partition for an
    O(n_b³) block predict today.

    Two tangent indices belong to the same block iff they are connected
    through any of the three coupling channels:

      * F couples them (a nonzero F[i, j] correlates δi' with δj);
      * a shared process-noise column drives both (L[:, k] nonzero in
        both rows ⇒ the same draw correlates them);
      * a sensor observes both (a measurement update mixes every column
        in its H support).

    This is a connected-components problem over those edges; the
    union-find below computes it without materializing the graph: each
    edge union(i, j) merges the endpoints' sets, with path compression
    keeping finds near O(1). The result is the list of components, each
    sorted, ordered by first index.
    """
    parent = list(range(n_tangent))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]      # path compression
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    rows, cols = np.nonzero(F_pattern)
    for i, j in zip(rows.tolist(), cols.tolist()):
        union(i, j)
    if L_pattern is not None:
        for k in range(L_pattern.shape[1]):
            rws = np.flatnonzero(L_pattern[:, k])
            for r in rws[1:]:
                union(int(rws[0]), int(r))
    for touched in h_supports:
        for c in touched[1:]:
            union(int(touched[0]), int(c))

    groups: dict[int, list[int]] = {}
    for i in range(n_tangent):
        groups.setdefault(find(i), []).append(i)
    blocks = [np.array(sorted(v)) for v in groups.values()]
    blocks.sort(key=lambda b: int(b[0]))
    return blocks
