"""Recompute acceptance intervals for the held-out INS qualification reports.

This checks reported final statistics; rerun the acquisition fixtures to
regenerate the trials. Primary acceptance checks were selected before the
held-out trials: joint attitude/bias, physical heading, and joint boundary
error where present. Supplemental controls use their stated joint ANEES gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from manta.estimation import chi2_quantile

PRIMARY = (
    "v2-calibrated-heldout64.json",
    "v2-weak-heldout256.json",
    "v2-noisy-heldout128.json",
)
CONTROLS = (
    "v2-zero-spin64.json",
    "v2-motion64.json",
    "v2-south200.json",
    "v2-equator500.json",
    "v2-matched-raw.json",
    "v2-matched-packet.json",
)


def verify(directory):
    rows = []
    for filename in PRIMARY + CONTROLS:
        report = json.loads((directory / filename).read_text())
        final = report["records"][-1]
        n = report["seeds"]
        tests = [("attitude_bias_anees", 9)]
        if filename in PRIMARY:
            assert report["seed"] == 104729
            assert (
                report["error_coordinates"]
                == "gravity_and_earth_referenced_swing_twist_v2"
            )
            tests += [
                ("physical_heading_anees", 1),
                ("attitude_bias_boundary_anees", 12),
            ]
        for metric, dof in tests:
            bounds = [chi2_quantile(dof * n, p) / n for p in (0.025, 0.975)]
            value = final[metric]
            rows.append(
                {
                    "report": filename,
                    "metric": metric,
                    "value": value,
                    "bounds": bounds,
                    "pass": bounds[0] <= value <= bounds[1],
                }
            )
    return {
        "scope": "synthetic final-epoch qualification, not hardware deployment",
        "pass": all(row["pass"] for row in rows),
        "checks": rows,
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = verify(a.directory)
    text = json.dumps(result, indent=2) + "\n"
    if a.output:
        a.output.write_text(text)
    print(text)
    raise SystemExit(0 if result["pass"] else 2)


if __name__ == "__main__":
    main()
