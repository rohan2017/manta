"""Qualify one-sample mounted packets with the runtime compiler profile."""

import argparse
import json
from pathlib import Path

from . import earth_ins_split as fixture


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = fixture.compile_functions

    def native(functions, **options):
        return original(functions, **{**options, "optimization": "runtime"})

    fixture.compile_functions = native
    report = fixture.run(
        covariance="nonlinear",
        mounted=True,
        gyro_density=0.001,
        bias_sigma=0.001,
        packet_samples=1,
        aiding_samples=10,
        seeds=16,
        seed=82719,
    )
    report["compiler_profile"] = "runtime: -O3 -march=native"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"acceptance": report["acceptance"], **report["records"][-1]}),
        flush=True,
    )
    raise SystemExit(0 if report["acceptance"] == "pass" else 2)


if __name__ == "__main__":
    main()
