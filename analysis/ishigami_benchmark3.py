#!/usr/bin/env python3
"""Print Benjamin Vingren's published Ishigami Sobol results against exact values.

Reads the existing JSON; does not train models or evaluate Ishigami. Standard library only.
Optionally runs the C++ TCI executable before displaying published results.
"""

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
from urllib.request import urlopen
import os

dirname = os.path.dirname(__file__)
RESULTS_FILE = os.path.join(dirname, 'output/sensitivity_analyses/noiseless_ishigami.json')
RESULTS_URL = (
    "https://raw.githubusercontent.com/benjamin-vingren/"
    "ishigami_sensitivity/main/output/sensitivity_analyses/noiseless_ishigami.json"
)


def read_json(source):
    if source.startswith(("https://", "http://")):
        with urlopen(source, timeout=20) as response:
            return json.load(response)
    with Path(source).open(encoding="utf-8") as stream:
        return json.load(stream)


def exact_sobol():
    a, b = 7.0, 0.1
    p = math.pi
    variance = 0.5 + a*a/8 + b*p**4/5 + b*b*p**8/18
    s1 = (0.5 + b*p**4/5 + b*b*p**8/50) / variance
    s2 = (a*a/8) / variance
    s13 = (8*b*b*p**8/225) / variance
    return dict(zip(("S1", "S2", "S3", "ST1", "ST2", "ST3", "S1,3"),
                    (s1, s2, 0.0, 1.0-s2, s2, s13, s13)))


def indices(result):
    """Order matches the seven indices reported in the C++ Ishigami benchmark."""
    if len(result["S1"]) != 3 or len(result["ST"]) != 3:
        raise ValueError("Expected three-variable Ishigami results")
    return dict(zip(("S1", "S2", "S3", "ST1", "ST2", "ST3", "S1,3"),
                    (*result["S1"], *result["ST"], result["S2"][0][2])))


def print_table(title, results, sizes, reference, details=False):
    print(f"\n=== {title} ===")
    print(f"{'method':<12}{'nEval':>8}{'RMSE':>14}{'maxAbs':>14}")
    for n, result in zip(sizes, results):
        values = indices(result)
        errors = [values[k] - reference[k] for k in reference]
        rmse = math.sqrt(sum(e*e for e in errors) / len(errors))
        maximum = max(abs(e) for e in errors)
        print(f"{title.split()[0]:<12}{n:>8d}{rmse:>14.4e}{maximum:>14.4e}")

    if details and results:
        print(f"\nFinal {title} indices (nEval={sizes[len(results)-1]}):")
        values = indices(results[-1])
        print(f"{'index':<10}{'estimated':>14}{'exact':>14}{'absError':>14}")
        for label, truth in reference.items():
            value = values[label]
            print(f"{label:<10}{value:>14.8f}{truth:>14.8f}{abs(value-truth):>14.3e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json",
                         default=RESULTS_FILE, metavar="PATH_OR_URL",
                        help="Published results JSON, local path or raw GitHub URL")
    parser.add_argument("--tci", metavar="EXE", help="Run this C++ TCI executable first")
    parser.add_argument("--show-indices", action="store_true",
                        help="Print final individual Sobol indices as well")
    args, tci_options = parser.parse_known_args()
    if tci_options and not args.tci:
        parser.error(f"Unrecognized options without --tci: {tci_options}")

    if args.tci:
        cmd = [str(Path(args.tci).resolve()), *tci_options]
        print(f"=== C++ TCI === {shlex.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)

    data = read_json(args.json)
    sizes = data["n_samples"]
    gp = data["gp_results"]
    direct = data["sal_results"]
    if len(sizes) != len(gp) or len(sizes) != len(direct):
        raise ValueError("n_samples, gp_results, and sal_results have different lengths")

    reference = exact_sobol()
    print("\n=== Benjamin's Ishigami results ===")
    print("Ishigami: a=7, b=0.1; independent uniform inputs on [-pi,pi]^3")
    print("Reference file: " + args.json)
    print("Seven-index error: S1, S2, S3, ST1, ST2, ST3, S1,3")
    print_table("SALib direct", direct, sizes, reference, args.show_indices)
    print_table("GP surrogate", gp, sizes, reference, args.show_indices)


if __name__ == "__main__":
    main()
