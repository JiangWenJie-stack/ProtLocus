#!/usr/bin/env python3
"""Run reproducible baseline, accuracy, and optional ablation experiments."""

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

try:
    from tools.evaluate_accuracy import evaluate, read_predictions, read_truth
except ImportError:
    from evaluate_accuracy import evaluate, read_predictions, read_truth


ROOT = Path(__file__).resolve().parents[1]
ABLATIONS = {
    "accuracy-no-repeat": "--acc-no-repeat",
    "accuracy-no-vid-chain": "--acc-no-vid-chain",
    "accuracy-no-rescue": "--acc-no-rescue",
    "accuracy-no-junc-refine": "--acc-no-junc-refine",
    "accuracy-no-rerank": "--acc-no-rerank",
}


def parse_profile(stderr_text):
    for line in stderr_text.splitlines():
        if not line.startswith("[M::profile]"):
            continue
        result = {}
        for item in line.split()[1:]:
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            try:
                result[key] = float(value) if "." in value else int(value)
            except ValueError:
                result[key] = value
        return result
    return {}


def run_experiment(name, binary, genome, proteins, output_dir, threads, extra_args, mode_args, spsc, truth):
    output_path = output_dir / (name + ".gff")
    log_path = output_dir / (name + ".stderr.log")
    command = [
        str(binary),
        "-I",
        "-u",
        "-t%d" % threads,
        "--gff",
        "--gff-delim=|",
        "--profile",
    ]
    command.extend(mode_args)
    if spsc is not None:
        command.append("--spsc=%s" % spsc)
    command.extend(extra_args)
    command.extend([str(genome), str(proteins)])
    started = time.time()
    with output_path.open("wb") as output:
        completed = subprocess.run(command, stdout=output, stderr=subprocess.PIPE)
    wall_seconds = time.time() - started
    stderr_text = completed.stderr.decode("utf-8", errors="replace")
    log_path.write_text(stderr_text, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError("experiment %s failed with exit code %d; see %s" % (name, completed.returncode, log_path))
    result = {
        "command": command,
        "command_text": " ".join(shlex.quote(item) for item in command),
        "output": str(output_path),
        "stderr": str(log_path),
        "wall_seconds": wall_seconds,
        "profile": parse_profile(stderr_text),
    }
    if truth is not None:
        result["accuracy"] = evaluate(read_truth(truth), read_predictions(output_path), 0, 0.1)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("genome", type=Path, help="genome FASTA or ProtLocus index")
    parser.add_argument("proteins", type=Path, help="query protein FASTA")
    parser.add_argument("--binary", type=Path, default=ROOT / "protlocus")
    parser.add_argument("--output-dir", type=Path, default=Path("accuracy-benchmark"))
    parser.add_argument("--truth", type=Path, help="query-associated CDS truth TSV")
    parser.add_argument("--spsc", type=Path, help="optional ProtLocus splice-score file")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ablations", action="store_true", help="run each accuracy feature ablation")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="additional ProtLocus argument; use --extra-arg=-j2 for arguments beginning with '-'",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.threads < 1:
        print("run_accuracy_benchmark: --threads must be positive", file=sys.stderr)
        return 2
    for path, label in ((args.binary, "binary"), (args.genome, "genome"), (args.proteins, "proteins")):
        if not path.exists():
            print("run_accuracy_benchmark: %s does not exist: %s" % (label, path), file=sys.stderr)
            return 2
    if args.truth is not None and not args.truth.exists():
        print("run_accuracy_benchmark: truth does not exist: %s" % args.truth, file=sys.stderr)
        return 2
    if args.spsc is not None and not args.spsc.exists():
        print("run_accuracy_benchmark: splice scores do not exist: %s" % args.spsc, file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes = [("baseline", []) , ("accuracy", ["--accuracy"])]
    if args.ablations:
        modes.extend((name, ["--accuracy", option]) for name, option in sorted(ABLATIONS.items()))
    summary = {}
    try:
        for name, mode_args in modes:
            print("[benchmark] %s" % name, file=sys.stderr)
            summary[name] = run_experiment(
                name,
                args.binary,
                args.genome,
                args.proteins,
                args.output_dir,
                args.threads,
                args.extra_arg,
                mode_args,
                args.spsc,
                args.truth,
            )
    except (OSError, RuntimeError, ValueError) as error:
        print("run_accuracy_benchmark: %s" % error, file=sys.stderr)
        return 1
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())