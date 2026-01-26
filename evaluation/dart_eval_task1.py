import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DART-Eval Task 1 (Prioritizing Known Regulatory Elements)"
    )
    parser.add_argument(
        "--dart_eval_dir",
        default=None,
        help="Path to DART-Eval repo (default: ../DART-Eval)",
    )
    parser.add_argument(
        "--dart_work_dir",
        default=None,
        help="Work dir used by DART-Eval (overrides $DART_WORK_DIR)",
    )
    parser.add_argument(
        "--ccre_bed",
        default=None,
        help="Path to ENCODE cCRE BED (ENCFF420VPZ.bed)",
    )
    parser.add_argument(
        "--processed_tsv",
        default=None,
        help="Output TSV for processed cCREs",
    )
    parser.add_argument(
        "--model",
        default="nucleotide_transformer",
        help="Model module name for zero-shot eval (e.g., nucleotide_transformer)",
    )
    parser.add_argument(
        "--run_dataset_gen",
        action="store_true",
        help="Run dataset generation step",
    )
    parser.add_argument(
        "--run_zero_shot",
        action="store_true",
        help="Run zero-shot likelihood eval",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    dart_eval_dir = (
        Path(args.dart_eval_dir).resolve()
        if args.dart_eval_dir
        else (script_dir.parent.parent / "DART-Eval").resolve()
    )

    dart_work_dir = args.dart_work_dir or os.environ.get("DART_WORK_DIR")
    if not dart_work_dir:
        raise ValueError("DART_WORK_DIR is required (set env var or --dart_work_dir)")

    dart_work_dir = Path(dart_work_dir).resolve()

    ccre_bed = (
        Path(args.ccre_bed).resolve()
        if args.ccre_bed
        else dart_work_dir / "task_1_ccre" / "input_data" / "ENCFF420VPZ.bed"
    )

    processed_tsv = (
        Path(args.processed_tsv).resolve()
        if args.processed_tsv
        else dart_work_dir
        / "task_1_ccre"
        / "processed_inputs"
        / "ENCFF420VPZ_processed.tsv"
    )

    env = os.environ.copy()
    env["DART_WORK_DIR"] = str(dart_work_dir)

    if args.run_dataset_gen:
        processed_tsv.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "dnalm_bench.task_1_paired_control.dataset_generators.encode_ccre",
            "--ccre_bed",
            str(ccre_bed),
            "--output_file",
            str(processed_tsv),
        ]
        print("Running dataset generation:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(dart_eval_dir), env=env, check=True)

    if args.run_zero_shot:
        module = f"dnalm_bench.task_1_paired_control.zero_shot.encode_ccre.{args.model}"
        cmd = [sys.executable, "-m", module]
        print("Running zero-shot eval:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(dart_eval_dir), env=env, check=True)


if __name__ == "__main__":
    main()
