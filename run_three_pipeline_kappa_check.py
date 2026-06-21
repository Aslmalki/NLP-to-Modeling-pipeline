#!/usr/bin/env python3
"""
Run run_blue_ocean_pipeline.py three times with a fresh codifier each time.
Exits 0 only if cohen_kappa matches on all three runs (from methodology_summary.json).

After each run, archives:
  outputs/validation/determinism_runs/run{N}/validation_labels.csv
  outputs/validation/determinism_runs/run{N}/methodology_summary.json
  outputs/validation/determinism_runs/run{N}/classification_report.txt
  outputs/validation/determinism_runs/run{N}/module_a_codifier_output.csv
"""
import json
import os
import shutil
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VALIDATION_DIR = os.path.join(PROJECT_ROOT, "outputs", "validation")
DETERMINISM_DIR = os.path.join(VALIDATION_DIR, "determinism_runs")
PIPELINE_SCRIPT = os.path.join(PROJECT_ROOT, "run_blue_ocean_pipeline.py")
CODIFIER_CSV = os.path.join(VALIDATION_DIR, "module_a_codifier_output.csv")
SUMMARY_JSON = os.path.join(VALIDATION_DIR, "methodology_summary.json")

ARCHIVE_FILES = (
    "validation_labels.csv",
    "methodology_summary.json",
    "classification_report.txt",
    "module_a_codifier_output.csv",
)


def _archive_run(run_i):
    dest = os.path.join(DETERMINISM_DIR, f"run{run_i}")
    os.makedirs(dest, exist_ok=True)
    for name in ARCHIVE_FILES:
        src = os.path.join(VALIDATION_DIR, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dest, name))
    print(f"Archived run {run_i} artifacts to {dest}", flush=True)


def main():
    env = os.environ.copy()
    for k, v in (
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
        ("NUMBA_NUM_THREADS", "1"),
    ):
        env[k] = v

    kappas = []
    paired = []
    for run_i in range(1, 4):
        if os.path.isfile(CODIFIER_CSV):
            os.remove(CODIFIER_CSV)
        print(
            f"\n{'=' * 70}\nTRIPLE-KAPPA-CHECK: starting pipeline run {run_i}/3\n{'=' * 70}\n",
            flush=True,
        )
        r = subprocess.run(
            [sys.executable, "-u", PIPELINE_SCRIPT],
            cwd=PROJECT_ROOT,
            env=env,
        )
        if r.returncode != 0:
            print(f"ERROR: pipeline run {run_i} exited with code {r.returncode}", file=sys.stderr)
            sys.exit(r.returncode)
        _archive_run(run_i)
        with open(SUMMARY_JSON, encoding="utf-8") as f:
            data = json.load(f)
        k = data.get("cohen_kappa")
        p = data.get("kappa_paired_rows")
        kappas.append(k)
        paired.append(p)
        print(f"Run {run_i}: cohen_kappa={k} kappa_paired_rows={p}", flush=True)

    k0 = kappas[0]
    if k0 is None or any(k != k0 for k in kappas):
        print("FAIL: Cohen's kappa differed across runs:", kappas, file=sys.stderr)
        print("paired rows per run:", paired, file=sys.stderr)
        sys.exit(2)
    if len(set(paired)) != 1:
        print("FAIL: kappa_paired_rows differed across runs:", paired, file=sys.stderr)
        sys.exit(3)

    print("\nOK: All three runs agree — cohen_kappa =", k0, "paired_rows =", paired[0], flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
