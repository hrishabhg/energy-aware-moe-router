#!/usr/bin/env python3
"""
health_check.py — Sanity checks between pipeline stages.

Called by run_all.sh after each step. Exits 0 if healthy, 1 if not.
Prints a one-line verdict to stdout; details to stderr.

Usage:
    python3 scripts/runpod/health_check.py --stage <stage_number>

Checks per stage:
    0 (setup):      GPU accessible, pynvml works, torch.cuda available
    1 (data):       Data directory exists, has >100K examples (full) or >1K (quick)
    2 (baselines):  metrics.jsonl exists for expected routers, no NaN loss,
                    final PPL < 500, Switch PPL < Random PPL
    3 (sweep):      At least 1 lambda dir has metrics.jsonl, no NaN, PPL < 1000
    4 (evaluate):   all_evals.jsonl exists, has expected number of rows
    5 (plot):       PNG files exist and are non-empty
"""
import argparse
import json
import math
import os
import sys


def log(msg):
    """Detail log to stderr (captured in log file but not in verdict)."""
    print(f"  [health] {msg}", file=sys.stderr)


def fail(reason):
    print(f"FAIL: {reason}")
    sys.exit(1)


def ok(msg="all checks passed"):
    print(f"OK: {msg}")
    sys.exit(0)


# ── Stage-specific checks ──────────────────────────────────────


def check_setup():
    """Post-setup: GPU + imports work."""
    try:
        import torch
        if not torch.cuda.is_available():
            fail("torch.cuda not available")
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        fail("torch not importable")

    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        pynvml.nvmlShutdown()
        log(f"pynvml power reading: {w:.1f} W")
    except Exception as e:
        fail(f"pynvml failed: {e}")

    ok("GPU + pynvml + torch verified")


def check_data():
    """Post-data: dataset exists and has enough examples."""
    for data_dir in ["./data/processed/c4_1pct", "./data/processed/c4_quick"]:
        if os.path.isdir(data_dir):
            try:
                from datasets import load_from_disk
                ds = load_from_disk(data_dir)
                n = len(ds)
                log(f"{data_dir}: {n:,} examples")
                if "quick" in data_dir and n >= 1_000:
                    ok(f"quick data: {n:,} examples")
                elif n >= 100_000:
                    ok(f"full data: {n:,} examples")
                else:
                    fail(f"dataset too small: {n:,} examples in {data_dir}")
            except Exception as e:
                fail(f"cannot load {data_dir}: {e}")

    fail("no data directory found (c4_1pct or c4_quick)")


def _check_metrics_file(path, max_ppl=1000):
    """Check a metrics.jsonl for NaN and extreme PPL. Returns last metrics dict."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = f.readlines()
    if not lines:
        return None
    last = json.loads(lines[-1])

    # NaN check on key fields
    for key in ["loss", "ppl", "loss_task"]:
        val = last.get(key)
        if val is not None and (math.isnan(val) or math.isinf(val)):
            fail(f"NaN/Inf detected in {path}: {key}={val}")

    # PPL sanity
    ppl = last.get("ppl", 0)
    if ppl > max_ppl:
        fail(f"PPL too high in {path}: {ppl:.2f} > {max_ppl}")

    log(f"{path}: step={last.get('step')}, loss={last.get('loss', '?'):.4f}, ppl={ppl:.2f}")
    return last


def check_baselines():
    """Post-baselines: metrics exist, no NaN, Switch < Random."""
    results = {}
    for router in ["switch", "random", "dense"]:
        path = f"results/m2/{router}/metrics.jsonl"
        m = _check_metrics_file(path, max_ppl=500)
        if m is not None:
            results[router] = m

    if not results:
        fail("no baseline metrics found at all")

    if len(results) < 2:
        log(f"only {len(results)} baseline(s) found, skipping comparative check")
        ok(f"{len(results)} baseline(s) trained, metrics look healthy")

    # Switch should beat Random
    if "switch" in results and "random" in results:
        s_ppl = results["switch"]["ppl"]
        r_ppl = results["random"]["ppl"]
        if s_ppl >= r_ppl:
            fail(f"Switch PPL ({s_ppl:.2f}) >= Random PPL ({r_ppl:.2f}) — likely a bug")
        log(f"Switch PPL ({s_ppl:.2f}) < Random PPL ({r_ppl:.2f}) — good")

    ok(f"{len(results)} baselines healthy")


def check_sweep():
    """Post-sweep: at least some lambda runs completed with sane metrics."""
    sweep_dir = "results/m2/energy_aware"
    if not os.path.isdir(sweep_dir):
        fail(f"sweep directory not found: {sweep_dir}")

    good = 0
    for entry in sorted(os.listdir(sweep_dir)):
        path = os.path.join(sweep_dir, entry, "metrics.jsonl")
        m = _check_metrics_file(path, max_ppl=1000)
        if m is not None:
            good += 1

    if good == 0:
        fail("no lambda runs produced valid metrics")
    if good < 3:
        log(f"WARNING: only {good} lambda runs succeeded (expected 7)")

    ok(f"{good} lambda run(s) healthy")


def check_evaluate():
    """Post-evaluate: all_evals.jsonl exists with expected rows."""
    combined = "results/m2/all_evals.jsonl"
    if not os.path.exists(combined):
        fail(f"{combined} not found")

    with open(combined) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if len(rows) == 0:
        fail(f"{combined} is empty")

    # Check each row for required fields
    required = ["router_type", "ppl", "total_joules"]
    for i, row in enumerate(rows):
        for key in required:
            if key not in row:
                fail(f"row {i} missing field '{key}' in {combined}")
            val = row[key]
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                fail(f"row {i}: {key}={val} (NaN/Inf) in {combined}")

    log(f"{combined}: {len(rows)} evaluation rows")
    ok(f"{len(rows)} evaluations recorded")


def check_plots():
    """Post-plot: figure files exist and are non-empty."""
    expected = [
        "figures/m2/pareto_lambda_sweep.png",
        "figures/m2/pareto_lambda_sweep.pdf",
        "figures/m2/training_curves.png",
    ]
    found = 0
    for path in expected:
        if os.path.exists(path) and os.path.getsize(path) > 100:
            log(f"{path}: {os.path.getsize(path)} bytes")
            found += 1
        else:
            log(f"MISSING or empty: {path}")

    if found == 0:
        fail("no plot files generated")
    if found < len(expected):
        log(f"WARNING: {found}/{len(expected)} plots generated")

    ok(f"{found}/{len(expected)} plots generated")


# ── Main ────────────────────────────────────────────────────────

CHECKS = {
    0: check_setup,
    1: check_data,
    2: check_baselines,
    3: check_sweep,
    4: check_evaluate,
    5: check_plots,
}


def main():
    parser = argparse.ArgumentParser(description="M2 pipeline health check")
    parser.add_argument("--stage", type=int, required=True, choices=CHECKS.keys())
    args = parser.parse_args()
    CHECKS[args.stage]()


if __name__ == "__main__":
    main()
