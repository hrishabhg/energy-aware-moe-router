#!/bin/bash
# ============================================================
# run_all.sh — Autonomous M2 pipeline orchestrator
# ============================================================
# Runs the full M2 pipeline (steps 00–06) unattended with:
#   - Budget tracking: auto-stops pod when spending nears limit
#   - Health checks: validates each stage output before proceeding
#   - Auto-stop: stops the pod when done (or on failure)
#   - Notifications: optional webhook ping on completion/failure
#   - Quick test: optional fast pre-flight before committing hours
#
# Workflow:
#   1. Fill in .env.runpod (from .env.runpod.template)
#   2. SSH into your RunPod pod
#   3. cd /workspace/energy-aware-moe-router
#   4. nohup bash scripts/runpod/run_all.sh &> logs/run_all.log &
#   5. Go to sleep. Check results in the morning.
#
# The script logs everything to logs/run_all.log and also writes
# a machine-readable status file at logs/pipeline_status.json.
#
# Usage:
#   bash scripts/runpod/run_all.sh              # interactive (see output)
#   bash scripts/runpod/run_all.sh --debug      # NO auto-stop on failure (for debugging)
#   nohup bash scripts/runpod/run_all.sh &      # background (go to sleep)
# ============================================================
set -uo pipefail
# Note: intentionally NOT set -e because we handle errors per-step.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# ── CLI flags ───────────────────────────────────────────────

DEBUG_MODE=false
if [[ "${1:-}" == "--debug" ]]; then
    DEBUG_MODE=true
    echo "*** DEBUG MODE: pod will NOT auto-stop on failure ***"
    echo "*** SSH in anytime to check logs/run_all.log ***"
    echo ""
fi

# ── Load config ──────────────────────────────────────────────

ENV_FILE="$SCRIPT_DIR/.env.runpod"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found."
    echo "Copy the template and fill it in:"
    echo "  cp $SCRIPT_DIR/.env.runpod.template $SCRIPT_DIR/.env.runpod"
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

# Defaults for optional values
BUDGET_LIMIT_USD="${BUDGET_LIMIT_USD:-40.00}"
GPU_RATE_PER_HOUR="${GPU_RATE_PER_HOUR:-1.49}"
ON_COMPLETE="${ON_COMPLETE:-stop}"
ON_FAILURE="${ON_FAILURE:-stop}"
QUICK_TEST_FIRST="${QUICK_TEST_FIRST:-true}"
NOTIFY_WEBHOOK_URL="${NOTIFY_WEBHOOK_URL:-}"
NOTIFY_PREFIX="${NOTIFY_PREFIX:-[energy-moe-m2]}"

# ── State tracking ───────────────────────────────────────────

START_TIME=$(date +%s)
STATUS_FILE="logs/pipeline_status.json"
mkdir -p logs

pycalc() {
    # Drop-in replacement for bc — always available on RunPod
    python3 -c "print(round($1, 2))"
}

write_status() {
    local status=$1
    local stage=$2
    local message=$3
    local now=$(date +%s)
    local elapsed=$(( now - START_TIME ))
    local hours_elapsed=$(pycalc "$elapsed / 3600")
    local cost_est=$(pycalc "$hours_elapsed * $GPU_RATE_PER_HOUR")

    cat > "$STATUS_FILE" <<STATUSEOF
{
    "status": "$status",
    "current_stage": "$stage",
    "message": "$message",
    "started_at": "$(date -d @$START_TIME 2>/dev/null || date -r $START_TIME 2>/dev/null || echo unknown)",
    "updated_at": "$(date)",
    "elapsed_seconds": $elapsed,
    "elapsed_hours": $hours_elapsed,
    "estimated_cost_usd": $cost_est,
    "budget_limit_usd": $BUDGET_LIMIT_USD
}
STATUSEOF
}

# ── Budget guard ─────────────────────────────────────────────

check_budget() {
    local now=$(date +%s)
    local elapsed=$(( now - START_TIME ))
    local hours_elapsed=$(pycalc "$elapsed / 3600")
    local cost_est=$(pycalc "$hours_elapsed * $GPU_RATE_PER_HOUR")

    # Python returns "True"/"False" — check if over budget
    local over=$(python3 -c "print(1 if $cost_est >= $BUDGET_LIMIT_USD else 0)")
    if [ "$over" -eq 1 ]; then
        echo ""
        echo "BUDGET LIMIT REACHED: \$$cost_est >= \$$BUDGET_LIMIT_USD"
        echo "Elapsed: ${hours_elapsed}h at \$$GPU_RATE_PER_HOUR/hr"
        return 1
    fi

    # Warn if within 20% of budget
    local near=$(python3 -c "print(1 if $cost_est >= $BUDGET_LIMIT_USD * 0.80 else 0)")
    if [ "$near" -eq 1 ]; then
        local pct=$(python3 -c "print(int($cost_est * 100 / $BUDGET_LIMIT_USD))")
        echo "  [budget] WARNING: \$$cost_est / \$$BUDGET_LIMIT_USD (${pct}%)"
    else
        echo "  [budget] \$$cost_est / \$$BUDGET_LIMIT_USD"
    fi
    return 0
}

# ── Notification ─────────────────────────────────────────────

notify() {
    local message="$1"
    local full_msg="$NOTIFY_PREFIX $message"

    echo "$full_msg"

    if [ -n "$NOTIFY_WEBHOOK_URL" ]; then
        # Try to detect webhook type and format accordingly
        if echo "$NOTIFY_WEBHOOK_URL" | grep -q "hooks.slack.com"; then
            # Slack format
            curl -s -X POST "$NOTIFY_WEBHOOK_URL" \
                -H 'Content-Type: application/json' \
                -d "{\"text\": \"$full_msg\"}" \
                > /dev/null 2>&1 || true
        elif echo "$NOTIFY_WEBHOOK_URL" | grep -q "ntfy"; then
            # ntfy.sh format
            curl -s -X POST "$NOTIFY_WEBHOOK_URL" \
                -d "$full_msg" \
                > /dev/null 2>&1 || true
        else
            # Generic POST
            curl -s -X POST "$NOTIFY_WEBHOOK_URL" \
                -H 'Content-Type: application/json' \
                -d "{\"text\": \"$full_msg\", \"message\": \"$full_msg\"}" \
                > /dev/null 2>&1 || true
        fi
    fi
}

# ── Pod control ──────────────────────────────────────────────

get_pod_id() {
    # RunPod injects RUNPOD_POD_ID into the container environment
    if [ -n "${RUNPOD_POD_ID:-}" ]; then
        echo "$RUNPOD_POD_ID"
        return
    fi
    # Fallback: check /etc/hostname or env
    if [ -f /etc/runpod_pod_id ]; then
        cat /etc/runpod_pod_id
        return
    fi
    echo ""
}

stop_pod() {
    if [ "$DEBUG_MODE" = "true" ]; then
        echo ""
        echo "============================================"
        echo " DEBUG MODE: skipping auto-stop."
        echo " Pod stays running. SSH in to inspect."
        echo " Remember to stop manually when done!"
        echo "============================================"
        return
    fi

    echo ""
    echo "============================================"
    echo " Auto-stopping pod..."
    echo "============================================"

    local pod_id
    pod_id=$(get_pod_id)

    # Method 1: RunPod API (most reliable)
    if [ -n "${RUNPOD_API_KEY:-}" ] && [ "$RUNPOD_API_KEY" != "your-api-key-here" ] && [ -n "$pod_id" ]; then
        echo "Stopping pod $pod_id via API..."
        curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
            -H 'Content-Type: application/json' \
            -d "{\"query\": \"mutation { podStop(input: {podId: \\\"$pod_id\\\"}) { id } }\"}" \
            2>&1 || true
        echo "Stop request sent."
        return
    fi

    # Method 2: runpodctl (if installed and configured)
    if command -v runpodctl &> /dev/null && [ -n "$pod_id" ]; then
        echo "Stopping pod $pod_id via runpodctl..."
        runpodctl stop pod "$pod_id" 2>/dev/null || true
        return
    fi

    # Method 3: Fallback — just shut down (this stops billing)
    echo "Could not auto-stop via API (pod_id=${pod_id:-unknown}). Shutting down system..."
    echo "If this doesn't work, the pod will keep running. Check RunPod dashboard."
    sudo shutdown -h +1 "M2 pipeline complete. Auto-shutdown in 1 minute." 2>/dev/null || true
}

# ── Run a pipeline step ──────────────────────────────────────

run_step() {
    local step_num=$1
    local script=$2
    local label=$3
    local extra_args="${4:-}"

    echo ""
    echo "============================================================"
    echo " STAGE $step_num: $label"
    echo " $(date)"
    echo "============================================================"

    write_status "running" "$step_num" "$label"

    # Budget check before starting
    if ! check_budget; then
        write_status "budget_exceeded" "$step_num" "Budget limit hit before $label"
        notify "BUDGET LIMIT (\$$BUDGET_LIMIT_USD) reached before stage $step_num ($label). Pod stopping."
        stop_pod
        exit 2
    fi

    # Run the script
    local step_start=$(date +%s)
    bash "$script" $extra_args 2>&1 | tee "logs/$(basename "$script" .sh).log"
    local exit_code=${PIPESTATUS[0]}
    local step_end=$(date +%s)
    local step_duration=$(( step_end - step_start ))
    local step_minutes=$(( step_duration / 60 ))

    echo "  [time] Stage $step_num took ${step_minutes}m"

    if [ $exit_code -ne 0 ]; then
        echo "  [FAIL] Stage $step_num exited with code $exit_code"

        # Run health check to get diagnostic info
        python3 scripts/runpod/health_check.py --stage "$step_num" 2>&1 || true

        write_status "failed" "$step_num" "$label failed (exit $exit_code) after ${step_minutes}m"
        notify "FAILED at stage $step_num ($label) after ${step_minutes}m. Exit code: $exit_code"

        if [ "$ON_FAILURE" = "stop" ]; then
            echo "ON_FAILURE=stop — stopping pod."
            # Package whatever we have before stopping
            bash scripts/runpod/06_package.sh 2>/dev/null || true
            stop_pod
            exit 1
        else
            echo "ON_FAILURE=continue_next — skipping to next stage."
            return 1
        fi
    fi

    # Health check
    echo "  [health] Running post-stage health check..."
    local health_verdict
    health_verdict=$(python3 scripts/runpod/health_check.py --stage "$step_num" 2>&1)
    local health_code=$?

    echo "  [health] $health_verdict"

    if [ $health_code -ne 0 ]; then
        write_status "health_fail" "$step_num" "Health check failed: $health_verdict"
        notify "HEALTH CHECK FAILED at stage $step_num ($label): $health_verdict"

        if [ "$ON_FAILURE" = "stop" ]; then
            bash scripts/runpod/06_package.sh 2>/dev/null || true
            stop_pod
            exit 1
        fi
        return 1
    fi

    write_status "completed_stage" "$step_num" "$label done in ${step_minutes}m"
    return 0
}

# ── Main pipeline ────────────────────────────────────────────

echo "============================================================"
echo " Energy-Aware MoE Router — M2 Autonomous Pipeline"
echo " $(date)"
echo " Budget: \$$BUDGET_LIMIT_USD at \$$GPU_RATE_PER_HOUR/hr"
echo " On complete: $ON_COMPLETE | On failure: $ON_FAILURE"
echo " Quick test: $QUICK_TEST_FIRST"
echo "============================================================"

write_status "starting" "0" "Pipeline starting"

# ── Quick test (optional) ────────────────────────────────────

if [ "$QUICK_TEST_FIRST" = "true" ]; then
    echo ""
    echo "============================================================"
    echo " QUICK TEST — verifying pipeline on tiny data (~15 min)"
    echo "============================================================"

    write_status "running" "quick_test" "Quick pre-flight test"

    # Setup
    bash scripts/runpod/00_setup.sh 2>&1 | tee logs/00_setup.log
    SETUP_EXIT=${PIPESTATUS[0]}
    if [ $SETUP_EXIT -ne 0 ]; then
        echo ""
        echo "!!! SETUP FAILED (exit code $SETUP_EXIT) !!!"
        echo "Last 20 lines of setup log:"
        tail -20 logs/00_setup.log 2>/dev/null || true
        notify "FAILED: setup failed (exit $SETUP_EXIT) during quick test"
        write_status "failed" "quick_test" "Setup failed (exit $SETUP_EXIT)"
        stop_pod
        exit 1
    fi

    # Quick data
    bash scripts/runpod/01_data.sh --quick 2>&1 | tee logs/01_data_quick.log
    DATA_EXIT=${PIPESTATUS[0]}
    if [ $DATA_EXIT -ne 0 ]; then
        echo ""
        echo "!!! DATA DOWNLOAD FAILED (exit code $DATA_EXIT) !!!"
        tail -20 logs/01_data_quick.log 2>/dev/null || true
        notify "FAILED: data download failed (exit $DATA_EXIT) during quick test"
        write_status "failed" "quick_test" "Quick data download failed (exit $DATA_EXIT)"
        stop_pod
        exit 1
    fi

    # Train just Switch on quick data to verify training loop
    echo ""
    echo "--- Quick test: training Switch on 10K examples ---"
    python3 train.py \
        --config configs/tiny_moe.yaml \
        --router switch \
        --seed 42 \
        2>&1 | tee logs/quick_test_train.log
    TRAIN_EXIT=${PIPESTATUS[0]}

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo ""
        echo "!!! TRAINING FAILED (exit code $TRAIN_EXIT) !!!"
        tail -20 logs/quick_test_train.log 2>/dev/null || true
        notify "FAILED: training loop broken (exit $TRAIN_EXIT) — fix before full run"
        write_status "failed" "quick_test" "Training failed (exit $TRAIN_EXIT)"
        stop_pod
        exit 1
    fi

    # Check the quick test produced something
    python3 -c "
import json, sys
try:
    with open('results/m2/switch/metrics.jsonl') as f:
        last = json.loads(f.readlines()[-1])
    ppl = last['ppl']
    if ppl > 10000:
        print(f'Quick test PPL={ppl:.0f} — suspiciously high but training ran')
    else:
        print(f'Quick test PPL={ppl:.2f} — looks good')
    sys.exit(0)
except Exception as e:
    print(f'Quick test check failed: {e}')
    sys.exit(1)
"
    if [ $? -ne 0 ]; then
        notify "FAILED: quick test metrics check failed"
        write_status "failed" "quick_test" "Quick test metrics invalid"
        stop_pod
        exit 1
    fi

    echo ""
    echo "============================================================"
    echo " Quick test PASSED. Starting full pipeline."
    echo "============================================================"
    notify "Quick test passed. Starting full pipeline (~9-14 hours)."

    # Clean up quick test artifacts before full run
    rm -rf results/m2/switch data/processed/c4_quick 2>/dev/null || true
fi

# ── Full pipeline ────────────────────────────────────────────

# Step 0: Setup (skip if quick test already ran it)
if [ "$QUICK_TEST_FIRST" = "true" ]; then
    echo ""
    echo "  [skip] Stage 0 (setup) — already done during quick test"
else
    run_step 0 "scripts/runpod/00_setup.sh" "Environment setup"
fi

# Step 1: Data (full download)
run_step 1 "scripts/runpod/01_data.sh" "C4 1% data download"

# Step 2: Baselines
run_step 2 "scripts/runpod/02_baselines.sh" "Baseline training (switch, random, dense)"

# Step 3: Energy sweep
run_step 3 "scripts/runpod/03_energy_sweep.sh" "Energy-aware lambda sweep (7 values)"

# Step 4: Evaluation
run_step 4 "scripts/runpod/04_evaluate.sh" "Evaluate all checkpoints"

# Step 5: Plots
run_step 5 "scripts/runpod/05_plot.sh" "Generate Pareto + training plots"

# Step 6: Package
bash scripts/runpod/06_package.sh 2>&1 | tee logs/06_package.log

# ── Done ─────────────────────────────────────────────────────

NOW=$(date +%s)
TOTAL_ELAPSED=$(( NOW - START_TIME ))
TOTAL_HOURS=$(pycalc "$TOTAL_ELAPSED / 3600")
TOTAL_COST=$(pycalc "$TOTAL_HOURS * $GPU_RATE_PER_HOUR")

echo ""
echo "============================================================"
echo " M2 PIPELINE COMPLETE"
echo " Total time: ${TOTAL_HOURS}h"
echo " Estimated cost: \$$TOTAL_COST"
echo " $(date)"
echo "============================================================"

write_status "completed" "done" "Pipeline finished in ${TOTAL_HOURS}h, est. cost \$$TOTAL_COST"
notify "DONE! Pipeline finished in ${TOTAL_HOURS}h. Est. cost: \$$TOTAL_COST. Pod will $ON_COMPLETE."

# Auto-stop/terminate
if [ "$ON_COMPLETE" = "stop" ] || [ "$ON_COMPLETE" = "terminate" ]; then
    echo ""
    echo "Results bundled at: results/m2/m2_results_bundle.tar.gz"
    echo "Download before terminating!"
    echo ""
    stop_pod
fi
