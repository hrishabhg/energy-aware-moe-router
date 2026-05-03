# RunPod Execution Scripts — M2

Run these scripts **in order** on your RunPod instance. Each script logs to `logs/` and prints a clear "run X next" message when done.

## Quick Reference

| Script | What it does | Time (A100) | Run once? |
|--------|-------------|-------------|-----------|
| `run_all.sh` | **Full autonomous pipeline** (recommended) | 9-14 hours  | Yes |
| `00_setup.sh` | Install deps, verify GPU + pynvml | 2 min       | Yes (persists on Stop) |
| `01_data.sh` | Download C4 subsample (500K examples, fast mode) | 15-25 min   | Yes (persists on Stop) |
| `02_baselines.sh` | Train Switch, Random, Dense | 2-3 hours   | Yes |
| `03_energy_sweep.sh` | Train 7 lambda values | 4-7 hours   | Yes |
| `04_evaluate.sh` | Evaluate all checkpoints | 10-20 min   | Yes |
| `05_plot.sh` | Generate Pareto + training plots | 1 min       | Re-run anytime |
| `06_package.sh` | Bundle results for download | 1 min       | Before terminating |

**Total estimated GPU time: ~9-14 hours on A100 SXM (~$13-21 at $1.49/hr community pricing)**

## Autonomous Mode (recommended — run overnight)

This is the "kick off and go to sleep" workflow. The orchestrator runs the full pipeline, monitors budget and health, and auto-stops the pod when done.

```bash
# 1. On your LOCAL machine: create the pod via RunPod web UI
#    Template: PyTorch 2.x | GPU: A100 SXM | Disk: 50GB container disk

# 2. SSH into the pod
ssh root@<pod-ip>

# 3. Clone the repo
cd /workspace
git clone https://github.com/hrishabhg/energy-aware-moe-router.git energy-aware-moe-router
cd energy-aware-moe-router

# 4. Configure
cp scripts/runpod/.env.runpod.template scripts/runpod/.env.runpod
nano scripts/runpod/.env.runpod   # fill in API key, budget, etc.

# 5. Launch in background and disconnect
nohup bash scripts/runpod/run_all.sh &> logs/run_all.log &
disown
exit    # safe to disconnect — pipeline keeps running
```

### What the orchestrator does

1. **Quick test** (optional, default on): trains Switch on 10K examples (~15 min) to verify the entire training loop works before committing to a 10+ hour run
2. **Budget tracking**: computes `elapsed_hours × GPU_rate` before each stage, auto-stops if approaching your limit
3. **Health checks**: after each stage, validates outputs (no NaN, PPL within bounds, Switch PPL < Random PPL, etc.)
4. **Auto-stop**: stops the pod when the pipeline finishes (or fails), so you don't burn money overnight
5. **Notifications** (optional): sends a webhook ping (Slack, ntfy.sh, Discord) on completion or failure
6. **Status file**: writes `logs/pipeline_status.json` so you can check progress via RunPod file manager

### Configuration (.env.runpod)

| Variable | Default | Description |
|----------|---------|-------------|
| `RUNPOD_API_KEY` | (required) | Your RunPod API key for auto-stop |
| `BUDGET_LIMIT_USD` | `40.00` | Auto-stop when spend approaches this |
| `GPU_RATE_PER_HOUR` | `1.49` | A100 SXM community rate (check RunPod dashboard) |
| `ON_COMPLETE` | `stop` | `stop` or `terminate` when pipeline finishes |
| `ON_FAILURE` | `stop` | `stop` or `continue_next` on failure |
| `QUICK_TEST_FIRST` | `true` | Run 15-min pre-flight before full pipeline |
| `NOTIFY_WEBHOOK_URL` | (empty) | Webhook URL for notifications |

### Checking progress remotely

```bash
# Option 1: Check the status file via RunPod web UI file manager
# Navigate to: logs/pipeline_status.json

# Option 2: SSH in and check
ssh root@<pod-ip> "cat /workspace/energy-aware-moe-router/logs/pipeline_status.json"

# Option 3: Tail the log
ssh root@<pod-ip> "tail -20 /workspace/energy-aware-moe-router/logs/run_all.log"
```

## Manual Mode (step-by-step)

If you prefer to run each step yourself:

```bash
# On RunPod (after creating pod with PyTorch 2.x template + A100)
cd /workspace
git clone https://github.com/hrishabhg/energy-aware-moe-router.git energy-aware-moe-router
cd energy-aware-moe-router

# Run in order
bash scripts/runpod/00_setup.sh 2>&1 | tee logs/00_setup.log
bash scripts/runpod/01_data.sh 2>&1 | tee logs/01_data.log
bash scripts/runpod/02_baselines.sh 2>&1 | tee logs/02_baselines.log
bash scripts/runpod/03_energy_sweep.sh 2>&1 | tee logs/03_energy_sweep.log
bash scripts/runpod/04_evaluate.sh 2>&1 | tee logs/04_evaluate.log
bash scripts/runpod/05_plot.sh 2>&1 | tee logs/05_plot.log
bash scripts/runpod/06_package.sh
```

## Quick test (before committing to the full run)

```bash
bash scripts/runpod/00_setup.sh
bash scripts/runpod/01_data.sh --quick     # 10K examples, 5 min
bash scripts/runpod/02_baselines.sh switch  # single baseline only
```

## Pod Lifecycle

- **Stop** (between sessions): Keeps disk, stops GPU billing (~$0.10/GB/month storage)
- **Terminate** (after M2): Destroys everything. Download bundle first!

## Known Gotchas

**C4 download mode matters hugely.** The hash-based sampling modes
(`--streaming-subsample` and the default full-download path) scan *all*
7,168 C4 shards to decide which examples to keep. On a single node this
takes 100+ hours — we burned 8 hours on an A100 and only reached shard
518/7168 (7.2%). Always use `--fast` mode, which reads shards sequentially
and stops at the target count. 500K examples (~15 min) gives ~100M tokens,
plenty for our 16M-param models.

**First tokenization is slow.** `data_loader.py` tokenizes and packs the
entire dataset on first use (~20-30 min for 500K examples). The result is
cached, so subsequent runs (including all training scripts) load instantly.
This happens during the smoke test in `01_data.sh` — don't kill it if logs
appear stuck.

**`bc` is not available on RunPod.** All arithmetic in `run_all.sh` uses
`python3 -c` instead. If you add new shell math, use the `pycalc()` helper.

**PyTorch `total_mem` vs `total_memory`.** Newer PyTorch versions (≥2.4)
renamed the property. Use `getattr(props, 'total_memory', None) or
getattr(props, 'total_mem', 0)` for compatibility.

**Pod GPU re-allocation.** If you Stop a pod and Start it later, RunPod may
assign a different GPU or fail to allocate one entirely. Use the Migrate
option in the dashboard if your pod shows "GPU unavailable."

## If something fails

Each script uses `set -euo pipefail` and stops on error. Check `logs/*.log` for the full output. You can re-run any script safely — they skip already-completed work where possible (e.g., 01_data.sh skips download if data exists).

The health checker (`health_check.py`) runs automatically in autonomous mode. You can also run it manually:

```bash
python3 scripts/runpod/health_check.py --stage 2   # check baselines
```
