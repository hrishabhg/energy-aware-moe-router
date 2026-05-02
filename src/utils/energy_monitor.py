"""
energy_monitor.py — Background GPU power sampler via pynvml.

Architecture
------------
A daemon thread polls nvmlDeviceGetPowerUsage at a configurable frequency
(default 10 Hz) and writes readings into a lock-free ring buffer. The
training loop reads the latest value without blocking, accepting up to
~100 ms staleness — GPU power state doesn't change faster than that.

Energy (joules) is computed by trapezoidal integration over the sampled
power readings for a given time window.

Usage
-----
    monitor = EnergyMonitor(device_indices=[0], sampling_hz=10)
    monitor.start()

    # ... training loop ...
    current_watts = monitor.latest_power(device=0)       # non-blocking
    joules_so_far = monitor.total_joules(device=0)        # since start()

    monitor.stop()
    summary = monitor.summary()   # dict with total joules, mean watts, etc.

Overhead target: < 3% of wall-clock. Profile with monitor.overhead_pct().

Notes
-----
- pynvml returns power in milliwatts; we convert to watts internally.
- On systems without NVIDIA GPUs (e.g., Mac for local dev), the monitor
  gracefully degrades: start() logs a warning and all power reads return 0.
- This module is deliberately simple — no wandb integration, no file I/O.
  The training loop is responsible for logging the numbers it cares about.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class PowerReading:
    """A single power measurement."""
    timestamp: float       # time.monotonic()
    watts: float           # instantaneous power draw


@dataclass
class DeviceState:
    """Per-device ring buffer and accumulated energy."""
    readings: list[PowerReading] = field(default_factory=list)
    total_joules: float = 0.0
    _last_reading: Optional[PowerReading] = field(default=None, repr=False)

    def append(self, reading: PowerReading) -> None:
        """Add a reading and integrate energy since the last one."""
        if self._last_reading is not None:
            dt = reading.timestamp - self._last_reading.timestamp
            if dt > 0:
                # Trapezoidal rule
                avg_watts = (reading.watts + self._last_reading.watts) / 2.0
                self.total_joules += avg_watts * dt
        self._last_reading = reading
        self.readings.append(reading)


class EnergyMonitor:
    """Background GPU power sampler with non-blocking reads.

    Parameters
    ----------
    device_indices : list of GPU indices to monitor (default: [0])
    sampling_hz : polling frequency in Hz (default: 10)
    tdp_watts : per-device TDP for normalization; if provided,
        ``latest_power_normalized()`` returns values in [0, 1].
    """

    def __init__(
        self,
        device_indices: list[int] | None = None,
        sampling_hz: float = 10.0,
        tdp_watts: float | None = None,
    ):
        self.device_indices = device_indices or [0]
        self.sampling_hz = sampling_hz
        self.tdp_watts = tdp_watts
        self._interval = 1.0 / sampling_hz

        self._devices: dict[int, DeviceState] = {
            idx: DeviceState() for idx in self.device_indices
        }
        self._nvml_handles: dict[int, object] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False
        self._nvml_available = False
        self._start_time: float = 0.0
        self._sample_count: int = 0
        self._sample_time_total: float = 0.0   # for overhead measurement

    # ── Lifecycle ───────────────────────────────────────────────

    def start(self) -> "EnergyMonitor":
        """Initialize pynvml and start the background sampler thread."""
        if self._started:
            log.warning("EnergyMonitor already started.")
            return self

        try:
            import pynvml
            pynvml.nvmlInit()
            for idx in self.device_indices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                self._nvml_handles[idx] = handle
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                log.info(f"EnergyMonitor: GPU {idx} = {name}")
            self._nvml_available = True
        except Exception as e:
            log.warning(
                f"pynvml not available ({e}). "
                "EnergyMonitor will return zeros for all power reads. "
                "This is expected on non-GPU machines (e.g., Mac for local dev)."
            )
            self._nvml_available = False

        self._start_time = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sample_loop, daemon=True, name="energy-monitor"
        )
        self._thread.start()
        self._started = True
        log.info(
            f"EnergyMonitor started: devices={self.device_indices}, "
            f"hz={self.sampling_hz}, nvml={'OK' if self._nvml_available else 'STUB'}"
        )
        return self

    def stop(self) -> dict:
        """Stop sampling and return a summary dict."""
        if not self._started:
            return {}
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._started = False

        if self._nvml_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass

        return self.summary()

    # ── Sampling loop (runs in daemon thread) ──────────────────

    def _sample_loop(self) -> None:
        """Poll pynvml at the configured frequency."""
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            self._take_sample()
            t1 = time.monotonic()

            self._sample_count += 1
            self._sample_time_total += (t1 - t0)

            # Sleep for the remainder of the interval
            elapsed = t1 - t0
            sleep_time = self._interval - elapsed
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

    def _take_sample(self) -> None:
        """Read power from all monitored devices."""
        ts = time.monotonic()
        if not self._nvml_available:
            for idx in self.device_indices:
                self._devices[idx].append(PowerReading(ts, 0.0))
            return

        import pynvml
        for idx in self.device_indices:
            try:
                mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handles[idx])
                watts = mw / 1000.0
            except Exception:
                watts = 0.0
            self._devices[idx].append(PowerReading(ts, watts))

    # ── Public read API (non-blocking) ─────────────────────────

    def latest_power(self, device: int = 0) -> float:
        """Return the most recent power reading in watts (non-blocking)."""
        state = self._devices.get(device)
        if state is None or state._last_reading is None:
            return 0.0
        return state._last_reading.watts

    def latest_power_normalized(self, device: int = 0) -> float:
        """Return latest power normalized to [0, 1] by TDP.

        Returns raw watts if tdp_watts was not set.
        """
        watts = self.latest_power(device)
        if self.tdp_watts is not None and self.tdp_watts > 0:
            return min(watts / self.tdp_watts, 1.0)
        return watts

    def latest_power_all(self) -> list[float]:
        """Return latest power reading for all monitored devices."""
        return [self.latest_power(idx) for idx in self.device_indices]

    def latest_power_all_normalized(self) -> list[float]:
        """Return latest normalized power for all monitored devices.

        This is the e_gpu vector used in the gating function:
            logits = x @ W_g - lambda * e_gpu
        """
        return [self.latest_power_normalized(idx) for idx in self.device_indices]

    def total_joules(self, device: int = 0) -> float:
        """Total energy consumed since start() in joules."""
        state = self._devices.get(device)
        return state.total_joules if state else 0.0

    def total_joules_all(self) -> list[float]:
        """Total joules for all monitored devices."""
        return [self.total_joules(idx) for idx in self.device_indices]

    # ── Diagnostics ────────────────────────────────────────────

    def overhead_pct(self) -> float:
        """Estimate sampling overhead as % of wall-clock since start().

        Target: < 3%. If this is higher, reduce sampling_hz.
        """
        wall = time.monotonic() - self._start_time
        if wall <= 0:
            return 0.0
        return (self._sample_time_total / wall) * 100.0

    def summary(self) -> dict:
        """Return a summary dict suitable for JSON logging."""
        wall = time.monotonic() - self._start_time
        result = {
            "wall_seconds": round(wall, 2),
            "samples_taken": self._sample_count,
            "sampling_overhead_pct": round(self.overhead_pct(), 4),
            "devices": {},
        }
        for idx in self.device_indices:
            state = self._devices[idx]
            readings = state.readings
            watts_list = [r.watts for r in readings]
            result["devices"][idx] = {
                "total_joules": round(state.total_joules, 2),
                "mean_watts": round(sum(watts_list) / len(watts_list), 2) if watts_list else 0.0,
                "max_watts": round(max(watts_list), 2) if watts_list else 0.0,
                "min_watts": round(min(watts_list), 2) if watts_list else 0.0,
                "num_readings": len(readings),
            }
        return result

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ── Quick self-test ────────────────────────────────────────────

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    print("Starting EnergyMonitor self-test (5 seconds) ...")
    with EnergyMonitor(device_indices=[0], sampling_hz=10) as mon:
        for _ in range(5):
            time.sleep(1.0)
            w = mon.latest_power(0)
            j = mon.total_joules(0)
            print(f"  Power: {w:.1f} W  |  Energy: {j:.1f} J")

        summary = mon.summary()

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nOverhead: {summary['sampling_overhead_pct']:.4f}%")
    print("Self-test complete.")
