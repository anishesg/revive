"""Cooling policy for the BMC-driven fan.

Runs on the host (dashboard server) each tick, reads live inference +
cluster-state signals, and returns the desired 0-255 PWM duty. Pushed to
the Arduino via ClusterController.set_fan_duty; the firmware applies it
and echoes STATUS FAN back.

The policy is deliberately simple — a few inputs, one output — because
this is a real-time loop and the cost of a wrong answer is either a
missed thermal margin (too cool is fine) or fan noise (too hot blows
harder). No ML, no PID overshoot; just clamps and an EMA to avoid
audible thrashing tick-to-tick.
"""
from __future__ import annotations

from dataclasses import dataclass


# Duty landmarks. 8-bit PWM so 255 = 100% on.
DUTY_OFF         = 0
DUTY_IDLE_HEALTHY = 30    # ~12% — barely-audible airflow for dust + sensors
DUTY_IDLE_DEGRADED = 180  # fewer workers sharing the load -> hotter per worker
DUTY_INFER_BASE  = 90     # ~35% baseline during any active inference
DUTY_INFER_MAX   = 230    # ceiling during active inference
DUTY_THERMAL_TAIL_SEC = 10.0  # how long to keep spinning after work stops

# Smoothing. Higher alpha -> more responsive, lower -> quieter.
EMA_ALPHA        = 0.35

# Throughput scaling. Aggregate tok/s across all workers maps into
# [INFER_BASE .. INFER_MAX]. Tuned so a single worker at ~20 tok/s lands
# mid-range and a saturated 3-worker ring at 50+ tok/s approaches the cap.
TPS_AT_MAX       = 55.0

# ms/step scaling — slow steps mean more work per token, so ramp up a bit.
STEP_MS_HOT      = 80.0   # anything above this bumps duty toward the cap


@dataclass
class FanInputs:
    """Everything the policy needs to make a decision."""
    bmc_state: str               # "down" | "healthy" | "degraded"
    inference_active: bool       # coordinator is currently generating
    seconds_since_last_token: float  # 0 while streaming; grows after EOS
    aggregate_tps: float         # sum of per-worker tok/s
    step_latency_ms: float       # last ring-step latency


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def compute_duty(inp: FanInputs) -> int:
    """Return the instantaneous desired duty for this tick (pre-EMA)."""
    if inp.bmc_state == "down":
        return DUTY_OFF

    degraded = inp.bmc_state == "degraded"

    # Decide if we're in a "work is happening right now" regime. The tail
    # window keeps the fan spinning for a bit after EOS so residual heat
    # from the last inference gets flushed.
    hot_tail = inp.seconds_since_last_token <= DUTY_THERMAL_TAIL_SEC
    working = inp.inference_active or hot_tail

    if not working:
        return DUTY_IDLE_DEGRADED if degraded else DUTY_IDLE_HEALTHY

    # Working. Scale between INFER_BASE and INFER_MAX based on throughput
    # (proxy for sustained work) and step latency (proxy for heavy step).
    tps_ratio = _clamp(inp.aggregate_tps / TPS_AT_MAX, 0.0, 1.0)
    step_ratio = _clamp(inp.step_latency_ms / STEP_MS_HOT, 0.0, 1.0)
    load = max(tps_ratio, step_ratio)

    # Fade in during the thermal tail so idle doesn't snap back immediately.
    if not inp.inference_active and hot_tail:
        fade = 1.0 - (inp.seconds_since_last_token / DUTY_THERMAL_TAIL_SEC)
        load = load * fade

    span = DUTY_INFER_MAX - DUTY_INFER_BASE
    duty = DUTY_INFER_BASE + int(load * span)
    if degraded:
        duty = max(duty, DUTY_IDLE_DEGRADED)
    return int(_clamp(duty, DUTY_OFF, 255))


class FanController:
    """Stateful wrapper: applies EMA smoothing + rate-limits duplicate sends."""

    def __init__(self):
        self._ema: float = 0.0
        self._last_sent: int = -1

    def step(self, inp: FanInputs) -> int:
        raw = compute_duty(inp)
        self._ema = (1 - EMA_ALPHA) * self._ema + EMA_ALPHA * raw
        return int(round(self._ema))

    def should_send(self, duty: int) -> bool:
        """Avoid flooding the BMC serial link with identical FAN commands.
        Send when the duty has actually moved by >=2 ticks, or when we've
        never sent one before."""
        if self._last_sent < 0:
            self._last_sent = duty
            return True
        if abs(duty - self._last_sent) >= 2:
            self._last_sent = duty
            return True
        return False
