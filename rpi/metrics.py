"""
REVIVE Device Metrics for Raspberry Pi
Collects CPU temperature, memory usage, and system health.
"""
import os
import time

import psutil


_start_time = time.time()


def cpu_temp_celsius() -> float:
    """Read SoC temperature from sysfs (Raspberry Pi specific)."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (FileNotFoundError, ValueError):
        temps = psutil.sensors_temperatures()
        if temps:
            first = next(iter(temps.values()))
            if first:
                return first[0].current
        return -1.0


def thermal_state() -> str:
    temp = cpu_temp_celsius()
    if temp < 0:
        return "unknown"
    if temp < 60:
        return "nominal"
    if temp < 70:
        return "fair"
    if temp < 80:
        return "serious"
    return "critical"


def memory_used_mb() -> int:
    return int(psutil.virtual_memory().used / (1024 * 1024))


def memory_total_mb() -> int:
    return int(psutil.virtual_memory().total / (1024 * 1024))


def cpu_percent() -> float:
    return psutil.cpu_percent(interval=0.1)


def uptime_seconds() -> int:
    return int(time.time() - _start_time)


def snapshot(tokens_generated: int = 0, tokens_per_second: float = 0.0,
             time_to_first_token_ms: int = 0, total_time_ms: int = 0) -> dict:
    return {
        "tokens_generated": tokens_generated,
        "tokens_per_second": tokens_per_second,
        "time_to_first_token_ms": time_to_first_token_ms,
        "total_time_ms": total_time_ms,
        "thermal_state": thermal_state(),
        "battery_percent": -1,
        "memory_used_mb": memory_used_mb(),
    }


def full_status(active_workers: int = 0, role: str = "aggregator") -> dict:
    return {
        "platform": "rpi",
        "role": role,
        "thermal_state": thermal_state(),
        "cpu_temp_c": cpu_temp_celsius(),
        "battery_percent": -1,
        "memory_used_mb": memory_used_mb(),
        "memory_total_mb": memory_total_mb(),
        "cpu_percent": cpu_percent(),
        "uptime_seconds": uptime_seconds(),
        "active_workers": active_workers,
    }
