"""Layer-to-worker assignment.

Two implementations:

  greedy_partition   — simple, fits in ~100 lines of C on an Arduino Uno.
                       Proportional to worker capability score. This is what
                       the BMC actually runs.

  pipeedge_dp        — PipeEdge-style dynamic programming (arxiv 2110.14895)
                       that minimizes max per-stage latency given measured
                       per-layer costs. Used as a reference / correctness
                       check. Not needed for the Arduino — it's the "what
                       would an ideal scheduler do" comparison.

Both produce the same structure: ordered list of (worker_id, start, end)
with contiguous layer ranges summing to num_layers.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class WorkerProfile:
    id: str
    capability_score: float  # higher = faster; relative units
    ram_mb: int
    ever_seen: bool = True
    alive: bool = True


def greedy_partition(num_layers: int, workers: list[WorkerProfile]) -> list[tuple[str, int, int]]:
    """Proportional split by capability score. Each worker gets a share of
    layers = num_layers * (their_score / total_score). Remainder is pushed
    to the last worker so the sum is exact."""
    alive = [w for w in workers if w.alive]
    if not alive:
        return []
    total = sum(w.capability_score for w in alive)
    if total <= 0:
        # uniform fallback
        share = num_layers // len(alive)
        rem = num_layers - share * len(alive)
        result = []
        cur = 0
        for i, w in enumerate(alive):
            n = share + (1 if i < rem else 0)
            result.append((w.id, cur, cur + n))
            cur += n
        return result

    result = []
    cur = 0
    for i, w in enumerate(alive):
        if i == len(alive) - 1:
            end = num_layers
        else:
            share = round(num_layers * w.capability_score / total)
            # minimum one layer per worker
            share = max(1, share)
            end = min(cur + share, num_layers - (len(alive) - i - 1))
        result.append((w.id, cur, end))
        cur = end
    return result


def pipeedge_dp(num_layers: int,
                workers: list[WorkerProfile],
                per_layer_compute: Optional[list[float]] = None) -> list[tuple[str, int, int]]:
    """Minimize max over stages of (compute_time_i). Pipeline throughput is
    bounded by the slowest stage, so minimizing the max is the right objective
    for single-stream decode.

    DP formulation:
      dp[k][i] = min over partitions of layers [0..i) into k stages of the
                 max per-stage compute time, using the best worker assignment
                 from the alive pool.

    For simplicity we assume workers[] is already in topology order and each
    worker gets one contiguous stage. This matches the greedy version and
    how the Arduino will enforce partitions.
    """
    alive = [w for w in workers if w.alive]
    N = len(alive)
    if N == 0:
        return []
    if per_layer_compute is None:
        per_layer_compute = [1.0] * num_layers

    # cost[w][i][j] = compute time for worker w running layers [i..j)
    # = sum(per_layer_compute[i:j]) / w.capability_score
    prefix = [0.0]
    for c in per_layer_compute:
        prefix.append(prefix[-1] + c)

    def cost(worker_idx: int, start: int, end: int) -> float:
        w = alive[worker_idx]
        layers_cost = prefix[end] - prefix[start]
        return layers_cost / max(w.capability_score, 1e-6)

    INF = float("inf")
    # dp[k][i] = (max_stage_time, back_ptr_i) for partitioning [0..i) into
    # first k workers (worker 0..k-1 assigned contiguously).
    dp: dict[tuple[int, int], float] = {}
    back: dict[tuple[int, int], int] = {}
    dp[(0, 0)] = 0.0

    for k in range(1, N + 1):
        min_layers_left = N - k  # each remaining worker needs ≥1 layer
        for i in range(k, num_layers - min_layers_left + 1):
            best = INF
            bj = -1
            for j in range(k - 1, i):
                prev = dp.get((k - 1, j), INF)
                if prev >= INF:
                    continue
                stage = cost(k - 1, j, i)
                m = max(prev, stage)
                if m < best:
                    best = m
                    bj = j
            if best < INF:
                dp[(k, i)] = best
                back[(k, i)] = bj

    final_key = (N, num_layers)
    if final_key not in dp:
        # infeasible (fewer than N layers); fall back to greedy
        return greedy_partition(num_layers, workers)

    # Reconstruct partition
    assignments = []
    i = num_layers
    for k in range(N, 0, -1):
        j = back[(k, i)]
        assignments.append((alive[k - 1].id, j, i))
        i = j
    assignments.reverse()
    return assignments


def validate_partition(partition: list[tuple[str, int, int]], num_layers: int) -> None:
    """Raise if the partition doesn't cover [0, num_layers) contiguously."""
    if not partition:
        raise ValueError("empty partition")
    assignments = sorted(partition, key=lambda x: x[1])
    if assignments[0][1] != 0:
        raise ValueError(f"first stage doesn't start at 0: {assignments}")
    for a, b in zip(assignments, assignments[1:]):
        if a[2] != b[1]:
            raise ValueError(f"gap/overlap between {a} and {b}")
    if assignments[-1][2] != num_layers:
        raise ValueError(f"last stage doesn't end at {num_layers}: {assignments}")
    for (_, s, e) in assignments:
        if s >= e:
            raise ValueError(f"empty stage {s}..{e}")
