"""Benchmark the distributed inference ring.

Runs N queries, reports throughput and per-stage breakdown. Output is
designed to be screenshot-able for the hackathon pitch.

Usage:
  python -m scripts.benchmark \\
      --model Qwen/Qwen3-0.6B \\
      --workers 127.0.0.1:50100 127.0.0.1:50101 127.0.0.1:50102 \\
      --runs 3 --tokens 60
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field

from pipeline.coordinator import PipelineCoordinator, WorkerEndpoint, _worker_id


PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis briefly.",
    "Write a short poem about autumn.",
    "What is 2+2 and why?",
    "Name three uses of water.",
    "Describe the color blue.",
    "Why is the sky blue?",
]


@dataclass
class RunStats:
    prompt: str
    tokens: int
    wall_s: float
    tps: float
    prefill_ms: float        # first step's total latency
    avg_decode_ms: float     # avg of step_latency_ms for non-first steps
    per_stage_avg_ms: list[float]  # avg latency per stage across all steps


async def run_one(coord: PipelineCoordinator, prompt: str, max_tokens: int) -> RunStats:
    t0 = time.time()
    n = 0
    step_lats: list[float] = []
    stage_lats: list[list[float]] = []
    prefill_ms = 0.0
    async for step in coord.generate(prompt, max_new_tokens=max_tokens,
                                      temperature=0.7, top_p=0.95, top_k=40):
        if n == 0:
            prefill_ms = step.step_latency_ms
        else:
            step_lats.append(step.step_latency_ms)
        stage_lats.append(step.stage_latencies_ms)
        n += 1
    wall = time.time() - t0
    avg_decode = statistics.mean(step_lats) if step_lats else 0.0
    # per-stage avg across all steps
    if stage_lats:
        per_stage = [
            statistics.mean(s[i] for s in stage_lats)
            for i in range(len(stage_lats[0]))
        ]
    else:
        per_stage = []
    return RunStats(
        prompt=prompt, tokens=n, wall_s=wall,
        tps=n / wall if wall > 0 else 0,
        prefill_ms=prefill_ms,
        avg_decode_ms=avg_decode,
        per_stage_avg_ms=per_stage,
    )


def fmt_table(rows: list[list[str]], align: list[str] = None) -> str:
    if not rows:
        return ""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    align = align or ["l"] * len(widths)
    lines = []
    for r_idx, r in enumerate(rows):
        cells = []
        for i, cell in enumerate(r):
            if align[i] == "r":
                cells.append(cell.rjust(widths[i]))
            else:
                cells.append(cell.ljust(widths[i]))
        lines.append("  ".join(cells))
        if r_idx == 0:
            lines.append("  ".join("─" * widths[i] for i in range(len(widths))))
    return "\n".join(lines)


def banner(title: str):
    print()
    print("═" * 72)
    print(f"  {title}")
    print("═" * 72)


async def run(args):
    endpoints = []
    for hp in args.workers:
        host, port = hp.rsplit(":", 1)
        endpoints.append(WorkerEndpoint(host=host, port=int(port)))

    async with PipelineCoordinator(endpoints, args.model) as coord:
        # Topology sanity report
        banner("CLUSTER TOPOLOGY")
        topo_rows = [["#", "WORKER", "HOST:PORT", "LAYERS", "FIRST", "LAST"]]
        for i, w in enumerate(coord.workers):
            topo_rows.append([
                str(i), _worker_id(w), f"{w.host}:{w.port}",
                f"[{w.layer_start}..{w.layer_end})",
                "✓" if w.is_first else "",
                "✓" if w.is_last else "",
            ])
        print(fmt_table(topo_rows, align=["r", "l", "l", "l", "c", "c"]))
        L = coord.workers[0].num_layers_total
        H = coord.workers[0].hidden_size
        print(f"\n  Model: {coord.workers[0].model}  "
              f"| {L} layers, hidden={H}  "
              f"| per-token wire payload: {H * 2 / 1024:.1f} KB (fp16)")

        # Warm up (first request always slow due to compile / cache fill)
        banner("WARMUP")
        ws = await run_one(coord, "Say hi.", max_tokens=10)
        print(f"  warmup: {ws.tokens} tokens in {ws.wall_s:.2f}s ({ws.tps:.1f} tok/s)")

        # Real runs
        banner(f"BENCHMARK RUNS  ({args.runs} runs × up to {args.tokens} tokens)")
        results: list[RunStats] = []
        rows = [["RUN", "PROMPT", "TOK", "WALL_S", "TOK/S", "PREFILL_MS", "DECODE_MS"]]
        for i in range(args.runs):
            prompt = PROMPTS[i % len(PROMPTS)]
            stats = await run_one(coord, prompt, args.tokens)
            results.append(stats)
            rows.append([
                str(i + 1),
                prompt[:32] + ("…" if len(prompt) > 32 else ""),
                str(stats.tokens),
                f"{stats.wall_s:.2f}",
                f"{stats.tps:.1f}",
                f"{stats.prefill_ms:.0f}",
                f"{stats.avg_decode_ms:.0f}",
            ])
        print(fmt_table(rows, align=["r", "l", "r", "r", "r", "r", "r"]))

        # Aggregates
        banner("AGGREGATE STATS")
        all_tps = [r.tps for r in results]
        all_decode = [r.avg_decode_ms for r in results]
        all_prefill = [r.prefill_ms for r in results]
        total_tokens = sum(r.tokens for r in results)
        total_wall = sum(r.wall_s for r in results)
        agg_rows = [["METRIC", "VALUE"]]
        agg_rows.append(["mean tok/s", f"{statistics.mean(all_tps):.2f}"])
        agg_rows.append(["median tok/s", f"{statistics.median(all_tps):.2f}"])
        if len(all_tps) > 1:
            agg_rows.append(["stdev tok/s", f"{statistics.stdev(all_tps):.2f}"])
        agg_rows.append(["mean prefill (ms)", f"{statistics.mean(all_prefill):.1f}"])
        agg_rows.append(["mean per-token (ms)", f"{statistics.mean(all_decode):.1f}"])
        agg_rows.append(["total tokens", str(total_tokens)])
        agg_rows.append(["total wall (s)", f"{total_wall:.2f}"])
        agg_rows.append(["aggregate tok/s", f"{total_tokens/total_wall:.2f}"])
        print(fmt_table(agg_rows, align=["l", "r"]))

        # Per-stage latency breakdown
        banner("PER-STAGE LATENCY BREAKDOWN  (avg across all decode steps)")
        # Combine across runs
        n_stages = len(results[0].per_stage_avg_ms) if results else 0
        if n_stages:
            stage_rows = [["#", "WORKER", "LAYERS", "AVG_MS", "% OF STEP", "BAR"]]
            sums = [0.0] * n_stages
            for r in results:
                for i, v in enumerate(r.per_stage_avg_ms):
                    sums[i] += v
            avgs = [s / len(results) for s in sums]
            total = sum(avgs) or 1.0
            BAR_WIDTH = 30
            for i, w in enumerate(coord.workers):
                pct = avgs[i] / total * 100
                fill = int(round(pct / 100 * BAR_WIDTH))
                bar = "█" * fill + "·" * (BAR_WIDTH - fill)
                stage_rows.append([
                    str(i),
                    _worker_id(w),
                    f"[{w.layer_start}..{w.layer_end})",
                    f"{avgs[i]:.1f}",
                    f"{pct:.1f}%",
                    bar,
                ])
            print(fmt_table(stage_rows, align=["r", "l", "l", "r", "r", "l"]))

            # Bottleneck call-out
            slowest = max(range(n_stages), key=lambda i: avgs[i])
            print(f"\n  → bottleneck stage: worker #{slowest} "
                  f"({_worker_id(coord.workers[slowest])}) "
                  f"at {avgs[slowest]:.1f}ms/step "
                  f"({avgs[slowest]/total*100:.0f}% of pipeline time)")
            # The "ideal" rate IF we could overlap stages. Possible during prefill
            # (Jupiter-style intra-sequence PP) or with speculative decoding;
            # NOT possible during single-user greedy decode where each token
            # depends on the prior being sampled. We report it as the bound,
            # not a claim about what we'll get.
            theoretical_max = total / max(avgs) * statistics.mean(all_tps)
            print(f"  → ideal rate (overlap-limited): ~{theoretical_max:.1f} tok/s")
            print(f"     reachable for prefill-heavy workloads or with speculative decoding;")
            print(f"     decode is serially dependent so single-user steady-state is bounded")
            print(f"     by sum(stage_times), not max(stage_times).")

        # Bandwidth math
        banner("WIRE BANDWIDTH USED  (per token, per ring hop)")
        bytes_per_token = H * 2  # fp16
        hops_per_token = len(coord.workers) - 1  # hidden state crosses N-1 wires
        bw_rows = [["METRIC", "VALUE"]]
        bw_rows.append(["hidden state size", f"{bytes_per_token} bytes ({H} × fp16)"])
        bw_rows.append(["ring hops per token", str(hops_per_token)])
        bw_rows.append(["bytes per token (total)", str(bytes_per_token * hops_per_token)])
        bw_rate_kbs = bytes_per_token * hops_per_token * statistics.mean(all_tps) / 1024
        bw_rows.append(["bandwidth at observed tok/s", f"{bw_rate_kbs:.1f} KB/s"])
        bw_rows.append(["headroom on 100Mbps WiFi", f"{12500 / max(bw_rate_kbs, 1e-3):.0f}× over"])
        print(fmt_table(bw_rows, align=["l", "r"]))
        print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--workers", nargs="+", required=True)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--tokens", type=int, default=40)
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(name)s] %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
