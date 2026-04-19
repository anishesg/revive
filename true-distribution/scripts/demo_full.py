"""End-to-end demo of the BMC-managed distributed inference cluster.

Assumes two worker processes are already running (via scripts/launch_local.sh)
and brings up the BMC simulator + controller + coordinator in-process, then:

  1. Runs a baseline query through the full ring
  2. Injects a failure into worker B via the BMC
  3. Tries another query and shows the cluster refusing to route
  4. Heals worker B, shows cluster recovery

Run:  python -m scripts.demo_full --model Qwen/Qwen3-0.6B
"""
import argparse
import asyncio
import logging
import sys
import time

from pipeline.bmc_sim import serve as bmc_serve
from pipeline.controller import ClusterController, TCPSimLink
from pipeline.coordinator import (
    ClusterDegraded,
    PipelineCoordinator,
    WorkerEndpoint,
    _worker_id,
)

log = logging.getLogger("demo")


async def stream_generate(coord: PipelineCoordinator, prompt: str,
                          max_tokens: int = 40, tag: str = ""):
    print(f"\n{'═' * 60}")
    print(f"  {tag}  prompt: {prompt!r}")
    print(f"{'═' * 60}")
    print("  ", end="", flush=True)
    t0 = time.time()
    n = 0
    text = ""
    try:
        async for step in coord.generate(prompt, max_new_tokens=max_tokens,
                                          temperature=0.7, top_p=0.95, top_k=40):
            print(step.token_text, end="", flush=True)
            text += step.token_text
            n += 1
        dt = time.time() - t0
        print(f"\n  → {n} tokens in {dt:.2f}s = {n/dt:.1f} tok/s\n")
        return text
    except ClusterDegraded as e:
        print(f"\n  ✗ GENERATION ABORTED: {e}\n")
        return None


async def run(model: str, worker_urls: list[str], bmc_port: int):
    # 1. Start BMC simulator
    print("[stage 1] starting BMC simulator ...")
    sim_task = asyncio.create_task(bmc_serve("127.0.0.1", bmc_port))
    await asyncio.sleep(0.3)

    # 2. Connect controller to BMC
    print("[stage 2] connecting controller to BMC ...")
    link = TCPSimLink("127.0.0.1", bmc_port)
    ctrl = ClusterController(link)
    await ctrl.start()

    # 3. Build coordinator, run baseline
    print("[stage 3] connecting coordinator to workers ...")
    endpoints = []
    for hp in worker_urls:
        host, port = hp.rsplit(":", 1)
        endpoints.append(WorkerEndpoint(host=host, port=int(port)))

    try:
        async with PipelineCoordinator(endpoints, model, controller=ctrl) as coord:
            print(f"\n  cluster partition (from BMC):")
            for (wid, s, e) in ctrl.view.partition:
                print(f"    {wid:10s}  layers [{s:>2}..{e:>2})")
            print(f"  BMC state: {ctrl.view.state}")

            # Baseline generation
            await stream_generate(coord, "What is the capital of France?",
                                   max_tokens=30, tag="[BASELINE]")

            # 4. Chaos: kill the last worker via BMC
            victim = _worker_id(endpoints[-1])
            print(f"\n╳ CHAOS: telling BMC to declare '{victim}' dead ...")
            await ctrl.inject_failure(victim)
            await asyncio.sleep(0.3)
            print(f"  BMC state now: {ctrl.view.state}")
            print(f"  dead workers: {ctrl.view.dead_workers}")
            print(f"  new partition: {ctrl.view.partition}")

            # 5. Try a query — should bail out
            await stream_generate(coord, "Write me a haiku about fall.",
                                   max_tokens=30, tag="[POST-FAILURE]")

            # 6. Heal: send a heartbeat from the killed worker
            print(f"\n✚ HEAL: sending heartbeat from '{victim}' ...")
            await ctrl.heartbeat(victim, tps=30.0, temp_c=45)
            await asyncio.sleep(0.3)
            print(f"  BMC state now: {ctrl.view.state}")
            print(f"  dead workers: {ctrl.view.dead_workers}")

            # 7. Try again — should succeed
            await stream_generate(coord, "Explain photosynthesis briefly.",
                                   max_tokens=40, tag="[RECOVERED]")

    finally:
        await ctrl.stop()
        sim_task.cancel()
        try:
            await sim_task
        except asyncio.CancelledError:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--workers", nargs="+", required=True,
                   help="host:port for each worker in layer order")
    p.add_argument("--bmc-port", type=int, default=45555)
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(name)s] %(message)s")
    asyncio.run(run(args.model, args.workers, args.bmc_port))


if __name__ == "__main__":
    main()
