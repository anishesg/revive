"""End-to-end test: BMC simulator + controller + partition decisions.

Exercises the full control plane without touching any LLM code.

  1. Start BMC sim on a random port
  2. Controller connects
  3. Register 3 workers with heterogeneous scores
  4. Set model layers=28
  5. Expect a PARTITION that sums to 28 and respects score proportions
  6. Inject failure on worker B
  7. Expect a new PARTITION without B, still summing to 28
  8. Send heartbeats from A, C (skip B)
  9. Let worker B's "timeout" elapse via FAIL injection (sim has 6s timeout
     in real use; we use FAIL here to keep the test fast)
"""
import asyncio
import random

from pipeline.bmc_sim import serve as bmc_serve
from pipeline.controller import ClusterController, TCPSimLink
from pipeline.partitioner import WorkerProfile, validate_partition


async def wait_for(predicate, timeout=5.0, interval=0.05):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


async def run():
    port = random.randint(40000, 50000)

    # Launch the BMC simulator as a task
    sim_task = asyncio.create_task(bmc_serve("127.0.0.1", port))
    await asyncio.sleep(0.2)  # let it bind the socket

    link = TCPSimLink("127.0.0.1", port)
    ctrl = ClusterController(link)
    await ctrl.start()

    try:
        partition_changes: list[int] = [0]

        async def on_change(view):
            partition_changes[0] += 1
            print(f"  [change #{partition_changes[0]}] state={view.state} "
                  f"partition={view.partition} dead={view.dead_workers}")

        ctrl.on_partition_change = on_change

        # 1. Tell BMC the model size
        await ctrl.set_num_layers(28)

        # 2. Register 3 workers with different scores
        await ctrl.register_worker(WorkerProfile("phoneA", capability_score=2.0, ram_mb=4096))
        await ctrl.register_worker(WorkerProfile("pi",     capability_score=0.5, ram_mb=2048))
        await ctrl.register_worker(WorkerProfile("phoneB", capability_score=1.5, ram_mb=4096))

        # 3. Wait for a valid partition covering all 28 layers with 3 stages
        ok = await wait_for(lambda: (
            ctrl.view.state == "healthy" and
            len(ctrl.view.partition) == 3 and
            ctrl.view.partition[-1][2] == 28
        ), timeout=3.0)
        assert ok, f"no healthy 3-stage partition: {ctrl.view}"

        validate_partition(ctrl.view.partition, 28)
        print(f"\n✓ Initial partition: {ctrl.view.partition}")

        # Verify proportions roughly: phoneA gets the most, pi the least
        spans = {wid: e - s for (wid, s, e) in ctrl.view.partition}
        assert spans["phoneA"] >= spans["phoneB"] >= spans["pi"], (
            f"proportion wrong: {spans}"
        )
        print(f"✓ Layer shares by score: {spans}")

        # 4. Chaos: kill the pi
        before_ver = ctrl.view.version
        await ctrl.inject_failure("pi")

        # Expect the BMC to emit DEAD + a new 2-stage partition
        ok = await wait_for(lambda: (
            "pi" in ctrl.view.dead_workers and
            len(ctrl.view.partition) == 2 and
            ctrl.view.version > before_ver
        ), timeout=2.0)
        assert ok, f"failover didn't happen: {ctrl.view}"

        validate_partition(ctrl.view.partition, 28)
        print(f"\n✓ Post-failure partition: {ctrl.view.partition}  state={ctrl.view.state}")
        assert ctrl.view.state == "degraded"

        # 5. Heartbeat from pi → should come back alive
        await ctrl.heartbeat("pi", tps=2.5, temp_c=52)
        ok = await wait_for(lambda: (
            "pi" not in ctrl.view.dead_workers and
            len(ctrl.view.partition) == 3 and
            ctrl.view.state == "healthy"
        ), timeout=2.0)
        assert ok, f"pi didn't come back: {ctrl.view}"
        print(f"\n✓ Pi recovered: {ctrl.view.partition}  state={ctrl.view.state}")

        print(f"\nTotal partition changes observed: {partition_changes[0]}")

    finally:
        await ctrl.stop()
        sim_task.cancel()
        try:
            await sim_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(run())
    print("\n═══════════════════════════════════")
    print("✓ BMC integration test PASSED")
    print("═══════════════════════════════════")
