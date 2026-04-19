"""Verify the partitioner produces valid, sensible assignments."""
from pipeline.partitioner import WorkerProfile, greedy_partition, pipeedge_dp, validate_partition


def test_uniform_workers():
    workers = [WorkerProfile(f"w{i}", 1.0, 1024) for i in range(3)]
    p = greedy_partition(36, workers)
    validate_partition(p, 36)
    assert len(p) == 3
    # Uniform workers + 36 layers → expect 12/12/12
    for _, s, e in p:
        assert e - s == 12


def test_heterogeneous_greedy():
    workers = [
        WorkerProfile("fast_phone", 2.0, 4096),
        WorkerProfile("pi", 0.5, 2048),
        WorkerProfile("slow_phone", 1.0, 2048),
    ]
    p = greedy_partition(28, workers)
    validate_partition(p, 28)
    # fast_phone should get the most layers
    spans = {w: e - s for (w, s, e) in p}
    assert spans["fast_phone"] > spans["pi"]


def test_pipeedge_beats_greedy_on_skew():
    # 10 layers, one worker is 5× faster. Greedy assigns proportionally;
    # PipeEdge will also concentrate layers on the fast worker to balance
    # per-stage time.
    workers = [
        WorkerProfile("fast", 5.0, 8192),
        WorkerProfile("slow", 1.0, 2048),
    ]
    p_g = greedy_partition(10, workers)
    p_dp = pipeedge_dp(10, workers)
    validate_partition(p_g, 10)
    validate_partition(p_dp, 10)

    # Compute max-stage-time for each
    def max_stage(partition, workers):
        wmap = {w.id: w for w in workers}
        return max((e - s) / wmap[wid].capability_score for wid, s, e in partition)

    t_g = max_stage(p_g, workers)
    t_dp = max_stage(p_dp, workers)
    # DP should be ≤ greedy (it's optimal for this objective)
    assert t_dp <= t_g + 1e-6, f"DP {t_dp} worse than greedy {t_g}"


def test_single_worker():
    w = [WorkerProfile("only", 1.0, 4096)]
    p = greedy_partition(28, w)
    validate_partition(p, 28)
    assert p == [("only", 0, 28)]


def test_dead_workers_excluded():
    workers = [
        WorkerProfile("a", 1.0, 1024, alive=True),
        WorkerProfile("b", 1.0, 1024, alive=False),
        WorkerProfile("c", 1.0, 1024, alive=True),
    ]
    p = greedy_partition(8, workers)
    validate_partition(p, 8)
    ids = {wid for wid, _, _ in p}
    assert "b" not in ids
    assert ids == {"a", "c"}


if __name__ == "__main__":
    test_uniform_workers(); print("✓ uniform")
    test_heterogeneous_greedy(); print("✓ heterogeneous greedy")
    test_pipeedge_beats_greedy_on_skew(); print("✓ DP ≤ greedy")
    test_single_worker(); print("✓ single worker")
    test_dead_workers_excluded(); print("✓ dead excluded")
    print("\nALL PARTITIONER TESTS PASSED")
