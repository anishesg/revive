"""Smoke test: load a single stage and run one forward pass.

This doesn't use HTTP — it imports PipelineStage directly and calls
.forward() to isolate layer-slicing + KV-cache correctness from networking.
"""
import numpy as np
import torch

from pipeline.protocol import Frame
from pipeline.worker import PipelineStage


MODEL = "Qwen/Qwen3-0.6B"


def test_first_stage_loads_and_forwards():
    stage = PipelineStage(MODEL, layer_start=0, layer_end=4, is_first=True, is_last=False)
    assert stage.embed is not None
    assert stage.lm_head is None

    tokens = [151644, 872, 198, 9707, 151645]  # arbitrary valid ids
    frame = Frame(
        seq_id="smoke-1",
        stage_kind="first",
        positions=list(range(len(tokens))),
        tensor=np.array(tokens, dtype=np.int32),
    )
    resp = stage.forward(frame)
    assert resp.token_id is None  # not last stage
    arr = np.frombuffer(resp.tensor, dtype=np.float16).reshape(resp.shape)
    assert arr.shape == (1, len(tokens), stage.hidden_size)
    assert np.isfinite(arr).all()


def test_last_stage_loads_and_samples():
    stage = PipelineStage(MODEL, layer_start=24, layer_end=28, is_first=False, is_last=True)
    assert stage.norm is not None
    assert stage.lm_head is not None
    # Fake hidden input of the right shape
    H = stage.hidden_size
    hidden = np.random.randn(1, 5, H).astype(np.float16) * 0.02
    frame = Frame(
        seq_id="smoke-2",
        stage_kind="last",
        positions=list(range(5)),
        tensor=hidden,
        temperature=1.0, top_k=10, top_p=0.9,
    )
    resp = stage.forward(frame)
    assert resp.token_id is not None
    assert 0 <= resp.token_id < stage.config.vocab_size


if __name__ == "__main__":
    test_first_stage_loads_and_forwards()
    print("first-stage smoke OK")
    test_last_stage_loads_and_samples()
    print("last-stage smoke OK")
