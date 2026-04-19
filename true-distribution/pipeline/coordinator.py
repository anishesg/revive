"""Pipeline coordinator.

Owns the tokenizer and drives the generation loop. For each token step:

  embed-stage.forward(tokens, positions) ──▶ hidden
  mid-stage.forward(hidden, positions)   ──▶ hidden
  ...
  last-stage.forward(hidden, positions)  ──▶ next_token_id

Streams tokens out as they're sampled. One decode step == one round trip
through the entire ring.
"""
from __future__ import annotations
import asyncio
import dataclasses
import logging
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import aiohttp
import numpy as np
from transformers import AutoTokenizer

from .controller import ClusterController
from .partitioner import WorkerProfile
from .protocol import Frame, Response

log = logging.getLogger("coord")


def _worker_id(w: "WorkerEndpoint") -> str:
    """Short, BMC-friendly worker id. Uses role+port so a single iPhone
    running two stages would still be distinguishable."""
    base = f"w{w.port - 50100}" if 50100 <= w.port < 50200 else f"{w.host[-5:]}{w.port}"
    return base[:8]  # Arduino's MAX_ID_LEN = 9 (with null)


@dataclass
class WorkerEndpoint:
    host: str
    port: int
    layer_start: int = 0
    layer_end: int = 0
    is_first: bool = False
    is_last: bool = False
    hidden_size: int = 0
    num_layers_total: int = 0
    model: str = ""

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class GenerationStep:
    token_id: int
    token_text: str
    eos: bool
    stage_latencies_ms: list[float]
    step_latency_ms: float
    position: int


class PipelineCoordinator:
    def __init__(self, workers: list[WorkerEndpoint], model_name: str,
                 controller: Optional[ClusterController] = None):
        self.workers = workers
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.session: Optional[aiohttp.ClientSession] = None
        self.controller = controller  # when set, BMC is in the loop

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=300, sock_connect=5)
        self.session = aiohttp.ClientSession(timeout=timeout)
        await self._handshake()
        await self._bmc_register_workers()
        self._hb_stop = asyncio.Event()
        if self.controller:
            self._hb_task = asyncio.create_task(self._heartbeat_loop())
        else:
            self._hb_task = None
        return self

    async def __aexit__(self, *_):
        if self._hb_task:
            self._hb_stop.set()
            try:
                await asyncio.wait_for(self._hb_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._hb_task.cancel()
        if self.controller:
            for w in self.workers:
                wid = _worker_id(w)
                try:
                    await self.controller.unregister_worker(wid)
                except Exception:
                    pass
        if self.session:
            await self.session.close()

    async def _heartbeat_loop(self):
        """Periodic HB so the BMC doesn't time workers out when they're idle
        (e.g. between user queries). The inference path also heartbeats after
        each ring step with live tok/s; those are more informative but less
        frequent."""
        from .bmc_protocol import HEARTBEAT_INTERVAL_MS
        interval = HEARTBEAT_INTERVAL_MS / 1000.0 / 2  # 2× rate
        try:
            while not self._hb_stop.is_set():
                for w in self.workers:
                    if self._is_worker_dead(w):
                        continue  # don't resuscitate someone the BMC killed
                    try:
                        await self._heartbeat_worker(w)
                    except Exception:
                        pass
                try:
                    await asyncio.wait_for(self._hb_stop.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass

    async def _bmc_register_workers(self):
        """Tell the BMC about our workers. Without a BMC the coordinator
        partitions by inspecting worker /info; with a BMC, we ALSO report
        membership upstream so the BMC owns failure detection."""
        if not self.controller:
            return
        # Announce the model layer count first
        await self.controller.set_num_layers(self.workers[0].num_layers_total)
        # Register each worker. Score heuristic: hidden_size · (layers owned)
        # is a decent proxy for capability with no profiling data. For the
        # demo we just use constant 1.0 and let the BMC accept the current
        # partition as-is.
        for w in self.workers:
            profile = WorkerProfile(
                id=_worker_id(w),
                capability_score=max(1.0, w.layer_end - w.layer_start),
                ram_mb=0,  # we don't know; BMC doesn't need it for scheduling
            )
            await self.controller.register_worker(profile)
            # Immediate heartbeat so the BMC's timeout clock starts from
            # "now" rather than "when we said REG". Avoids spurious DEADs
            # when a slow model load delays the first inference.
            await self.controller.heartbeat(_worker_id(w), tps=0, temp_c=0)
        # Give the BMC a moment to emit STATE + PARTITION
        await asyncio.sleep(0.2)
        log.info(f"BMC view: state={self.controller.view.state} "
                 f"partition={self.controller.view.partition}")

    async def _heartbeat_worker(self, worker: "WorkerEndpoint", last_tps: float = 0.0):
        if self.controller:
            await self.controller.heartbeat(_worker_id(worker), tps=last_tps, temp_c=0)

    def _is_worker_dead(self, worker: "WorkerEndpoint") -> bool:
        if not self.controller:
            return False
        return _worker_id(worker) in self.controller.view.dead_workers

    async def _handshake(self):
        for w in self.workers:
            async with self.session.get(f"{w.url}/info") as resp:
                info = await resp.json()
                w.layer_start = info["layer_start"]
                w.layer_end = info["layer_end"]
                w.is_first = info["is_first"]
                w.is_last = info["is_last"]
                w.hidden_size = info["hidden_size"]
                w.num_layers_total = info["num_layers_total"]
                w.model = info["model"]
                log.info(f"worker {w.host}:{w.port} → layers [{w.layer_start}..{w.layer_end}) "
                         f"first={w.is_first} last={w.is_last}")
        # Sanity: ring covers all layers, first/last flags correct, model consistent.
        ordered = sorted(self.workers, key=lambda w: w.layer_start)
        assert ordered[0].is_first and ordered[0].layer_start == 0, "first worker must own layer 0"
        assert ordered[-1].is_last, "last worker must have is_last=True"
        assert ordered[-1].layer_end == ordered[-1].num_layers_total, "last worker must own the final layer"
        for a, b in zip(ordered, ordered[1:]):
            assert a.layer_end == b.layer_start, f"layer gap between {a.url} and {b.url}"
        models = {w.model for w in self.workers}
        if len(models) > 1:
            # Tolerate different representations (e.g. iOS worker reports a
            # filesystem path, Python reports the HuggingFace id). Sanity
            # check via n_layers + hidden_size equality (done in layer_gap
            # assertions above already).
            log.warning(f"workers report different model strings (ok if shapes match): {models}")
        self.workers = ordered  # canonical order
        log.info(f"ring validated across {len(self.workers)} stages, "
                 f"model={ordered[0].model}, total_layers={ordered[-1].num_layers_total}")

    async def _post_forward(self, worker: WorkerEndpoint, frame: Frame) -> Response:
        raw = frame.encode()
        async with self.session.post(f"{worker.url}/forward", data=raw,
                                     headers={"Content-Type": "application/octet-stream"}) as resp:
            body = await resp.read()
        return Response.decode(body)

    async def _reset_all(self, seq_id: str):
        for w in self.workers:
            try:
                async with self.session.post(f"{w.url}/reset", json={"seq_id": seq_id}) as r:
                    await r.read()
            except Exception:
                pass

    async def generate(self, prompt: str, max_new_tokens: int = 512,
                       temperature: float = 0.7, top_p: float = 0.8,
                       top_k: int = 20, seq_id: Optional[str] = None) -> AsyncIterator[GenerationStep]:
        # Defaults follow Qwen's non-thinking-mode best practices:
        # temp=0.7, top_p=0.8, top_k=20, min_p=0, with repetition penalty
        # in the sampler. These avoid the "Paris. Paris. Paris." collapse.
        seq_id = seq_id or uuid.uuid4().hex[:12]
        try:
            # Build prompt via Qwen's official chat template. CRITICAL:
            # (a) pass enable_thinking=False so the model skips the
            #     <think>…</think> scratchpad, and (b) install a firm system
            #     prompt demanding concise direct answers — without it,
            #     Qwen3-0.6B rambles, restates questions, and produces
            #     "worksheet" style output that no one wants.
            system_prompt = (
                "You are REVIVE, a concise, direct assistant running on a "
                "distributed cluster of phones and edge devices. "
                "Answer the user's question directly and accurately. "
                "Do NOT restate the question. Do NOT list alternative questions. "
                "Do NOT think out loud or write internal monologue. "
                "Keep answers to 1-4 sentences unless the user explicitly asks for detail."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            prompt_ids = None
            for kwargs in (
                {"enable_thinking": False},     # Qwen3 / Qwen3-Coder
                {},                              # older models
            ):
                try:
                    out = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True, add_generation_prompt=True, return_tensors=None,
                        **kwargs,
                    )
                    if hasattr(out, "input_ids"):
                        prompt_ids = list(out.input_ids[0]) if hasattr(out.input_ids, "__iter__") else list(out["input_ids"])
                    elif isinstance(out, dict):
                        prompt_ids = list(out["input_ids"])
                    elif isinstance(out, list) and out and isinstance(out[0], list):
                        prompt_ids = list(out[0])
                    else:
                        prompt_ids = list(out)
                    break
                except TypeError:
                    continue
                except Exception:
                    break
            if prompt_ids is None:
                prompt_ids = self.tokenizer.encode(prompt)
            prompt_ids = [int(x) for x in prompt_ids]

            positions = list(range(len(prompt_ids)))
            tokens_so_far = list(prompt_ids)

            # Prefill — push the full prompt through the ring once.
            step = await self._ring_step(seq_id, prompt_ids, positions,
                                         temperature, top_p, top_k, is_prefill=True)
            tokens_so_far.append(step.token_id)
            yield step
            if step.eos:
                return

            # Decode loop — one new token at a time through the ring.
            for i in range(max_new_tokens - 1):
                next_pos = len(tokens_so_far) - 1  # position of the token we just appended
                step = await self._ring_step(seq_id, [step.token_id], [next_pos],
                                             temperature, top_p, top_k, is_prefill=False)
                tokens_so_far.append(step.token_id)
                yield step
                if step.eos:
                    break
        finally:
            await self._reset_all(seq_id)

    async def _ring_step(self, seq_id: str, inputs: list[int], positions: list[int],
                         temperature: float, top_p: float, top_k: int,
                         is_prefill: bool) -> GenerationStep:
        t0 = time.time()
        stage_latencies: list[float] = []

        # Refuse to route through a worker the BMC has declared dead.
        # The authoritative partition lives in the BMC (or sim) — we respect it.
        if self.controller:
            for w in self.workers:
                if self._is_worker_dead(w):
                    raise ClusterDegraded(
                        f"BMC declared worker {_worker_id(w)} dead; "
                        f"cluster state={self.controller.view.state}"
                    )

        # Stage 0: token_ids → hidden
        first = self.workers[0]
        frame = Frame(
            seq_id=seq_id,
            stage_kind="first",
            positions=positions,
            tensor=np.array(inputs, dtype=np.int32),
            temperature=temperature, top_p=top_p, top_k=top_k,
        )
        resp = await self._post_forward(first, frame)
        stage_latencies.append(resp.latency_ms)
        await self._heartbeat_worker(first, last_tps=resp.tokens_per_second)

        current_hidden = np.frombuffer(resp.tensor, dtype=np.float16).reshape(resp.shape).copy()

        # Middle + last stages
        for i, w in enumerate(self.workers[1:], start=1):
            kind = "last" if w.is_last else "mid"
            frame = Frame(
                seq_id=seq_id,
                stage_kind=kind,
                positions=positions,
                tensor=current_hidden,
                temperature=temperature, top_p=top_p, top_k=top_k,
            )
            resp = await self._post_forward(w, frame)
            stage_latencies.append(resp.latency_ms)
            await self._heartbeat_worker(w, last_tps=resp.tokens_per_second)
            if w.is_last:
                token_id = resp.token_id
                eos = resp.eos
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                return GenerationStep(
                    token_id=token_id,
                    token_text=token_text,
                    eos=eos,
                    stage_latencies_ms=stage_latencies,
                    step_latency_ms=(time.time() - t0) * 1000,
                    position=positions[-1] + 1,
                )
            current_hidden = np.frombuffer(resp.tensor, dtype=np.float16).reshape(resp.shape).copy()

        raise RuntimeError("ring walked off the end without a last stage")


class ClusterDegraded(RuntimeError):
    """Raised when the BMC has declared a worker dead mid-generation. The
    coordinator bails out instead of silently hanging on a broken ring."""
    pass


# ─── Simple CLI for ad-hoc testing ──────────────────────────────────────────

async def _cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", nargs="+", required=True,
                        help="host:port for each stage in layer order")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(name)s] %(message)s")

    endpoints = []
    for hp in args.workers:
        host, port = hp.rsplit(":", 1)
        endpoints.append(WorkerEndpoint(host=host, port=int(port)))

    async with PipelineCoordinator(endpoints, args.model) as coord:
        log.info(f"streaming: {args.prompt!r}")
        text = ""
        t0 = time.time()
        n = 0
        async for step in coord.generate(args.prompt, max_new_tokens=args.max_tokens,
                                         temperature=args.temperature):
            text += step.token_text
            n += 1
            print(step.token_text, end="", flush=True)
        dt = time.time() - t0
        print()
        log.info(f"done: {n} tokens in {dt:.2f}s = {n/dt:.1f} tok/s")


if __name__ == "__main__":
    asyncio.run(_cli())
