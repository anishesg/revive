"""Dashboard server for the BMC-managed distributed inference cluster.

Runs the whole stack in one process:
  - BMC simulator (or connects to real Arduino later)
  - ClusterController
  - PipelineCoordinator (pre-launched workers registered via HTTP)
  - Web UI + SSE stream of BMC events, worker state, token stream

Endpoints:
  GET  /                     dashboard html
  GET  /events               SSE stream of state updates
  POST /query                {"prompt": "..."}
  POST /chaos/fail           {"worker_id": "w1"}
  POST /chaos/heal           {"worker_id": "w1"}
  POST /workers/register     {"host": "...", "port": 50100}  (for manual ring assembly)
  GET  /bmc/log              recent BMC protocol events

Launch:
  python -m dashboard.server --model Qwen/Qwen3-0.6B --workers 127.0.0.1:50100 127.0.0.1:50101
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import asdict
from typing import Optional

from aiohttp import web

from pipeline.bmc_sim import serve as bmc_serve, serve_many as bmc_serve_many
from pipeline.controller import (
    BMCLink,
    ClusterController,
    SerialLink,
    TCPSimLink,
    find_arduino_device,
)
from pipeline.coordinator import (
    ClusterDegraded,
    PipelineCoordinator,
    WorkerEndpoint,
    _worker_id,
)
from pipeline.multi_bmc import MultiBMCController
from pipeline.net_utils import get_lan_ip

log = logging.getLogger("dashboard")


# ─── shared app state ─────────────────────────────────────────────────────

class Chat:
    """One conversation thread. Messages kept in order."""
    def __init__(self, chat_id: str, title: str = "New chat"):
        self.id = chat_id
        self.title = title
        self.created_at = time.time()
        self.updated_at = time.time()
        self.messages: list[dict] = []

    def to_dict(self, include_messages: bool = True) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": self.messages if include_messages else [],
            "message_count": len(self.messages),
        }


class AppState:
    def __init__(self):
        self.controller: Optional[ClusterController] = None
        self.coord: Optional[PipelineCoordinator] = None
        self.workers: list[WorkerEndpoint] = []
        self.sse_clients: list[web.StreamResponse] = []
        self.bmc_events: list[dict] = []   # recent BMC protocol events
        self.total_tokens = 0
        self.last_collective_tps = 0.0
        self.busy = False
        self.bmc_source = "unknown"
        self.lan_ip: str = "127.0.0.1"
        self.public_url: str = ""
        self.dashboard_port: int = 4100

        # Chats
        self.chats: dict[str, Chat] = {}
        self.active_chat_id: Optional[str] = None
        self._seed_initial_chat()

    def _seed_initial_chat(self):
        cid = uuid.uuid4().hex[:8]
        self.chats[cid] = Chat(cid, title="New chat")
        self.active_chat_id = cid

    def active_chat(self) -> Optional[Chat]:
        if self.active_chat_id is None:
            return None
        return self.chats.get(self.active_chat_id)

    def create_chat(self, title: str = "New chat") -> Chat:
        cid = uuid.uuid4().hex[:8]
        chat = Chat(cid, title=title)
        self.chats[cid] = chat
        self.active_chat_id = cid
        return chat

    def delete_chat(self, chat_id: str) -> bool:
        if chat_id not in self.chats:
            return False
        del self.chats[chat_id]
        if self.active_chat_id == chat_id:
            # Pick next available, or seed a new one
            remaining = sorted(self.chats.values(), key=lambda c: c.updated_at, reverse=True)
            if remaining:
                self.active_chat_id = remaining[0].id
            else:
                self._seed_initial_chat()
        return True

    def snapshot(self) -> dict:
        ctrl = self.controller
        bmc = {
            "state": ctrl.view.state if ctrl else "disconnected",
            "partition": ctrl.view.partition if ctrl else [],
            "dead_workers": list(ctrl.view.dead_workers) if ctrl else [],
            "version": ctrl.view.version if ctrl else 0,
        }
        # Multi-BMC HA info (None when running single-BMC mode)
        bmc_ha = None
        if isinstance(ctrl, MultiBMCController):
            snap = ctrl.snapshot()
            bmc_ha = {
                "leader": snap["leader"],
                "replicas": snap["replicas"],
            }
        workers_info = []
        for w in self.workers:
            wid = _worker_id(w)
            workers_info.append({
                "id": wid,
                "host": w.host,
                "port": w.port,
                "layer_start": w.layer_start,
                "layer_end": w.layer_end,
                "is_first": w.is_first,
                "is_last": w.is_last,
                "dead": wid in (ctrl.view.dead_workers if ctrl else set()),
            })
        # Chat metadata for sidebar (light — no messages)
        chat_list = [c.to_dict(include_messages=False)
                     for c in sorted(self.chats.values(),
                                      key=lambda c: c.updated_at, reverse=True)]
        active = self.active_chat()
        active_messages = active.messages[-100:] if active else []

        return {
            "bmc": bmc,
            "bmc_ha": bmc_ha,
            "bmc_source": self.bmc_source,
            "lan_ip": self.lan_ip,
            "public_url": self.public_url,
            "workers": workers_info,
            "chats": chat_list,
            "active_chat_id": self.active_chat_id,
            "messages": active_messages,
            "bmc_events": self.bmc_events[-40:],
            "total_tokens": self.total_tokens,
            "collective_tps": self.last_collective_tps,
            "num_layers": self.workers[0].num_layers_total if self.workers else 0,
            "model": self.workers[0].model if self.workers else "",
            "busy": self.busy,
        }


# ─── SSE ──────────────────────────────────────────────────────────────────

async def broadcast(state: AppState):
    payload = ("data: " + json.dumps(state.snapshot()) + "\n\n").encode()
    dead = []
    for resp in list(state.sse_clients):
        try:
            await resp.write(payload)
        except Exception:
            dead.append(resp)
    for r in dead:
        if r in state.sse_clients:
            state.sse_clients.remove(r)


async def handle_sse(request: web.Request) -> web.StreamResponse:
    state: AppState = request.app["state"]
    resp = web.StreamResponse(headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Access-Control-Allow-Origin": "*",
        "X-Accel-Buffering": "no",
    })
    await resp.prepare(request)
    state.sse_clients.append(resp)
    await resp.write(("data: " + json.dumps(state.snapshot()) + "\n\n").encode())
    try:
        while True:
            await asyncio.sleep(20)
            await resp.write(b": ping\n\n")
    except Exception:
        pass
    finally:
        if resp in state.sse_clients:
            state.sse_clients.remove(resp)
    return resp


# ─── query + chaos endpoints ──────────────────────────────────────────────

async def handle_query(request: web.Request) -> web.Response:
    state: AppState = request.app["state"]
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    max_tokens = int(body.get("max_tokens", 512))  # big default — Qwen naturally hits <|im_end|> way before this
    chat_id = body.get("chat_id") or state.active_chat_id
    if not prompt:
        return web.json_response({"ok": False, "error": "empty prompt"}, status=400)
    if chat_id and chat_id != state.active_chat_id:
        state.active_chat_id = chat_id

    if state.busy:
        return web.json_response({"ok": False, "error": "already generating"}, status=409)
    asyncio.ensure_future(_run_query(state, prompt, max_tokens))
    return web.json_response({"ok": True})


import re as _re
_THINK_RE = _re.compile(r"<think>.*?</think>\s*", _re.DOTALL | _re.IGNORECASE)
_LEAK_HEADS = (
    "here's the prompt:", "here is the prompt:", "the prompt is:",
    "the answer should be", "also include examples",
    "okay, so i need to", "let me think", "let me start by", "let me recall",
    "first, i need to", "i should",
)


def _clean_output(text: str) -> str:
    """Sanitize generated text before it's shown to the user.

    Strips:
      1. <think>...</think> blocks (thinking-mode leakage)
      2. The whole tail of the output starting at obvious reasoning/meta
         patterns like "Okay, so I need to..." — Qwen3-0.6B tends to
         continue into a second round of meta-reasoning that ruins the
         answer. We keep only the first clean chunk.
    """
    if not text:
        return text
    # 1) remove <think> blocks
    text = _THINK_RE.sub("", text)

    # 2) cut at the first leak heading (case-insensitive).
    lower = text.lower()
    cut = len(text)
    for head in _LEAK_HEADS:
        idx = lower.find(head)
        if idx >= 0 and idx < cut:
            cut = idx
    # also cut at an obvious "Here's the prompt: \"..." if we spot it mid-text
    text = text[:cut]
    return text.strip()


async def _run_query(state: AppState, prompt: str, max_tokens: int):
    state.busy = True
    chat = state.active_chat()
    if chat is None:
        chat = state.create_chat()

    # Auto-title the chat from the first prompt
    if chat.title == "New chat" and not chat.messages:
        chat.title = prompt if len(prompt) <= 40 else prompt[:38] + "…"

    await broadcast(state)
    qid = uuid.uuid4().hex[:8]
    chat.messages.append({
        "role": "user", "content": prompt, "qid": qid, "ts": time.time()
    })
    chat.updated_at = time.time()
    await broadcast(state)
    assistant_entry = {
        "role": "assistant", "content": "", "qid": qid, "ts": time.time(),
        "tps": 0.0, "tokens": 0, "aborted": False, "abort_reason": "",
        "stage_latencies_ms": [],
        "avg_stage_latencies_ms": [],
        "step_latency_ms": 0.0,
        "streaming": True,
    }
    chat.messages.append(assistant_entry)

    t0 = time.time()
    tokens = 0
    stage_lat_sums: list[float] = []
    last_broadcast = 0.0
    try:
        async for step in state.coord.generate(prompt, max_new_tokens=max_tokens,
                                                temperature=0.7, top_p=0.95, top_k=40):
            assistant_entry["content"] += step.token_text
            tokens += 1
            state.total_tokens += 1
            dt = time.time() - t0
            assistant_entry["tokens"] = tokens
            assistant_entry["tps"] = tokens / dt if dt > 0 else 0
            assistant_entry["stage_latencies_ms"] = step.stage_latencies_ms
            assistant_entry["step_latency_ms"] = step.step_latency_ms
            # Rolling average per stage
            if not stage_lat_sums:
                stage_lat_sums = list(step.stage_latencies_ms)
            else:
                for i, v in enumerate(step.stage_latencies_ms):
                    if i < len(stage_lat_sums):
                        stage_lat_sums[i] += v
                    else:
                        stage_lat_sums.append(v)
            assistant_entry["avg_stage_latencies_ms"] = [s / tokens for s in stage_lat_sums]
            state.last_collective_tps = assistant_entry["tps"]
            # Per-token broadcast, but coalesce if SSE clients can't keep up.
            # Hard cap: at most 30 broadcasts/sec to avoid drowning the browser.
            now = time.time()
            if (now - last_broadcast) >= 0.033 or step.eos:
                await broadcast(state)
                last_broadcast = now
        assistant_entry["tps"] = tokens / max(time.time() - t0, 1e-6)
        # Final sanitization pass — removes any thinking-leak tail
        assistant_entry["content"] = _clean_output(assistant_entry["content"])
    except ClusterDegraded as e:
        assistant_entry["aborted"] = True
        assistant_entry["abort_reason"] = str(e)
    finally:
        assistant_entry["streaming"] = False
        state.busy = False
        chat.updated_at = time.time()
        await broadcast(state)


async def handle_chaos_fail(request: web.Request) -> web.Response:
    state: AppState = request.app["state"]
    body = await request.json()
    wid = body.get("worker_id", "").strip()
    if not wid:
        return web.json_response({"ok": False, "error": "missing worker_id"}, status=400)
    await state.controller.inject_failure(wid)
    return web.json_response({"ok": True})


async def handle_chaos_heal(request: web.Request) -> web.Response:
    state: AppState = request.app["state"]
    body = await request.json()
    wid = body.get("worker_id", "").strip()
    if not wid:
        return web.json_response({"ok": False, "error": "missing worker_id"}, status=400)
    await state.controller.heartbeat(wid, tps=0, temp_c=0)
    return web.json_response({"ok": True})


async def handle_bmc_kill(request: web.Request) -> web.Response:
    """Forcibly disconnect from a BMC replica — simulates yanking the
    Arduino's USB cable. Only meaningful in multi-BMC / HA mode."""
    state: AppState = request.app["state"]
    body = await request.json()
    rid = body.get("replica_id", "").strip()
    if not rid:
        return web.json_response({"ok": False, "error": "missing replica_id"}, status=400)
    if not isinstance(state.controller, MultiBMCController):
        return web.json_response({"ok": False, "error": "not in HA mode"}, status=409)
    try:
        await state.controller.kill_replica(rid)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)
    await broadcast(state)
    return web.json_response({"ok": True})


async def handle_bmc_revive(request: web.Request) -> web.Response:
    """Re-open the link to a BMC replica (simulates plugging the Arduino
    back in). Note: a just-revived replica has empty cluster state until
    a future heartbeat/register storm; it will not be elected leader unless
    the incumbent has lower priority or has died since."""
    state: AppState = request.app["state"]
    body = await request.json()
    rid = body.get("replica_id", "").strip()
    if not rid:
        return web.json_response({"ok": False, "error": "missing replica_id"}, status=400)
    if not isinstance(state.controller, MultiBMCController):
        return web.json_response({"ok": False, "error": "not in HA mode"}, status=409)
    try:
        await state.controller.revive_replica(rid)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)
    await broadcast(state)
    return web.json_response({"ok": True})


async def handle_index(request: web.Request) -> web.StreamResponse:
    here = os.path.dirname(os.path.abspath(__file__))
    # Disable caching so dashboard edits show on a normal refresh — avoids
    # "reload is stuck" / "I see the old UI" issues during iteration.
    resp = web.FileResponse(os.path.join(here, "index.html"))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


async def handle_bmc_log(request: web.Request) -> web.Response:
    state: AppState = request.app["state"]
    log_lines = state.controller.recent_events(n=80) if state.controller else []
    return web.json_response([
        {"ts": ts, "dir": d.strip(), "line": line}
        for (ts, d, line) in log_lines
    ])


async def handle_chat_new(request: web.Request) -> web.Response:
    state: AppState = request.app["state"]
    chat = state.create_chat()
    await broadcast(state)
    return web.json_response({"ok": True, "chat_id": chat.id})


async def handle_chat_delete(request: web.Request) -> web.Response:
    state: AppState = request.app["state"]
    chat_id = request.match_info["chat_id"]
    ok = state.delete_chat(chat_id)
    await broadcast(state)
    return web.json_response({"ok": ok})


async def handle_chat_activate(request: web.Request) -> web.Response:
    state: AppState = request.app["state"]
    chat_id = request.match_info["chat_id"]
    if chat_id not in state.chats:
        return web.json_response({"ok": False, "error": "unknown chat"}, status=404)
    state.active_chat_id = chat_id
    await broadcast(state)
    return web.json_response({"ok": True})


async def handle_discover(request: web.Request) -> web.Response:
    """Lightweight JSON endpoint for phones/iPads to auto-discover the cluster.
    A worker app can hit http://<mac_ip>:4100/discover to find out how to
    join. Complements the Arduino's serial-level DISCOVER reply."""
    state: AppState = request.app["state"]
    return web.json_response({
        "public_url": state.public_url,
        "lan_ip": state.lan_ip,
        "port": state.dashboard_port,
        "register_worker": f"{state.public_url}/worker-register",
        "events_stream": f"{state.public_url}/events",
        "bmc_state": state.controller.view.state if state.controller else "disconnected",
        "model": state.workers[0].model if state.workers else None,
        "num_workers_connected": len(state.workers),
    })


# ─── app startup / shutdown ───────────────────────────────────────────────

async def _wire_bmc_events_to_broadcast(state: AppState):
    """Wrap controller callbacks so every BMC event triggers a dashboard push."""
    orig_log = state.controller._log
    def logged(direction, line):
        orig_log(direction, line)
        state.bmc_events.append({"ts": time.time(), "dir": direction.strip(), "line": line})
        if len(state.bmc_events) > 200:
            state.bmc_events = state.bmc_events[-200:]
        # Schedule a broadcast on the running loop
        asyncio.ensure_future(broadcast(state))
    state.controller._log = logged

    async def on_change(view):
        await broadcast(state)
    state.controller.on_partition_change = on_change

    # Multi-BMC: also hook the leader-change callback so the dashboard
    # animates the failover moment.
    if isinstance(state.controller, MultiBMCController):
        async def on_leader(new_leader, prev):
            state.bmc_events.append({
                "ts": time.time(), "dir": "**",
                "line": f"LEADER_CHANGED {prev or '(none)'} → {new_leader or '(none)'}"
            })
            await broadcast(state)
        state.controller.on_leader_change = on_leader


async def on_startup(app: web.Application):
    cfg = app["config"]
    state: AppState = app["state"]

    # Detect LAN IP so phones/iPads on the same network can reach the dashboard.
    state.lan_ip = get_lan_ip()
    state.dashboard_port = int(cfg.get("dashboard_port", 4100))
    state.public_url = f"http://{state.lan_ip}:{state.dashboard_port}"
    log.info(f"dashboard LAN URL: {state.public_url}")

    bmc_count = int(cfg.get("bmc_count", 1))

    # ─── HA / multi-BMC path ─────────────────────────────────────────────
    if bmc_count > 1:
        if cfg["serial_device"]:
            # Future: multiple physical Arduinos over serial. For now, HA
            # requires the simulator; we'd wire this by passing a comma-sep
            # list of /dev paths and building N SerialLinks.
            raise RuntimeError("--bmc-count > 1 not yet supported with --serial-device")
        start_port = cfg["bmc_port"]
        log.info(f"starting {bmc_count} embedded BMC replicas on ports "
                 f"{start_port}..{start_port + bmc_count - 1}")
        app["bmc_task"] = asyncio.create_task(
            bmc_serve_many("127.0.0.1", start_port, bmc_count)
        )
        await asyncio.sleep(0.4)

        links = []
        for i in range(bmc_count):
            port = start_port + i
            rid = f"bmc{i}"
            links.append((TCPSimLink("127.0.0.1", port), rid,
                          f"tcp://127.0.0.1:{port}"))
        state.controller = MultiBMCController(links)
        await state.controller.start()
        state.bmc_source = f"simulator-ha:{bmc_count}x"
        await _wire_bmc_events_to_broadcast(state)
    else:
        # ─── single BMC (existing path) ──────────────────────────────────
        link: BMCLink
        if cfg["serial_device"]:
            log.info(f"connecting to physical Arduino BMC at {cfg['serial_device']}")
            link = SerialLink(cfg["serial_device"])
            state.bmc_source = f"arduino:{cfg['serial_device']}"
        else:
            if cfg["start_bmc_sim"]:
                log.info(f"starting embedded BMC simulator on port {cfg['bmc_port']}")
                app["bmc_task"] = asyncio.create_task(bmc_serve("127.0.0.1", cfg["bmc_port"]))
                await asyncio.sleep(0.3)
            link = TCPSimLink("127.0.0.1", cfg["bmc_port"])
            state.bmc_source = f"simulator:tcp://127.0.0.1:{cfg['bmc_port']}"
        state.controller = ClusterController(link)
        await state.controller.start()
        await _wire_bmc_events_to_broadcast(state)

    # Push our LAN URL to the BMC so DISCOVER returns it. Works for both
    # single-controller and multi-BMC modes (same method signature).
    try:
        await state.controller.set_coordinator_url(state.public_url)
    except Exception as e:
        log.warning(f"could not push coordinator URL to BMC: {e}")

    # 3. Build coordinator
    endpoints = []
    for hp in cfg["worker_urls"]:
        host, port = hp.rsplit(":", 1)
        endpoints.append(WorkerEndpoint(host=host, port=int(port)))
    state.workers = endpoints
    state.coord = PipelineCoordinator(endpoints, cfg["model"], controller=state.controller)
    await state.coord.__aenter__()
    log.info(f"dashboard ready — BMC state={state.controller.view.state} "
             f"partition={state.controller.view.partition}")


async def on_cleanup(app: web.Application):
    state: AppState = app["state"]
    if state.coord:
        try:
            await state.coord.__aexit__(None, None, None)
        except Exception:
            pass
    if state.controller:
        try:
            await state.controller.stop()
        except Exception:
            pass
    bmc_task = app.get("bmc_task")
    if bmc_task:
        bmc_task.cancel()


def build_app(config: dict) -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["config"] = config
    app["state"] = AppState()
    app.router.add_get("/", handle_index)
    app.router.add_get("/events", handle_sse)
    app.router.add_post("/query", handle_query)
    app.router.add_post("/chaos/fail", handle_chaos_fail)
    app.router.add_post("/chaos/heal", handle_chaos_heal)
    app.router.add_post("/chaos/bmc_kill", handle_bmc_kill)
    app.router.add_post("/chaos/bmc_revive", handle_bmc_revive)
    app.router.add_get("/bmc/log", handle_bmc_log)
    app.router.add_get("/discover", handle_discover)
    app.router.add_post("/chats/new", handle_chat_new)
    app.router.add_delete("/chats/{chat_id}", handle_chat_delete)
    app.router.add_post("/chats/{chat_id}/activate", handle_chat_activate)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--workers", nargs="+", required=True)
    p.add_argument("--bmc-port", type=int, default=45555,
                   help="first BMC port. If --bmc-count > 1, replicas take "
                        "consecutive ports starting here.")
    p.add_argument("--bmc-count", type=int, default=1,
                   help="number of BMC replicas to run. >1 enables HA mode with "
                        "leader election and replica-level chaos engineering.")
    p.add_argument("--no-start-bmc", action="store_true",
                   help="don't spawn an embedded BMC simulator; connect to an existing one")
    p.add_argument("--serial-device", default=None,
                   help="path to a real Arduino BMC (e.g. /dev/tty.usbmodem14101); "
                        "overrides the simulator path when set")
    p.add_argument("--auto-serial", action="store_true",
                   help="try to auto-detect an Arduino on a USB serial port and use it")
    p.add_argument("--host", default="0.0.0.0",
                   help="bind address — default 0.0.0.0 so phones/iPads on the "
                        "same WiFi can reach the dashboard via the Mac's LAN IP")
    p.add_argument("--port", type=int, default=4100)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(name)s] %(message)s")

    serial_device = args.serial_device
    if not serial_device and args.auto_serial:
        serial_device = find_arduino_device()
        if serial_device:
            print(f"[auto] detected Arduino at {serial_device}")
        else:
            print("[auto] no Arduino detected on any serial port; falling back to simulator")

    config = {
        "model": args.model,
        "worker_urls": args.workers,
        "bmc_port": args.bmc_port,
        "bmc_count": args.bmc_count,
        "start_bmc_sim": not args.no_start_bmc and not serial_device,
        "serial_device": serial_device,
        "dashboard_port": args.port,
    }
    app = build_app(config)
    if args.bmc_count > 1:
        last = args.bmc_port + args.bmc_count - 1
        source = f"HA cluster of {args.bmc_count} simulated BMCs (ports {args.bmc_port}..{last})"
    else:
        source = f"Arduino @ {serial_device}" if serial_device else f"Simulator @ tcp://127.0.0.1:{args.bmc_port}"
    print(f"\nREVIVE BMC Dashboard → http://{args.host}:{args.port}")
    print(f"    BMC source: {source}\n")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
