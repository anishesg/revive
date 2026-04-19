#!/usr/bin/env python3
"""
REVIVE Swarm Dashboard
Run: python3 server.py
Open: http://localhost:4000
"""
import asyncio, json, time, sys, os, socket
from aiohttp import web, ClientSession, ClientTimeout

# ── mDNS discovery (optional, zeroconf) ──────────────────────────────────────
try:
    from zeroconf import ServiceBrowser, Zeroconf
    from zeroconf.asyncio import AsyncZeroconf
    HAS_ZEROCONF = True
except ImportError:
    HAS_ZEROCONF = False

workers: dict[str, dict] = {}
messages: list[dict] = []
sse_clients: list[web.StreamResponse] = []
total_tokens = 0

# ── Discovery ─────────────────────────────────────────────────────────────────

class ReviveListener:
    def add_service(self, zc, type_, name):
        info = zc.get_service_info(type_, name)
        if not info: return
        host = socket.inet_ntoa(info.addresses[0]) if info.addresses else None
        if not host: return
        port = info.port
        props = {k.decode(): v.decode() for k, v in info.properties.items()}
        role  = props.get('role', 'worker')
        model = props.get('model', 'unknown')
        ram   = int(props.get('ram', 0))
        wname = name.replace('._revive._tcp.local.', '').replace('._revive._tcp.', '')
        if wname not in workers:
            print(f"[discovery] {role} joined @ {host}:{port}")
            workers[wname] = dict(name=wname, host=host, port=port, role=role,
                                  model=model, ram_mb=ram, tps=0, status='idle', tokens=0)
            broadcast_state()

    def remove_service(self, zc, type_, name):
        wname = name.replace('._revive._tcp.local.', '').replace('._revive._tcp.', '')
        if wname in workers:
            del workers[wname]
            broadcast_state()

    def update_service(self, zc, type_, name): pass

async def start_discovery():
    if not HAS_ZEROCONF:
        print("[discovery] zeroconf not available — use manual add")
        return
    azc = AsyncZeroconf()
    ServiceBrowser(azc.zeroconf, "_revive._tcp.local.", ReviveListener())
    print("[discovery] Browsing _revive._tcp …")

# ── Health polling ────────────────────────────────────────────────────────────

async def health_loop():
    async with ClientSession() as session:
        while True:
            for w in list(workers.values()):
                try:
                    async with session.get(
                        f"http://{w['host']}:{w['port']}/health",
                        timeout=ClientTimeout(total=2)
                    ) as r:
                        w['status'] = 'idle' if w['status'] != 'generating' else 'generating'
                except Exception:
                    w['status'] = 'offline'
            broadcast_state()
            await asyncio.sleep(5)

# ── Inference ─────────────────────────────────────────────────────────────────

async def query_worker(session, worker, prompt):
    url = f"http://{worker['host']}:{worker['port']}/v1/chat/completions"
    t0 = time.time()
    worker['status'] = 'generating'
    broadcast_state()
    try:
        async with session.post(url, json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200, "temperature": 0.7,
        }, timeout=ClientTimeout(total=45)) as resp:
            data = await resp.json(content_type=None)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            # no cleaning — show raw output
            metrics = data.get("metrics", {})
            tps    = metrics.get("tokens_per_second", 0)
            tokens = metrics.get("tokens_generated", 0)
            worker.update(tps=round(tps, 1), status='idle',
                          tokens=worker.get('tokens', 0) + tokens)
            return dict(role=worker['role'], model=worker['model'], content=content,
                        tps=tps, tokens=tokens,
                        latency_ms=round((time.time()-t0)*1000),
                        worker=worker['name'])
    except Exception as e:
        worker['status'] = 'offline'
        return dict(role=worker['role'], model=worker['model'],
                    content=f"[offline: {e}]", tps=0, tokens=0,
                    latency_ms=0, worker=worker['name'])

async def run_query(query: str):
    global total_tokens
    import uuid
    qid = str(uuid.uuid4())[:8]
    live = [w for w in workers.values() if w['status'] != 'offline']
    if not live:
        messages.append(dict(role='system', content='No workers online.', ts=time.time(), qid=qid))
        broadcast_state(); return

    messages.append(dict(role='user', content=query, ts=time.time(), qid=qid))
    broadcast_state()

    async with ClientSession() as session:
        results = await asyncio.gather(*[query_worker(session, w, query) for w in live])

    for r in results:
        total_tokens += r['tokens']
        messages.append(dict(ts=time.time(), qid=qid, **r))

    broadcast_state()

# ── SSE ───────────────────────────────────────────────────────────────────────

def build_state():
    live = list(workers.values())
    return dict(workers=live, messages=messages[-200:],
                total_tokens=total_tokens,
                collective_tps=round(sum(w.get('tps',0) for w in live if w['status']!='offline'), 1),
                worker_count=len([w for w in live if w['status']!='offline']))

def broadcast_state():
    payload = ("data: " + json.dumps(build_state()) + "\n\n").encode()
    asyncio.ensure_future(_broadcast(payload))

async def _broadcast(payload: bytes):
    dead = []
    for resp in list(sse_clients):
        try:
            await resp.write(payload)
        except Exception:
            dead.append(resp)
    for resp in dead:
        if resp in sse_clients:
            sse_clients.remove(resp)

async def handle_sse(request):
    resp = web.StreamResponse(headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Access-Control-Allow-Origin": "*",
    })
    await resp.prepare(request)
    sse_clients.append(resp)
    await resp.write(("data: " + json.dumps(build_state()) + "\n\n").encode())
    try:
        while True:
            await asyncio.sleep(15)
            await resp.write(b": ping\n\n")
    except Exception: pass
    finally:
        if resp in sse_clients: sse_clients.remove(resp)
    return resp

# ── HTTP handlers ─────────────────────────────────────────────────────────────

async def handle_query(request):
    data = await request.json()
    asyncio.ensure_future(run_query(data.get("query", "")))
    return web.json_response({"ok": True})

async def handle_add_worker(request):
    d = await request.json()
    name = f"{d.get('role','worker')}-{d['host']}:{d.get('port',50001)}"
    workers[name] = dict(name=name, host=d['host'], port=int(d.get('port', 50001)),
                         role=d.get('role','worker'), model=d.get('model','unknown'),
                         ram_mb=0, tps=0, status='idle', tokens=0)
    broadcast_state()
    return web.json_response({"ok": True})

async def handle_index(request):
    return web.FileResponse(os.path.join(os.path.dirname(__file__), 'index.html'))

async def on_startup(app):
    asyncio.ensure_future(start_discovery())
    asyncio.ensure_future(health_loop())

def main():
    app = web.Application()
    app.on_startup.append(on_startup)
    app.router.add_get('/', handle_index)
    app.router.add_get('/events', handle_sse)
    app.router.add_post('/query', handle_query)
    app.router.add_post('/workers/add', handle_add_worker)
    print("REVIVE Dashboard → http://localhost:4000")
    web.run_app(app, port=4000, print=None)

if __name__ == '__main__':
    main()
