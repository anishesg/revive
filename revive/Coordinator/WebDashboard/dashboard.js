// REVIVE Dashboard — Real-time swarm visualization
// Polls /api/state every 1.5s, renders canvas topology, agent cards, and chat.

const API = '';  // same origin

// ── State ─────────────────────────────────────────────────────────────────────
let state = {
  workers: [],         // [{id, role, model, host, port, status, tps, weight, color}]
  chatMessages: [],    // [{role, content, color, model, tps}]
  mode: 'swarm',
  isQuerying: false,
  totalQueries: 0,
};

let nodes = [];        // canvas node positions, derived from workers
let particles = [];    // flying-token particles during generation

// ── Canvas ────────────────────────────────────────────────────────────────────
const canvas = document.getElementById('swarm-canvas');
const ctx = canvas.getContext('2d');

function resizeCanvas() {
  canvas.width = canvas.offsetWidth * devicePixelRatio;
  canvas.height = canvas.offsetHeight * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
}
window.addEventListener('resize', () => { resizeCanvas(); layoutNodes(); });
resizeCanvas();

function layoutNodes() {
  const W = canvas.offsetWidth;
  const H = canvas.offsetHeight;
  const n = state.workers.length;
  if (n === 0) { nodes = []; return; }

  const cx = W / 2, cy = H / 2;
  const r = Math.min(cx, cy) * 0.68;

  nodes = state.workers.map((w, i) => {
    const angle = (2 * Math.PI * i / n) - Math.PI / 2;
    return {
      x: cx + r * Math.cos(angle),
      y: cy + r * Math.sin(angle),
      worker: w,
      pulseT: 0,
    };
  });

  // iPad coordinator always at center
  nodes.unshift({
    x: cx, y: cy,
    worker: { role: 'coordinator', model: 'Aggregator', status: state.isQuerying ? 'generating' : 'idle',
              tps: 0, color: '#4a90d9', displayName: 'iPad' },
    pulseT: 0,
    isCoord: true,
  });
}

function spawnParticles(from, to) {
  for (let i = 0; i < 6; i++) {
    particles.push({
      x: from.x, y: from.y,
      tx: to.x, ty: to.y,
      t: Math.random() * 0.3,  // stagger
      speed: 0.008 + Math.random() * 0.006,
      color: from.worker.color || '#00ff88',
      size: 2 + Math.random() * 2,
    });
  }
}

// ── Render Loop ───────────────────────────────────────────────────────────────
let lastFrame = 0;
function render(ts) {
  const dt = Math.min((ts - lastFrame) / 1000, 0.05);
  lastFrame = ts;

  const W = canvas.offsetWidth;
  const H = canvas.offsetHeight;
  ctx.clearRect(0, 0, W, H);

  const coordNode = nodes.find(n => n.isCoord);

  // Draw edges (coord ↔ each worker)
  if (coordNode) {
    for (const node of nodes) {
      if (node.isCoord) continue;
      ctx.beginPath();
      ctx.moveTo(coordNode.x, coordNode.y);
      ctx.lineTo(node.x, node.y);
      const active = node.worker.status === 'generating';
      ctx.strokeStyle = active ? (node.worker.color || '#00ff88') + '44' : '#222';
      ctx.lineWidth = active ? 1.5 : 0.5;
      ctx.stroke();
    }
  }

  // Update and draw particles
  particles = particles.filter(p => p.t <= 1);
  for (const p of particles) {
    p.t += p.speed;
    const t = Math.min(p.t, 1);
    const x = p.x + (p.tx - p.x) * t;
    const y = p.y + (p.ty - p.y) * t;
    ctx.beginPath();
    ctx.arc(x, y, p.size, 0, Math.PI * 2);
    ctx.fillStyle = p.color + Math.floor((1 - t) * 255).toString(16).padStart(2, '0');
    ctx.fill();
  }

  // Draw nodes
  for (const node of nodes) {
    const { x, y, worker, isCoord } = node;
    const color = worker.color || '#4a90d9';
    const r = isCoord ? 22 : 16;
    const active = worker.status === 'generating';

    // Pulse ring
    if (active) {
      node.pulseT = (node.pulseT || 0) + dt * 2;
      const pulseR = r + 8 * Math.abs(Math.sin(node.pulseT));
      ctx.beginPath();
      ctx.arc(x, y, pulseR, 0, Math.PI * 2);
      ctx.strokeStyle = color + '55';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Glow
    const grd = ctx.createRadialGradient(x, y, 0, x, y, r * 2);
    grd.addColorStop(0, color + (active ? '33' : '11'));
    grd.addColorStop(1, 'transparent');
    ctx.beginPath();
    ctx.arc(x, y, r * 2, 0, Math.PI * 2);
    ctx.fillStyle = grd;
    ctx.fill();

    // Circle
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = '#111';
    ctx.fill();
    ctx.strokeStyle = active ? color : (color + '88');
    ctx.lineWidth = active ? 2 : 1;
    ctx.stroke();

    // Label
    const label = isCoord ? 'COORD' : (worker.role || '').slice(0, 6).toUpperCase();
    ctx.fillStyle = active ? color : '#888';
    ctx.font = `bold ${isCoord ? 8 : 7}px "SF Mono", monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, x, y - 1);

    if (!isCoord && worker.tps > 0) {
      ctx.fillStyle = '#555';
      ctx.font = '6px "SF Mono", monospace';
      ctx.fillText(`${worker.tps.toFixed(1)}t/s`, x, y + 8);
    }
  }

  // Spawn particles when querying
  if (state.isQuerying && coordNode && Math.random() < 0.15) {
    const workerNodes = nodes.filter(n => !n.isCoord && n.worker.status === 'generating');
    if (workerNodes.length > 0) {
      const rnd = workerNodes[Math.floor(Math.random() * workerNodes.length)];
      spawnParticles(rnd, coordNode);
    }
  }

  requestAnimationFrame(render);
}
requestAnimationFrame(render);

// ── State polling ─────────────────────────────────────────────────────────────
async function pollState() {
  try {
    const res = await fetch(`${API}/api/state`);
    if (!res.ok) return;
    const data = await res.json();

    const prevQueryCount = state.totalQueries;
    state.workers = data.workers || [];
    state.chatMessages = data.messages || [];
    state.mode = data.mode || 'swarm';
    state.isQuerying = data.isQuerying || false;
    state.totalQueries = data.totalQueries || 0;

    // Re-layout nodes when worker count changes
    layoutNodes();

    // Update stats bar
    const totalTPS = state.workers.reduce((s, w) => s + (w.tps || 0), 0);
    document.getElementById('stat-nodes').textContent = `${state.workers.length} nodes`;
    document.getElementById('stat-tps').textContent = `${totalTPS.toFixed(0)} tok/s`;
    document.getElementById('stat-queries').textContent = `${state.totalQueries} queries`;

    renderAgentCards();
    renderChat();
  } catch (_) {
    // Server not reachable yet
  }
}

setInterval(pollState, 1500);
pollState();

// ── Agent cards ───────────────────────────────────────────────────────────────
function renderAgentCards() {
  const grid = document.getElementById('agents-grid');
  grid.innerHTML = '';

  for (const w of state.workers) {
    const card = document.createElement('div');
    card.className = 'agent-card' + (w.status === 'generating' ? ' active' : '');
    card.style.borderColor = w.status === 'generating' ? w.color : '#222';

    const pct = Math.min((w.tps || 0) / 40, 1) * 100;

    card.innerHTML = `
      <div class="agent-role" style="color:${w.color || '#888'}">${escHtml(w.role?.toUpperCase() || 'WORKER')}</div>
      <div class="agent-model">${escHtml(w.model || '')}</div>
      <div class="agent-status">
        <div class="status-dot ${w.status || 'idle'}"></div>
        <span style="font-size:9px;color:#666">${escHtml(w.status || 'idle')}</span>
      </div>
      <div class="agent-tps">${(w.tps || 0).toFixed(1)}</div>
      <div class="agent-meta">tok/s · ${escHtml(w.host || '')}:${w.port || ''}</div>
      <div class="progress-bar"><div class="progress-fill" style="width:${pct}%;background:${w.color || '#00ff88'}"></div></div>
    `;
    grid.appendChild(card);
  }

  if (state.workers.length === 0) {
    grid.innerHTML = '<div style="color:#444;font-size:11px;padding:12px">No workers connected. Launch REVIVE in Worker mode on each phone and connect to the same WiFi.</div>';
  }
}

// ── Chat ──────────────────────────────────────────────────────────────────────
let renderedCount = 0;

function renderChat() {
  const log = document.getElementById('chat-log');
  const messages = state.chatMessages;

  // Only append new messages
  for (let i = renderedCount; i < messages.length; i++) {
    const msg = messages[i];
    const el = document.createElement('div');
    el.className = 'chat-message';

    const isUser  = msg.role === 'user';
    const isSwarm = msg.role === 'swarm';
    const roleLabel = isUser ? 'YOU' : isSwarm ? 'SWARM · MoA' : (msg.role || '').toUpperCase();
    const color = isUser ? '#4a90d9' : (msg.color || '#888');
    const bubbleCls = isUser ? 'user' : isSwarm ? 'assistant' : 'agent';

    const meta = msg.tps ? `${msg.tps.toFixed(1)} tok/s` : '';
    const model = msg.model ? `· ${msg.model}` : '';

    el.innerHTML = `
      <div class="chat-role" style="color:${color}">${escHtml(roleLabel)} <span style="color:#333">${escHtml(model)}</span></div>
      <div class="chat-content ${bubbleCls}" style="${bubbleCls === 'agent' ? `border-color:${color}` : ''}">${escHtml(msg.content || '')}</div>
      <div class="chat-meta">${escHtml(meta)}</div>
    `;
    log.appendChild(el);
    renderedCount++;
  }

  // Auto-scroll
  log.scrollTop = log.scrollHeight;
}

// ── Query submission ──────────────────────────────────────────────────────────
async function sendQuery() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text || state.isQuerying) return;
  input.value = '';

  try {
    await fetch(`${API}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: text, mode: state.mode }),
    });
  } catch (_) {
    // If fetch fails, the server may not be running yet
  }
}

// ── Mode toggle ───────────────────────────────────────────────────────────────
function setMode(m) {
  state.mode = m;
  document.getElementById('btn-swarm').classList.toggle('active', m === 'swarm');
  document.getElementById('btn-speed').classList.toggle('active', m === 'speed');
  fetch(`${API}/api/mode`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode: m }),
  }).catch(() => {});
}

// ── Impact slider ─────────────────────────────────────────────────────────────
function updateImpact(val) {
  const pct = parseInt(val);
  document.getElementById('pct-val').textContent = pct;

  const totalPhones = 5300;  // millions
  const nodes = Math.round(totalPhones * pct / 100);
  const gpusDisplaced = Math.round(nodes / 5);
  const powerSavedGW = (nodes * 3 / 1000).toFixed(1);  // 3W per phone avg

  document.getElementById('impact-result').textContent =
    `${nodes}M nodes · ${gpusDisplaced}M GPUs displaced · ${powerSavedGW} GW saved`;
}

// Update power/savings from live state
function updateImpactLive() {
  const workerCount = state.workers.length;
  const wattageEstimate = workerCount * 3;  // ~3W per phone inference
  document.getElementById('impact-power').textContent = `${wattageEstimate}W`;

  const pctSaved = workerCount > 0 ? Math.round((1 - wattageEstimate / 300) * 100) : 0;
  document.getElementById('impact-saving').textContent = `${Math.max(pctSaved, 0)}%`;
}

setInterval(updateImpactLive, 3000);

// ── Utilities ─────────────────────────────────────────────────────────────────
function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
