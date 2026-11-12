/* ═══════════════════════════════════════════════════════
   CFD UI — Frontend logic
═══════════════════════════════════════════════════════ */

// ── Fluid & speed preset tables ───────────────────────
const FLUID_PRESETS = {
  air:    { omega: 1.8 },
  water:  { omega: 1.2 },
  oil:    { omega: 0.9 },
  custom: null,
};

const SPEED_PRESETS = {
  fpv_race:  { u_in: 0.06 },
  fpv_hover: { u_in: 0.04 },
  car:       { u_in: 0.05 },
  cycling:   { u_in: 0.025 },
  walking:   { u_in: 0.01 },
  custom:    null,
};

// ── State ─────────────────────────────────────────────
let uploadedStl = null;
let currentJobId = null;
let socket       = null;

// ── Sparkline chart helper ────────────────────────────
const MAX_POINTS = 60;

function makeSparkline(id, color) {
  const ctx = document.getElementById(id);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: Array(MAX_POINTS).fill(''),
      datasets: [{ data: Array(MAX_POINTS).fill(0), borderColor: color,
                   borderWidth: 1.5, fill: false, pointRadius: 0, tension: 0.3 }]
    },
    options: {
      responsive: false,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: { display: false, min: 0, max: 100 }
      },
      elements: { line: { borderCapStyle: 'round' } }
    }
  });
}

const gpuChart = makeSparkline('chartGpuUtil', '#4f8ef7');
const cpuChart = makeSparkline('chartCpu',     '#22c55e');

function pushPoint(chart, val) {
  if (!chart) return;
  chart.data.datasets[0].data.push(val);
  chart.data.datasets[0].data.shift();
  chart.update('none');
}

// ── System monitor polling ────────────────────────────
async function pollMonitor() {
  try {
    const d = await fetch('/api/monitor').then(r => r.json());
    setText('valCpu',  d.cpu_pct.toFixed(0));
    setText('valRam',  d.ram_pct.toFixed(0));
    pushPoint(cpuChart, d.cpu_pct);

    if (d.gpu && d.gpu.available) {
      setText('valGpuUtil', d.gpu.gpu_pct);
      setText('valVram',    d.gpu.mem_used_mb + '/' + d.gpu.mem_total_mb);
      setText('valGpuTemp', d.gpu.temp_c + '°C');
      pushPoint(gpuChart, d.gpu.gpu_pct);
    } else {
      setText('valGpuUtil', 'N/A');
    }
  } catch (_) {}
}

setInterval(pollMonitor, 1000);
pollMonitor();

// ── Drag-and-drop STL upload ──────────────────────────
const dropzone = document.getElementById('dropzone');
const stlInput = document.getElementById('stlFile');

['dragover', 'dragenter'].forEach(e =>
  dropzone.addEventListener(e, ev => { ev.preventDefault(); dropzone.classList.add('over'); }));
['dragleave', 'drop'].forEach(e =>
  dropzone.addEventListener(e, ev => { dropzone.classList.remove('over'); }));

dropzone.addEventListener('drop', ev => {
  ev.preventDefault();
  const file = ev.dataTransfer.files[0];
  if (file) uploadStl(file);
});

stlInput.addEventListener('change', () => {
  if (stlInput.files[0]) uploadStl(stlInput.files[0]);
});

dropzone.addEventListener('click', e => {
  if (e.target.tagName !== 'LABEL') stlInput.click();
});

async function uploadStl(file) {
  const form = new FormData();
  form.append('file', file);
  try {
    const res = await fetch('/api/upload-stl', { method: 'POST', body: form });
    if (!res.ok) { alert(await res.text()); return; }
    const d = await res.json();
    uploadedStl = d.filename;
    document.getElementById('stlName').textContent = '✓ ' + d.filename;
  } catch (err) {
    alert('Upload failed: ' + err);
  }
}

// ── View buttons ──────────────────────────────────────
document.getElementById('viewBtns').addEventListener('click', e => {
  if (!e.target.dataset.view) return;
  document.querySelectorAll('#viewBtns button').forEach(b => b.classList.remove('active'));
  e.target.classList.add('active');
});

// ── Grid size buttons ─────────────────────────────────
document.getElementById('gridBtns').addEventListener('click', e => {
  const btn = e.target;
  if (!btn.dataset.nx) return;
  document.querySelectorAll('#gridBtns button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('nx').value = btn.dataset.nx;
  document.getElementById('ny').value = btn.dataset.ny;
});

// ── Sliders with live value display ──────────────────
function bindSlider(id, displayId, suffix = '°') {
  const el = document.getElementById(id);
  const dv = document.getElementById(displayId);
  if (!el || !dv) return;
  el.addEventListener('input', () => { dv.textContent = el.value + suffix; updatePhysics(); });
}

bindSlider('heading', 'headingVal');
bindSlider('r3x', 'r3xV');
bindSlider('r3y', 'r3yV');
bindSlider('r3z', 'r3zV');

document.getElementById('omega').addEventListener('input', function() {
  document.getElementById('omegaVal').textContent = (+this.value).toFixed(2);
  updatePhysics();
});
document.getElementById('uin').addEventListener('input', function() {
  document.getElementById('uinVal').textContent = (+this.value).toFixed(3);
  updatePhysics();
});

// ── Preset selectors ──────────────────────────────────
document.getElementById('fluidPreset').addEventListener('change', function() {
  const p = FLUID_PRESETS[this.value];
  const box = document.getElementById('customOmega');
  if (p) {
    box.style.display = 'none';
    document.getElementById('omega').value = p.omega;
    document.getElementById('omegaVal').textContent = p.omega.toFixed(2);
  } else {
    box.style.display = 'block';
  }
  updatePhysics();
});

document.getElementById('speedPreset').addEventListener('change', function() {
  const p = SPEED_PRESETS[this.value];
  const box = document.getElementById('customUin');
  if (p) {
    box.style.display = 'none';
    document.getElementById('uin').value = p.u_in;
    document.getElementById('uinVal').textContent = p.u_in.toFixed(3);
  } else {
    box.style.display = 'block';
  }
  updatePhysics();
});

function getOmega() { return parseFloat(document.getElementById('omega').value); }
function getUin()   { return parseFloat(document.getElementById('uin').value); }

function updatePhysics() {
  const omega = getOmega();
  const u_in  = getUin();
  const tau   = 1 / omega;
  const nu    = (1/3) * (tau - 0.5);
  const cs    = Math.sqrt(1/3);
  const Ma    = u_in / cs;

  // Rough Re: assume obstacle ~30% of grid height
  const ny    = parseInt(document.getElementById('ny').value) || 512;
  const L     = Math.round(ny * 0.3);
  const Re    = (u_in * L) / nu;

  document.getElementById('physicsInfo').innerHTML =
    `ω = <span>${omega.toFixed(2)}</span>  &nbsp; τ = <span>${tau.toFixed(3)}</span><br>` +
    `ν_LBM = <span>${nu.toFixed(4)}</span><br>` +
    `Ma = <span>${Ma.toFixed(3)}</span>  ${Ma > 0.1 ? '⚠ >0.1' : '✓'}<br>` +
    `Re ≈ <span>${Math.round(Re)}</span>  (L=${L}px)`;
}

updatePhysics();

// ── Progress bars ─────────────────────────────────────
function setBar(n, pct, state) {
  const stage = document.getElementById('stage' + n);
  const bar   = document.getElementById('bar'   + n);
  bar.style.width = pct + '%';
  stage.className = 'stage ' + (state || '');
}

function setStageTime(n, s) {
  document.getElementById('time' + n).textContent = s ? s + 's' : '';
}

function resetBars() {
  for (let i = 1; i <= 4; i++) {
    setBar(i, 0, '');
    setStageTime(i, '');
  }
  document.getElementById('log').textContent = '';
}

// ── WebSocket handler ─────────────────────────────────
let _pollTimer = null;

function connectWs(jobId) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  socket = new WebSocket(`${proto}://${location.host}/ws/${jobId}`);

  socket.onmessage = ev => {
    const msg = JSON.parse(ev.data);

    if (msg.type === 'progress') {
      const n = msg.stage_n;
      if (n >= 1 && n <= 4) setBar(n, msg.pct, 'active');
      for (let i = 1; i < n; i++) setBar(i, 100, 'done');
      if (msg.log) appendLog(msg.log);
    }

    if (msg.type === 'done') {
      stopPoll();
      for (let i = 1; i <= 4; i++) setBar(i, 100, 'done');
      appendLog('[Done] Simulation complete.');
      document.getElementById('btnRun').disabled = false;
      showResults(msg.results, jobId);
    }

    if (msg.type === 'error') {
      stopPoll();
      appendLog('[ERROR] ' + msg.message, true);
      document.getElementById('btnRun').disabled = false;
    }

    // On WS connect the server sends the full job state — apply it immediately
    if (msg.type === 'state' && msg.data) {
      applyJobState(msg.data, jobId);
    }
  };

  socket.onerror = () => appendLog('[WS error] Check server console.', true);

  // Polling fallback — catches errors that arrived before WS connected
  startPoll(jobId);
}

function startPoll(jobId) {
  stopPoll();
  _pollTimer = setInterval(async () => {
    try {
      const d = await fetch(`/api/status/${jobId}`).then(r => r.json());
      applyJobState(d, jobId);
      if (d.status === 'done' || d.status === 'error') stopPoll();
    } catch (_) {}
  }, 3000);
}

function stopPoll() {
  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
}

function applyJobState(d, jobId) {
  // Restore elapsed times
  if (d.elapsed) {
    const names = ['Compiling CUDA','Generating mask','Running CUDA sim','Rendering MP4'];
    Object.entries(d.elapsed).forEach(([k, v]) => {
      const idx = names.indexOf(k) + 1;
      if (idx > 0) setStageTime(idx, v);
    });
  }
  // Restore active stage bar
  if (d.stage_n >= 1 && d.stage_n <= 4) {
    setBar(d.stage_n, d.pct || 0, d.status === 'error' ? 'error' : 'active');
    for (let i = 1; i < d.stage_n; i++) setBar(i, 100, 'done');
  }
  // Show error if job already failed
  if (d.status === 'error' && d.error) {
    appendLog('[ERROR] ' + d.error, true);
    document.getElementById('btnRun').disabled = false;
  }
  // Show completion
  if (d.status === 'done') {
    for (let i = 1; i <= 4; i++) setBar(i, 100, 'done');
    document.getElementById('btnRun').disabled = false;
    if (d.results && d.results.length) showResults(d.results, jobId);
  }
}

// ── Log helpers ───────────────────────────────────────
function appendLog(line, isErr = false) {
  // Client-side safety: drop carriage-return progress-bar lines that slipped through
  if (line.includes('\r') || /\d+%[|█]/.test(line)) return;
  const el = document.getElementById('log');
  const span = document.createElement('span');
  span.textContent = line + '\n';
  if (isErr) span.style.color = '#ef4444';
  el.appendChild(span);
  el.scrollTop = el.scrollHeight;
}

// ── Results ───────────────────────────────────────────
function showResults(paths, jobId) {
  const player  = document.getElementById('videoPlayer');
  const gallery = document.getElementById('gallery');
  gallery.innerHTML = '';

  const mp4 = paths.find(p => p.endsWith('.mp4'));
  if (mp4) {
    player.style.display = 'block';
    player.src = `/api/video/${jobId}`;
    player.load();
  }

  // Download links — backend already filters out intermediate frame PNGs
  if (paths.length === 0) {
    gallery.innerHTML = '<p style="color:var(--muted);font-size:12px">No output files.</p>';
    return;
  }
  paths.forEach(p => {
    const name = p.split(/[\\/]/).pop();
    const ext  = name.split('.').pop().toLowerCase();
    const icon = ext === 'mp4' ? '🎞' : ext === 'csv' ? '📊' : '📄';
    const row  = document.createElement('div');
    row.className = 'result-row';
    row.innerHTML = `<span>${icon} ${name}</span><a href="/api/results/${jobId}/${name}" download>⬇ Download</a>`;
    gallery.appendChild(row);
  });
}

// ── Clean old jobs ────────────────────────────────────
async function cleanJobs() {
  const btn = document.getElementById('btnClean');
  btn.disabled = true;
  btn.textContent = 'Cleaning…';
  try {
    const res = await fetch('/api/jobs', { method: 'DELETE' });
    const d   = await res.json();
    appendLog(`[Clean] Removed ${d.count} completed job(s).`);
    document.getElementById('videoPlayer').style.display = 'none';
    document.getElementById('gallery').innerHTML = '';
  } catch (err) {
    appendLog('[Clean] Error: ' + err, true);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Clean Jobs';
  }
}

// ── Start run ─────────────────────────────────────────
async function startRun() {
  if (!uploadedStl) { alert('Please upload an STL file first.'); return; }

  const view = document.querySelector('#viewBtns button.active')?.dataset.view || 'side_left';

  const body = {
    stl_filename:  uploadedStl,
    view,
    heading_deg:   parseFloat(document.getElementById('heading').value),
    rotate3d_x:    parseFloat(document.getElementById('r3x').value),
    rotate3d_y:    parseFloat(document.getElementById('r3y').value),
    rotate3d_z:    parseFloat(document.getElementById('r3z').value),
    flip_x:        document.getElementById('flipX').checked,
    flip_y:        document.getElementById('flipY').checked,
    flip_z:        document.getElementById('flipZ').checked,
    nx:            parseInt(document.getElementById('nx').value),
    ny:            parseInt(document.getElementById('ny').value),
    u_in:          getUin(),
    omega:         getOmega(),
    iterations:    parseInt(document.getElementById('iters').value),
    save_interval: parseInt(document.getElementById('saveInterval').value),
    inflow_dir:    parseInt(document.getElementById('inflowDir').value),
    margin:        0.5,
  };

  resetBars();
  document.getElementById('btnRun').disabled = true;
  document.getElementById('videoPlayer').style.display = 'none';
  document.getElementById('gallery').innerHTML = '';

  try {
    const res = await fetch('/api/run', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    if (!res.ok) { alert(await res.text()); document.getElementById('btnRun').disabled = false; return; }
    const d = await res.json();
    currentJobId = d.job_id;
    appendLog(`Job ${d.job_id} started…`);
    connectWs(d.job_id);
  } catch (err) {
    alert('Failed to start: ' + err);
    document.getElementById('btnRun').disabled = false;
  }
}

// ── Utility ───────────────────────────────────────────
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}
