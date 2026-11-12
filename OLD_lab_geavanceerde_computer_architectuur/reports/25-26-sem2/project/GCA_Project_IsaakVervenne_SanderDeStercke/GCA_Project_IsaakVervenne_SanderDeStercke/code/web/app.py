"""
CFD Web UI — FastAPI backend
Run: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import BINARY, JOBS_DIR, MAKEFILE_DIR, PROJECT_CODE_DIR, PYTHON_DIR, UPLOAD_DIR

# ---------------------------------------------------------------------------
# GPU monitoring (optional — degrades gracefully without nvidia card)
# ---------------------------------------------------------------------------
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_ok = True
except Exception:
    _nvml_ok = False


def gpu_stats() -> Dict[str, Any]:
    if not _nvml_ok:
        return {"available": False}
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util  = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem   = pynvml.nvmlDeviceGetMemoryInfo(h)
        temp  = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        name  = pynvml.nvmlDeviceGetName(h)
        return {
            "available":  True,
            "name":       name if isinstance(name, str) else name.decode(),
            "gpu_pct":    util.gpu,
            "mem_pct":    round(mem.used / mem.total * 100, 1),
            "mem_used_mb": mem.used // (1024 * 1024),
            "mem_total_mb": mem.total // (1024 * 1024),
            "temp_c":     temp,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


psutil.cpu_percent()   # first call always returns 0 — prime it at import time


def system_stats() -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "cpu_pct":    psutil.cpu_percent(interval=None),
        "ram_pct":    vm.percent,
        "ram_used_mb": vm.used // (1024 * 1024),
        "ram_total_mb": vm.total // (1024 * 1024),
        "gpu":        gpu_stats(),
    }


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------
jobs: Dict[str, Dict] = {}   # job_id -> state dict
job_ws: Dict[str, list] = {} # job_id -> list of connected WebSocket clients


def _job(job_id: str) -> Dict:
    if job_id not in jobs:
        raise HTTPException(404, f"Job {job_id!r} not found")
    return jobs[job_id]


async def _broadcast(job_id: str, msg: Dict):
    dead = []
    for ws in job_ws.get(job_id, []):
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        job_ws[job_id].remove(ws)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="CFD Web UI")

_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(str(_static / "index.html"))


# ---------------------------------------------------------------------------
# System monitor
# ---------------------------------------------------------------------------
@app.get("/api/monitor")
async def monitor():
    return system_stats()


@app.get("/api/debug")
async def debug():
    """Show resolved paths and current jobs — useful for diagnosing config issues."""
    return {
        "PROJECT_CODE_DIR": str(PROJECT_CODE_DIR),
        "PYTHON_DIR":       str(PYTHON_DIR),
        "BINARY":           str(BINARY),
        "MAKEFILE_DIR":     str(MAKEFILE_DIR),
        "UPLOAD_DIR":       str(UPLOAD_DIR),
        "JOBS_DIR":         str(JOBS_DIR),
        "binary_exists":    BINARY.exists(),
        "makefile_exists":  (MAKEFILE_DIR / "Makefile").exists(),
        "python_dir_exists": PYTHON_DIR.exists(),
        "jobs":  {jid: {"status": j["status"], "stage": j["stage"], "error": j["error"]}
                  for jid, j in jobs.items()},
    }


# ---------------------------------------------------------------------------
# STL upload
# ---------------------------------------------------------------------------
@app.post("/api/upload-stl")
async def upload_stl(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".stl"):
        raise HTTPException(400, "Only .stl files accepted")
    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": file.filename, "path": str(dest)}


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
class RunRequest(BaseModel):
    stl_filename: str
    view: str = "side_left"
    heading_deg: float = -40.0
    rotate3d_x: float = 0.0
    rotate3d_y: float = 0.0
    rotate3d_z: float = 0.0
    flip_x: bool = False
    flip_y: bool = False
    flip_z: bool = False
    nx: int = Field(1024, ge=64, le=4096)
    ny: int = Field(512, ge=32, le=2048)
    u_in: float = Field(0.06, ge=0.001, le=0.15)
    omega: float = Field(1.8, ge=0.51, le=1.99)
    iterations: int = Field(50000, ge=100, le=500000)
    save_interval: int = Field(100, ge=1)
    margin: float = Field(0.5, ge=0.0, le=2.0)
    inflow_dir: int = Field(1, ge=0, le=3)


@app.post("/api/run")
async def run_pipeline(req: RunRequest):
    stl_path = UPLOAD_DIR / req.stl_filename
    if not stl_path.exists():
        raise HTTPException(404, f"STL not found: {req.stl_filename}")

    job_id  = str(uuid.uuid4())[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir()

    jobs[job_id] = {
        "id":      job_id,
        "status":  "queued",
        "stage":   "",
        "stage_n": 0,
        "pct":     0,
        "elapsed": {},
        "error":   None,
        "results": [],
        "params":  req.model_dump(),
    }
    job_ws[job_id] = []

    asyncio.create_task(_run_job(job_id, stl_path, job_dir, req))
    return {"job_id": job_id}


# ---------------------------------------------------------------------------
# Status / results
# ---------------------------------------------------------------------------
@app.get("/api/status/{job_id}")
async def status(job_id: str):
    return _job(job_id)


@app.get("/api/results/{job_id}")
async def results(job_id: str):
    j = _job(job_id)
    return {"results": j["results"]}


@app.get("/api/video/{job_id}")
async def video(job_id: str):
    j = _job(job_id)
    mp4 = next((r for r in j["results"] if r.endswith(".mp4")), None)
    if not mp4:
        raise HTTPException(404, "No video yet")
    return FileResponse(mp4, media_type="video/mp4")


@app.get("/api/results/{job_id}/{filename}")
async def download_result(job_id: str, filename: str):
    _job(job_id)  # validate job exists
    file_path = JOBS_DIR / job_id / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(str(file_path))


@app.delete("/api/jobs")
async def cleanup_jobs():
    """Remove all completed/errored job directories and clear them from memory."""
    removed = []
    to_delete = [jid for jid, j in list(jobs.items()) if j["status"] in ("done", "error")]
    for jid in to_delete:
        job_dir = JOBS_DIR / jid
        if job_dir.exists():
            shutil.rmtree(job_dir)
        jobs.pop(jid, None)
        job_ws.pop(jid, None)
        removed.append(jid)
    return {"removed": removed, "count": len(removed)}


# ---------------------------------------------------------------------------
# WebSocket — real-time progress
# ---------------------------------------------------------------------------
@app.websocket("/ws/{job_id}")
async def ws_progress(ws: WebSocket, job_id: str):
    await ws.accept()
    job_ws.setdefault(job_id, []).append(ws)
    # Send current state immediately
    if job_id in jobs:
        await ws.send_json({"type": "state", "data": jobs[job_id]})
    try:
        while True:
            await asyncio.sleep(30)   # keep-alive ping
            await ws.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        if job_id in job_ws and ws in job_ws[job_id]:
            job_ws[job_id].remove(ws)


# ---------------------------------------------------------------------------
# Pipeline execution (runs as asyncio Task)
# ---------------------------------------------------------------------------
async def _run_job(job_id: str, stl_path: Path, job_dir: Path, req: RunRequest):
    j = jobs[job_id]

    # Give the browser time to open the WebSocket before we start broadcasting.
    await asyncio.sleep(1.0)

    async def update(stage: str, n: int, pct: int, log: str = ""):
        j["stage"]  = stage
        j["stage_n"] = n
        j["pct"]    = pct
        j["status"] = "running"
        msg = {"type": "progress", "stage": stage, "stage_n": n, "pct": pct, "log": log}
        await _broadcast(job_id, msg)

    async def run_cmd(cmd: list, cwd: Path, stage: str, stage_n: int,
                      progress_re: Optional[re.Pattern] = None):
        """Run a subprocess, stream its stdout/stderr to the WebSocket."""
        t0 = time.time()
        print(f"[{job_id}] Running: {' '.join(str(c) for c in cmd)}", flush=True)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Command not found: {cmd[0]} — {exc}") from exc
        last_lines: list = []
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            last_lines = (last_lines + [line])[-20:]   # keep last 20 lines for error context
            pct = j["pct"]
            if progress_re:
                m = progress_re.search(line)
                if m:
                    pct = int(m.group(1))
            # Suppress noisy progress-bar lines (tqdm %| and CUDA \r[====] bars)
            noisy = '\r' in line or bool(re.search(r'\d+%[\|█]', line))
            await update(stage, stage_n, pct, "" if noisy else line)
        await proc.wait()
        j["elapsed"][stage] = round(time.time() - t0, 2)
        if proc.returncode != 0:
            context = "\n".join(last_lines[-5:])
            raise RuntimeError(
                f"Stage '{stage}' exited with code {proc.returncode}\n{context}"
            )

    try:
        # ── Stage 1: Compile ──────────────────────────────────────────────
        await update("Compiling CUDA", 1, 0)
        await run_cmd(["make", "fpv_cfd_stl"], MAKEFILE_DIR, "Compiling CUDA", 1)

        # ── Stage 2: Generate mask ────────────────────────────────────────
        await update("Generating mask", 2, 0)
        stl_base = stl_path.stem
        mask_csv = job_dir / f"obstacle_mask_{stl_base}_{req.view}_hdg{int(req.heading_deg)}.csv"

        mask_cmd = [
            sys.executable, str(PYTHON_DIR / "stl_to_mask.py"),
            "--stl", str(stl_path),
            "--view", req.view,
            "--nx", str(req.nx), "--ny", str(req.ny),
            "--heading-deg", str(req.heading_deg),
            "--margin", str(req.margin),
            "--out", str(mask_csv),
        ]
        # Only pass 3D orientation flags when non-default so the command works
        # with both the updated and original versions of stl_to_mask.py.
        if abs(req.rotate3d_x) > 0.01: mask_cmd += ["--rotate3d-x", str(req.rotate3d_x)]
        if abs(req.rotate3d_y) > 0.01: mask_cmd += ["--rotate3d-y", str(req.rotate3d_y)]
        if abs(req.rotate3d_z) > 0.01: mask_cmd += ["--rotate3d-z", str(req.rotate3d_z)]
        if req.flip_x: mask_cmd.append("--flip-x")
        if req.flip_y: mask_cmd.append("--flip-y")
        if req.flip_z: mask_cmd.append("--flip-z")
        await run_cmd(mask_cmd, PROJECT_CODE_DIR, "Generating mask", 2)

        # ── Stage 3: CUDA simulation ──────────────────────────────────────
        await update("Running CUDA sim", 3, 0)
        vel_csv = job_dir / "velocity.csv"
        cuda_progress = re.compile(r"\]\s+(\d+)%")
        sim_cmd = [
            str(BINARY),
            str(mask_csv), str(vel_csv),
            str(req.nx), str(req.ny),
            str(req.u_in), str(req.iterations),
            str(req.inflow_dir), str(req.save_interval),
            str(req.omega),
        ]
        await run_cmd(sim_cmd, job_dir, "Running CUDA sim", 3, cuda_progress)

        # ── Stage 4: Render animation ─────────────────────────────────────
        await update("Rendering MP4", 4, 0)
        mp4_path = job_dir / "cfd_animation.mp4"
        anim_cmd = [
            sys.executable, str(PYTHON_DIR / "animate_cfd.py"),
            "--temp-dir", str(job_dir),
            "--out-mp4",  str(mp4_path),
            "--nx", str(req.nx), "--ny", str(req.ny),
        ]
        tqdm_re = re.compile(r"(\d+)%\|")
        await run_cmd(anim_cmd, job_dir, "Rendering MP4", 4, tqdm_re)

        # ── Done ──────────────────────────────────────────────────────────
        # Exclude intermediate animation frame PNGs (frame_NNNN.png) — only keep meaningful outputs
        j["results"] = [str(p) for p in sorted(job_dir.glob("*"))
                        if p.suffix in (".mp4", ".csv") or
                           (p.suffix == ".png" and not re.match(r"frame_\d+\.png", p.name))]
        j["status"]  = "done"
        j["pct"]     = 100
        await _broadcast(job_id, {"type": "done", "results": j["results"]})

    except Exception as exc:
        print(f"[{job_id}] ERROR: {exc}", flush=True)
        j["status"] = "error"
        j["error"]  = str(exc)
        await _broadcast(job_id, {"type": "error", "message": str(exc)})
