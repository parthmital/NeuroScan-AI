"""Helpers shared by the unified notebooks: worker launch and live monitoring, progress
reporting from the workers, JSON files, and the Holm correction.

Each notebook writes this file to /kaggle/working/src; training runs in one subprocess per GPU
and the notebook turns the workers' status files into live progress bars.
"""

import faulthandler
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

STALL_SECONDS = 1800  # no new status or event for this long counts as a possible hang


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=float))
    os.replace(tmp, path)


def launch_workers(script, config_path, log_dir, gpu_count):
    processes = []
    for gpu in range(gpu_count):
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=str(gpu),
            PYTHONUNBUFFERED="1",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            OMP_NUM_THREADS=str(max(1, (os.cpu_count() or 2) // gpu_count)),
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        )
        log = open(Path(log_dir) / f"worker_gpu{gpu}.log", "a")
        command = [sys.executable, str(script), "--config", str(config_path), "--gpu", str(gpu)]
        processes.append(
            (subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log)
        )
    return processes


def log_tail(log_dir, gpu, lines=40):
    path = Path(log_dir) / f"worker_gpu{gpu}.log"
    return (
        "\n".join(path.read_text(errors="replace").splitlines()[-lines:]) if path.exists() else ""
    )


def stop_workers(processes, timeout=60):
    for process, _ in processes:
        if process.poll() is None:
            process.terminate()
    for process, _ in processes:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout)


def monitor_workers(
    processes, log_dir, total_jobs, finished_jobs, poll_seconds=3, heartbeat_seconds=300
):
    """Live progress: one bar for all jobs, one bar per GPU, epoch lines as they arrive, and a status
    line per GPU every `heartbeat_seconds` (the bars are widgets, which the log of a saved Kaggle run
    does not show). Warns when a worker reports nothing for STALL_SECONDS, and as soon as a worker
    exits with an error stops the others and raises with the end of its log."""
    from tqdm.auto import tqdm

    log_dir = Path(log_dir)
    overall = tqdm(total=total_jobs, desc="Jobs", unit="job", position=0)
    overall.update(finished_jobs())
    bars = [
        tqdm(total=1, desc=f"GPU {gpu}: waiting", unit="step", position=gpu + 1, leave=True)
        for gpu in range(len(processes))
    ]
    offsets = [0] * len(processes)
    current_job = [None] * len(processes)
    last_heartbeat = time.time()
    last_change = [time.time()] * len(processes)
    stamps = [None] * len(processes)
    while True:
        for gpu, bar in enumerate(bars):
            status = read_json(log_dir / f"status_gpu{gpu}.json", {})
            if status.get("job"):
                if status["job"] != current_job[gpu]:
                    current_job[gpu] = status["job"]
                    bar.reset(total=max(1, status.get("total", 1)))
                    bar.set_description(f"GPU {gpu}: {status['job']}")
                bar.n = min(status.get("step", 0), bar.total)
                bar.set_postfix(status.get("postfix", {}), refresh=False)
                bar.refresh()
            events = log_dir / f"events_gpu{gpu}.jsonl"
            if events.exists():
                with open(events, "rb") as handle:
                    handle.seek(offsets[gpu])
                    data = handle.read()
                complete = data[: data.rfind(b"\n") + 1]  # a line still being written waits
                for line in complete.splitlines():
                    tqdm.write(json.loads(line)["message"])
                offsets[gpu] += len(complete)
        for gpu, (process, _) in enumerate(processes):
            stamp = tuple(
                path.stat().st_mtime if path.exists() else 0
                for path in (log_dir / f"status_gpu{gpu}.json", log_dir / f"events_gpu{gpu}.jsonl")
            )
            if stamp != stamps[gpu] or process.poll() is not None:
                stamps[gpu], last_change[gpu] = stamp, time.time()
            elif time.time() - last_change[gpu] >= STALL_SECONDS:
                last_change[gpu] = time.time()
                tqdm.write(
                    f"WARNING GPU {gpu} worker: no progress for {STALL_SECONDS // 60} min; end of its log:\n{log_tail(log_dir, gpu)}"
                )
        if time.time() - last_heartbeat >= heartbeat_seconds:
            last_heartbeat = time.time()
            for gpu in range(len(processes)):
                status = read_json(log_dir / f"status_gpu{gpu}.json", {})
                if status.get("job"):
                    tqdm.write(
                        f"[GPU {gpu}] {status['job']}: step {status.get('step', 0)}/{status.get('total', '?')} {status.get('postfix', {})}"
                    )
        done = finished_jobs()
        overall.update(done - overall.n)
        failed = [
            gpu for gpu, (process, _) in enumerate(processes) if process.poll() not in (None, 0)
        ]
        if failed or all(process.poll() is not None for process, _ in processes):
            break
        time.sleep(poll_seconds)
    for bar in bars:
        bar.close()
    overall.close()
    if failed:
        tqdm.write(f"GPU {failed} worker failed; stopping the other workers")
        stop_workers(processes)
    for _, log in processes:
        log.close()
    if failed:
        raise RuntimeError(
            "\n\n".join(
                f"GPU {gpu} worker exited with {processes[gpu][0].returncode}:\n{log_tail(log_dir, gpu)}"
                for gpu in failed
            )
            + "\n\nFinished runs and per-epoch checkpoints are in /kaggle/working; set RESUME_DIR to this version's output to continue."
        )


class Reporter:
    """Status file (polled by the notebook) and epoch events written by a worker."""

    def __init__(self, log_dir, gpu):
        self.status_path = Path(log_dir) / f"status_gpu{gpu}.json"
        self.events_path = Path(log_dir) / f"events_gpu{gpu}.jsonl"
        self.gpu = gpu

    def status(self, **fields):
        write_json(self.status_path, fields)
        faulthandler.dump_traceback_later(STALL_SECONDS, repeat=True)  # re-armed by every report

    def event(self, message, **fields):
        with open(self.events_path, "a") as handle:
            handle.write(
                json.dumps(
                    {"time": time.time(), "message": f"[GPU {self.gpu}] {message}", **fields}
                )
                + "\n"
            )
        print(message, flush=True)
        faulthandler.dump_traceback_later(STALL_SECONDS, repeat=True)


def holm(p_values):
    """Holm-Bonferroni adjusted p-values, same order as the input."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted
