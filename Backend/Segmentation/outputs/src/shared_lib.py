"""Helpers shared by the unified notebooks: worker launch and live monitoring, progress
reporting from the workers, JSON files, and the Holm correction.

Each notebook writes this file to /kaggle/working/src; training runs in one subprocess per GPU
and the notebook turns the workers' status files into live progress bars.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np


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
    import subprocess

    processes = []
    for gpu in range(gpu_count):
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=str(gpu),
            PYTHONUNBUFFERED="1",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            OMP_NUM_THREADS=str(max(1, (os.cpu_count() or 2) // gpu_count)),
        )
        log = open(Path(log_dir) / f"worker_gpu{gpu}.log", "a")
        command = [sys.executable, str(script), "--config", str(config_path), "--gpu", str(gpu)]
        processes.append(
            (subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log)
        )
    return processes


def monitor_workers(processes, log_dir, total_jobs, finished_jobs, poll_seconds=3):
    """Live progress: one bar for all jobs, one bar per GPU, epoch lines as they arrive."""
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
                with open(events) as handle:
                    handle.seek(offsets[gpu])
                    for line in handle:
                        tqdm.write(json.loads(line)["message"])
                    offsets[gpu] = handle.tell()
        done = finished_jobs()
        overall.update(done - overall.n)
        if all(process.poll() is not None for process, _ in processes):
            break
        time.sleep(poll_seconds)
    for bar in bars:
        bar.close()
    overall.close()
    failures = []
    for gpu, (process, log) in enumerate(processes):
        log.close()
        if process.returncode != 0:
            tail = (log_dir / f"worker_gpu{gpu}.log").read_text().splitlines()[-40:]
            failures.append(
                f"GPU {gpu} worker exited with {process.returncode}:\n" + "\n".join(tail)
            )
    if failures:
        raise RuntimeError("\n\n".join(failures))


class Reporter:
    """Status file (polled by the notebook) and epoch events written by a worker."""

    def __init__(self, log_dir, gpu):
        self.status_path = Path(log_dir) / f"status_gpu{gpu}.json"
        self.events_path = Path(log_dir) / f"events_gpu{gpu}.jsonl"
        self.gpu = gpu

    def status(self, **fields):
        write_json(self.status_path, fields)

    def event(self, message, **fields):
        with open(self.events_path, "a") as handle:
            handle.write(
                json.dumps(
                    {"time": time.time(), "message": f"[GPU {self.gpu}] {message}", **fields}
                )
                + "\n"
            )
        print(message, flush=True)


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
