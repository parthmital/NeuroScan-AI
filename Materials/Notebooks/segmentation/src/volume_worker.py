"""Training worker for the hybrid-ensemble segmentation notebook: one process per GPU.

Each worker claims (model, seed) jobs and trains on random native-resolution patches with
checkpoints every epoch (so a later session resumes from the last epoch), keeps the epoch with
the best sliding-window validation Dice, then predicts every validation, internal test and
external case with test-time augmentation, stores the probabilities, tunes the enhancing-tumour
size threshold on the validation cases, scores the test and external cases, measures how much
each MRI sequence matters (sequence-drop ablation) and times inference.
"""

import argparse
import faulthandler
import gc
import itertools
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shared_lib  # noqa: E402
import volume_lib as lib  # noqa: E402


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def probe_micro_batch(model, start, size, weights, device, headroom=0.9):
    """Largest micro-batch whose forward and backward pass peak below `headroom` of the GPU memory
    (the rest is left for augmentation, fragmentation and evaluation); nothing is stepped."""
    model.train()
    capacity = torch.cuda.get_device_properties(device).total_memory
    candidate = start
    while candidate >= 1:
        torch.cuda.reset_peak_memory_stats(device)
        try:
            x = torch.randn(candidate, 4, size, size, size, device=device)
            y = torch.zeros(candidate, 3, size, size, size, device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lib.supervised_loss(model(x), y, weights)
            loss.backward()
            torch.cuda.synchronize(device)
            if candidate == 1 or torch.cuda.max_memory_allocated(device) <= headroom * capacity:
                return candidate
        except torch.OutOfMemoryError:
            pass
        finally:
            x = y = loss = None
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        candidate //= 2
    raise RuntimeError("Even a micro-batch of 1 does not fit in GPU memory")


def case_input(cfg, case, device):
    return torch.from_numpy(np.asarray(lib.load_image(cfg["cache_dir"], case), dtype=np.float32))[
        None
    ].to(device)


@torch.no_grad()
def validate(model, val, cfg, device):
    """Mean Dice per region over the validation cases: whole cropped volumes, sliding windows, no TTA."""
    rows = []
    for case in val["case"]:
        probs, _ = lib.sliding_probs(
            model, case_input(cfg, case, device), cfg["patch"], cfg["sw_overlap"], False
        )
        pred = lib.regions_from_probs(probs[0])
        ref = torch.from_numpy(lib.load_regions(cfg["cache_dir"], case)).to(device)
        rows.append([lib.dice(pred[i], ref[i]) for i in range(3)])
    return np.array(rows).mean(axis=0)


def predict_split(model, frame, cfg, device, out_dir, reporter, label):
    """Stores the TTA probabilities of every case of a split (cases already stored are skipped)."""
    out_dir = Path(out_dir)
    start = time.perf_counter()
    for position, case in enumerate(frame["case"], start=1):
        path = out_dir / f"{case}.npz"
        if not path.exists():
            mean, std = lib.sliding_probs(
                model, case_input(cfg, case, device), cfg["patch"], cfg["sw_overlap"], cfg["tta"]
            )
            wt = mean[0, 0] > 0.5
            spread = float(std[0, 0][wt].mean()) if wt.any() else 0.0
            lib.save_probs(path, mean[0], spread)
        reporter.status(job=label, step=position, total=len(frame), postfix={"split": out_dir.name})
    reporter.event(
        f"{label}: {len(frame)} {out_dir.name} cases predicted in {time.perf_counter() - start:.0f}s"
    )


def tune_et_threshold(frame, cfg, prob_dir):
    """Validation ET Dice for each candidate ET size threshold."""
    scores = {threshold: [] for threshold in cfg["et_candidates"]}
    for case in frame["case"]:
        probs = torch.from_numpy(lib.load_probs(Path(prob_dir) / f"{case}.npz")[0]).float() / 255
        ref_et = torch.from_numpy(lib.load_regions(cfg["cache_dir"], case)[2])
        for threshold in scores:
            scores[threshold].append(lib.dice(lib.regions_from_probs(probs, threshold)[2], ref_et))
    return {threshold: float(np.mean(values)) for threshold, values in scores.items()}


def score_split(frame, cfg, prob_dir, et_min):
    """Per-case metrics of a split from its stored probabilities."""
    rows = []
    for case in frame.itertuples():
        stored, spread = lib.load_probs(Path(prob_dir) / f"{case.case}.npz")
        pred = lib.regions_from_probs(torch.from_numpy(stored).float() / 255, et_min).numpy()
        ref = lib.load_regions(cfg["cache_dir"], case.case)
        spacing = tuple(float(v) for v in case.spacing.split("x"))
        metrics = lib.case_metrics(pred, ref, spacing, int(case.native_voxels))
        rows.append(
            {
                "case": case.case,
                "dataset": case.dataset,
                "grade": case.grade,
                **metrics,
                "tta_std_in_wt": spread,
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def modality_ablation(model, frame, cfg, device, et_min, reporter, label):
    """Dice per region of every test case with all sequences and with each sequence set to zero
    (the mean of its z-scored brain voxels), without TTA."""
    rows = []
    for position, case in enumerate(frame["case"], start=1):
        reporter.status(
            job=label, step=position, total=len(frame), postfix={"stage": "sequence drop"}
        )
        x = case_input(cfg, case, device)
        ref = torch.from_numpy(lib.load_regions(cfg["cache_dir"], case)).to(device)
        for dropped in ("none", *lib.MODALITIES):
            inputs = x.clone()
            if dropped != "none":
                inputs[:, lib.MODALITIES.index(dropped)] = 0
            probs, _ = lib.sliding_probs(model, inputs, cfg["patch"], cfg["sw_overlap"], False)
            pred = lib.regions_from_probs(probs[0], et_min)
            rows.append(
                {
                    "case": case,
                    "dropped": dropped,
                    **{f"dice_{r}": lib.dice(pred[i], ref[i]) for i, r in enumerate(lib.REGIONS)},
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def time_inference(model, shape, cfg, device, tta, iterations):
    x = torch.randn(1, 4, *shape, device=device)
    lib.sliding_probs(model, x, cfg["patch"], cfg["sw_overlap"], tta)
    torch.cuda.synchronize(device)
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        lib.sliding_probs(model, x, cfg["patch"], cfg["sw_overlap"], tta)
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)
    return float(np.mean(timings)), float(np.std(timings))


def run_job(job, cfg, cases, device, reporter, max_micro):
    name, seed = job["model"], job["seed"]
    rid = lib.run_id(name, seed)
    recipe = lib.RECIPES[name]
    patch, epochs, iters = cfg["patch"], cfg["epochs"][name], cfg["iters_per_epoch"]
    seed_everything(seed, cfg["deterministic"])
    run_dir = Path(cfg["runs_dir"]) / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(cfg["weights_dir"]) / lib.slug(name)
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = weights_dir / f"{rid}.pth", weights_dir / f"{rid}_last.pth"

    model = lib.build_model(name).to(device)
    weights = recipe["ds_weights"]
    micro = probe_micro_batch(model, max_micro, patch, weights, device)
    accumulation = math.ceil(cfg["batch"] / micro)
    samples_per_epoch = iters * accumulation * micro
    train = cases[cases["role"] == "train"]
    val = cases[cases["role"] == "val"]
    dataset = lib.PatchDataset(
        cfg["cache_dir"],
        train["case"],
        cfg["fg_path"],
        cfg["source"],
        patch,
        cfg["fg_probability"],
        seed,
    )
    loader_kwargs = {"num_workers": cfg["loader_workers"], "pin_memory": True}
    if cfg["loader_workers"] > 0:
        loader_kwargs["prefetch_factor"] = 4
    total_steps = epochs * iters
    warmup_steps = recipe["warmup_epochs"] * iters
    if recipe["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=recipe["lr"],
            momentum=0.99,
            nesterov=True,
            weight_decay=recipe["weight_decay"],
        )

        def lr_factor(step):
            return (1 - min(step, total_steps - 1) / total_steps) ** 0.9

    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"]
        )

        def lr_factor(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda")
    state = {
        "epoch": 0,
        "step": 0,
        "history": [],
        "best_dice": -1.0,
        "best_epoch": 0,
        "train_seconds": 0.0,
        "peak_memory_gb": 0.0,
    }
    if last_path.exists():  # resume from the end of the last finished epoch
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        state = checkpoint["state"]
        torch.set_rng_state(checkpoint["cpu_rng"].cpu())
        torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu(), device)
        reporter.event(f"{rid}: resumed after epoch {state['epoch']}")
    reporter.event(
        f"{rid}: {len(train)} train / {len(val)} val cases, micro-batch {micro} x {accumulation} accumulation, "
        f"{total_steps} steps over {epochs} epochs of {samples_per_epoch} patches"
    )

    torch.cuda.reset_peak_memory_stats(device)
    start_epoch = state["epoch"]
    if state["epoch"] < epochs:  # one loader for the remaining epochs: its workers start once
        sampler = range(state["epoch"] * samples_per_epoch, epochs * samples_per_epoch)
        batches = iter(
            torch.utils.data.DataLoader(
                dataset, batch_size=micro, sampler=sampler, drop_last=True, **loader_kwargs
            )
        )
    while state["epoch"] < epochs:
        # The next epoch and the evaluation of the member must both end before the deadline.
        needed = state.get("eval_estimate", 0.0) + (
            state["history"][-1]["epoch_seconds"] if state["history"] else 0.0
        )
        if time.time() + needed > cfg["deadline"]:
            reporter.event(
                f"{rid}: stopping after epoch {state['epoch']}/{epochs} so that evaluation ends within the session budget"
            )
            break
        epoch = state["epoch"] + 1
        model.train()
        start, losses = time.perf_counter(), []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (image, bits) in enumerate(
            itertools.islice(batches, samples_per_epoch // micro), start=1
        ):
            image, label = lib.augment(
                image.to(device, non_blocking=True), bits.to(device, non_blocking=True), patch
            )
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(image)
            loss = lib.supervised_loss(outputs, label, weights)
            scaler.scale(loss / accumulation).backward()
            losses.append(loss.item())
            if batch_index % accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                state["step"] += 1
                reporter.status(
                    job=rid,
                    step=state["step"],
                    total=total_steps,
                    postfix={
                        "epoch": epoch,
                        "loss": round(float(np.mean(losses)), 4),
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                        "mem_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2),
                    },
                )
        train_seconds = time.perf_counter() - start
        state["train_seconds"] += train_seconds
        state["peak_memory_gb"] = max(
            state["peak_memory_gb"], torch.cuda.max_memory_allocated(device) / 1e9
        )
        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": float(np.mean(losses)),
            "samples_per_s": samples_per_epoch / train_seconds,
            **{f"val_dice_{region}": np.nan for region in lib.REGIONS},
            "val_dice_mean": np.nan,
        }
        improved = False
        if epoch % cfg["val_every"] == 0 or epoch == epochs:
            val_start = time.perf_counter()
            val_dice = validate(model, val, cfg, device)
            # Evaluation predicts every val/test/external case with TTA and runs the sequence-drop
            # ablation on the test cases (one pass per sequence plus the full input), without TTA.
            per_case = (time.perf_counter() - val_start) / len(val)
            passes = (len(lib.FLIP_SETS) if cfg["tta"] else 1) * sum(
                cases["role"].isin(["val", "test", "external"])
            )
            passes += (1 + len(lib.MODALITIES)) * int((cases["role"] == "test").sum())
            state["eval_estimate"] = 1.2 * per_case * passes
            row.update(
                {f"val_dice_{region}": float(value) for region, value in zip(lib.REGIONS, val_dice)}
            )
            row["val_dice_mean"] = float(val_dice.mean())
            improved = row["val_dice_mean"] > state["best_dice"]
            if improved:
                state["best_dice"], state["best_epoch"] = row["val_dice_mean"], epoch
                torch.save(
                    {k: v.detach().float().cpu() for k, v in model.state_dict().items()}, best_path
                )
        row["epoch_seconds"] = time.perf_counter() - start
        state["history"].append(row)
        state["epoch"] = epoch
        pd.DataFrame(state["history"]).to_csv(run_dir / "history.csv", index=False)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "state": state,
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device),
            },
            last_path,
        )
        message = f"{rid} epoch {epoch}/{epochs}: train loss {row['train_loss']:.4f} | {row['samples_per_s']:.2f} patches/s"
        if not np.isnan(row["val_dice_mean"]):
            message += f" | val Dice WT {row['val_dice_WT']:.4f} TC {row['val_dice_TC']:.4f} ET {row['val_dice_ET']:.4f} mean {row['val_dice_mean']:.4f}"
        reporter.event(message + (" | best" if improved else ""))

    batches = None  # stops the loader workers before evaluation
    truncated = state["epoch"] < epochs
    if truncated and not best_path.exists():
        reporter.event(f"{rid}: no validated checkpoint yet; left for the next session")
        return False
    prob_root = Path(cfg["probs_dir"]) / rid
    if state["epoch"] > start_epoch:  # predictions of an earlier session's checkpoint are stale
        shutil.rmtree(prob_root, ignore_errors=True)
    # Evaluation of the best epoch: probabilities of every validation, test and external case
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    splits = {role: cases[cases["role"] == role] for role in ("val", "test", "external")}
    predict_seconds = {}
    for role, frame in splits.items():
        start = time.perf_counter()
        predict_split(model, frame, cfg, device, prob_root / role, reporter, f"{rid} {role}")
        predict_seconds[role] = time.perf_counter() - start
    inference_memory = torch.cuda.max_memory_allocated(device) / 1e9
    tuning = tune_et_threshold(splits["val"], cfg, prob_root / "val")
    et_min = max(tuning, key=lambda t: (tuning[t], -t))
    for role, frame in splits.items():
        score_split(frame, cfg, prob_root / role, et_min).to_csv(
            run_dir / f"{role}_cases.csv", index=False
        )
    reporter.event(f"{rid}: sequence-drop ablation on {len(splits['test'])} test cases")
    modality_ablation(
        model, splits["test"], cfg, device, et_min, reporter, f"{rid} ablation"
    ).to_csv(run_dir / "modality_ablation.csv", index=False)
    shape = tuple(
        int(np.median([int(s.split("x")[i]) for s in splits["test"]["crop_shape"]]))
        for i in range(3)
    )
    latency, latency_sd = time_inference(model, shape, cfg, device, False, cfg["latency_iters"])
    latency_tta, latency_tta_sd = time_inference(
        model, shape, cfg, device, True, max(2, cfg["latency_iters"] // 2)
    )
    test_scores = pd.read_csv(run_dir / "test_cases.csv")
    external_scores = pd.read_csv(run_dir / "external_cases.csv")
    summary = {
        "run_id": rid,
        "model": name,
        "family": recipe["family"],
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "micro_batch": micro,
        "accumulation": accumulation,
        "optimizer_steps": state["step"],
        "epochs_run": state["epoch"],
        "truncated": truncated,
        "patches_seen": state["epoch"] * samples_per_epoch,
        "best_epoch": state["best_epoch"],
        "best_val_dice": state["best_dice"],
        "et_min_voxels": et_min,
        "et_tuning": {str(k): v for k, v in tuning.items()},
        "train_seconds": state["train_seconds"],
        "predict_seconds": predict_seconds,
        "peak_train_memory_gb": state["peak_memory_gb"],
        "peak_inference_memory_gb": inference_memory,
        "latency_shape": "x".join(map(str, shape)),
        "latency_s": latency,
        "latency_s_sd": latency_sd,
        "latency_tta_s": latency_tta,
        "latency_tta_s_sd": latency_tta_sd,
        "test_dice_mean": float(test_scores["dice_mean"].mean()),
        "external_dice_mean": (
            float(external_scores["dice_mean"].mean()) if len(external_scores) else float("nan")
        ),
        "gpu": torch.cuda.get_device_name(device),
    }
    shared_lib.write_json(run_dir / "summary.json", summary)
    if not truncated:
        last_path.unlink(missing_ok=True)  # optimiser state is no longer needed
    reporter.event(
        f"{rid} done{' (stopped early by the time budget)' if truncated else ''}: test mean Dice {summary['test_dice_mean']:.4f}, external {summary['external_dice_mean']:.4f}, ET size threshold {et_min} voxels"
    )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    faulthandler.enable()  # a crash prints its traceback to the worker log
    cfg = json.loads(Path(args.config).read_text())
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES selects the physical GPU
    torch.cuda.set_device(device)
    reporter = shared_lib.Reporter(cfg["log_dir"], args.gpu)
    cases = pd.read_csv(cfg["case_table"])
    claims = Path(cfg["claims_dir"])
    for job in cfg["jobs"]:
        rid = lib.run_id(job["model"], job["seed"])
        summary = shared_lib.read_json(Path(cfg["runs_dir"]) / rid / "summary.json")
        if summary and not summary.get("truncated"):
            continue  # a run stopped by the budget continues from its last checkpoint
        if time.time() > cfg["deadline"]:
            reporter.event(f"Time budget reached; {rid} and later jobs left for the next session")
            break
        try:
            os.close(os.open(claims / rid, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        max_micro = lib.RECIPES[job["model"]]["micro_batch"]
        while True:
            try:
                run_job(job, cfg, cases, device, reporter, max_micro)
                break
            except torch.OutOfMemoryError as error:
                if max_micro == 1:
                    raise
                reporter.event(
                    f"{rid}: out of GPU memory ({str(error).splitlines()[0][:120]}); retrying from the last checkpoint with micro-batch {max_micro // 2}"
                )
                max_micro //= 2
            finally:  # the finished or failed job's tensors are unreachable now
                gc.collect()
                torch.cuda.empty_cache()
    reporter.status(job=None)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
