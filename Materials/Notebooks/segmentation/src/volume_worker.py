"""Training worker for the unified segmentation notebook: one process per GPU.

Each worker claims (model, seed) jobs, trains with checkpoints every epoch (so a later session
resumes from the last epoch), selects the best epoch on validation mean Dice, tunes the
enhancing-tumour size threshold on the validation cases, and evaluates the test cases at their
native resolution with test-time augmentation.
"""

import argparse
import json
import math
import os
import random
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


def probe_micro_batch(model, start, size, weights, device):
    """Largest micro-batch whose forward and backward pass fit; nothing is stepped."""
    model.train()
    candidate = start
    while candidate >= 1:
        try:
            x = torch.randn(candidate, 4, size, size, size, device=device)
            y = torch.zeros(candidate, 3, size, size, size, device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lib.supervised_loss(model(x), y, weights)
            loss.backward()
            torch.cuda.synchronize(device)
            return candidate
        except torch.OutOfMemoryError:
            candidate //= 2
        finally:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    raise RuntimeError("Even a micro-batch of 1 does not fit in GPU memory")


@torch.no_grad()
def validate(model, loader, weights, device):
    model.eval()
    losses, rows = [], []
    for image, label in loader:
        image, label = image.to(device, non_blocking=True), label.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(image)
        losses.append(lib.supervised_loss(outputs, label, weights).item())
        pred = torch.sigmoid(outputs[0].float()) > 0.5
        for p, r in zip(pred, label.bool()):
            rows.append([lib.dice(p[i], r[i]) for i in range(3)])
    rows = np.array(rows)
    return float(np.mean(losses)), rows.mean(axis=0)


def evaluate_split(model, cases, cfg, device, tta, et_candidates=None, et_min=0, probs_out=None):
    """Native-resolution evaluation of a set of cases.

    With et_candidates, returns the mean ET Dice for each candidate size threshold (tuning).
    Otherwise returns one metrics row per case and optionally stores the probabilities.
    """
    images = np.load(cfg["images_path"], mmap_mode="r")
    tuning = {threshold: [] for threshold in (et_candidates or [])}
    rows = []
    for position, case in enumerate(cases.itertuples()):
        x = torch.from_numpy(np.asarray(images[case.cache_index], dtype=np.float32))[None].to(
            device
        )
        mean, std = lib.predict_probs(model, x, tta)
        mean, std = mean[0], std[0]
        reference, spacing = lib.load_reference(case.seg)
        native = lib.to_native(mean, reference.shape[1:])
        if et_candidates is not None:
            ref_et = torch.from_numpy(reference[2]).to(device)
            for threshold in et_candidates:
                tuning[threshold].append(
                    lib.dice(lib.regions_from_probs(native, threshold)[2], ref_et)
                )
            continue
        pred = lib.regions_from_probs(native, et_min).cpu().numpy()
        wt_pred = mean[0] > 0.5
        uncertainty = float(std[0][wt_pred].mean()) if wt_pred.any() else 0.0
        rows.append(
            {
                "case": case.case,
                "grade": case.grade,
                **lib.case_metrics(pred, reference, spacing),
                "tta_std_in_wt": uncertainty,
            }
        )
        if probs_out is not None:
            probs_out[position] = (mean.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    if et_candidates is not None:
        return {threshold: float(np.mean(values)) for threshold, values in tuning.items()}
    return pd.DataFrame(rows)


@torch.no_grad()
def time_inference(model, size, device, tta, iterations, warmup):
    x = torch.randn(1, 4, size, size, size, device=device)
    for _ in range(warmup):
        lib.predict_probs(model, x, tta)
    torch.cuda.synchronize(device)
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        lib.predict_probs(model, x, tta)
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)
    return float(np.mean(timings)), float(np.std(timings))


def run_job(job, cfg, cases, device, reporter):
    name, seed = job["model"], job["seed"]
    rid = lib.run_id(name, seed)
    recipe = lib.RECIPES[name]
    size, epochs = cfg["img_size"], cfg["epochs"]
    seed_everything(seed, cfg["deterministic"])
    run_dir = Path(cfg["runs_dir"]) / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(cfg["weights_dir"]) / lib.slug(name)
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = weights_dir / f"{rid}.pth", weights_dir / f"{rid}_last.pth"

    model = lib.build_model(name, size).to(device)
    weights = recipe["ds_weights"]
    micro = probe_micro_batch(model, recipe["micro_batch"], size, weights, device)
    accumulation = math.ceil(cfg["effective_batch"] / micro)
    train = cases[cases["role"] == "train"]
    val = cases[cases["role"] == "val"]
    loader_kwargs = {
        "num_workers": cfg["loader_workers"],
        "pin_memory": True,
        "persistent_workers": cfg["loader_workers"] > 0,
    }
    train_loader = torch.utils.data.DataLoader(
        lib.CachedCases(cfg["images_path"], cfg["labels_path"], train["cache_index"].to_numpy()),
        batch_size=micro,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        lib.CachedCases(cfg["images_path"], cfg["labels_path"], val["cache_index"].to_numpy()),
        batch_size=micro,
        **loader_kwargs,
    )
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = epochs * steps_per_epoch
    warmup_steps = recipe["warmup_epochs"] * steps_per_epoch
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
        "stale": 0,
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
        reporter.event(f"{rid}: resumed after epoch {state['epoch']}")
    reporter.event(
        f"{rid}: {len(train)} train / {len(val)} val cases, micro-batch {micro} x {accumulation} accumulation, "
        f"{total_steps} steps over {epochs} epochs"
    )

    torch.cuda.reset_peak_memory_stats(device)
    while state["epoch"] < epochs:
        if time.time() > cfg["deadline"]:
            reporter.event(
                f"{rid}: time budget reached after epoch {state['epoch']}; resume in the next session"
            )
            return False
        epoch = state["epoch"] + 1
        model.train()
        start, losses = time.perf_counter(), []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (image, label) in enumerate(train_loader, start=1):
            image, label = lib.augment(
                image.to(device, non_blocking=True), label.to(device, non_blocking=True)
            )
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(image)
            loss = lib.supervised_loss(outputs, label, weights)
            scaler.scale(loss / accumulation).backward()
            losses.append(loss.item())
            if batch_index % accumulation == 0 or batch_index == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
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
        val_loss, val_dice = validate(model, val_loader, weights, device)
        state["train_seconds"] += train_seconds
        state["peak_memory_gb"] = max(
            state["peak_memory_gb"], torch.cuda.max_memory_allocated(device) / 1e9
        )
        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            **{f"val_dice_{region}": float(value) for region, value in zip(lib.REGIONS, val_dice)},
            "val_dice_mean": float(val_dice.mean()),
            "epoch_seconds": time.perf_counter() - start,
        }
        state["history"].append(row)
        improved = row["val_dice_mean"] > state["best_dice"]
        if improved:
            state["best_dice"], state["best_epoch"], state["stale"] = row["val_dice_mean"], epoch, 0
            torch.save(
                {k: v.detach().float().cpu() for k, v in model.state_dict().items()}, best_path
            )
        else:
            state["stale"] += 1
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
            },
            last_path,
        )
        reporter.event(
            f"{rid} epoch {epoch}/{epochs}: train loss {row['train_loss']:.4f} | val loss {val_loss:.4f} | "
            f"val Dice WT {val_dice[0]:.4f} TC {val_dice[1]:.4f} ET {val_dice[2]:.4f} mean {row['val_dice_mean']:.4f}"
            + (" | best" if improved else "")
        )
        if cfg["early_stop_patience"] and state["stale"] >= cfg["early_stop_patience"]:
            reporter.event(f"{rid}: early stop after {state['stale']} epochs without improvement")
            break

    # Evaluation of the best epoch
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    tuning = evaluate_split(model, val, cfg, device, cfg["tta"], et_candidates=cfg["et_candidates"])
    et_min = max(tuning, key=lambda t: (tuning[t], -t))
    test = cases[cases["role"] == "test"]
    probs_path = Path(cfg["probs_dir"]) / f"{rid}.npy"
    probs_out = np.lib.format.open_memmap(
        probs_path, mode="w+", dtype=np.uint8, shape=(len(test), 3, size, size, size)
    )
    metrics = evaluate_split(
        model, test, cfg, device, cfg["tta"], et_min=et_min, probs_out=probs_out
    )
    probs_out.flush()
    del probs_out
    metrics.to_csv(run_dir / "test_cases.csv", index=False)
    inference_memory = torch.cuda.max_memory_allocated(device) / 1e9
    latency, latency_sd = time_inference(model, size, device, False, cfg["latency_iters"], 5)
    latency_tta, latency_tta_sd = time_inference(
        model, size, device, True, max(3, cfg["latency_iters"] // 5), 1
    )
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
        "best_epoch": state["best_epoch"],
        "best_val_dice": state["best_dice"],
        "et_min_voxels": et_min,
        "et_tuning": {str(k): v for k, v in tuning.items()},
        "train_seconds": state["train_seconds"],
        "peak_train_memory_gb": state["peak_memory_gb"],
        "peak_inference_memory_gb": inference_memory,
        "latency_s": latency,
        "latency_s_sd": latency_sd,
        "latency_tta_s": latency_tta,
        "latency_tta_s_sd": latency_tta_sd,
        "test_dice_mean": float(metrics["dice_mean"].mean()),
        "gpu": torch.cuda.get_device_name(device),
    }
    shared_lib.write_json(run_dir / "summary.json", summary)
    last_path.unlink(missing_ok=True)  # optimiser state is no longer needed
    reporter.event(
        f"{rid} done: test mean Dice {summary['test_dice_mean']:.4f}, ET size threshold {et_min} voxels"
    )
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES selects the physical GPU
    torch.cuda.set_device(device)
    reporter = shared_lib.Reporter(cfg["log_dir"], args.gpu)
    cases = pd.read_csv(cfg["case_table"])
    claims = Path(cfg["claims_dir"])
    for job in cfg["jobs"]:
        rid = lib.run_id(job["model"], job["seed"])
        if (Path(cfg["runs_dir"]) / rid / "summary.json").exists():
            continue
        if time.time() > cfg["deadline"]:
            reporter.event(f"Time budget reached; {rid} and later jobs left for the next session")
            break
        try:
            os.close(os.open(claims / rid, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        run_job(job, cfg, cases, device, reporter)
    reporter.status(job=None)


if __name__ == "__main__":
    main()
