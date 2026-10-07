"""Training worker for the unified 2D notebooks: one process per GPU.

Each worker claims jobs (model, fold, input mode) from a shared list, trains, selects the
checkpoint on the validation fold, and writes predictions, history and a summary. A job is
skipped when its summary already exists, so a rerun or a resumed session continues where
the previous one stopped.
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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import image_lib as lib  # noqa: E402
import shared_lib  # noqa: E402


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def loss_sum(logits, labels, binary):
    if binary:
        return F.binary_cross_entropy_with_logits(
            logits[:, 0].float(), labels.float(), reduction="sum"
        )
    return F.cross_entropy(logits.float(), labels, reduction="sum")


def probe_micro_batch(model, prep_size, start, binary, num_outputs, device, channels_last):
    """Largest micro-batch whose forward and backward pass fit, halving on out-of-memory.

    Weights and normalisation statistics are restored afterwards, so probing does not
    change the model that is trained.
    """
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.train()
    candidate = start
    try:
        while candidate >= 1:
            try:
                x = torch.randn(candidate, 3, prep_size, prep_size, device=device)
                if channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)
                y = torch.zeros(candidate, dtype=torch.long, device=device)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = loss_sum(model(x), y, binary)
                loss.backward()
                torch.cuda.synchronize(device)
                return candidate
            except torch.OutOfMemoryError:
                candidate //= 2
            finally:
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
        raise RuntimeError("Even a micro-batch of 1 does not fit in GPU memory")
    finally:
        model.load_state_dict(state)


def evaluate(model, prep, images, masks, labels, mode, binary):
    logits = lib.predict_logits(model, prep, images, masks, mode)
    labels_t = torch.from_numpy(labels)
    loss = loss_sum(torch.from_numpy(logits), labels_t, binary).item() / len(labels)
    probs = lib.logits_to_probs(logits)
    accuracy = float((probs.argmax(1) == labels).mean())
    if binary:
        auc = roc_auc_score(labels, probs[:, 1])
    else:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    return logits, loss, accuracy, float(auc)


def run_job(job, cfg, arrays, frame, device, reporter):
    name, fold, mode = job["model"], job["fold"], job["mode"]
    rid = lib.run_id(name, fold, mode)
    binary = cfg["num_outputs"] == 1
    spec = lib.MODEL_SPECS[name]
    seed_everything(
        cfg["seed"] + 1000 * fold + (1 if mode == "background" else 0), cfg["deterministic"]
    )
    images, masks = arrays

    train_rows = frame[(frame["role"] == "dev") & (frame["fold"] != fold)]
    val_rows = frame[(frame["role"] == "dev") & (frame["fold"] == fold)]
    test_rows = frame[frame["role"] == "test"]
    external_rows = frame[frame["role"] == "external"]

    def take(rows):
        index = rows["cache_index"].to_numpy()
        return images[index], masks[index], rows["label"].to_numpy()

    train_images, train_masks, train_labels = take(train_rows)
    val_images, val_masks, val_labels = take(val_rows)
    test_images, test_masks, test_labels = take(test_rows)
    # Pinned host tensors: each step only copies one micro-batch to the GPU.
    train_images_t = torch.from_numpy(np.ascontiguousarray(train_images)).pin_memory()
    train_masks_t = torch.from_numpy(np.ascontiguousarray(train_masks)).pin_memory()
    train_labels_t = torch.from_numpy(train_labels).long()

    channels_last = name not in lib.TOKEN_MODELS
    model = lib.build_model(name, cfg["num_outputs"], pretrained=True).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    prep = lib.GpuPreprocessor(spec["input_size"], device, channels_last)

    effective = cfg["effective_batch"]
    micro = probe_micro_batch(
        model,
        spec["input_size"],
        min(spec["micro_batch"], effective),
        binary,
        cfg["num_outputs"],
        device,
        channels_last,
    )
    steps_per_epoch = math.ceil(len(train_rows) / effective)
    total_steps = cfg["epochs"] * steps_per_epoch
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda")
    generator = torch.Generator().manual_seed(cfg["seed"] + fold)

    run_dir = Path(cfg["runs_dir"]) / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = Path(cfg["weights_dir"]) / lib.slug(name) / f"{rid}.pth"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    reporter.event(
        f"{rid}: {len(train_rows)} train / {len(val_rows)} val / {len(test_rows)} test, "
        f"micro-batch {micro} x {math.ceil(effective / micro)} accumulation = {effective}, {total_steps} steps"
    )

    torch.cuda.reset_peak_memory_stats(device)
    history, best_loss, best_epoch, step = [], float("inf"), 0, 0
    train_start = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_start = time.perf_counter()
        seen, loss_total, correct = 0, 0.0, 0
        for chunk in torch.randperm(len(train_rows), generator=generator).split(effective):
            optimizer.zero_grad(set_to_none=True)
            pieces = list(chunk.split(micro))
            if len(pieces) > 1 and len(pieces[-1]) == 1:  # no single-sample batch-norm steps
                pieces[-2] = torch.cat(pieces[-2:])
                pieces.pop()
            for piece in pieces:
                x = prep(train_images_t[piece], train_masks_t[piece], mode=mode, train=True)
                y = train_labels_t[piece].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(x)
                loss = loss_sum(logits, y, binary)
                scaler.scale(loss / len(chunk)).backward()
                loss_total += loss.item()
                predicted = (logits[:, 0] > 0).long() if binary else logits.argmax(1)
                correct += int((predicted == y).sum())
                seen += len(piece)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            if step % 5 == 0 or step == total_steps:
                reporter.status(
                    job=rid,
                    step=step,
                    total=total_steps,
                    postfix={
                        "epoch": epoch,
                        "loss": round(loss_total / seen, 4),
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                        "mem_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2),
                    },
                )
        train_seconds = time.perf_counter() - epoch_start
        _, val_loss, val_acc, val_auc = evaluate(
            model, prep, val_images, val_masks, val_labels, mode, binary
        )
        row = {
            "epoch": epoch,
            "step": step,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": loss_total / seen,
            "train_acc": correct / seen,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "train_images_per_second": seen / train_seconds,
            "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch = val_loss, epoch
            torch.save(
                {k: v.detach().float().cpu() for k, v in model.state_dict().items()}, weights_path
            )
        reporter.event(
            f"{rid} epoch {epoch}/{cfg['epochs']}: train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} AUC {val_auc:.4f}"
            + (" | best" if improved else "")
        )
    train_seconds = time.perf_counter() - train_start
    peak_train_memory = torch.cuda.max_memory_allocated(device) / 1e9

    # Predictions of the checkpoint selected on validation loss.
    lib.load_weights(model, weights_path, device)
    outputs = {
        "val_index": val_rows["cache_index"].to_numpy(),
        "val_logits": lib.predict_logits(model, prep, val_images, val_masks, mode),
        "test_index": test_rows["cache_index"].to_numpy(),
        "test_logits": lib.predict_logits(model, prep, test_images, test_masks, mode),
    }
    if mode == "full":  # inference-only masking test on the same checkpoint
        outputs["test_logits_background_only"] = lib.predict_logits(
            model, prep, test_images, test_masks, "background"
        )
        outputs["test_logits_head_only"] = lib.predict_logits(
            model, prep, test_images, test_masks, "head"
        )
        if len(external_rows):
            ext_images, ext_masks, _ = take(external_rows)
            outputs["external_index"] = external_rows["cache_index"].to_numpy()
            outputs["external_logits"] = lib.predict_logits(
                model, prep, ext_images, ext_masks, mode
            )
    np.savez_compressed(Path(cfg["predictions_dir"]) / f"{rid}.npz", **outputs)

    summary = {
        "run_id": rid,
        "model": name,
        "family": spec["family"],
        "fold": fold,
        "mode": mode,
        "input_size": spec["input_size"],
        "parameters": lib.count_parameters(model),
        "micro_batch": micro,
        "accumulation": math.ceil(effective / micro),
        "effective_batch": effective,
        "optimizer_steps": step,
        "epochs": cfg["epochs"],
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "train_seconds": train_seconds,
        "train_images_per_second": len(train_rows) * cfg["epochs"] / train_seconds,
        "peak_train_memory_gb": peak_train_memory,
        "train_images": len(train_rows),
        "val_images": len(val_rows),
        "gpu": torch.cuda.get_device_name(device),
    }
    shared_lib.write_json(run_dir / "summary.json", summary)
    reporter.event(f"{rid} done in {train_seconds / 60:.1f} min, best epoch {best_epoch}")
    del model, optimizer, train_images_t, train_masks_t
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES selects the physical GPU
    torch.cuda.set_device(device)
    reporter = shared_lib.Reporter(cfg["log_dir"], args.gpu)
    frame = pd.read_csv(cfg["split_table"])
    arrays = (np.load(cfg["images_path"], mmap_mode="r"), np.load(cfg["masks_path"], mmap_mode="r"))
    claims = Path(cfg["claims_dir"])
    for job in cfg["jobs"]:
        rid = lib.run_id(job["model"], job["fold"], job["mode"])
        if (Path(cfg["runs_dir"]) / rid / "summary.json").exists():
            continue
        if time.time() > cfg["deadline"]:
            reporter.event(f"Time budget reached; {rid} and later jobs left for the next session")
            break
        try:
            os.close(os.open(claims / rid, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue  # the other GPU took this job
        run_job(job, cfg, arrays, frame, device, reporter)
    reporter.status(job=None)


if __name__ == "__main__":
    main()
