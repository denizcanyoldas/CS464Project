"""Scratch CNN V2: cosine learning-rate decay and validation-based early stopping."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from .cnn import (
    _balanced_weights,
    _loaders,
    _predict,
    build_model,
    resolve_device,
)


def train_scratch_v2(train, validation, root, labels, config, out_dir):
    """Train a scratch CNN and select its checkpoint using validation macro-F1."""
    if config["model"] != "scratch":
        raise ValueError("Scratch CNN V2 only supports model='scratch'.")

    device = resolve_device(config["device"])
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_loader, val_loader = _loaders(train, validation, root, labels, config)
    model = build_model("scratch", len(labels)).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )
    class_weights = (
        _balanced_weights(train, labels, device)
        if config["class_weight"] == "balanced"
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    truth = [val_loader.dataset.index[row["label"]] for row in validation]

    best_score = -1.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(config["epochs"]):
        train_loader.dataset.epoch = epoch
        started = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]

        model.train()
        loss_total = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            loss_total += loss.detach().item() * targets.size(0)

        train_loss = loss_total / len(train_loader.dataset)
        predictions = _predict(model, val_loader, device, tta=False)["single"]
        score = f1_score(
            truth,
            predictions.argmax(axis=1),
            labels=range(len(labels)),
            average="macro",
            zero_division=0,
        )
        scheduler.step()

        epoch_seconds = time.perf_counter() - started
        row = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "validation_macro_f1": float(score),
            "learning_rate": float(current_lr),
            "seconds": float(epoch_seconds),
        }
        history.append(row)
        print(
            f"Epoch {epoch + 1}/{config['epochs']} on {device}: "
            f"loss={train_loss:.4f}, validation macro-F1={score:.4f}, "
            f"lr={current_lr:.6g}, {epoch_seconds:.1f}s",
            flush=True,
        )

        if score > best_score + config["min_delta"]:
            best_score = float(score)
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config["patience"]:
            print(
                f"Early stopping after epoch {epoch + 1}; "
                f"best epoch was {best_epoch}.",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    torch.save(
        {
            "model": "scratch",
            "weights": best_state,
            "labels": labels,
            "image_side": config["image_side"],
            "batch_size": config["batch_size"],
            "threads": config["threads"],
        },
        Path(out_dir) / "checkpoint.pt",
    )

    timing = {}
    final_predictions = _predict(
        model, val_loader, device, config["tta"], timing
    )

    training_log = {
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score,
        "selection_metric": "single_view_validation_macro_f1",
        "early_stopping_patience": config["patience"],
        "epochs_completed": len(history),
        "stopped_early": len(history) < config["epochs"],
        "class_weight": config["class_weight"],
        "scheduler": "cosine_annealing",
        "epochs": history,
        "inference_timing": timing,
    }
    if device.type == "cuda":
        training_log["cuda_peak_memory_mb"] = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )

    (Path(out_dir) / "training_log.json").write_text(
        json.dumps(training_log, indent=2) + "\n", encoding="utf-8"
    )
    return final_predictions
