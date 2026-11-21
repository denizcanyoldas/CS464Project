"""Scratch CNN and EfficientNet-B0 with validation-selected checkpoints."""

from collections import Counter
import json
from pathlib import Path
import random
import time

import numpy as np
from sklearn.metrics import f1_score

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as error:
    raise ImportError("Install requirements-cnn.txt to run the CNN experiments.") from error

from .images import d4_view, load_canvas


CHANNEL_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
CHANNEL_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class RoiImages(Dataset):
    def __init__(self, rows, root, labels, side, *, augment=False, seed=0):
        self.rows = rows
        self.root = Path(root)
        self.index = {label: index for index, label in enumerate(labels)}
        self.side = side
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = load_canvas(self.root / row["path"], self.side)
        if self.augment:
            transform = random.Random(f"{self.seed}:{self.epoch}:{index}").randrange(8)
            image = d4_view(image, transform)
        rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255
        return (rgb - CHANNEL_MEAN) / CHANNEL_STD, self.index[row["label"]]


def resolve_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this process. Use --device auto or cpu.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("PyTorch Metal is unavailable in this process. Use --device auto or cpu.")
    return torch.device(name)


def _block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(4, out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


def build_model(name, classes, *, pretrained=False):
    if name == "scratch":
        return nn.Sequential(
            _block(3, 24),
            _block(24, 48),
            _block(48, 96),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, classes),
        )
    if name == "efficientnet-b0":
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, classes)
        return model
    raise ValueError(f"Unknown CNN: {name}")


def _tensor_view(batch, index):
    image = batch.flip(dims=(-1,)) if index >= 4 else batch
    return image.rot90(index % 4, dims=(-2, -1)) if index % 4 else image


def _predict(model, loader, device, tta):
    model.eval()
    single_view, pooled = [], []
    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device)
            single = model(images).softmax(dim=1)
            single_view.append(single.cpu())
            if tta:
                views = [single]
                for index in range(1, 8):
                    views.append(model(_tensor_view(images, index)).softmax(dim=1))
                pooled.append(torch.stack(views).mean(dim=0).cpu())
    predictions = {"single": torch.cat(single_view).numpy()}
    if tta:
        predictions["d4"] = torch.cat(pooled).numpy()
    return predictions


def _balanced_weights(rows, labels, device):
    support = Counter(row["label"] for row in rows)
    if set(labels) != set(support):
        raise ValueError("CNN training requires at least one example of every label.")
    return torch.tensor([len(rows) / (len(labels) * support[label]) for label in labels], device=device)


def _loaders(train, validation, root, labels, config):
    seed, side = config["seed"], config["image_side"]
    training = RoiImages(train, root, labels, side, augment=config["augment"] == "d4", seed=seed)
    evaluation = RoiImages(validation, root, labels, side)
    training_loader = DataLoader(
        training, batch_size=config["batch_size"], num_workers=config["workers"], shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    evaluation_loader = DataLoader(evaluation, batch_size=config["batch_size"], num_workers=config["workers"])
    return training_loader, evaluation_loader


def _fit_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    loss_total = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), targets)
        loss.backward()
        optimizer.step()
        loss_total += loss.detach().item() * targets.size(0)
    return loss_total / len(loader.dataset)


def train_cnn(model_name, train, validation, root, labels, config, out_dir):
    """Use single-view validation macro-F1 for selection, then score requested views."""
    device = resolve_device(config["device"])
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    if model_name == "efficientnet-b0" and config["image_side"] <= 32:
        batch = config["batch_size"]
        if batch == 1 or len(train) % batch == 1:
            raise ValueError("EfficientNet at this resolution needs at least two images per training batch for batch normalization.")

    train_loader, val_loader = _loaders(train, validation, root, labels, config)
    model = build_model(model_name, len(labels), pretrained=model_name == "efficientnet-b0").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    weight = _balanced_weights(train, labels, device) if config["class_weight"] == "balanced" else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    truth = [val_loader.dataset.index[row["label"]] for row in validation]
    best_score = -1.0
    history = []
    for epoch in range(config["epochs"]):
        train_loader.dataset.epoch = epoch
        start = time.perf_counter()
        loss = _fit_epoch(model, train_loader, opt, criterion, device)
        predictions = _predict(model, val_loader, device, tta=False)["single"]
        score = f1_score(truth, predictions.argmax(axis=1), labels=range(len(labels)),
                         average="macro", zero_division=0)
        row = {"epoch": epoch + 1, "train_loss": loss, "validation_macro_f1": float(score),
               "seconds": time.perf_counter() - start}
        history.append(row)
        print(f"Epoch {epoch + 1}/{config['epochs']} on {device}: "
              f"loss={loss:.4f}, validation macro-F1={score:.4f}, {row['seconds']:.1f}s", flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save({
        "model": model_name, "weights": best_state, "labels": labels,
        "image_side": config["image_side"], "batch_size": config["batch_size"], "threads": config["threads"],
    }, out_dir / "checkpoint.pt")
    training_log = {
        "device": str(device), "best_epoch": best_epoch, "selection_metric": "single_view_validation_macro_f1",
        "class_weight": config["class_weight"], "epochs": history,
    }
    (out_dir / "training_log.json").write_text(json.dumps(training_log, indent=2) + "\n", encoding="utf-8")
    return _predict(model, val_loader, device, config["tta"])


def predict_cnn(out_dir, rows, root, tta, device):
    checkpoint = torch.load(out_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    torch.set_num_threads(checkpoint["threads"])
    device = resolve_device(device)
    model = build_model(checkpoint["model"], len(checkpoint["labels"]))
    model.load_state_dict(checkpoint["weights"])
    model.to(device)
    images = RoiImages(rows, root, checkpoint["labels"], checkpoint["image_side"])
    loader = DataLoader(images, batch_size=checkpoint["batch_size"])
    return _predict(model, loader, device, tta)
