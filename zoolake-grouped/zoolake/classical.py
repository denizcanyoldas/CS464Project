"""Handcrafted image features and small scikit-learn/XGBoost baselines."""

from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .images import d4_views, load_canvas


def image_features(image: Image.Image) -> np.ndarray:
    """Summarize intensity, occupancy, shape, edges and coarse spatial layout."""
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    foreground = gray > 12 / 255
    y, x = np.nonzero(foreground)
    height, width = gray.shape
    pixels = rgb[foreground]
    if not pixels.size:
        pixels = np.zeros((1, 3), dtype=np.float32)
        geometry = (0.0,) * 6
    else:
        geometry = (
            foreground.mean(),
            (x.max() - x.min() + 1) / width,
            (y.max() - y.min() + 1) / height,
            x.std() / width,
            y.std() / height,
            np.mean(((x - (width - 1) / 2) / width) ** 2 + ((y - (height - 1) / 2) / height) ** 2),
        )
    color_histogram = np.stack([
        np.bincount(channel, minlength=8)
        for channel in np.minimum((pixels * 8).astype(np.int32), 7).T
    ]) / len(pixels)
    horizontal_edges = np.abs(np.diff(gray, axis=1))
    vertical_edges = np.abs(np.diff(gray, axis=0))
    coarse = image.convert("L").resize((8, 8), Image.Resampling.BOX)
    return np.concatenate((
        np.asarray(coarse, dtype=np.float32).reshape(-1) / 255.0,
        color_histogram.ravel(),
        pixels.mean(axis=0),
        pixels.std(axis=0),
        np.asarray(geometry),
        [horizontal_edges.mean(), vertical_edges.mean(), horizontal_edges.std(), vertical_edges.std()],
    )).astype(np.float32)


def _local_path(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise ValueError(f"Image from the split is missing: {path}. Check --data-root and the archive version.")
    return path


def _feature_array(root, rows, side, multi_view):
    features = []
    for row in rows:
        canvas = load_canvas(_local_path(root, row["path"]), side)
        views = d4_views(canvas) if multi_view else (canvas,)
        features.append([image_features(view) for view in views])
    return np.asarray(features)


def _build_model(name, config):
    balance = "balanced" if config["class_weight"] == "balanced" else None
    if name == "lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=config["regularization_c"], class_weight=balance,
                max_iter=2000, random_state=config["seed"],
            ),
        )
    if name == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(
                C=config["regularization_c"], kernel="rbf", gamma="scale",
                class_weight=balance, probability=True, random_state=config["seed"],
            ),
        )
    if name != "xgb":
        raise ValueError(f"Unknown classical model: {name}")
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise RuntimeError("Install requirements-xgb.txt to run XGBoost.") from error
    return XGBClassifier(
        n_estimators=120, max_depth=5, learning_rate=0.1, tree_method="hist",
        objective="multi:softprob", random_state=config["seed"], n_jobs=config["threads"],
    )


def _predict(bundle, rows, root, tta, threads):
    model = bundle["model"]
    features = _feature_array(root, rows, bundle["image_side"], tta)
    with threadpool_limits(limits=threads):
        if tta:
            probabilities = model.predict_proba(features.reshape(-1, features.shape[-1]))
            probabilities = probabilities.reshape(len(rows), 8, -1)
            return {"single": probabilities[:, 0, :], "d4": probabilities.mean(axis=1)}
        return {"single": model.predict_proba(features[:, 0, :])}


def train_classical(model_name, train, validation, root, labels, config, out_dir):
    augment = config["augment"] == "d4"
    features = _feature_array(root, train, config["image_side"], augment)
    encoder = LabelEncoder().fit(labels)
    if encoder.classes_.tolist() != labels:
        raise ValueError("Split labels must be sorted to preserve probability-column identity.")
    target = encoder.transform([row["label"] for row in train])
    target = np.repeat(target, features.shape[1])
    if np.unique(target).size != len(labels):
        raise ValueError("Every label needs at least one training image.")
    model = _build_model(model_name, config)
    fit_options = {}
    # Each source image contributes the same loss weight in both augmentation arms.
    if augment or model_name == "xgb":
        weights = np.full(len(target), 1.0 / features.shape[1])
        if model_name == "xgb":
            if config["class_weight"] == "balanced":
                weights *= len(target) / (len(labels) * np.bincount(target)[target])
            fit_options["sample_weight"] = weights
        else:
            fit_options[f"{model.steps[-1][0]}__sample_weight"] = weights
    with threadpool_limits(limits=config["threads"]):
        model.fit(features.reshape(-1, features.shape[-1]), target, **fit_options)
    bundle = {"model": model, "image_side": config["image_side"], "labels": labels}
    probabilities = _predict(bundle, validation, root, config["tta"], config["threads"])
    joblib.dump(bundle, out_dir / "model.joblib")
    return probabilities


def predict_classical(out_dir, rows, root, tta, threads):
    bundle = joblib.load(out_dir / "model.joblib")
    return _predict(bundle, rows, root, tta, threads)
