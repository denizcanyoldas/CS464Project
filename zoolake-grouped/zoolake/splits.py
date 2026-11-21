"""Versioned split and subset manifests for an audited dataset."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
from random import Random
from tempfile import NamedTemporaryFile

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


PARTITIONS = ("train", "validation", "test")
PROPORTIONS = np.array([0.70, 0.15, 0.15])
PERCENTAGES = (10, 25, 50, 100)
# (Per-class relative tolerance, overall split tolerance as share of all images, minimum width)
STRATIFICATION_BANDS = ((0.20, 0.0, 0), (0.30, 0.0025, 1), (0.50, 0.01, 2))


def _quarantine_conflicts(records):
    by_hash = defaultdict(list)
    for row in records:
        by_hash[row["sha256"]].append(row)

    conflicts = {
        digest: [{"path": row["path"], "label": row["label"]} for row in rows]
        for digest, rows in by_hash.items()
        if len({row["label"] for row in rows}) > 1
    }
    clean = [row for row in records if row["sha256"] not in conflicts]
    return clean, conflicts


def _components(records, group_by):
    parents = list(range(len(records)))

    def find(index):
        while index != parents[index]:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    keys = ("sha256", "group", "capture_frame")
    if group_by == "acquisition_day":
        keys += ("acquisition_day",)
    previous = {}
    for index, row in enumerate(records):
        for key in keys:
            value = row.get(key)
            if value:
                first = previous.setdefault((key, value), index)
                parents[find(index)] = find(first)

    groups = defaultdict(list)
    for index, row in enumerate(records):
        groups[find(index)].append(row)
    return list(groups.values())


def _component_support(components, labels):
    support = Counter(label for group in components for label in {r["label"] for r in group})
    return {label: support[label] for label in labels}


def _provider_diagnostics(records, components, labels, raw_provider):
    missing = [row["path"] for row in records if row.get("split") not in PARTITIONS]
    crossing = []
    for group in components:
        memberships = {row.get("split") for row in group}.intersection(PARTITIONS)
        if len(memberships) > 1:
            crossing.append({
                "component": group[0]["path"],
                "partitions": sorted(memberships),
                "size": len(group),
            })
    counts = Counter((row.get("split"), row["label"]) for row in records)
    absent = {
        partition: [label for label in labels if counts[partition, label] == 0]
        for partition in PARTITIONS
    }
    issues = raw_provider.get("issues", [])
    return {
        "eligible": not (missing or crossing or any(absent.values()) or issues),
        "missing_paths_after_quarantine": missing,
        "cross_partition_components_after_quarantine": crossing,
        "absent_labels_after_quarantine": absent,
        "audited_manifest_counts": raw_provider.get("manifest_counts", {}),
        "audited_issues": issues,
    }


def _stratify(components, labels, seed, time_limit_seconds):
    label_index = {label: index for index, label in enumerate(labels)}
    components_by_signature = defaultdict(list)
    for group in components:
        counts = Counter(label_index[row["label"]] for row in group)
        components_by_signature[tuple(sorted(counts.items()))].append(group)

    signatures = sorted(components_by_signature)
    rng = Random(seed)
    rng.shuffle(signatures)
    signatures_counts = np.array([len(components_by_signature[key]) for key in signatures])
    signature_labels = np.zeros((len(labels), len(signatures)))
    for index, signature in enumerate(signatures):
        for label, count in signature:
            signature_labels[label, index] = count

    # The integer variables assign interchangeable components to partitions. A
    # feasibility solve gives a reproducible assignment without a timed incumbent.
    n_signatures = len(signatures)
    n_partitions = len(PARTITIONS)
    n_assignments = n_signatures * n_partitions
    n_measures = (len(labels) + 1) * n_partitions
    matrix = lil_matrix((n_signatures + n_measures, n_assignments))

    for index in range(n_signatures):
        for partition in range(n_partitions):
            column = partition * n_signatures + index
            matrix[index, column] = 1

    signature_sizes = signature_labels.sum(axis=0)
    measure_vectors = np.vstack([signature_sizes, signature_labels])
    targets = measure_vectors @ signatures_counts
    for partition in range(n_partitions):
        columns = slice(partition * n_signatures, (partition + 1) * n_signatures)
        rows = slice(n_signatures + partition * len(targets), n_signatures + (partition + 1) * len(targets))
        matrix[rows, columns] = measure_vectors

    matrix = matrix.tocsr()
    targets_by_partition = PROPORTIONS[:, None] * targets
    # A dominant component may itself exceed the training budget of a class.
    # Increase the count margin for that class to reflect the indivisible block.
    concentration = np.maximum(0, signature_labels.max(axis=1) - targets_by_partition[0, 1:])
    concentration_tolerance = np.ceil(concentration).astype(int)
    tried_bands = []
    for class_tolerance, overall_tolerance, min_overall_width in STRATIFICATION_BANDS:
        overall_width = max(min_overall_width, ceil(targets[0] * overall_tolerance))
        widths = np.maximum(1, np.ceil(class_tolerance * targets_by_partition))
        widths[:, 1:] = np.maximum(widths[:, 1:], concentration_tolerance)
        widths[:, 0] = overall_width
        lower = np.maximum(1, np.floor(targets_by_partition + 1e-9) - widths)
        upper = np.ceil(targets_by_partition - 1e-9) + widths
        outcome = milp(
            np.zeros(n_assignments),
            integrality=np.ones(n_assignments, dtype=int),
            bounds=Bounds(0, np.tile(signatures_counts, n_partitions)),
            constraints=LinearConstraint(
                matrix,
                np.concatenate([signatures_counts, lower.ravel()]),
                np.concatenate([signatures_counts, upper.ravel()]),
            ),
            options={"time_limit": time_limit_seconds},
        )
        tried_bands.append({
            "class_relative_tolerance": class_tolerance,
            "overall_image_tolerance": overall_width,
            "status": outcome.message,
        })
        if outcome.status == 0:
            break
        if outcome.status != 2:
            raise ValueError(
                "The feasibility solver did not finish this split. Increase the solver time "
                f"limit and try again with identical data and settings. {outcome.message}"
            )
    else:
        raise ValueError("No all-label grouped split satisfies the configured stratification bands.")

    assignments = np.rint(outcome.x[:n_assignments]).astype(int).reshape(n_partitions, n_signatures)
    if (assignments < 0).any() or not np.array_equal(assignments.sum(axis=0), signatures_counts):
        raise ValueError("The split solver did not return a valid integral assignment.")
    if (assignments @ signature_labels.T < 1).any():
        raise ValueError("The split solver returned an assignment that omits a label.")

    memberships = {}
    for index, signature in enumerate(signatures):
        groups = components_by_signature[signature]
        rng.shuffle(groups)
        start = 0
        for partition, name in enumerate(PARTITIONS):
            end = start + assignments[partition, index]
            memberships.update((group[0]["path"], name) for group in groups[start:end])
            start = end
    return memberships, {
        "message": outcome.message,
        "scipy_version": scipy.__version__,
        "attempts": tried_bands,
        "concentrated_class_tolerances": {
            label: int(width) for label, width in zip(labels, concentration_tolerance) if width
        },
        "time_limit_seconds": time_limit_seconds,
    }


def _subsets(records, labels, seed):
    rng = Random(seed)
    rows_by_label = defaultdict(list)
    for row in records:
        rows_by_label[row["label"]].append(row)
    # A seeded within-class order determines membership at every fraction.
    for label in labels:
        rng.shuffle(rows_by_label[label])
    tie_order = labels.copy()
    rng.shuffle(tie_order)

    size = len(records)
    class_counts = {label: len(rows_by_label[label]) for label in labels}
    selected = {label: 1 for label in labels}
    by_fraction = {}
    statistics = {}
    selected_total = len(labels)
    for fraction in PERCENTAGES:
        requested = (size * fraction + 50) // 100
        budget = max(requested, len(labels))
        expected = {label: budget * class_counts[label] / size for label in labels}
        while selected_total < budget:
            label = max(
                (label for label in tie_order if selected[label] < class_counts[label]),
                key=lambda label: expected[label] - selected[label],
            )
            selected[label] += 1
            selected_total += 1
        rows = [row for label in labels for row in rows_by_label[label][:selected[label]]]
        by_fraction[str(fraction)] = sorted(row["path"] for row in rows)
        statistics[str(fraction)] = {
            "requested_count": requested,
            "actual_count": len(rows),
            "actual_percent": 100 * len(rows) / size,
            "distinct_decoded_images": len({row["sha256"] for row in rows}),
            "min_class_count": min(selected.values()),
            "class_counts": selected.copy(),
            "labels_below_one_proportional_image": [
                label for label in labels if fraction * class_counts[label] < 100
            ],
        }
    return by_fraction, statistics


def _write_summary(result, output):
    records = result["records"]
    counts = result["partition_class_counts"]
    provider = result["provider_diagnostics"]
    grouping = (result["audit_group_by"] or "audit group") if result["group_by"] == "group" else (
        result["group_by"]
    )
    grouping = grouping.replace("_", " ")
    lines = [
        "# ZooLake split and subset report",
        "",
        f"Strategy **{result['strategy']}**, seed **{result['split_seed']}**, grouping **{grouping}**.",
        f"Assigned **{len(records):,}** readable images across **{len(result['labels'])}** labels.",
        f"Quarantined **{result['quarantined_images']}** images in "
        f"**{len(result['quarantined_conflicting_hashes'])}** decoded hashes shared by conflicting labels.",
        f"{result['images_without_requested_group']:,} assigned images lacked the requested grouping field.",
        "",
        "Exact decoded duplicates and records linked by capture frames or chosen groups share a partition.",
        "Different frames or time blocks can still contain related crops from one capture session.",
        "",
        "## Provider partition assessment",
        "",
        f"Provider folds eligible under the selected grouping policy **{provider['eligible']}**.",
        f"After quarantine, **{len(provider['missing_paths_after_quarantine'])}** images lacked "
        f"an unambiguous provider assignment and **{len(provider['cross_partition_components_after_quarantine'])}** "
        "grouped components crossed provider boundaries.",
        "",
        "## Fixed folds",
        "",
        "| Partition | Images | Image share | Smallest class | Grouped components |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in PARTITIONS:
        rows = [row for row in records if row["split"] == name]
        lines.append(
            f"| {name} | {len(rows):,} | {len(rows) / len(records):.2%} | "
            f"{min(counts[name].values()):,} | {len({row['component'] for row in rows}):,} |"
        )
    lines.extend([
        "",
        "| Class | Independent components | Train | Validation | Test |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for label in result["labels"]:
        support = result["components_per_class"][label]
        lines.append(
            f"| {label} | {support:,} | {counts['train'][label]:,} | "
            f"{counts['validation'][label]:,} | {counts['test'][label]:,} |"
        )
    lines.extend([
        "",
        "## Nested training subsets",
        "",
        "Each label contributes at least one image. Remaining places are allocated by proportional deficit.",
        "Budgets are based on the fixed training fold and expanded only if needed to cover all labels.",
        "",
        "| Seed | Fraction | Requested | Actual | Distinct decoded images | Smallest class | Labels with expected support <1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for seed, statistics in result["subset_statistics"].items():
        for fraction, stat in statistics.items():
            below = ", ".join(stat["labels_below_one_proportional_image"]) or "None"
            lines.append(
                f"| {seed} | {fraction}% | {stat['requested_count']:,} | {stat['actual_count']:,} | "
                f"{stat['distinct_decoded_images']:,} | {stat['min_class_count']} | {below} |"
            )
    if result["solver"]:
        solver = result["solver"]
        chosen_band = solver["attempts"][-1]
        lines.extend([
            "",
            f"Feasibility solver status **{solver['message']}**. Overall fold counts are within "
            f"**{chosen_band['overall_image_tolerance']}** images of the closest integer bounds. "
            "Per-class tolerances beyond the closest integer bounds are at least one image, "
            f"or **{chosen_band['class_relative_tolerance']:.0%}** of the target rounded up.",
        ])
        if solver["concentrated_class_tolerances"]:
            concentrated = ", ".join(
                f"{label} ({margin} images)" for label, margin in solver["concentrated_class_tolerances"].items()
            )
            lines.append("Dominant grouped components allow a wider count tolerance for " + concentrated + ".")
    _write_text(output, "\n".join(lines) + "\n")


def _write_text(path, text):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def create_splits(
    audit_path: Path,
    output_dir: Path,
    strategy: str,
    split_seed: int,
    subset_seeds: list[int],
    *,
    group_by: str = "group",
    time_limit_seconds: float = 30,
) -> dict:
    if strategy not in {"provider", "stratified"}:
        raise ValueError("Choose split strategy 'provider' or 'stratified'.")
    if group_by not in {"group", "acquisition_day"}:
        raise ValueError("Choose grouping 'group' (audit group) or 'acquisition_day'.")
    if not subset_seeds or len(set(subset_seeds)) != len(subset_seeds):
        raise ValueError("Specify at least one distinct subset seed.")
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    if audit.get("status") != "ok":
        raise ValueError(f"Audit status is {audit.get('status')!r}. Run a successful dataset audit first.")
    labels = list(audit["labels"])
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("The dataset audit must provide an ordered list of unique labels.")
    paths = {row["path"] for row in audit["records"]}
    if len(paths) != len(audit["records"]):
        raise ValueError("Audit records contain duplicate image paths.")
    if {row["label"] for row in audit["records"]} - set(labels):
        raise ValueError("Audit records use labels outside the audit label mapping.")

    records, conflicts = _quarantine_conflicts(sorted(audit["records"], key=lambda row: row["path"]))
    if group_by == "acquisition_day" and not any(row.get(group_by) for row in records):
        raise ValueError("No acquisition-day metadata is available in the usable audit records.")
    components = _components(records, group_by)
    support = _component_support(components, labels)
    infeasible = ", ".join(f"{label} ({count})" for label, count in support.items() if count < 3)
    if infeasible:
        raise ValueError(f"At least three independent components per class are required: {infeasible}.")

    provider = _provider_diagnostics(records, components, labels, audit.get("provider", {}))
    solver = None
    if strategy == "provider":
        if not provider["eligible"]:
            raise ValueError(
                "Provider partitions fail validation after quarantine: "
                f"{len(provider['missing_paths_after_quarantine'])} ambiguous/missing assignments, "
                f"{len(provider['cross_partition_components_after_quarantine'])} cross-fold components, "
                f"missing labels {provider['absent_labels_after_quarantine']}, "
                f"provider manifest issues {provider['audited_issues']}."
            )
        assignments = {group[0]["path"]: group[0]["split"] for group in components}
    else:
        assignments, solver = _stratify(components, labels, split_seed, time_limit_seconds)

    manifest = []
    for group in components:
        component = group[0]["path"]
        for row in group:
            manifest.append({
                **row, "provider_split": row.get("split"), "split": assignments[component],
                "component": component,
            })
    manifest.sort(key=lambda row: row["path"])

    train = [row for row in manifest if row["split"] == "train"]
    subset_results = {str(seed): _subsets(train, labels, seed) for seed in subset_seeds}
    fold_label_counts = Counter((row["split"], row["label"]) for row in manifest)
    result = {
        "schema_version": 1,
        "root": audit["root"],
        "audit_sha256": sha256(audit_bytes).hexdigest(),
        "labels": labels,
        "records": manifest,
        "strategy": strategy,
        "split_seed": split_seed,
        "group_by": group_by,
        "audit_group_by": audit.get("group_by"),
        "images_without_requested_group": sum(not row.get(group_by) for row in records),
        "quarantined_conflicting_hashes": conflicts,
        "quarantined_images": sum(map(len, conflicts.values())),
        "target_ratios": dict(zip(PARTITIONS, PROPORTIONS.tolist())),
        "components_per_class": support,
        "provider_diagnostics": provider,
        "solver": solver,
        "partition_class_counts": {
            name: {label: fold_label_counts[name, label] for label in labels} for name in PARTITIONS
        },
        "subsets": {seed: value[0] for seed, value in subset_results.items()},
        "subset_statistics": {seed: value[1] for seed, value in subset_results.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_text(output_dir / "splits.json", json.dumps(result, indent=2) + "\n")
    _write_summary(result, output_dir / "split_summary.md")
    return result
