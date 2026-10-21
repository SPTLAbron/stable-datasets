"""Export compact training evidence and render PlantVillage/Intel learning curves.

Export local runs (no checkpoint tensors or dataset images are copied)::

    python -m benchmarks.plantvillage_intel_results --artifact-root benchmark-runs

Regenerate plots using only the committed epoch CSV::

    python -m benchmarks.plantvillage_intel_results --plot-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path


OUTPUT = Path("docs/source/datasets/experiments/plantvillage_intel")
METRICS = {
    "linear_top1": "eval/linear_probe_top1_epoch",
    "linear_top5": "eval/linear_probe_top5_epoch",
    "knn_top1": "eval/knn_probe_top1",
    "knn_top5": "eval/knn_probe_top5",
    "train_loss": "fit/loss_epoch",
}


def read_epochs(paths):
    """Merge sparse CSV records; later segments replace replayed epochs entirely."""
    merged = {}
    for path in paths:
        segment = {}
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                if not row.get("epoch"):
                    continue
                epoch = int(float(row["epoch"])) + 1
                metrics = segment.setdefault(epoch, {})
                for name, key in METRICS.items():
                    if row.get(key):
                        metrics[name] = float(row[key])
        merged.update(segment)
    return {epoch: values for epoch, values in sorted(merged.items()) if "linear_top1" in values}


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash(file)


def _hash(file):
    digest = hashlib.sha256()
    for block in iter(lambda: file.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def export(artifact_root, output):
    original = artifact_root / "full-simclr-20261020-141753/intelimageclassification"
    revised = artifact_root / "simclr-50epochs-20261020-172814"
    seed1 = artifact_root / "plantvillage-seed1-10epochs-20261021-084010"
    experiments = [
        ("intelimageclassification", 0, 50, 150, [original, revised / "intelimageclassification_seed0"]),
        ("plantvillage", 0, 10, 50, [revised / "plantvillage_seed0"]),
        ("plantvillage", 1, 10, 50, [seed1]),
    ]
    output.mkdir(parents=True, exist_ok=True)
    table, runs = [], []
    for dataset, seed, epochs, scheduler_epochs, segments in experiments:
        metrics_paths = [segment / "run/metrics.csv" for segment in segments]
        history = read_epochs(metrics_paths)
        assert list(history) == list(range(1, epochs + 1)), (dataset, seed, list(history))
        for epoch, values in history.items():
            table.append({"dataset": dataset, "seed": seed, "epoch": epoch, **values})
        final = history[epochs]
        sources = [{"path": str(path.relative_to(artifact_root)), "sha256": sha256(path)} for path in metrics_paths]
        source_env = json.loads((segments[-1] / "run/environment.json").read_text())
        packages = source_env["packages"]["pip_freeze"].splitlines()
        versions = [
            line
            for line in packages
            if re.match(
                r"^(torch|torchvision|torchmetrics|lightning|stable-pretraining|numpy|pillow|timm|hydra-core|omegaconf)[=@ ]",
                line,
                re.I,
            )
        ]
        if dataset == "intelimageclassification":
            log = (original / "console.log").read_text()
            original_seconds = sum(float(t) for t in re.findall(r"\[Epoch \d+/150\] done in ([\d.]+)s", log))
            recovery_seconds = json.loads((segments[-1] / "exit.json").read_text())["runtime_seconds"]
            runtime = original_seconds + recovery_seconds
            runtime_note = "Sum of original completed epoch durations plus recovery process wall time; includes two replayed epochs."
        else:
            runtime = json.loads((segments[-1] / "exit.json").read_text())["runtime_seconds"]
            runtime_note = "Full training process wall time, including startup and checkpoint writes."
        checkpoints = list(segments[0].rglob("final.ckpt"))
        checkpoint = checkpoints[0]
        runs.append(
            {
                "dataset": dataset,
                "seed": seed,
                "epochs": epochs,
                "scheduler_epochs": scheduler_epochs,
                "final": final,
                "runtime_seconds": runtime,
                "runtime_note": runtime_note,
                "best_linear_top1": max(values["linear_top1"] for values in history.values()),
                "best_knn_top1": max(values["knn_top1"] for values in history.values()),
                "metrics_sources": sources,
                "final_checkpoint_sha256": sha256(checkpoint),
                "python": source_env["python"]["version"].split()[0],
                "packages": versions,
                "stopping": "validation >=70% at a multiple of 5"
                if dataset == "plantvillage" and seed == 0
                else "fixed completed epoch budget",
            }
        )
    fields = ["dataset", "seed", "epoch", *METRICS]
    with (output / "epochs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    plant = [run for run in runs if run["dataset"] == "plantvillage"]
    aggregate = {
        metric: {
            "mean": statistics.mean(run["final"][metric] for run in plant),
            "sample_std": statistics.stdev(run["final"][metric] for run in plant),
        }
        for metric in ["linear_top1", "linear_top5", "knn_top1", "knn_top5"]
    }
    evidence = {
        "metric_definition": "Macro-averaged per-class top-k validation accuracy; values are fractions.",
        "hardware": "Apple M1 Max, 32 GB unified memory, MPS, FP32",
        "backbone": "ViT-S/16, 224x224, randomly initialized",
        "batch_size": 32,
        "accumulate_grad_batches": 1,
        "optimizer": {"name": "AdamW", "lr": 0.0003, "weight_decay": 0.05, "betas": [0.9, 0.999]},
        "probe": "Online linear head: default LARS lr0.1 constant schedule; detached encoder features. kNN k10, queue20000.",
        "runs": runs,
        "plantvillage_two_seed_descriptive_summary": aggregate,
        "limitations": [
            "Validation results, not held-out test estimates.",
            "PlantVillage seed0 stopping selected epoch10 using validation; seed1 then repeated that fixed budget.",
            "Intel50 uses a150-epoch cosine schedule. Replay epochs49/50 replace earlier values.",
            "PlantVillage image-level splits are not leaf-group-disjoint; adjacent captures may leak.",
            "Two PlantVillage seeds do not establish broad statistical robustness.",
        ],
    }
    (output / "results.json").write_text(json.dumps(evidence, indent=2) + "\n")


def plot(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (output / "epochs.csv").open() as file:
        rows = list(csv.DictReader(file))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    for ax, dataset, title in zip(
        axes,
        ["intelimageclassification", "plantvillage"],
        ["Intel · 6 classes · seed 0", "PlantVillage · 38 classes · seeds 0, 1"],
    ):
        for seed, style in [(0, "-"), (1, "--")]:
            values = [r for r in rows if r["dataset"] == dataset and int(r["seed"]) == seed]
            if not values:
                continue
            for metric, color, label in [("linear_top1", "#2468c9", "Linear"), ("knn_top1", "#da6724", "k-NN")]:
                x = [int(r["epoch"]) for r in values]
                y = [100 * float(r[metric]) for r in values]
                ax.plot(x, y, style, color=color, linewidth=1.7, label=f"{label}, seed {seed}")
                ax.scatter(x[-1], y[-1], color=color, s=22)
        ax.axhline(
            100 / (6 if dataset == "intelimageclassification" else 38), color="#888888", linestyle=":", label="Chance"
        )
        ax.set(title=title, xlabel="Completed epoch", ylabel="Validation balanced top-1 (%)", ylim=(0, 100))
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("SimCLR · ViT-S/16 · 224 px · batch 32 · M1 Max")
    fig.savefig(output / "accuracy.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.7), layout="constrained")
    for ax, dataset, title in zip(axes, ["intelimageclassification", "plantvillage"], ["Intel", "PlantVillage"]):
        for seed in (0, 1):
            values = [r for r in rows if r["dataset"] == dataset and int(r["seed"]) == seed and r["train_loss"]]
            if values:
                ax.plot(
                    [int(r["epoch"]) for r in values], [float(r["train_loss"]) for r in values], label=f"Seed {seed}"
                )
        ax.set(title=title, xlabel="Completed epoch", ylabel="Logged SimCLR epoch loss")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(output / "loss.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 2.7), layout="constrained")
    for ax, title, counts in zip(
        axes, ["PlantVillage · 38 classes", "Intel · 6 classes"], [(43429, 5428, 5448), (12632, 1402, 3000)]
    ):
        bars = ax.barh(["Train", "Validation", "Test"], counts, color=["#2468c9", "#639c63", "#da6724"])
        ax.invert_yaxis()
        ax.set(title=title, xlabel="Images", xlim=(0, max(counts) * 1.25))
        ax.spines[["top", "right"]].set_visible(False)
        for bar, count in zip(bars, counts):
            ax.text(
                count + max(counts) * 0.02, bar.get_y() + bar.get_height() / 2, f"{count:,}", va="center", fontsize=9
            )
    fig.savefig(output / "splits.png", dpi=170)
    plt.close(fig)


def audit_sources(raw_root, output):
    """Record the source snapshots and split digests without extracting archives."""
    import zipfile

    from stable_datasets.images import IntelImageClassification, PlantVillage

    records = []
    for builder, pattern in [
        (PlantVillage, "master*.zip"),
        (IntelImageClassification, "intel-image-classification*.zip"),
    ]:
        paths = sorted(raw_root.glob(pattern))
        if len(paths) != 1:
            raise ValueError(f"Expected exactly one {pattern} in {raw_root}")
        path = paths[0]
        with zipfile.ZipFile(path) as archive:
            splits = builder._split_members(archive)
        records.append(
            {
                "builder": builder.__name__,
                "archive_name": path.name,
                "archive_sha256": sha256(path),
                "archive_bytes": path.stat().st_size,
                "url": builder.SOURCE.assets["images"].url,
                "split_seed": builder.SPLIT_SEED,
                "splits": {
                    split: {
                        "count": len(names),
                        "sorted_member_paths_sha256": hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest(),
                    }
                    for split, names in splits.items()
                },
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "sources.json").write_text(json.dumps(records, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("benchmark-runs"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--source-archives", type=Path)
    args = parser.parse_args()
    if not args.plot_only:
        export(args.artifact_root, args.output)
    if args.source_archives:
        audit_sources(args.source_archives, args.output)
    plot(args.output)


if __name__ == "__main__":
    main()
