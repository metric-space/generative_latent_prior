"""Run the GLP sentiment evaluation pipeline from a YAML config.

Example:
    python glp_eval.py config=configs/eval_sentiment.yaml

OmegaConf command-line overrides can be appended, for example:
    python glp_eval.py config=configs/eval_sentiment.yaml device=cuda:1 stages.judge=false
"""

from dataclasses import dataclass, field
import glob
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd

from glp.utils_judge import judge_parquets
from integrations.sentiment.sentiment_steer import (
    grade_parquets,
    load_summary,
    plot_scaling,
    plot_tradeoff,
    sentiment_steering,
)


@dataclass
class StageConfig:
    """Evaluation stages that can be run independently."""

    steer: bool = True
    judge: bool = True
    grade: bool = True
    summarize: bool = True
    plot_concept_fluency: bool = True
    plot_tradeoffs: bool = False
    plot_scaling: bool = False


@dataclass
class EvalConfig:
    # Runtime
    device: str = "cuda:0"
    stages: StageConfig = field(default_factory=StageConfig)

    # Models and checkpoints
    model_name: str = "meta-llama/Llama-3.2-1B"
    weights_folders: list[str] = field(
        default_factory=lambda: ["generative-latent-prior/glp-llama1b-d3"]
    )
    ckpt_names: list[str] = field(default_factory=lambda: ["final"])

    # Input and output data
    sst5_train: str = "integrations/sentiment/cached_inputs/sst5_train.csv"
    sst5_test: Optional[str] = "integrations/sentiment/cached_inputs/sst5_test.csv"
    prefix_file: str = (
        "integrations/sentiment/cached_inputs/openwebtext_neutral_100_prefixes.csv"
    )
    save_root: str = "runs/sentiment"

    # Steering
    tasks: list[str] = field(
        default_factory=lambda: ["positive_sentiment", "negative_sentiment"]
    )
    methods: list[str] = field(default_factory=lambda: ["diffmean", "glp"])
    alphas: list[float] = field(
        default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    )
    act_norm: Optional[float] = 5.2087
    max_new_tokens: int = 20
    temperature: float = 0.0
    batch_size: int = 100
    seed: int = 42

    # GLP post-processing
    u: float = 0.5
    num_timesteps: int = 20
    record_direction_trace: bool = False
    dtype: str = "bfloat16"

    # LLM judge
    judge_model: str = "gpt-4o-mini"
    max_concurrency: int = 100
    overwrite_judge: bool = False


def load_config():
    """Merge defaults, a YAML file, and command-line overrides."""

    defaults = OmegaConf.structured(EvalConfig)
    OmegaConf.set_struct(defaults, False)
    cli = OmegaConf.from_cli()
    config_path = cli.pop("config", None)
    file_config = OmegaConf.load(config_path) if config_path else OmegaConf.create()
    config = OmegaConf.merge(defaults, file_config, cli)
    OmegaConf.resolve(config)
    return config


def validate_config(config) -> None:
    if not config.weights_folders:
        raise ValueError("weights_folders must contain at least one GLP path or Hub repo ID")
    if not config.ckpt_names:
        raise ValueError("ckpt_names must contain at least one checkpoint name")
    if not config.tasks:
        raise ValueError("tasks must not be empty")
    if not config.methods:
        raise ValueError("methods must not be empty")
    if any(method not in {"diffmean", "glp"} for method in config.methods):
        raise ValueError("methods may contain only 'diffmean' and 'glp'")
    if config.num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if config.stages.steer:
        for path_name in ("sst5_train", "prefix_file"):
            path = Path(config[path_name])
            if not path.is_file():
                raise FileNotFoundError(f"{path_name} does not exist: {path}")
        if config.sst5_test and not Path(config.sst5_test).is_file():
            raise FileNotFoundError(f"sst5_test does not exist: {config.sst5_test}")

    if config.stages.judge and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required when stages.judge=true")


def output_folder(save_root: str, weights_folder: str, ckpt_name: str) -> str:
    """Return a stable output directory for a model/checkpoint pair."""

    weights_name = Path(weights_folder.rstrip("/")).name
    suffix = "" if ckpt_name == "final" else f"-{ckpt_name.replace('/', '-')}"
    return str(Path(save_root) / f"{weights_name}{suffix}")


def result_files(save_folders: list[str]) -> list[str]:
    """Find result Parquets only for runs selected by this invocation."""

    return sorted(
        file
        for folder in save_folders
        for file in glob.glob(str(Path(folder) / "*" / "*.parquet"))
    )


def padded_score_limits(values, padding_fraction: float = 0.08, minimum_span: float = 0.2):
    """Return tight, padded limits for judge scores whose valid range is 0--2."""

    values = np.asarray(values, dtype=float)
    data_min = float(np.nanmin(values))
    data_max = float(np.nanmax(values))
    span = max(data_max - data_min, minimum_span)
    center = (data_min + data_max) / 2
    lower = max(0.0, center - span / 2 - span * padding_fraction)
    upper = min(2.0, center + span / 2 + span * padding_fraction)
    return lower, upper


def plot_concept_fluency_comparison(runs, save_root: str) -> None:
    """Plot concept against fluency, with steering alpha tracing each curve.

    One DiffMean baseline is shown per task because it is independent of the
    GLP checkpoint. Each GLP model/checkpoint gets its own curve. The judged
    per-example Parquets remain unchanged; only an aggregate CSV and figures
    are written under ``save_root``.
    """

    frames = []
    diffmean_tasks = set()
    required_columns = {"alpha", "fluency_llm_score", "concept_llm_score"}

    for weights_folder, ckpt_name, save_folder in runs:
        model_name = Path(weights_folder.rstrip("/")).name
        model_label = model_name if ckpt_name == "final" else f"{model_name} [{ckpt_name}]"
        for file_name in result_files([save_folder]):
            path = Path(file_name)
            task = path.parent.name
            method = path.stem
            if method not in {"diffmean", "glp"}:
                continue
            if method == "diffmean" and task in diffmean_tasks:
                continue

            df = pd.read_parquet(path)
            missing = required_columns - set(df.columns)
            if missing:
                raise RuntimeError(
                    f"Cannot plot {path}: missing judged columns {sorted(missing)}. "
                    "Enable stages.judge or use already judged Parquets."
                )

            if method == "diffmean":
                curve = "DiffMean"
                diffmean_tasks.add(task)
            else:
                curve = f"GLP: {model_label}"
            frames.append(
                df.assign(task=task, curve=curve)[
                    ["task", "curve", "alpha", "fluency_llm_score", "concept_llm_score"]
                ]
            )

    if not frames:
        raise FileNotFoundError("No DiffMean or GLP result Parquets were found to plot")

    rows = pd.concat(frames, ignore_index=True)
    summary = (
        rows.groupby(["task", "curve", "alpha"], as_index=False)[
            ["fluency_llm_score", "concept_llm_score"]
        ]
        .mean()
        .sort_values(["task", "curve", "alpha"], key=lambda col: col.abs() if col.name == "alpha" else col)
    )

    output_dir = Path(save_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "concept_fluency_summary.csv", index=False)

    tasks = list(dict.fromkeys(summary["task"].tolist()))
    with plt.rc_context({"text.usetex": False}):
        fig, axes = plt.subplots(
            1,
            len(tasks),
            figsize=(6.5 * len(tasks), 5.5),
            dpi=200,
            squeeze=False,
        )
        for ax, task in zip(axes[0], tasks):
            task_rows = summary[summary["task"] == task]
            curves = list(dict.fromkeys(task_rows["curve"].tolist()))
            colors = ["tab:orange"] + list(plt.cm.Blues(np.linspace(0.5, 0.95, max(1, len(curves) - 1))))
            for color, curve in zip(colors, curves):
                curve_rows = task_rows[task_rows["curve"] == curve].copy()
                curve_rows = curve_rows.iloc[np.argsort(curve_rows["alpha"].abs().to_numpy())]
                ax.plot(
                    curve_rows["fluency_llm_score"],
                    curve_rows["concept_llm_score"],
                    marker="o",
                    linewidth=1.8,
                    color=color,
                    label=curve,
                )
                for row in curve_rows.itertuples():
                    ax.annotate(
                        f"{abs(row.alpha):g}",
                        (row.fluency_llm_score, row.concept_llm_score),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=8,
                        color=color,
                    )

            ax.set_title(task.replace("_", " ").title())
            ax.set_xlabel("Mean fluency score")
            ax.set_ylabel("Mean concept score")
            ax.set_xlim(padded_score_limits(task_rows["fluency_llm_score"]))
            ax.set_ylim(padded_score_limits(task_rows["concept_llm_score"]))
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(fontsize=9)

        fig.suptitle("Concept–fluency trade-off (point labels are |alpha|)")
        fig.tight_layout()
        fig.savefig(output_dir / "concept_fluency.png", bbox_inches="tight")
        fig.savefig(output_dir / "concept_fluency.pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {output_dir / 'concept_fluency.png'}")


def main() -> None:
    config = load_config()
    validate_config(config)

    runs = [
        (
            weights_folder,
            ckpt_name,
            output_folder(config.save_root, weights_folder, ckpt_name),
        )
        for weights_folder in config.weights_folders
        for ckpt_name in config.ckpt_names
    ]
    save_folders = [save_folder for _, _, save_folder in runs]

    print("Evaluation runs:")
    for weights_folder, ckpt_name, save_folder in runs:
        print(f"  {weights_folder} [{ckpt_name}] -> {save_folder}")

    if config.stages.steer:
        for weights_folder, ckpt_name, save_folder in runs:
            sentiment_steering(
                config,
                weights_folder,
                ckpt_name,
                save_folder,
                device=config.device,
            )

    files = result_files(save_folders)
    if any((config.stages.judge, config.stages.grade, config.stages.summarize)) and not files:
        raise FileNotFoundError(
            "No result Parquets were found. Run with stages.steer=true first, "
            "or point save_root at existing results."
        )

    if config.stages.judge:
        judge_parquets(
            files,
            model=config.judge_model,
            max_concurrency=config.max_concurrency,
            overwrite=config.overwrite_judge,
        )

    if config.stages.grade:
        grade_parquets(config, files, device=config.device)

    if config.stages.summarize:
        for save_folder in save_folders:
            if not result_files([save_folder]):
                continue
            try:
                load_summary(save_folder)
            except KeyError as exc:
                raise RuntimeError(
                    f"Cannot summarize {save_folder}: LLM-judge score columns are missing. "
                    "Enable stages.judge or use already judged Parquets."
                ) from exc

    if config.stages.plot_concept_fluency:
        plot_concept_fluency_comparison(runs, config.save_root)

    if config.stages.plot_tradeoffs:
        for save_folder in save_folders:
            if result_files([save_folder]):
                plot_tradeoff(config, save_folder)

    if config.stages.plot_scaling:
        plot_scaling(config)


if __name__ == "__main__":
    main()
