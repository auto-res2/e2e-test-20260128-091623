import subprocess
import sys

import hydra
from omegaconf import DictConfig, OmegaConf


def _has_path(cfg: DictConfig, path: str) -> bool:
    return OmegaConf.select(cfg, path) is not None


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_id = cfg.run
    overrides = [f"run={run_id}", f"results_dir={cfg.results_dir}", f"mode={cfg.mode}"]

    if cfg.mode == "trial":
        overrides.append("wandb.mode=disabled")
        if _has_path(cfg, "runs.optuna.n_trials"):
            overrides.append("runs.optuna.n_trials=0")
        if _has_path(cfg, "runs.dataset.split.calibration_size"):
            overrides.append("runs.dataset.split.calibration_size=2")
        if _has_path(cfg, "runs.dataset.split.test_size"):
            overrides.append("runs.dataset.split.test_size=2")
        if _has_path(cfg, "runs.decoding.max_new_tokens"):
            overrides.append("runs.decoding.max_new_tokens=16")
        if _has_path(cfg, "runs.training.epochs"):
            overrides.append("runs.training.epochs=1")
        if _has_path(cfg, "runs.training.batch_size"):
            overrides.append("runs.training.batch_size=1")
        if _has_path(cfg, "runs.decoding.budget.B"):
            overrides.append("runs.decoding.budget.B=2")
        if _has_path(cfg, "runs.decoding.budget.k"):
            overrides.append("runs.decoding.budget.k=2")
        if _has_path(cfg, "runs.decoding.budget.b"):
            overrides.append("runs.decoding.budget.b=1")
        if _has_path(cfg, "runs.decoding.budget.rollback_w"):
            overrides.append("runs.decoding.budget.rollback_w=2")
        if _has_path(cfg, "runs.decoding.budget.lookahead"):
            overrides.append("runs.decoding.budget.lookahead=2")
        if _has_path(cfg, "runs.decoding.budget.J"):
            overrides.append("runs.decoding.budget.J=1")
    elif cfg.mode == "full":
        overrides.append("wandb.mode=online")
    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}")

    cmd = [sys.executable, "-m", "src.train"] + overrides
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
