import math
import os
import random
import warnings
from typing import Callable, Dict, List, Tuple

import hydra
import numpy as np
import optuna
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from src.model import (
    DecodeResult,
    RiskModel,
    conformal_threshold,
    decode_cot_first_token,
    decode_cr2ad,
    decode_greedy,
    decode_rahd,
    decode_ut_cot,
    features_from_trace,
    greedy_rollout,
    predict_risk_scores,
    risk_score,
)
from src.preprocess import get_normalization_fn, load_dataset, make_prompt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_strategy(run_cfg: DictConfig) -> str:
    strategy = OmegaConf.select(run_cfg, "decoding.strategy")
    if strategy:
        return str(strategy).lower()
    method = str(OmegaConf.select(run_cfg, "method") or "").lower().replace("²", "2")
    if "cr2ad" in method or "conformal" in method:
        return "cr2ad"
    if "ut-cot" in method or "ut cot" in method:
        return "ut_cot"
    if "rahd" in method:
        return "rahd"
    if "cot" in method:
        return "cot"
    if "greedy" in method:
        return "greedy"
    return "greedy"


def _safe_min(cfg: DictConfig, path: str, value: int | float) -> None:
    current = OmegaConf.select(cfg, path)
    if current is not None:
        OmegaConf.update(cfg, path, min(current, value))


def apply_mode_overrides(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    run_cfg = cfg.runs
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        if OmegaConf.select(run_cfg, "optuna.n_trials") is not None:
            run_cfg.optuna.n_trials = 0
        _safe_min(run_cfg, "dataset.split.calibration_size", 2)
        _safe_min(run_cfg, "dataset.split.test_size", 2)
        _safe_min(run_cfg, "training.epochs", 1)
        _safe_min(run_cfg, "training.batch_size", 1)
        _safe_min(run_cfg, "decoding.max_new_tokens", 16)
        _safe_min(run_cfg, "decoding.budget.B", 2)
        _safe_min(run_cfg, "decoding.budget.k", 2)
        _safe_min(run_cfg, "decoding.budget.b", 1)
        _safe_min(run_cfg, "decoding.budget.rollback_w", 2)
        _safe_min(run_cfg, "decoding.budget.lookahead", 2)
        _safe_min(run_cfg, "decoding.budget.J", 1)
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}")


def ensure_optuna_defaults(run_cfg: DictConfig) -> None:
    if OmegaConf.select(run_cfg, "optuna") is None:
        run_cfg.optuna = {"n_trials": 0, "search_spaces": []}
    if OmegaConf.select(run_cfg, "optuna.n_trials") is None:
        run_cfg.optuna.n_trials = 0
    if OmegaConf.select(run_cfg, "optuna.search_spaces") is None:
        run_cfg.optuna.search_spaces = []


def serialize_label(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    return str(value)


class WandbLogger:
    def __init__(self, cfg: DictConfig, run_id: str) -> None:
        self.enabled = cfg.wandb.mode != "disabled"
        self.run = None
        if self.enabled:
            self.run = wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                id=run_id,
                config=OmegaConf.to_container(cfg, resolve=True),
                resume="allow",
                mode=cfg.wandb.mode,
            )
            print(self.run.get_url())

    def log(self, data: Dict, step: int | None = None) -> None:
        if self.enabled:
            wandb.log(data, step=step)

    def summary_set(self, key: str, value) -> None:
        if self.enabled:
            wandb.summary[key] = value

    def finish(self) -> None:
        if self.enabled:
            wandb.finish()


def resolve_feature_params(feature_cfg: DictConfig) -> Tuple[List[str], int, int, float]:
    feature_names = list(getattr(feature_cfg, "features", []))
    if not feature_names:
        raise ValueError("Risk model feature list is empty")
    feature_T = int(getattr(feature_cfg, "T", 24))
    if feature_T <= 0:
        feature_T = 24
    local_window = OmegaConf.select(feature_cfg, "local_window")
    if local_window is None:
        local_window = min(8, feature_T)
    low_margin_threshold = OmegaConf.select(feature_cfg, "low_margin_threshold")
    if low_margin_threshold is None:
        low_margin_threshold = 0.08
    return feature_names, feature_T, int(local_window), float(low_margin_threshold)


@torch.no_grad()
def compute_greedy_features(
    data: List[Tuple[str, int | str]],
    model,
    tokenizer,
    parse_fn: Callable[[str], int | str | None],
    max_new_tokens: int,
    max_length: int,
    device: torch.device,
    feature_names: List[str],
    feature_T: int,
    local_window: int,
    low_margin_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not feature_names:
        raise ValueError("Risk model feature list is empty")
    features: List[np.ndarray] = []
    errors: List[int] = []
    for question, gold in tqdm(data, desc="Greedy calibration", leave=False):
        prompt = make_prompt(question)
        text, margins, entropies, _, _ = greedy_rollout(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            device=device,
        )
        pred = parse_fn(text)
        errors.append(int(pred != gold))
        feat = features_from_trace(
            margins,
            entropies,
            feature_names=feature_names,
            T=feature_T,
            local_window=local_window,
            low_margin_threshold=low_margin_threshold,
        )
        features.append(feat)
    if not features:
        raise ValueError("No features extracted from calibration data")
    return np.stack(features), np.array(errors, dtype=np.float32)


def resolve_risk_model_type(risk_cfg: DictConfig | None) -> str | None:
    if risk_cfg is None:
        return None
    raw = str(getattr(risk_cfg, "type", "logistic_regression")).lower()
    if raw in {"logistic_regression", "torch", "torch_logistic_regression"}:
        return "torch_logistic_regression"
    if raw in {"sklearn", "sklearn_logistic", "sklearn_logistic_regression"}:
        return "sklearn_logistic_regression"
    if raw in {"isotonic", "isotonic_regression"}:
        return "isotonic_regression"
    if raw in {"none", "heuristic"}:
        return "none"
    raise ValueError(f"Unsupported risk_model.type: {raw}")


def build_optimizer(
    name: str,
    params,
    lr: float,
    weight_decay: float,
    momentum: float,
) -> torch.optim.Optimizer:
    name = (name or "adamw").lower()
    if name in {"adamw", "adam_w", "adamw_torch"}:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name in {"adam"}:
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name in {"sgd"}:
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f"Unsupported optimizer: {name}")


def fit_risk_model(
    features: torch.Tensor,
    labels: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    grad_accum_steps: int,
    optimizer_name: str,
    warmup_steps: int,
    momentum: float,
    device: torch.device,
    logger: WandbLogger,
) -> RiskModel:
    model = RiskModel(features.shape[1]).to(device)
    assert model.linear.out_features == 1, "Risk model output dimension mismatch"

    mean = features.mean(dim=0)
    std = features.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    model.set_normalization(mean.to(device), std.to(device))

    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = build_optimizer(optimizer_name, model.parameters(), lr, weight_decay, momentum)

    steps_per_epoch = max(1, math.ceil(len(loader) / max(1, grad_accum_steps)))
    total_steps = max(1, steps_per_epoch * max(1, epochs))
    warmup_steps = min(max(0, int(warmup_steps)), total_steps)
    scheduler = (
        get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        if warmup_steps > 0
        else None
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    global_step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        for step, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            if epoch == 0 and step == 0:
                assert x.ndim == 2 and y.ndim == 1, "Batch shapes invalid"
                assert x.shape[0] == y.shape[0], "Batch size mismatch"
            logits = model(x)
            loss = criterion(logits, y) / max(1, grad_accum_steps)
            assert torch.isfinite(loss).item(), "Loss is not finite"

            grads = torch.autograd.grad(
                loss, model.parameters(), create_graph=False, retain_graph=True
            )
            grad_norm = torch.sqrt(
                sum(g.detach().pow(2).sum() for g in grads if g is not None)
            )
            loss.backward()

            if (step + 1) % max(1, grad_accum_steps) == 0 or step == len(loader) - 1:
                total_grad = 0.0
                for param in model.parameters():
                    assert param.grad is not None, "Gradient missing before optimizer step"
                    total_grad += float(param.grad.detach().abs().sum().item())
                assert total_grad > 0.0, "Zero gradient detected before optimizer step"
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y).float().mean().item()
            lr_now = (
                scheduler.get_last_lr()[0]
                if scheduler is not None
                else optimizer.param_groups[0]["lr"]
            )
            logger.log(
                {
                    "risk_train_loss": loss.item() * max(1, grad_accum_steps),
                    "risk_train_acc": acc,
                    "risk_grad_norm": grad_norm.item(),
                    "risk_lr": lr_now,
                    "risk_epoch": epoch,
                },
                step=global_step,
            )
            global_step += 1

    model.eval()
    return model


def fit_sklearn_logistic(
    features_np: np.ndarray, labels_np: np.ndarray, max_iter: int
):
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required for sklearn_logistic_regression risk models"
        ) from exc
    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(features_np, labels_np.astype(int))
    clf.model_type = "sklearn_logistic_regression"
    return clf


def fit_isotonic_regression(
    features_np: np.ndarray, labels_np: np.ndarray, feature_index: int
):
    try:
        from sklearn.isotonic import IsotonicRegression
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required for isotonic_regression risk models"
        ) from exc
    iso = IsotonicRegression(out_of_bounds="clip")
    proxy = features_np[:, feature_index]
    iso.fit(proxy, labels_np.astype(float))
    iso.model_type = "isotonic_regression"
    iso.feature_index = feature_index
    return iso


def sample_optuna_params(trial: optuna.trial.Trial, search_spaces: List[Dict]) -> Dict:
    params = {}
    for space in search_spaces:
        name = space["param_name"]
        dist = space["distribution_type"]
        if dist == "categorical":
            params[name] = trial.suggest_categorical(name, space["choices"])
        elif dist == "uniform":
            params[name] = trial.suggest_float(name, space["low"], space["high"])
        else:
            raise ValueError(f"Unsupported distribution type: {dist}")
    return params


def split_indices(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    perm = np.random.permutation(n)
    if n < 3:
        train_idx = perm[:1]
        calib_idx = perm[1:] if n > 1 else perm[:1]
        val_idx = calib_idx
        return train_idx, calib_idx, val_idx
    train_end = max(1, int(0.6 * n))
    calib_end = max(train_end + 1, int(0.8 * n))
    train_idx = perm[:train_end]
    calib_idx = perm[train_end:calib_end]
    val_idx = perm[calib_end:]
    if len(val_idx) == 0:
        val_idx = calib_idx
    if len(calib_idx) == 0:
        calib_idx = train_idx
    return train_idx, calib_idx, val_idx


def _budget_param(run_cfg: DictConfig, name: str, default: int) -> int:
    value = OmegaConf.select(run_cfg, f"decoding.budget.{name}")
    return int(value) if value is not None else default


def _decoding_param(run_cfg: DictConfig, name: str, default) -> float:
    value = OmegaConf.select(run_cfg, f"decoding.{name}")
    return float(value) if value is not None else float(default)


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    apply_mode_overrides(cfg)
    run_cfg = cfg.runs
    ensure_optuna_defaults(run_cfg)
    run_id = str(OmegaConf.select(run_cfg, "run_id") or cfg.run)

    os.makedirs(cfg.results_dir, exist_ok=True)
    set_seed(int(run_cfg.dataset.split.split_seed))

    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    parse_fn = get_normalization_fn(run_cfg.dataset.preprocessing.normalization)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if str(run_cfg.model.precision).lower() == "bf16" and device.type == "cuda"
        else torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(
        run_cfg.model.name, cache_dir=".cache/", trust_remote_code=True
    )
    added_tokens = False
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            added_tokens = True
    assert tokenizer.pad_token_id is not None, "Tokenizer pad_token_id missing"

    model = AutoModelForCausalLM.from_pretrained(
        run_cfg.model.name,
        cache_dir=".cache/",
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    if added_tokens:
        model.resize_token_embeddings(len(tokenizer))
    model.eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    assert model.config.vocab_size > 0, "Model vocabulary size invalid"

    with torch.no_grad():
        dummy_ids = tokenizer("Hello", return_tensors="pt").input_ids.to(device)
        dummy_out = model(dummy_ids)
        assert (
            dummy_out.logits.shape[-1] == model.config.vocab_size
        ), "Model output dimension mismatch"

    logger = WandbLogger(cfg, run_id=run_id)

    calib_data, test_data = load_dataset(
        run_cfg.dataset.name,
        calibration_size=int(run_cfg.dataset.split.calibration_size),
        test_size=int(run_cfg.dataset.split.test_size),
        split_seed=int(run_cfg.dataset.split.split_seed),
        cache_dir=".cache/",
    )

    max_new_tokens = int(run_cfg.decoding.max_new_tokens)
    max_length = int(run_cfg.dataset.preprocessing.max_length)

    strategy = infer_strategy(run_cfg)

    risk_model = None
    t_alpha = None
    risk_threshold = None
    feature_names: List[str] = []
    feature_T = 24
    local_window = 8
    low_margin_threshold = 0.08

    risk_cfg = OmegaConf.select(run_cfg, "decoding.risk_model")
    risk_model_type = resolve_risk_model_type(risk_cfg)

    needs_risk_model = strategy in {"cr2ad", "rahd"} or (
        strategy == "ut_cot" and OmegaConf.select(run_cfg, "decoding.use_risk_model")
    )

    if needs_risk_model:
        if risk_cfg is None:
            raise ValueError("Risk model configuration is required for this strategy")
        if risk_model_type in {"none"} and strategy in {"cr2ad", "rahd"}:
            raise ValueError("Risk model type 'none' is not valid for CR2AD/RAHD")
        feature_names, feature_T, local_window, low_margin_threshold = resolve_feature_params(
            risk_cfg
        )
        features_np, errors_np = compute_greedy_features(
            calib_data,
            model,
            tokenizer,
            parse_fn,
            max_new_tokens,
            max_length,
            device,
            feature_names,
            feature_T,
            local_window,
            low_margin_threshold,
        )
        features = torch.tensor(features_np, dtype=torch.float32)
        errors = torch.tensor(errors_np, dtype=torch.float32)
        assert features.shape[0] == errors.shape[0], "Features/labels size mismatch"

        train_idx, calib_idx, val_idx = split_indices(len(features_np))

        if risk_model_type == "torch_logistic_regression":
            risk_model = fit_risk_model(
                features[train_idx],
                errors[train_idx],
                epochs=int(run_cfg.training.epochs),
                lr=float(run_cfg.training.learning_rate),
                weight_decay=float(run_cfg.training.weight_decay),
                batch_size=max(1, int(run_cfg.training.batch_size)),
                grad_accum_steps=max(1, int(run_cfg.training.gradient_accumulation_steps)),
                optimizer_name=str(run_cfg.training.optimizer),
                warmup_steps=int(run_cfg.training.warmup_steps),
                momentum=float(OmegaConf.select(run_cfg, "training.momentum") or 0.0),
                device=device,
                logger=logger,
            )
            risk_model.model_type = "torch_logistic_regression"
            risk_model.eval()
        elif risk_model_type == "sklearn_logistic_regression":
            warnings.warn(
                "training.optimizer and warmup_steps are ignored for sklearn logistic regression",
                RuntimeWarning,
            )
            max_iter = max(100, int(run_cfg.training.epochs) * 200)
            risk_model = fit_sklearn_logistic(features_np[train_idx], errors_np[train_idx], max_iter)
        elif risk_model_type == "isotonic_regression":
            warnings.warn(
                "training.optimizer and warmup_steps are ignored for isotonic regression",
                RuntimeWarning,
            )
            feature_index = (
                feature_names.index("min_margin") if "min_margin" in feature_names else 0
            )
            risk_model = fit_isotonic_regression(
                features_np[train_idx], errors_np[train_idx], feature_index
            )
        elif risk_model_type == "none":
            risk_model = None
        else:
            raise ValueError(f"Unsupported risk model type: {risk_model_type}")

        if risk_model is not None:
            calib_features = (
                features[calib_idx]
                if isinstance(risk_model, RiskModel)
                else features_np[calib_idx]
            )
            calib_risks = predict_risk_scores(risk_model, calib_features, device)
        else:
            calib_risks = None
        calib_errors = errors_np[calib_idx]

        if strategy == "cr2ad":
            if calib_risks is None:
                raise ValueError("CR2AD requires risk scores for conformal thresholding")
            t_alpha = conformal_threshold(calib_risks, calib_errors, float(run_cfg.decoding.alpha))
            logger.log(
                {
                    "calibration_threshold": t_alpha,
                    "calibration_alpha": float(run_cfg.decoding.alpha),
                }
            )
        elif strategy == "rahd":
            configured = OmegaConf.select(run_cfg, "decoding.risk_threshold")
            if configured is None:
                configured = OmegaConf.select(run_cfg, "decoding.rho")
            if configured is None:
                if calib_risks is None:
                    raise ValueError("RAHD requires risk scores to set risk threshold")
                risk_threshold = float(np.quantile(calib_risks, 0.8))
            else:
                risk_threshold = float(configured)

        optuna_cfg = getattr(run_cfg, "optuna", None)
        n_trials = int(OmegaConf.select(run_cfg, "optuna.n_trials") or 0)

        if strategy == "cr2ad" and optuna_cfg is not None and n_trials > 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            val_data = [calib_data[i] for i in val_idx]
            eval_subset = val_data[: min(len(val_data), 100)]

            def objective(trial: optuna.trial.Trial) -> float:
                params = sample_optuna_params(trial, optuna_cfg.search_spaces)
                alpha = params.get("alpha", float(run_cfg.decoding.alpha))
                t_trial = conformal_threshold(calib_risks, calib_errors, alpha)
                decode_params = {
                    "B": int(params.get("B", _budget_param(run_cfg, "B", 10))),
                    "k": int(params.get("k", _budget_param(run_cfg, "k", 6))),
                    "b": int(params.get("b", _budget_param(run_cfg, "b", 3))),
                    "J": int(params.get("J", _budget_param(run_cfg, "J", 2))),
                    "rollback_w": int(
                        params.get("rollback_w", _budget_param(run_cfg, "rollback_w", 8))
                    ),
                    "lookahead": int(
                        params.get("lookahead", _budget_param(run_cfg, "lookahead", 8))
                    ),
                    "lambda_logp": float(
                        params.get("lambda_logp", _decoding_param(run_cfg, "lambda_logp", 0.02))
                    ),
                }
                correct = 0
                for question, gold in eval_subset:
                    prompt = make_prompt(question)
                    result = decode_cr2ad(
                        model,
                        tokenizer,
                        prompt,
                        parse_fn,
                        risk_model=risk_model,
                        t_alpha=t_trial,
                        max_new_tokens=max_new_tokens,
                        max_length=max_length,
                        feature_names=feature_names,
                        feature_T=feature_T,
                        local_window=local_window,
                        low_margin_threshold=low_margin_threshold,
                        device=device,
                        **decode_params,
                    )
                    correct += int(result.answer == gold)
                return correct / max(1, len(eval_subset))

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_trial.params

            if "alpha" in best_params:
                run_cfg.decoding.alpha = best_params["alpha"]
                t_alpha = conformal_threshold(
                    calib_risks, calib_errors, float(run_cfg.decoding.alpha)
                )
            for key in ["B", "k", "b", "J", "rollback_w", "lookahead"]:
                if key in best_params:
                    OmegaConf.update(run_cfg, f"decoding.budget.{key}", best_params[key])
            if "lambda_logp" in best_params:
                run_cfg.decoding.lambda_logp = best_params["lambda_logp"]

            logger.log({f"optuna_best_{k}": v for k, v in best_params.items()})
            logger.log(
                {
                    "calibration_threshold": t_alpha,
                    "calibration_alpha": float(run_cfg.decoding.alpha),
                }
            )

    total_correct = 0
    correct_steps: List[int] = []
    forward_passes_list: List[int] = []
    expanded_list: List[int] = []
    rollback_counts: List[int] = []
    any_rollback_flags: List[bool] = []
    abs_errors: List[float] = []
    signed_errors: List[float] = []
    extractable = 0
    selected_risks: List[float] = []

    for step, (question, gold) in enumerate(tqdm(test_data, desc="Test", leave=False)):
        if step == 0:
            assert isinstance(question, str), "Input question must be a string"
            assert gold is not None, "Gold label missing"
        prompt = make_prompt(question)

        result: DecodeResult
        if strategy == "cr2ad":
            result = decode_cr2ad(
                model,
                tokenizer,
                prompt,
                parse_fn,
                risk_model=risk_model,
                t_alpha=float(t_alpha) if t_alpha is not None else float(run_cfg.decoding.alpha),
                B=_budget_param(run_cfg, "B", 10),
                k=_budget_param(run_cfg, "k", 6),
                b=_budget_param(run_cfg, "b", 3),
                J=_budget_param(run_cfg, "J", 2),
                rollback_w=_budget_param(run_cfg, "rollback_w", 8),
                lookahead=_budget_param(run_cfg, "lookahead", 8),
                lambda_logp=_decoding_param(run_cfg, "lambda_logp", 0.02),
                max_new_tokens=max_new_tokens,
                max_length=max_length,
                feature_names=feature_names,
                feature_T=feature_T,
                local_window=local_window,
                low_margin_threshold=low_margin_threshold,
                device=device,
            )
        elif strategy == "rahd":
            result = decode_rahd(
                model,
                tokenizer,
                prompt,
                parse_fn,
                risk_model=risk_model,
                risk_threshold=float(risk_threshold) if risk_threshold is not None else 0.5,
                B=_budget_param(run_cfg, "B", 10),
                k=_budget_param(run_cfg, "k", 6),
                b=_budget_param(run_cfg, "b", 3),
                J=_budget_param(run_cfg, "J", 2),
                lookahead=_budget_param(run_cfg, "lookahead", 8),
                lambda_logp=_decoding_param(run_cfg, "lambda_logp", 0.02),
                max_new_tokens=max_new_tokens,
                max_length=max_length,
                feature_names=feature_names,
                feature_T=feature_T,
                local_window=local_window,
                low_margin_threshold=low_margin_threshold,
                device=device,
            )
        elif strategy == "ut_cot":
            result = decode_ut_cot(
                model,
                tokenizer,
                prompt,
                parse_fn,
                risk_model=risk_model,
                tau=_decoding_param(run_cfg, "tau", 0.5),
                B=_budget_param(run_cfg, "B", 6),
                k=_budget_param(run_cfg, "k", 6),
                lambda_logp=_decoding_param(run_cfg, "lambda_logp", 0.02),
                max_new_tokens=max_new_tokens,
                max_length=max_length,
                feature_names=feature_names,
                feature_T=feature_T,
                local_window=local_window,
                low_margin_threshold=low_margin_threshold,
                device=device,
            )
        elif strategy == "cot":
            result = decode_cot_first_token(
                model,
                tokenizer,
                prompt,
                parse_fn,
                k=_budget_param(run_cfg, "k", 10),
                B=_budget_param(run_cfg, "B", 10),
                lambda_logp=_decoding_param(run_cfg, "lambda_logp", 0.02),
                max_new_tokens=max_new_tokens,
                max_length=max_length,
                device=device,
            )
        elif strategy == "greedy":
            result = decode_greedy(
                model,
                tokenizer,
                prompt,
                parse_fn,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
                device=device,
                risk_model=risk_model,
                feature_names=feature_names,
                feature_T=feature_T,
                local_window=local_window,
                low_margin_threshold=low_margin_threshold,
            )
        else:
            raise ValueError(f"Unsupported decoding strategy: {strategy}")

        pred = result.answer
        correct = int(pred == gold)
        total_correct += correct
        correct_steps.append(correct)
        forward_passes_list.append(result.forward_passes)
        expanded_list.append(result.expanded_hypotheses)
        rollback_counts.append(result.rollback_events)
        any_rollback_flags.append(result.any_rollback)

        if pred is not None:
            extractable += 1
            if isinstance(pred, int) and isinstance(gold, int):
                error = float(pred - gold)
                signed_errors.append(error)
                abs_errors.append(abs(error))
            else:
                error = None
        else:
            error = None

        running_acc = total_correct / max(1, step + 1)
        risk_val = float(result.selected_risk) if result.selected_risk is not None else float("nan")
        selected_risks.append(risk_val)
        no_rollback = int(not result.any_rollback)

        logger.log(
            {
                "test_accuracy_step": correct,
                "test_accuracy_running": running_acc,
                "test_forward_passes": result.forward_passes,
                "test_expanded_hypotheses": result.expanded_hypotheses,
                "test_rollback_events": result.rollback_events,
                "test_any_rollback": int(result.any_rollback),
                "test_selected_risk": risk_val,
                "test_no_rollback": no_rollback,
                "test_no_rollback_correct": int(no_rollback and correct == 1),
                "test_error": error,
                "test_abs_error": abs(error) if error is not None else None,
                "test_extractable": int(pred is not None),
                "test_pred": serialize_label(pred),
                "test_gold": serialize_label(gold),
            },
            step=step,
        )

    num_samples = len(test_data)
    accuracy = total_correct / max(1, num_samples)
    mean_forward_passes = float(np.mean(forward_passes_list)) if forward_passes_list else 0.0
    avg_expanded = float(np.mean(expanded_list)) if expanded_list else 0.0
    avg_rollback_events = float(np.mean(rollback_counts)) if rollback_counts else 0.0

    no_rb_idx = [i for i, flag in enumerate(any_rollback_flags) if not flag]
    coverage_no_rb = len(no_rb_idx) / max(1, num_samples)
    if no_rb_idx:
        selective_error_rate = (
            sum(1 - correct_steps[i] for i in no_rb_idx) / len(no_rb_idx)
        )
    else:
        selective_error_rate = 0.0

    extractable_rate = extractable / max(1, num_samples)
    median_abs_error = float(np.median(abs_errors)) if abs_errors else 0.0
    mean_abs_error = float(np.mean(abs_errors)) if abs_errors else 0.0
    mean_selected_risk = (
        float(np.nanmean(selected_risks)) if selected_risks else float("nan")
    )

    logger.log(
        {
            "test_accuracy": accuracy,
            "mean_forward_passes": mean_forward_passes,
            "avg_expanded_hypotheses": avg_expanded,
            "avg_rollback_events": avg_rollback_events,
            "selective_error_rate_no_rollback": selective_error_rate,
            "coverage_no_rollback": coverage_no_rb,
            "median_abs_error": median_abs_error,
            "mean_abs_error": mean_abs_error,
            "extractable_rate": extractable_rate,
            "mean_selected_risk": mean_selected_risk,
        },
        step=num_samples,
    )

    logger.summary_set("accuracy", accuracy)
    logger.summary_set("mean_forward_passes", mean_forward_passes)
    logger.summary_set("avg_expanded_hypotheses", avg_expanded)
    logger.summary_set("avg_rollback_events", avg_rollback_events)
    logger.summary_set("selective_error_rate_no_rollback", selective_error_rate)
    logger.summary_set("coverage_no_rollback", coverage_no_rb)
    logger.summary_set("median_abs_error", median_abs_error)
    logger.summary_set("mean_abs_error", mean_abs_error)
    logger.summary_set("extractable_rate", extractable_rate)
    logger.summary_set("mean_selected_risk", mean_selected_risk)
    logger.summary_set("num_test_samples", num_samples)

    logger.finish()


if __name__ == "__main__":
    main()
