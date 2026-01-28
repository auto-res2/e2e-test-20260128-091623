import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn


class ForwardCounter:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, model, input_ids: torch.Tensor):
        self.count += 1
        return model(input_ids)

    def reset(self) -> None:
        self.count = 0


def safe_eos_token_id(tokenizer) -> Optional[int]:
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    return None


@torch.no_grad()
def next_token_stats(model, input_ids: torch.Tensor, counter: ForwardCounter):
    output = counter(model, input_ids)
    logits = output.logits[:, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1)
    margin = (top2.values[0, 0] - top2.values[0, 1]).item()
    entropy = float(-(probs * (probs + 1e-12).log()).sum().item())
    return probs, margin, entropy


def compute_feature_values(
    margins: List[float],
    entropies: List[float],
    T: int,
    local_window: Optional[int] = None,
    low_margin_threshold: Optional[float] = 0.08,
) -> dict:
    if len(margins) == 0:
        return {
            "min_margin": 0.0,
            "mean_margin": 0.0,
            "mean_entropy": 0.0,
            "local_min_margin": 0.0,
            "slope_margin": 0.0,
            "first_low_margin_pos": 0.0,
        }
    tT = min(T, len(margins)) if T > 0 else len(margins)
    m = np.array(margins[:tT], dtype=np.float32)
    e = np.array(entropies[:tT], dtype=np.float32) if len(entropies) > 0 else np.zeros_like(m)
    mm = float(m.min())
    am = float(m.mean())
    me = float(e.mean()) if len(e) > 0 else 0.0
    if local_window is None:
        local_window = min(8, tT)
    w = min(max(1, int(local_window)), tT)
    local_min = float(m[-w:].min())
    slope = float(m[-1] - m[0]) if tT > 1 else 0.0
    threshold = 0.08 if low_margin_threshold is None else float(low_margin_threshold)
    first_low = float(next((i for i, x in enumerate(m) if x < threshold), tT))
    return {
        "min_margin": mm,
        "mean_margin": am,
        "mean_entropy": me,
        "local_min_margin": local_min,
        "slope_margin": slope,
        "first_low_margin_pos": first_low,
    }


def features_from_trace(
    margins: List[float],
    entropies: List[float],
    feature_names: List[str],
    T: int,
    local_window: Optional[int] = None,
    low_margin_threshold: Optional[float] = 0.08,
) -> np.ndarray:
    if not feature_names:
        raise ValueError("feature_names must be non-empty")
    if len(margins) == 0:
        return np.zeros(len(feature_names), dtype=np.float32)
    values = compute_feature_values(margins, entropies, T, local_window, low_margin_threshold)
    feats = []
    for name in feature_names:
        if name not in values:
            raise ValueError(f"Unknown feature name: {name}")
        feats.append(values[name])
    return np.array(feats, dtype=np.float32)


class RiskModel(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("std", torch.ones(in_dim))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean.clone().detach()
        self.std = std.clone().detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / (self.std + 1e-6)
        return self.linear(x).squeeze(-1)


def _as_numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


@torch.no_grad()
def predict_risk_scores(risk_model, features, device: torch.device) -> np.ndarray:
    if risk_model is None:
        raise ValueError("risk_model must be provided")
    if isinstance(risk_model, RiskModel):
        feats = (
            features
            if isinstance(features, torch.Tensor)
            else torch.tensor(features, dtype=torch.float32)
        )
        logits = risk_model(feats.to(device))
        probs = torch.sigmoid(logits)
        return probs.detach().cpu().numpy()
    if hasattr(risk_model, "predict_proba"):
        feats = _as_numpy(features)
        return risk_model.predict_proba(feats)[:, 1]
    if hasattr(risk_model, "predict"):
        feats = _as_numpy(features)
        feature_index = getattr(risk_model, "feature_index", None)
        if feature_index is not None and feats.ndim > 1:
            feats = feats[:, feature_index]
        preds = risk_model.predict(feats)
        preds = np.asarray(preds, dtype=np.float32)
        return np.clip(preds, 0.0, 1.0)
    raise ValueError("Unsupported risk model interface")


def risk_score(risk_model, feature_vec: np.ndarray, device: torch.device) -> float:
    if risk_model is None:
        raise ValueError("risk_model must be provided")
    if isinstance(risk_model, RiskModel):
        feat = torch.tensor(feature_vec, dtype=torch.float32, device=device).unsqueeze(0)
        logits = risk_model(feat)
        return float(torch.sigmoid(logits).item())
    if hasattr(risk_model, "predict_proba"):
        feat = _as_numpy(feature_vec).reshape(1, -1)
        return float(risk_model.predict_proba(feat)[0, 1])
    if hasattr(risk_model, "predict"):
        feat = _as_numpy(feature_vec)
        feature_index = getattr(risk_model, "feature_index", None)
        if feature_index is not None:
            feat_val = float(feat[feature_index])
            pred = risk_model.predict(np.array([feat_val]))[0]
        else:
            pred = risk_model.predict(feat.reshape(1, -1))[0]
        return float(np.clip(pred, 0.0, 1.0))
    raise ValueError("Unsupported risk model interface")


def conformal_threshold(risks: np.ndarray, errors: np.ndarray, alpha: float) -> float:
    pairs = sorted(zip(risks, errors), key=lambda x: x[0])
    best_t = 0.0
    err = 0
    for i, (r, e) in enumerate(pairs, start=1):
        err += int(e)
        if err / i <= alpha:
            best_t = float(r)
    return float(best_t)


def fallback_risk_from_margin(margins: List[float]) -> float:
    if not margins:
        return 0.5
    mm = float(np.min(margins))
    return float(1.0 / (1.0 + math.exp(30.0 * (mm - 0.08))))


def estimate_risk(
    risk_model,
    margins: List[float],
    entropies: List[float],
    feature_names: List[str],
    feature_T: int,
    local_window: Optional[int],
    low_margin_threshold: Optional[float],
    device: torch.device,
) -> float:
    if risk_model is None:
        return fallback_risk_from_margin(margins)
    if not feature_names:
        raise ValueError("feature_names must be provided when using a risk model")
    feat = features_from_trace(
        margins,
        entropies,
        feature_names=feature_names,
        T=feature_T,
        local_window=local_window,
        low_margin_threshold=low_margin_threshold,
    )
    return risk_score(risk_model, feat, device)


@dataclass
class Hypothesis:
    input_ids: torch.Tensor
    margins: List[float]
    entropies: List[float]
    logp: float
    branch_events: int
    ended: bool


@dataclass
class DecodeResult:
    answer: int | str | None
    forward_passes: int
    expanded_hypotheses: int
    rollback_events: int
    selected_risk: float | None
    any_rollback: bool = False


@torch.no_grad()
def greedy_extend(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    margins: List[float],
    entropies: List[float],
    logp: float,
    max_new_tokens: int,
    max_length: int,
    device: torch.device,
    counter: ForwardCounter,
) -> Tuple[torch.Tensor, List[float], List[float], float]:
    eos_id = safe_eos_token_id(tokenizer)
    for _ in range(max_new_tokens):
        if input_ids.shape[1] >= max_length:
            break
        probs, margin, entropy = next_token_stats(model, input_ids, counter)
        margins.append(margin)
        entropies.append(entropy)
        next_token = probs.argmax(dim=-1, keepdim=True)
        logp += float(torch.log(probs.gather(-1, next_token) + 1e-12).item())
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if eos_id is not None and next_token.item() == eos_id:
            break
    return input_ids, margins, entropies, logp


@torch.no_grad()
def greedy_rollout(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    max_length: int,
    device: torch.device,
) -> Tuple[str, List[float], List[float], float, int]:
    counter = ForwardCounter()
    counter.reset()
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).input_ids.to(device)
    margins: List[float] = []
    entropies: List[float] = []
    logp = 0.0
    input_ids, margins, entropies, logp = greedy_extend(
        model,
        tokenizer,
        input_ids,
        margins,
        entropies,
        logp,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        device=device,
        counter=counter,
    )
    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return text, margins, entropies, logp, counter.count


def select_best_answer(
    hyps: List[Hypothesis],
    tokenizer,
    parse_fn: Callable[[str], int | str | None],
    risk_model,
    feature_names: List[str],
    feature_T: int,
    local_window: Optional[int],
    low_margin_threshold: Optional[float],
    lambda_logp: float,
    device: torch.device,
) -> Tuple[int | str | None, float | None, int]:
    score_by_ans: dict = {}
    best_weight_by_ans: dict = {}
    risk_by_ans: dict = {}
    branch_by_ans: dict = {}

    for hyp in hyps:
        text = tokenizer.decode(hyp.input_ids[0], skip_special_tokens=True)
        ans = parse_fn(text)
        if ans is None:
            continue
        risk = None
        if risk_model is not None:
            risk = estimate_risk(
                risk_model,
                hyp.margins,
                hyp.entropies,
                feature_names,
                feature_T,
                local_window,
                low_margin_threshold,
                device,
            )
        weight = math.exp(lambda_logp * hyp.logp)
        if risk is not None:
            weight *= (1.0 - risk)
        score_by_ans[ans] = score_by_ans.get(ans, 0.0) + weight
        if weight > best_weight_by_ans.get(ans, -float("inf")):
            best_weight_by_ans[ans] = weight
            risk_by_ans[ans] = risk
            branch_by_ans[ans] = hyp.branch_events

    if not score_by_ans:
        return None, None, 0
    best_ans = max(score_by_ans.items(), key=lambda kv: kv[1])[0]
    return best_ans, risk_by_ans.get(best_ans), branch_by_ans.get(best_ans, 0)


@torch.no_grad()
def decode_greedy(
    model,
    tokenizer,
    prompt: str,
    parse_fn: Callable[[str], int | str | None],
    max_new_tokens: int,
    max_length: int,
    device: torch.device,
    risk_model=None,
    feature_names: Optional[List[str]] = None,
    feature_T: int = 24,
    local_window: Optional[int] = None,
    low_margin_threshold: Optional[float] = None,
) -> DecodeResult:
    text, margins, entropies, logp, forward_passes = greedy_rollout(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        device=device,
    )
    pred = parse_fn(text)
    selected_risk = None
    if risk_model is not None and feature_names:
        feat = features_from_trace(
            margins,
            entropies,
            feature_names,
            T=feature_T,
            local_window=local_window,
            low_margin_threshold=low_margin_threshold,
        )
        selected_risk = risk_score(risk_model, feat, device)
    return DecodeResult(pred, forward_passes, 1, 0, selected_risk, any_rollback=False)


@torch.no_grad()
def decode_cot_first_token(
    model,
    tokenizer,
    prompt: str,
    parse_fn: Callable[[str], int | str | None],
    k: int,
    B: int,
    lambda_logp: float,
    max_new_tokens: int,
    max_length: int,
    device: torch.device,
) -> DecodeResult:
    counter = ForwardCounter()
    counter.reset()
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).input_ids.to(device)
    probs, margin_now, entropy_now = next_token_stats(model, input_ids, counter)
    topk = torch.topk(probs, k=min(k, probs.shape[-1]), dim=-1)
    num_candidates = min(topk.indices.shape[1], max(1, B))
    hyps: List[Hypothesis] = []
    for i in range(num_candidates):
        next_token = topk.indices[:, i : i + 1]
        pref = torch.cat([input_ids, next_token], dim=-1)
        logp = float(torch.log(probs.gather(-1, next_token) + 1e-12).item())
        margins = [margin_now]
        entropies = [entropy_now]
        remaining = max(0, max_new_tokens - 1)
        pref, margins, entropies, logp = greedy_extend(
            model,
            tokenizer,
            pref,
            margins,
            entropies,
            logp,
            remaining,
            max_length,
            device,
            counter,
        )
        hyps.append(Hypothesis(pref, margins, entropies, logp, 0, True))

    best_ans, best_risk, _ = select_best_answer(
        hyps,
        tokenizer,
        parse_fn,
        risk_model=None,
        feature_names=[],
        feature_T=24,
        local_window=None,
        low_margin_threshold=None,
        lambda_logp=lambda_logp,
        device=device,
    )
    expanded = max(1, num_candidates)
    return DecodeResult(best_ans, counter.count, expanded, 0, best_risk, any_rollback=False)


@torch.no_grad()
def decode_ut_cot(
    model,
    tokenizer,
    prompt: str,
    parse_fn: Callable[[str], int | str | None],
    risk_model,
    tau: float,
    B: int,
    k: int,
    lambda_logp: float,
    max_new_tokens: int,
    max_length: int,
    feature_names: List[str],
    feature_T: int,
    local_window: Optional[int],
    low_margin_threshold: Optional[float],
    device: torch.device,
) -> DecodeResult:
    counter = ForwardCounter()
    counter.reset()
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).input_ids.to(device)
    margins: List[float] = []
    entropies: List[float] = []
    logp = 0.0
    eos_id = safe_eos_token_id(tokenizer)
    expanded = 1
    branched = False

    for step in range(max_new_tokens):
        if input_ids.shape[1] >= max_length:
            break
        probs, margin, entropy = next_token_stats(model, input_ids, counter)
        temp_m = margins + [margin]
        temp_e = entropies + [entropy]
        risk_val = estimate_risk(
            risk_model,
            temp_m,
            temp_e,
            feature_names,
            feature_T,
            local_window,
            low_margin_threshold,
            device,
        )
        if (not branched) and risk_val > tau and expanded < B:
            topk = torch.topk(probs, k=min(k, probs.shape[-1]), dim=-1)
            remaining = max(0, B - expanded)
            num_candidates = min(topk.indices.shape[1], remaining)
            if num_candidates > 0:
                hyps: List[Hypothesis] = []
                for i in range(num_candidates):
                    next_token = topk.indices[:, i : i + 1]
                    pref = torch.cat([input_ids, next_token], dim=-1)
                    logp2 = logp + float(
                        torch.log(probs.gather(-1, next_token) + 1e-12).item()
                    )
                    m2 = margins + [margin]
                    e2 = entropies + [entropy]
                    remaining_tokens = max(0, max_new_tokens - step - 1)
                    pref, m2, e2, logp2 = greedy_extend(
                        model,
                        tokenizer,
                        pref,
                        m2,
                        e2,
                        logp2,
                        remaining_tokens,
                        max_length,
                        device,
                        counter,
                    )
                    hyps.append(Hypothesis(pref, m2, e2, logp2, 1, True))
                expanded += num_candidates
                best_ans, best_risk, _ = select_best_answer(
                    hyps,
                    tokenizer,
                    parse_fn,
                    risk_model,
                    feature_names,
                    feature_T,
                    local_window,
                    low_margin_threshold,
                    lambda_logp,
                    device,
                )
                return DecodeResult(best_ans, counter.count, expanded, 0, best_risk, any_rollback=False)
            branched = True

        next_token = probs.argmax(dim=-1, keepdim=True)
        logp += float(torch.log(probs.gather(-1, next_token) + 1e-12).item())
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        margins.append(margin)
        entropies.append(entropy)
        if eos_id is not None and next_token.item() == eos_id:
            break

    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    pred = parse_fn(text)
    selected_risk = None
    if risk_model is not None and feature_names:
        selected_risk = estimate_risk(
            risk_model,
            margins,
            entropies,
            feature_names,
            feature_T,
            local_window,
            low_margin_threshold,
            device,
        )
    return DecodeResult(pred, counter.count, expanded, 0, selected_risk, any_rollback=False)


@torch.no_grad()
def decode_rahd(
    model,
    tokenizer,
    prompt: str,
    parse_fn: Callable[[str], int | str | None],
    risk_model,
    risk_threshold: float,
    B: int,
    k: int,
    b: int,
    J: int,
    lookahead: int,
    lambda_logp: float,
    max_new_tokens: int,
    max_length: int,
    feature_names: List[str],
    feature_T: int,
    local_window: Optional[int],
    low_margin_threshold: Optional[float],
    device: torch.device,
) -> DecodeResult:
    counter = ForwardCounter()
    counter.reset()
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).input_ids.to(device)
    base_len = input_ids.shape[1]
    eos_id = safe_eos_token_id(tokenizer)

    hyps = [Hypothesis(input_ids, [], [], 0.0, 0, False)]
    expanded = 1

    for _ in range(max_new_tokens):
        new_hyps: List[Hypothesis] = []
        all_ended = True
        for hyp in hyps:
            if hyp.ended:
                new_hyps.append(hyp)
                continue
            all_ended = False

            risk_val = estimate_risk(
                risk_model,
                hyp.margins,
                hyp.entropies,
                feature_names,
                feature_T,
                local_window,
                low_margin_threshold,
                device,
            )
            can_branch = (
                risk_val > risk_threshold
                and hyp.branch_events < J
                and expanded < B
            )
            if can_branch:
                probs, margin_now, entropy_now = next_token_stats(model, hyp.input_ids, counter)
                topk = torch.topk(probs, k=min(k, probs.shape[-1]), dim=-1)
                remaining = max(0, B - expanded)
                num_candidates = min(topk.indices.shape[1], remaining)
                if num_candidates > 0:
                    candidates = []
                    for i in range(num_candidates):
                        next_token = topk.indices[:, i : i + 1]
                        pref2 = torch.cat([hyp.input_ids, next_token], dim=-1)
                        logp2 = hyp.logp + float(
                            torch.log(probs.gather(-1, next_token) + 1e-12).item()
                        )
                        m2 = hyp.margins + [margin_now]
                        e2 = hyp.entropies + [entropy_now]

                        tmp_ids = pref2
                        tmp_m, tmp_e, tmp_logp = list(m2), list(e2), logp2
                        for _ in range(lookahead):
                            if tmp_ids.shape[1] >= max_length:
                                break
                            probs2, margin2, entropy2 = next_token_stats(model, tmp_ids, counter)
                            tmp_m.append(margin2)
                            tmp_e.append(entropy2)
                            next_token2 = probs2.argmax(dim=-1, keepdim=True)
                            tmp_logp += float(
                                torch.log(probs2.gather(-1, next_token2) + 1e-12).item()
                            )
                            tmp_ids = torch.cat([tmp_ids, next_token2], dim=-1)
                            if eos_id is not None and next_token2.item() == eos_id:
                                break
                        risk2 = estimate_risk(
                            risk_model,
                            tmp_m,
                            tmp_e,
                            feature_names,
                            feature_T,
                            local_window,
                            low_margin_threshold,
                            device,
                        )
                        score = (risk_val - risk2) + lambda_logp * (
                            tmp_logp / max(1, tmp_ids.shape[1] - base_len)
                        )
                        candidates.append((score, pref2, m2, e2, logp2))
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    keep = candidates[: min(b, len(candidates))]
                    for _, pref2, m2, e2, lp2 in keep:
                        new_hyps.append(Hypothesis(pref2, m2, e2, lp2, hyp.branch_events + 1, False))
                    expanded += num_candidates
                    continue

            if hyp.input_ids.shape[1] >= max_length:
                new_hyps.append(Hypothesis(hyp.input_ids, hyp.margins, hyp.entropies, hyp.logp, hyp.branch_events, True))
                continue

            probs, margin, entropy = next_token_stats(model, hyp.input_ids, counter)
            next_token = probs.argmax(dim=-1, keepdim=True)
            logp2 = hyp.logp + float(torch.log(probs.gather(-1, next_token) + 1e-12).item())
            ids2 = torch.cat([hyp.input_ids, next_token], dim=-1)
            ended = eos_id is not None and next_token.item() == eos_id
            new_hyps.append(
                Hypothesis(ids2, hyp.margins + [margin], hyp.entropies + [entropy], logp2, hyp.branch_events, ended)
            )

        hyps = new_hyps
        if len(hyps) > B:
            hyps.sort(key=lambda h: h.logp, reverse=True)
            hyps = hyps[:B]
        if all_ended:
            break

    best_ans, best_risk, _ = select_best_answer(
        hyps,
        tokenizer,
        parse_fn,
        risk_model,
        feature_names,
        feature_T,
        local_window,
        low_margin_threshold,
        lambda_logp,
        device,
    )
    return DecodeResult(best_ans, counter.count, expanded, 0, best_risk, any_rollback=False)


@torch.no_grad()
def decode_cr2ad(
    model,
    tokenizer,
    prompt: str,
    parse_fn: Callable[[str], int | str | None],
    risk_model,
    t_alpha: float,
    B: int,
    k: int,
    b: int,
    J: int,
    rollback_w: int,
    lookahead: int,
    lambda_logp: float,
    max_new_tokens: int,
    max_length: int,
    feature_names: List[str],
    feature_T: int,
    local_window: Optional[int],
    low_margin_threshold: Optional[float],
    device: torch.device,
) -> DecodeResult:
    assert risk_model is not None, "Risk model required for CR2AD"
    counter = ForwardCounter()
    counter.reset()
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).input_ids.to(device)
    base_len = input_ids.shape[1]
    eos_id = safe_eos_token_id(tokenizer)

    hyps = [Hypothesis(input_ids, [], [], 0.0, 0, False)]
    expanded = 1
    rollback_events_total = 0

    for _ in range(max_new_tokens):
        new_hyps: List[Hypothesis] = []
        all_ended = True
        for hyp in hyps:
            if hyp.ended:
                new_hyps.append(hyp)
                continue
            all_ended = False

            risk_val = estimate_risk(
                risk_model,
                hyp.margins,
                hyp.entropies,
                feature_names,
                feature_T,
                local_window,
                low_margin_threshold,
                device,
            )
            can_rethink = (
                risk_val > t_alpha
                and hyp.branch_events < J
                and expanded < B
                and len(hyp.margins) > 0
            )

            if can_rethink:
                w = min(rollback_w, len(hyp.margins))
                local_idx = int(np.argmin(np.array(hyp.margins[-w:], dtype=np.float32)))
                g = (len(hyp.margins) - w) + local_idx
                trunc_len = min(base_len + g, hyp.input_ids.shape[1])
                prefix = hyp.input_ids[:, :trunc_len]
                if prefix.shape[1] >= max_length:
                    new_hyps.append(hyp)
                    continue

                probs, margin_now, entropy_now = next_token_stats(model, prefix, counter)
                topk = torch.topk(probs, k=min(k, probs.shape[-1]), dim=-1)

                remaining = max(0, B - expanded)
                num_candidates = min(topk.indices.shape[1], remaining)
                if num_candidates > 0:
                    rollback_events_total += 1
                    candidates = []
                    for i in range(num_candidates):
                        next_token = topk.indices[:, i : i + 1]
                        pref2 = torch.cat([prefix, next_token], dim=-1)
                        logp2 = hyp.logp + float(
                            torch.log(probs.gather(-1, next_token) + 1e-12).item()
                        )
                        margins2 = hyp.margins[:g] + [margin_now]
                        entropies2 = hyp.entropies[:g] + [entropy_now]

                        tmp_ids = pref2
                        tmp_m, tmp_e, tmp_logp = list(margins2), list(entropies2), logp2
                        for _ in range(lookahead):
                            if tmp_ids.shape[1] >= max_length:
                                break
                            probs2, margin2, entropy2 = next_token_stats(model, tmp_ids, counter)
                            tmp_m.append(margin2)
                            tmp_e.append(entropy2)
                            next_token2 = probs2.argmax(dim=-1, keepdim=True)
                            tmp_logp += float(
                                torch.log(probs2.gather(-1, next_token2) + 1e-12).item()
                            )
                            tmp_ids = torch.cat([tmp_ids, next_token2], dim=-1)
                            if eos_id is not None and next_token2.item() == eos_id:
                                break
                        risk2 = estimate_risk(
                            risk_model,
                            tmp_m,
                            tmp_e,
                            feature_names,
                            feature_T,
                            local_window,
                            low_margin_threshold,
                            device,
                        )
                        score = (risk_val - risk2) + lambda_logp * (
                            tmp_logp / max(1, tmp_ids.shape[1] - base_len)
                        )
                        candidates.append((score, pref2, margins2, entropies2, logp2))

                    candidates.sort(key=lambda x: x[0], reverse=True)
                    keep = candidates[: min(b, len(candidates))]
                    for _, pref2, m2, e2, lp2 in keep:
                        new_hyps.append(Hypothesis(pref2, m2, e2, lp2, hyp.branch_events + 1, False))
                    expanded += num_candidates
                    continue

            if hyp.input_ids.shape[1] >= max_length:
                new_hyps.append(Hypothesis(hyp.input_ids, hyp.margins, hyp.entropies, hyp.logp, hyp.branch_events, True))
                continue

            probs, margin, entropy = next_token_stats(model, hyp.input_ids, counter)
            next_token = probs.argmax(dim=-1, keepdim=True)
            logp2 = hyp.logp + float(torch.log(probs.gather(-1, next_token) + 1e-12).item())
            ids2 = torch.cat([hyp.input_ids, next_token], dim=-1)
            ended = eos_id is not None and next_token.item() == eos_id
            new_hyps.append(
                Hypothesis(ids2, hyp.margins + [margin], hyp.entropies + [entropy], logp2, hyp.branch_events, ended)
            )

        hyps = new_hyps
        if len(hyps) > B:
            hyps.sort(key=lambda h: h.logp, reverse=True)
            hyps = hyps[:B]
        if all_ended:
            break

    best_ans, best_risk, _ = select_best_answer(
        hyps,
        tokenizer,
        parse_fn,
        risk_model,
        feature_names,
        feature_T,
        local_window,
        low_margin_threshold,
        lambda_logp,
        device,
    )
    any_rollback = rollback_events_total > 0
    return DecodeResult(
        best_ans,
        counter.count,
        expanded,
        rollback_events_total,
        best_risk,
        any_rollback=any_rollback,
    )
