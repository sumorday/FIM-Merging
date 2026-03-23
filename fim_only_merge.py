"""
FIM-Only Proxy Merge (Ablation)
================================
Same as FIM-Merging but uses ONLY diagonal FIM as layer importance signal,
WITHOUT the ||delta||^2 weighting.

This is used to ablate the contribution of the task vector norm term in
Proposition 1, which states that merging error is bounded by:
    E(alpha) <= alpha(1-alpha)/2 * ||delta||^2 * ||H_f||

If FIM x ||delta||^2 > FIM-only, it validates that BOTH terms matter.

Usage:
    python fim_only_merge.py \
        --base_model /path/to/base \
        --ft_model /path/to/ft \
        --save_path /path/to/save \
        --merging_method ties \
        --threshold_ratio 0.2 \
        --device cpu
"""

import torch
import re
import numpy as np
import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_layer_number(name):
    if "embed" in name:
        return -1
    elif "model.norm" in name:
        return -2
    elif "lm_head" in name:
        return -3
    else:
        nums = re.findall(r'\d+', name)
        if not nums:
            return None
        return int(nums[0])


def _is_gate_proj(name):
    return "gate_proj" in name


def compute_fim_scores(base_model, base_model_path,
                       n_samples=8, seq_len=64, device='cpu'):
    logger.info("Computing diagonal FIM scores...")
    config = AutoConfig.from_pretrained(base_model_path)
    vocab_size = config.vocab_size

    model = base_model.to(device)
    model.train()

    torch.manual_seed(42)
    random_input_ids = torch.randint(0, vocab_size, (n_samples, seq_len))

    fisher_diag = {name: torch.zeros_like(param.data, dtype=torch.float32)
                   for name, param in model.named_parameters()}

    for i in range(n_samples):
        model.zero_grad()
        input_ids = random_input_ids[i:i+1].to(device)
        outputs = model(input_ids, labels=input_ids)
        outputs.loss.backward()
        logger.info(f"  FIM sample {i+1}/{n_samples} done")
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_diag[name] += param.grad.detach().float() ** 2

    for name in fisher_diag:
        fisher_diag[name] /= n_samples

    model.cpu()
    torch.cuda.empty_cache()

    layer_fim = {}
    for name, fim in fisher_diag.items():
        number = _get_layer_number(name)
        if number is None:
            continue
        layer_fim.setdefault(number, []).append(fim.mean().item())

    layer_fim_final = {k: float(np.mean(v)) for k, v in layer_fim.items()}

    logger.info("FIM per layer:")
    for k in sorted(layer_fim_final):
        logger.info(f"  Layer {k:3d}: {layer_fim_final[k]:.6e}")

    return layer_fim_final


def compute_delta_norms(base_model, ft_model):
    logger.info("Computing ||Delta||^2 per layer...")
    base_params = dict(base_model.named_parameters())
    ft_params   = dict(ft_model.named_parameters())

    layer_delta = {}
    for name in base_params:
        if name not in ft_params:
            continue
        if base_params[name].shape != ft_params[name].shape:
            continue
        number = _get_layer_number(name)
        if number is None:
            continue
        delta = ft_params[name].float() - base_params[name].float()
        layer_delta.setdefault(number, []).append((delta ** 2).mean().item())

    return {k: float(np.mean(v)) for k, v in layer_delta.items()}


def compute_alpha_fim_only(layer_fim, layer_delta):
    """
    FIM-ONLY proxy: uses ONLY diagonal FIM, ignores ||delta||^2.
    This is the ablation version of compute_alpha.

    In the full FIM-Merging:
        importance = FIM * ||delta||^2   (Proposition 1 motivated)

    In this ablation:
        importance = FIM only            (ignores task vector norm)
    """
    eps = 1e-10

    keys = set(layer_fim) & set(layer_delta)

    # KEY DIFFERENCE: only use FIM, not FIM * delta^2
    importance = {k: layer_fim[k] for k in keys}

    log_imp  = {k: float(np.log(v + eps)) for k, v in importance.items()}
    t_log    = {k: v for k, v in log_imp.items() if k >= 0}
    t_vals   = np.array(list(t_log.values()))

    median_log   = float(np.median(t_vals))
    log_imp_norm = {k: v - median_log for k, v in log_imp.items()}

    t_norm_vals = np.array([log_imp_norm[k] for k in t_log])
    t_range     = float(t_norm_vals.max() - t_norm_vals.min())
    theta       = 1.0 / (t_range + eps)
    max_t_norm  = float(t_norm_vals.max())

    logger.info(f"[FIM-ONLY] Transformer importance: median={median_log:.4f}, "
                f"range={t_range:.4f}, theta={theta:.6f}")

    layer_alpha = {}
    for k, lnorm in log_imp_norm.items():
        t_x = 1.0 / (1.0 + np.exp(-theta * (lnorm - max_t_norm)))
        layer_alpha[k] = float(1.0 - t_x)

    logger.info("[FIM-ONLY] Alpha per layer:")
    for k in sorted(layer_alpha):
        logger.info(f"  Layer {k:3d}: alpha={layer_alpha[k]:.4f}")

    return layer_alpha


def ties_resolve(delta, fim_weight=None, threshold_ratio=0.4):
    importance = (fim_weight * delta.abs()) if fim_weight is not None else delta.abs()
    flat  = importance.flatten()
    k     = max(1, int(flat.numel() * threshold_ratio))
    thresh = flat.kthvalue(flat.numel() - k + 1).values.item()
    trimmed = delta.clone()
    trimmed[importance < thresh] = 0.0
    return trimmed


def residual_norm_calibration(merged_w, base_w, n_probe=4, drift_threshold=0.05):
    torch.manual_seed(0)
    in_dim = base_w.shape[1]
    probe  = torch.randn(n_probe, in_dim)
    base_norm   = (probe @ base_w.T).norm(dim=-1).mean().item()
    merged_norm = (probe @ merged_w.T).norm(dim=-1).mean().item()
    if base_norm < 1e-8:
        return merged_w
    drift = abs(merged_norm - base_norm) / base_norm
    if drift > drift_threshold:
        scale    = base_norm / (merged_norm + 1e-10)
        merged_w = merged_w * scale
    return merged_w


def fim_only_merge(base_model_path, ft_model_path, save_path,
                   merging_method='ties', device='cpu',
                   n_samples=8, seq_len=64,
                   threshold_ratio=0.2,
                   gate_protection_factor=0.7,
                   norm_drift_threshold=0.05):
    """
    FIM-Only Merge for ablation study.
    Uses FIM alone (no ||delta||^2) as layer importance signal.
    All other settings identical to full FIM-Merging.
    """
    logger.info("=" * 60)
    logger.info("FIM-ONLY PROXY MERGE (Ablation)")
    logger.info("Importance = FIM only (NOT FIM x ||delta||^2)")
    logger.info("=" * 60)

    logger.info(f"Base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map='cpu')
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    logger.info(f"FT model: {ft_model_path}")
    ft_model = AutoModelForCausalLM.from_pretrained(
        ft_model_path, torch_dtype=torch.bfloat16, device_map='cpu')

    fim_scores  = compute_fim_scores(base_model, base_model_path,
                                     n_samples=n_samples, seq_len=seq_len,
                                     device=device)
    delta_norms = compute_delta_norms(base_model, ft_model)

    # KEY: use FIM-only alpha, not the full FIM x delta^2
    layer_alpha = compute_alpha_fim_only(fim_scores, delta_norms)

    logger.info(f"Merging method: {merging_method.upper()}")
    logger.info(f"TIES threshold_ratio: {threshold_ratio}")
    logger.info(f"Gate protection factor: {gate_protection_factor}")

    base_mods = {n: m for n, m in base_model.named_modules()
                 if hasattr(m, 'weight') and m.weight is not None}
    ft_mods   = {n: m for n, m in ft_model.named_modules()
                 if hasattr(m, 'weight') and m.weight is not None}

    new_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map='cpu')
    new_mods  = {n: m for n, m in new_model.named_modules()
                 if hasattr(m, 'weight') and m.weight is not None}

    norm_calibrated = 0

    for name in new_mods:
        if name not in base_mods or name not in ft_mods:
            continue
        number = _get_layer_number(name)
        if number is None or number not in layer_alpha:
            continue

        alpha = layer_alpha[number]
        if _is_gate_proj(name):
            alpha = alpha * gate_protection_factor

        base_w = base_mods[name].weight.data.float()
        ft_w   = ft_mods[name].weight.data.float()
        delta  = ft_w - base_w

        if merging_method == 'ties':
            fim_val = fim_scores.get(number, None)
            fim_t   = torch.full_like(delta, fim_val) if fim_val is not None else None
            delta   = ties_resolve(delta, fim_weight=fim_t,
                                   threshold_ratio=threshold_ratio)

        merged_w = base_w + alpha * delta

        if base_w.ndim == 2:
            merged_w_cal = residual_norm_calibration(
                merged_w, base_w, drift_threshold=norm_drift_threshold)
            if not torch.allclose(merged_w_cal, merged_w, atol=1e-6):
                norm_calibrated += 1
            merged_w = merged_w_cal

        new_mods[name].weight.data = merged_w.to(torch.bfloat16)

        if (hasattr(base_mods[name], 'bias') and base_mods[name].bias is not None
                and hasattr(ft_mods[name], 'bias') and ft_mods[name].bias is not None):
            base_b = base_mods[name].bias.data.float()
            ft_b   = ft_mods[name].bias.data.float()
            new_mods[name].bias.data = (
                base_b + alpha * (ft_b - base_b)).to(torch.bfloat16)

    logger.info(f"Norm calibration applied to {norm_calibrated} modules")
    logger.info(f"Saving to {save_path}")
    new_model.save_pretrained(save_path, torch_dtype=torch.bfloat16)
    tokenizer.save_pretrained(save_path)
    logger.info("Done!")

    return new_model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FIM-Only Proxy Merge for ablation study")
    parser.add_argument("--base_model",             type=str, required=True)
    parser.add_argument("--ft_model",               type=str, required=True)
    parser.add_argument("--save_path",              type=str, required=True)
    parser.add_argument("--merging_method",         type=str, default="ties",
                        choices=["ta", "ties"])
    parser.add_argument("--device",                 type=str, default="cpu")
    parser.add_argument("--n_samples",              type=int, default=8)
    parser.add_argument("--seq_len",                type=int, default=64)
    parser.add_argument("--threshold_ratio",        type=float, default=0.2,
                        help="0.2 for 1.5B")
    parser.add_argument("--gate_protection_factor", type=float, default=0.7)
    parser.add_argument("--norm_drift_threshold",   type=float, default=0.05)
    args = parser.parse_args()

    fim_only_merge(
        base_model_path=args.base_model,
        ft_model_path=args.ft_model,
        save_path=args.save_path,
        merging_method=args.merging_method,
        device=args.device,
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        threshold_ratio=args.threshold_ratio,
        gate_protection_factor=args.gate_protection_factor,
        norm_drift_threshold=args.norm_drift_threshold,
    )
