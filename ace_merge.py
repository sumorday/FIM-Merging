"""
ACE-Merging for Long-to-Short (Two-Model Scenario)
====================================================
Based on: "Adaptive Covariance Estimation for Data-Free Model Merging"
arXiv:2603.02945

Core idea: Use the covariance of weight displacements (delta W) as a
data-free proxy for input covariance, then merge via weighted least squares.

Theorem 1 (ACE): Input covariance Sigma_t is proportional to the covariance
of weight displacements:
    Sigma_t proportional to Cov(Delta_W_t)

For two-model L2S merging (base + ft), the ACE solution simplifies to:
    W_merged = W_base + delta * sigma / (sigma + cagg)

Usage:
    python ace_merge.py \
        --base_model /path/to/base \
        --ft_model /path/to/ft \
        --save_path /path/to/save \
        --threshold_ratio 0.2 \
        --epsilon 0.01
"""

import torch
import re
import numpy as np
import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def compute_ace_covariance(delta_w, epsilon=0.01, gamma_threshold=0.3):
    """
    Compute ACE covariance estimate from weight displacement delta_w.
    Returns sigma_diag [d_in] and gamma (heterogeneity metric).
    """
    d_out, d_in = delta_w.shape

    # Column-wise mean and centering
    mu = delta_w.mean(dim=0, keepdim=True)
    delta_centered = delta_w - mu

    # Diagonal covariance approximation (memory efficient)
    sigma_diag = (delta_centered ** 2).mean(dim=0)  # [d_in]

    # Heterogeneity metric gamma
    col_norms_sq = (delta_w ** 2).sum(dim=0)  # [d_in]
    gamma = col_norms_sq.var().item() / (col_norms_sq.mean().item() ** 2 + 1e-10)

    # Adaptive normalization
    trace_val = sigma_diag.sum().item()
    if gamma > gamma_threshold and trace_val > 1e-10:
        sigma_diag = sigma_diag / trace_val
        eps_adjusted = epsilon / trace_val
    else:
        eps_adjusted = epsilon

    # Tikhonov regularization
    sigma_diag = sigma_diag + eps_adjusted

    return sigma_diag, gamma


def compute_collective_prior(all_sigma_diags):
    """
    Collective Structural Prior (CSP): mean sigma across all layers.
    Returns a scalar (mean of all means) for cross-layer dimension mismatch.
    """
    if not all_sigma_diags:
        return None
    all_means = [s.mean().item() for s in all_sigma_diags]
    return float(np.mean(all_means))


def ace_merge_layer(base_w, ft_w, epsilon=0.01, gamma_threshold=0.3,
                    cagg_scalar=None):
    """
    ACE merge for a single weight matrix.
    W_merged = W_base + delta * (sigma / (sigma + cagg))
    """
    delta = ft_w - base_w

    sigma_diag, gamma = compute_ace_covariance(
        delta.float(), epsilon=epsilon, gamma_threshold=gamma_threshold)

    if cagg_scalar is not None:
        weight = sigma_diag / (sigma_diag + cagg_scalar + 1e-10)
    else:
        weight = sigma_diag / (sigma_diag + epsilon + 1e-10)

    # Apply per-column weight: merged = base + delta * weight
    merged_w = base_w.float() + delta.float() * weight.unsqueeze(0)

    return merged_w, gamma, weight.mean().item()


def ties_trimming(delta, threshold_ratio=0.2):
    """Standard TIES trimming: keep top threshold_ratio by magnitude."""
    flat = delta.abs().flatten()
    k = max(1, int(flat.numel() * threshold_ratio))
    thresh = flat.kthvalue(flat.numel() - k + 1).values.item()
    trimmed = delta.clone()
    trimmed[delta.abs() < thresh] = 0.0
    return trimmed


def residual_norm_calibration(merged_w, base_w, n_probe=4, drift_threshold=0.05):
    torch.manual_seed(0)
    in_dim = base_w.shape[1]
    probe = torch.randn(n_probe, in_dim)
    base_norm = (probe @ base_w.T).norm(dim=-1).mean().item()
    merged_norm = (probe @ merged_w.T).norm(dim=-1).mean().item()
    if base_norm < 1e-8:
        return merged_w
    drift = abs(merged_norm - base_norm) / base_norm
    if drift > drift_threshold:
        scale = base_norm / (merged_norm + 1e-10)
        merged_w = merged_w * scale
    return merged_w


def ace_merge(base_model_path, ft_model_path, save_path,
              epsilon=0.01, gamma_threshold=0.3,
              threshold_ratio=0.2,
              gate_protection_factor=0.7,
              norm_drift_threshold=0.05,
              use_ties=True):
    """ACE-Merging for two-model L2S scenario."""
    logger.info("=" * 60)
    logger.info("ACE-MERGING (Data-Free Covariance Estimation)")
    logger.info(f"epsilon={epsilon}, gamma_threshold={gamma_threshold}")
    logger.info(f"TIES threshold_ratio={threshold_ratio}")
    logger.info("=" * 60)

    logger.info(f"Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map='cpu')
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    logger.info(f"Loading FT model: {ft_model_path}")
    ft_model = AutoModelForCausalLM.from_pretrained(
        ft_model_path, torch_dtype=torch.bfloat16, device_map='cpu')

    new_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map='cpu')

    base_mods = {n: m for n, m in base_model.named_modules()
                 if hasattr(m, 'weight') and m.weight is not None}
    ft_mods = {n: m for n, m in ft_model.named_modules()
               if hasattr(m, 'weight') and m.weight is not None}
    new_mods = {n: m for n, m in new_model.named_modules()
                if hasattr(m, 'weight') and m.weight is not None}

    # Pass 1: Collect sigma diags for CSP
    logger.info("Pass 1: Computing per-layer covariances for CSP...")
    all_sigma_diags = []

    for name in base_mods:
        if name not in ft_mods:
            continue
        base_w = base_mods[name].weight.data.float()
        ft_w = ft_mods[name].weight.data.float()
        if base_w.shape != ft_w.shape or base_w.ndim != 2:
            continue
        number = _get_layer_number(name)
        if number is None:
            continue
        delta = ft_w - base_w
        sigma_diag, _ = compute_ace_covariance(
            delta, epsilon=epsilon, gamma_threshold=gamma_threshold)
        all_sigma_diags.append(sigma_diag)

    cagg_scalar = compute_collective_prior(all_sigma_diags)
    logger.info(f"CSP computed from {len(all_sigma_diags)} layers")
    logger.info(f"CSP scalar (mean sigma)={cagg_scalar:.6e}")

    # Pass 2: Merge
    logger.info("Pass 2: Merging layers with ACE...")
    norm_calibrated = 0
    avg_weights = []

    for name in new_mods:
        if name not in base_mods or name not in ft_mods:
            continue
        base_w = base_mods[name].weight.data.float()
        ft_w = ft_mods[name].weight.data.float()
        if base_w.shape != ft_w.shape:
            continue

        if base_w.ndim == 2:
            delta = ft_w - base_w

            if use_ties:
                delta = ties_trimming(delta, threshold_ratio=threshold_ratio)

            if _is_gate_proj(name):
                delta = delta * gate_protection_factor

            merged_w, gamma, avg_w = ace_merge_layer(
                base_w, base_w + delta,
                epsilon=epsilon,
                gamma_threshold=gamma_threshold,
                cagg_scalar=cagg_scalar)

            avg_weights.append(avg_w)

            merged_w_cal = residual_norm_calibration(
                merged_w, base_w, drift_threshold=norm_drift_threshold)
            if not torch.allclose(merged_w_cal, merged_w, atol=1e-6):
                norm_calibrated += 1
            merged_w = merged_w_cal

        else:
            merged_w = (base_w + ft_w) / 2.0

        new_mods[name].weight.data = merged_w.to(torch.bfloat16)

        if (hasattr(base_mods[name], 'bias') and
                base_mods[name].bias is not None and
                hasattr(ft_mods[name], 'bias') and
                ft_mods[name].bias is not None):
            base_b = base_mods[name].bias.data.float()
            ft_b = ft_mods[name].bias.data.float()
            new_mods[name].bias.data = (
                (base_b + ft_b) / 2.0).to(torch.bfloat16)

    logger.info(f"Average ACE weight: {np.mean(avg_weights):.4f} "
                f"(std={np.std(avg_weights):.4f})")
    logger.info(f"Norm calibration applied to {norm_calibrated} modules")
    logger.info(f"Saving to {save_path}")
    new_model.save_pretrained(save_path, torch_dtype=torch.bfloat16)
    tokenizer.save_pretrained(save_path)
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ACE-Merging for L2S")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--ft_model", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--gamma_threshold", type=float, default=0.3)
    parser.add_argument("--threshold_ratio", type=float, default=0.2)
    parser.add_argument("--gate_protection_factor", type=float, default=0.7)
    parser.add_argument("--norm_drift_threshold", type=float, default=0.05)
    parser.add_argument("--no_ties", action="store_true")
    args = parser.parse_args()

    ace_merge(
        base_model_path=args.base_model,
        ft_model_path=args.ft_model,
        save_path=args.save_path,
        epsilon=args.epsilon,
        gamma_threshold=args.gamma_threshold,
        threshold_ratio=args.threshold_ratio,
        gate_protection_factor=args.gate_protection_factor,
        norm_drift_threshold=args.norm_drift_threshold,
        use_ties=not args.no_ties,
    )