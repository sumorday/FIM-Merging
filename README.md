# FIM-Merging: Data-Free Layer-Adaptive Merging via Fisher Information for Long-to-Short Reasoning LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2503.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2603.21705)

## Overview

FIM-Merging is a **data-free** layer-adaptive model merging method for Long-to-Short (L2S) reasoning LLMs. It uses the diagonal Fisher Information Matrix (FIM) computed on random token inputs to assign per-layer merging coefficients, without requiring any domain-specific calibration data.

**Key results:**
- MATH500: **90.2** (+6.2 over ACM-TIES) on 7B scale
- AIME24: **36.7%** with self-consistency (n=16), surpassing ACM-TIES (33.3%) without any calibration data
- Response length reduced by **92.6%** on 1.5B scale while improving accuracy

## Method

FIM-Merging computes layer importance as:

$$s^l = \log(\hat{\mathcal{F}}^l \cdot \|\delta^l\|^2)$$

where $\hat{\mathcal{F}}^l$ is the diagonal FIM computed with random token inputs and $\|\delta^l\|^2$ is the task vector norm. This directly instantiates the Hessian bound from Proposition 1 in our paper.

## Installation

```bash
pip install torch transformers vllm
```

## Usage

### Step 1: Merge Models (FIM-TIES)

```bash
# For 1.5B models
python fim_merge.py \
    --base_model /path/to/Qwen2.5-Math-1.5B \
    --ft_model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
    --save_path /path/to/merged_model \
    --merging_method ties \
    --threshold_ratio 0.2 \
    --n_samples 8 \
    --seq_len 64 \
    --device cpu

# For 7B models
python fim_merge.py \
    --base_model /path/to/Qwen2.5-Math-7B \
    --ft_model /path/to/DeepSeek-R1-Distill-Qwen-7B \
    --save_path /path/to/merged_model \
    --merging_method ties \
    --threshold_ratio 0.4 \
    --n_samples 8 \
    --seq_len 64 \
    --device cpu
```

### Step 2: Evaluate

```bash
# Full benchmark evaluation
bash run_eval.sh \
    /path/to/merged_model \
    /path/to/merged_model/math_eval \
    eval.log \
    0

# Self-consistency on AIME24 (recommended n=16, temperature=0.3)
python eval_self_consistency.py \
    --model_path /path/to/merged_model \
    --data_dir /path/to/evaluation/data \
    --output_dir /path/to/results \
    --n_samples 16 \
    --temperature 0.3 \
    --device cuda:0
```

## Key Hyperparameters

| Parameter | 1.5B | 7B | Description |
|-----------|------|----|-------------|
| `threshold_ratio` | 0.2 | 0.4 | TIES pruning ratio (scale-dependent) |
| `gate_protection_factor` | 0.7 | 0.7 | Gate projection protection |
| `n_samples` | 8 | 8 | Number of random inputs for FIM |
| `seq_len` | 64 | 64 | Sequence length for FIM computation |

## Main Results

### 7B Scale

| Method | GSM8K | MATH500 | Minerva | Olympiad | College | AIME24 |
|--------|-------|---------|---------|----------|---------|--------|
| ACM-TIES | 92.2 | 84.0 | 38.6 | 46.4 | 40.3 | 33.3 |
| **FIM-TIES (ours)** | **92.2** | **90.2** | **41.9** | **47.9** | **40.7** | 26.7 |
| **FIM-TIES + SC (n=16)** | - | - | - | - | - | **36.7** |

### 1.5B Scale

| Method | GSM8K | MATH500 | Minerva | Olympiad | College | AIME24 | Avg. Length |
|--------|-------|---------|---------|----------|---------|--------|-------------|
| ACM-TIES | 78.4 | 71.4 | 28.7 | 33.8 | 37.9 | 10.0 | 1,489 |
| **FIM-TIES (ours)** | **81.6** | **74.9** | **30.8** | **36.3** | **40.5** | **20.0** | **411** |

## Models

- Base model: [Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B)
- FT model: [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)

## Citation

```bibtex
@article{xia2025fim,
  title={Data-Free Layer-Adaptive Merging via Fisher Information for Long-to-Short Reasoning LLMs},
  author={Xia, Tian},
  journal={arXiv preprint},
  year={2025}
}
```

## Acknowledgements

We thank Dr. Yilun Wu (National Yang Ming Chiao Tung University) for his guidance
and Dr. Jiahuan Pei (Vrije Universiteit Amsterdam) for her support.
