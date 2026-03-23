"""
Self-Generated Dummy Thinking Shells for FIM Computation
=========================================================
完全data-free版本：让base model自己生成带<think>...</think>结构的序列，
用这些序列替代random tokens计算FIM。

核心思路（受RAIN-Merging启发）：
- RAIN用null-space projection保护thinking format
- 我们用thinking format序列计算FIM，让FIM感知reasoning pathway的激活模式
- 完全不需要任何真实问题数据

相比random tokens的优势：
- 激活了<think>...</think>的reasoning subspace
- FIM能更准确反映long-CoT路径的参数重要性
- 对AIME24等多步推理任务更有针对性

运行方式：
    python generate_synthetic_cot.py \
        --base_model /path/to/base_model \
        --ft_model /path/to/ft_model \
        --output_file synthetic_cot.json \
        --n_samples 32 \
        --max_new_tokens 256 \
        --device cuda

注意：用base model和ft model各生成一半，混合使用
"""

import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# 16种不同的thinking shell起始prompt
# 让模型自由生成thinking内容，不给任何数学问题
THINKING_SEEDS = [
    "<|im_start|>assistant\n<think>\n",
    "<|im_start|>assistant\n<think>\nLet me think step by step.\n",
    "<|im_start|>assistant\n<think>\nFirst, I need to analyze this carefully.\n",
    "<|im_start|>assistant\n<think>\nLet me break this down.\n",
    "<|im_start|>assistant\n<think>\nHmm, let me consider the key points.\n",
    "<|im_start|>assistant\n<think>\nI'll work through this systematically.\n",
    "<|im_start|>assistant\n<think>\nLet me verify my reasoning.\n",
    "<|im_start|>assistant\n<think>\nOkay, so the key insight here is\n",
    "<|im_start|>assistant\n<think>\nWait, let me reconsider.\n",
    "<|im_start|>assistant\n<think>\nActually, I think the approach should be\n",
    "<|im_start|>assistant\n<think>\nLet me check if this is correct.\n",
    "<|im_start|>assistant\n<think>\nThe important thing to notice is\n",
    "<|im_start|>assistant\n<think>\nSo we have the following situation:\n",
    "<|im_start|>assistant\n<think>\nBy carefully examining this,\n",
    "<|im_start|>assistant\n<think>\nLet me approach this from a different angle.\n",
    "<|im_start|>assistant\n<think>\nTo solve this, I need to consider\n",
]


def generate_thinking_shell(model, tokenizer, seed_prompt,
                             max_new_tokens=256, device='cuda'):
    """
    让模型从thinking seed开始自由生成，
    产生带reasoning结构的dummy序列
    """
    inputs = tokenizer(seed_prompt, return_tensors='pt',
                       add_special_tokens=False).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return full_text


def main():
    parser = argparse.ArgumentParser(
        description="Generate dummy thinking shells for FIM computation")
    parser.add_argument('--base_model', type=str,
                        default='/home/t640/models/Qwen2.5-Math-1.5B/Qwen/Qwen2.5-Math-1.5B')
    parser.add_argument('--ft_model', type=str,
                        default='/home/t640/models/DeepSeek-R1-1.5B/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
    parser.add_argument('--output_file', type=str, default='synthetic_cot.json')
    parser.add_argument('--n_samples', type=int, default=32)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    results = []
    n_per_model = args.n_samples // 2

    # ── 用FT model生成前一半（DeepSeek-R1有更丰富的thinking结构）──
    print(f"\n[1/2] Loading FT model: {args.ft_model}")
    ft_tokenizer = AutoTokenizer.from_pretrained(
        args.ft_model, trust_remote_code=True)
    ft_model = AutoModelForCausalLM.from_pretrained(
        args.ft_model, torch_dtype=torch.bfloat16,
        device_map=args.device)
    ft_model.eval()

    seeds_ft = THINKING_SEEDS[:n_per_model]
    for i, seed in enumerate(seeds_ft):
        print(f"  FT [{i+1}/{n_per_model}] seed: {seed[:50].strip()!r}")
        text = generate_thinking_shell(
            ft_model, ft_tokenizer, seed,
            max_new_tokens=args.max_new_tokens,
            device=args.device)
        results.append({
            'source': 'ft_model',
            'seed': seed,
            'full_text': text,
            'n_tokens': len(ft_tokenizer.encode(text))
        })
        print(f"    -> {len(text.split())} words generated")

    del ft_model
    torch.cuda.empty_cache()
    print("  FT model unloaded.")

    # ── 用Base model生成后一半 ──
    print(f"\n[2/2] Loading Base model: {args.base_model}")
    base_tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map=args.device)
    base_model.eval()

    seeds_base = THINKING_SEEDS[n_per_model:n_per_model * 2]
    for i, seed in enumerate(seeds_base):
        print(f"  Base [{i+1}/{n_per_model}] seed: {seed[:50].strip()!r}")
        text = generate_thinking_shell(
            base_model, base_tokenizer, seed,
            max_new_tokens=args.max_new_tokens,
            device=args.device)
        results.append({
            'source': 'base_model',
            'seed': seed,
            'full_text': text,
            'n_tokens': len(base_tokenizer.encode(text))
        })
        print(f"    -> {len(text.split())} words generated")

    del base_model
    torch.cuda.empty_cache()
    print("  Base model unloaded.")

    # ── 保存 ──
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    avg_tokens = sum(r['n_tokens'] for r in results) / len(results)
    print(f"\nSaved {len(results)} thinking shells to {args.output_file}")
    print(f"  Average length: {avg_tokens:.0f} tokens")
    print(f"  From FT model:   {n_per_model} samples")
    print(f"  From Base model: {n_per_model} samples")
    print(f"\nNext step:")
    print(f"  python fim_synthetic_merge.py \\")
    print(f"    --base_model {args.base_model} \\")
    print(f"    --ft_model {args.ft_model} \\")
    print(f"    --save_path /home/t640/models/l2s_1b_fim_ties_plus \\")
    print(f"    --synthetic_data {args.output_file} \\")
    print(f"    --merging_method ties --threshold_ratio 0.2 --device cpu")


if __name__ == '__main__':
    main()