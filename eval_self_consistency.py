"""
Self-Consistency Evaluation for AIME24
=======================================
Generate multiple answers per problem and take majority vote.
This is a post-merge inference technique that improves accuracy
on hard reasoning tasks like AIME24 without changing the merge method.

Usage:
    python eval_self_consistency.py \
        --model_path /home/t640/models/l2s_1b_fim_ties_v4 \
        --n_samples 8 \
        --temperature 0.7 \
        --data_dir /home/t640/Long-to-Short-via-Model-Merging/Qwen2.5-Math-main/evaluation/data \
        --output_dir ./sc_results \
        --device cuda:1
"""

import json
import os
import re
import argparse
import torch
from collections import Counter
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_aime24(data_dir):
    """Load AIME24 test problems."""
    data_path = os.path.join(data_dir, "aime24", "test.jsonl")
    problems = []
    with open(data_path) as f:
        for line in f:
            d = json.loads(line)
            problems.append(d)
    print(f"Loaded {len(problems)} AIME24 problems")
    return problems


def build_prompt(problem):
    """Build prompt in qwen25-math-cot format."""
    return (
        "<|im_start|>system\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
        "<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def extract_answer(text):
    """Extract answer from \\boxed{} in generated text."""
    # Try to find \boxed{...}
    patterns = [
        r'\\boxed\{([^}]+)\}',
        r'\\boxed\s*\{([^}]+)\}',
        r'boxed\{([^}]+)\}',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    return None


def majority_vote(answers):
    """
    Take majority vote from a list of answers.
    Returns (best_answer, vote_count, total_count)
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return None, 0, len(answers)
    counter = Counter(valid)
    best, count = counter.most_common(1)[0]
    return best, count, len(answers)


def normalize_answer(ans):
    """Normalize answer for comparison."""
    if ans is None:
        return None
    ans = ans.strip()
    # Remove trailing zeros after decimal
    try:
        f = float(ans)
        if f == int(f):
            return str(int(f))
        return str(f)
    except:
        return ans.lower().replace(' ', '')


def evaluate_self_consistency(model_path, data_dir, output_dir,
                               n_samples=8, temperature=0.7,
                               max_tokens=2048, device='cuda:1',
                               gpu_memory_utilization=0.85):
    os.makedirs(output_dir, exist_ok=True)

    # Load problems
    problems = load_aime24(data_dir)

    # Set GPU
    gpu_id = int(device.split(':')[-1]) if ':' in device else 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Load model with vllm
    print(f"Loading model: {model_path}")
    llm = LLM(
        model=model_path,
        dtype='bfloat16',
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=4096,
        trust_remote_code=True,
    )

    # Build prompts
    prompts = [build_prompt(p['question'] if 'question' in p else p['problem'])
               for p in problems]
    gt_answers = [str(p['answer']).strip() for p in problems]

    # Greedy (n=1) baseline
    print(f"\n--- Greedy decoding (n=1) ---")
    greedy_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        n=1,
    )
    greedy_outputs = llm.generate(prompts, greedy_params)
    greedy_correct = 0
    greedy_results = []
    for i, output in enumerate(greedy_outputs):
        pred = extract_answer(output.outputs[0].text)
        gt = normalize_answer(gt_answers[i])
        pred_norm = normalize_answer(pred)
        correct = (pred_norm == gt)
        if correct:
            greedy_correct += 1
        greedy_results.append({
            'idx': i,
            'gt': gt_answers[i],
            'pred': pred,
            'correct': correct,
        })
    greedy_acc = greedy_correct / len(problems) * 100
    print(f"Greedy accuracy: {greedy_correct}/{len(problems)} = {greedy_acc:.1f}%")

    # Self-consistency (n samples)
    print(f"\n--- Self-consistency (n={n_samples}, temp={temperature}) ---")
    sc_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=n_samples,
        top_p=0.95,
    )
    sc_outputs = llm.generate(prompts, sc_params)

    sc_correct = 0
    sc_results = []
    vote_details = []

    for i, output in enumerate(sc_outputs):
        # Extract answers from all samples
        all_answers = [extract_answer(o.text) for o in output.outputs]
        best_ans, vote_count, total = majority_vote(all_answers)
        gt = normalize_answer(gt_answers[i])
        best_norm = normalize_answer(best_ans)
        correct = (best_norm == gt)
        if correct:
            sc_correct += 1

        sc_results.append({
            'idx': i,
            'gt': gt_answers[i],
            'majority_answer': best_ans,
            'vote_count': vote_count,
            'total_samples': total,
            'all_answers': all_answers,
            'correct': correct,
        })
        vote_details.append(
            f"  [{i:2d}] GT={gt_answers[i]:>4s} | "
            f"Vote={best_ans}({vote_count}/{total}) | "
            f"All={[a for a in all_answers if a]} | "
            f"{'✓' if correct else '✗'}"
        )

    sc_acc = sc_correct / len(problems) * 100
    print(f"SC accuracy: {sc_correct}/{len(problems)} = {sc_acc:.1f}%")
    print(f"\nImprovement: {greedy_acc:.1f}% -> {sc_acc:.1f}% "
          f"({sc_acc - greedy_acc:+.1f}%)")

    # Print vote details
    print("\nPer-problem breakdown:")
    for line in vote_details:
        print(line)

    # Save results
    results = {
        'model': model_path,
        'n_samples': n_samples,
        'temperature': temperature,
        'greedy_acc': greedy_acc,
        'greedy_correct': greedy_correct,
        'sc_acc': sc_acc,
        'sc_correct': sc_correct,
        'total': len(problems),
        'improvement': sc_acc - greedy_acc,
        'greedy_results': greedy_results,
        'sc_results': sc_results,
    }

    out_file = os.path.join(output_dir, 'aime24_sc_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")

    # Summary table for different n_samples
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Model: {model_path}")
    print(f"Greedy (n=1):          {greedy_correct}/30 = {greedy_acc:.1f}%")
    print(f"Self-consistency (n={n_samples}): {sc_correct}/30 = {sc_acc:.1f}%")
    print(f"Improvement:           {sc_acc - greedy_acc:+.1f} points")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='/home/t640/models/l2s_1b_fim_ties_v4')
    parser.add_argument('--data_dir', type=str,
                        default='/home/t640/Long-to-Short-via-Model-Merging/'
                                'Qwen2.5-Math-main/evaluation/data')
    parser.add_argument('--output_dir', type=str, default='./sc_results')
    parser.add_argument('--n_samples', type=int, default=8,
                        help='Number of samples for majority vote')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85)
    args = parser.parse_args()

    evaluate_self_consistency(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        device=args.device,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


if __name__ == '__main__':
    main()
