"""
Negotiation Evaluation Script for GRPO checkpoints.

Plays multi-game negotiations (selfplay with frozen opponent) and evaluates
outcomes using GPT-4o-mini, producing metrics comparable to Luca's CSV format.

Usage:
  python evaluations/run_negotiation_eval.py --repetitions 5
  python evaluations/run_negotiation_eval.py --repetitions 10 --checkpoint output/grpo-multigame-self-only/checkpoint-560
"""

import sys
import os
import argparse
import json
import time
from statistics import mean, stdev
from collections import defaultdict

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel

# Add repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.negotiation.env import NegotiationEnv
from envs.negotiation.games import Game
from evaluator.evaluator import Evaluator
from evaluator.openai_model import OpenAIModel
from evaluations.adversarial_personas import PERSONAS, apply_persona


# ============================================================
# Model loading
# ============================================================

def load_model_and_tokenizer(checkpoint_path, model_name="OpenPipe/Qwen3-14B-Instruct"):
    """Load base model with optional LoRA checkpoint."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    has_lora = False
    if checkpoint_path and checkpoint_path != "none":
        print(f"Loading LoRA from: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        has_lora = True
    else:
        print("No checkpoint — running base model only.")

    model.eval()
    return model, tokenizer, has_lora


# ============================================================
# Generation
# ============================================================

def gen_response(model, tokenizer, messages, max_new_tokens=200, temperature=1.0):
    """Generate a single turn response."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1800).to(model.device)

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def play_negotiation(model, tokenizer, prompt_agent, prompt_opponent,
                     agent_starts=True, max_rounds=5, has_lora=False,
                     max_new_tokens=200, temperature=1.0):
    """Play a full multi-turn negotiation (selfplay with frozen opponent)."""
    agent_history = [{"role": "system", "content": prompt_agent}]
    opponent_history = [{"role": "system", "content": prompt_opponent}]
    conversation = []

    unwrapped = model

    for round_num in range(max_rounds):
        speakers = ["agent", "opponent"] if agent_starts else ["opponent", "agent"]

        for speaker in speakers:
            if speaker == "agent":
                if has_lora:
                    unwrapped.enable_adapter_layers()
                response = gen_response(model, tokenizer, agent_history,
                                       max_new_tokens=max_new_tokens, temperature=temperature)
                agent_history.append({"role": "assistant", "content": response})
                opponent_history.append({"role": "user", "content": response})
                conversation.append({"role": "assistant", "content": response})
            else:
                if has_lora:
                    unwrapped.disable_adapter_layers()
                response = gen_response(model, tokenizer, opponent_history,
                                       max_new_tokens=max_new_tokens, temperature=temperature)
                if has_lora:
                    unwrapped.enable_adapter_layers()
                opponent_history.append({"role": "assistant", "content": response})
                agent_history.append({"role": "user", "content": response})
                conversation.append({"role": "user", "content": response})

    if has_lora:
        unwrapped.enable_adapter_layers()
    return conversation


# ============================================================
# Evaluation
# ============================================================

def evaluate_checkpoint(model, tokenizer, has_lora, env, num_games,
                        max_rounds=5, temperature=1.0, seed=42, repetitions=1,
                        opponent_persona="cooperative", sample_start_idx=0):
    """Run negotiations and evaluate with GPT-4o-mini.

    Iterates over a slice of the curated eval dataset's *flat sample list*. The
    flat list has two samples per scenario (role 1 + role 2), so for the
    multi-game eval set with 14 scenarios there are 28 samples (indices 0..27).
    sample_start_idx lets you eval a contiguous slice (e.g. samples 10..15)
    so a previous run on samples 0..9 can be extended without re-rolling those.
    """
    eval_dataset = env.create_eval_dataset()
    end_idx = min(sample_start_idx + num_games, len(eval_dataset))
    n_samples = end_idx - sample_start_idx
    if n_samples <= 0:
        raise ValueError(
            f"Empty sample slice: start={sample_start_idx}, num_games={num_games}, "
            f"dataset size={len(eval_dataset)}"
        )

    all_results = []
    all_U_A, all_U_B = [], []
    all_ratio_self, all_ratio_welfare, all_ratio_nash, all_ratio_rcoop = [], [], [], []
    all_agreed = []
    archetype_data = defaultdict(lambda: {"U_A": [], "U_B": [], "ratio_self": [],
                                           "ratio_welfare": [], "ratio_nash": [],
                                           "ratio_rcoop": [], "agreed": []})

    total = n_samples * repetitions
    print(f"\nPlaying {total} negotiations ({n_samples} samples x {repetitions} reps, "
          f"sample indices {sample_start_idx}..{end_idx - 1}, max {max_rounds} rounds each)...")

    game_counter = 0
    for rep in range(repetitions):
        rep_seed = seed + rep
        torch.manual_seed(rep_seed)
        np.random.seed(rep_seed)

        for i in range(sample_start_idx, end_idx):
            sample = eval_dataset[i]
            prompt_agent = sample["prompt"]
            prompt_opponent = apply_persona(sample["prompt_2"], opponent_persona)
            agent_starts = sample["starting_agent"]
            game_config = sample["game_config"]
            negotiation_role = sample["negotiation_role"]
            archetype = sample["archetype"]

            t0 = time.time()
            conversation = play_negotiation(
                model, tokenizer, prompt_agent, prompt_opponent,
                agent_starts=agent_starts, max_rounds=max_rounds,
                has_lora=has_lora, temperature=temperature,
            )
            gen_time = time.time() - t0

            # Evaluate with GPT-4o-mini
            game = Game(**game_config)
            eval_model = OpenAIModel(model_provider="openai", model_name="gpt-4o-mini")
            evaluator = Evaluator(model=eval_model, game=game, game_type=env.game_type)

            starting_agent = 0 if agent_starts else 1
            evaluation = evaluator.evaluate(conversation, starting_agent=starting_agent, get_payoffs=True)

            # Compute metrics
            max_metrics = env.compute_max_metrics(game, negotiation_role)

            if evaluation is None or "payoffs" not in evaluation:
                U_A, U_B = 0.0, 0.0
                ratio_self = ratio_welfare = ratio_nash = ratio_rcoop = 0.0
                agreed = False
            else:
                p1 = evaluation["payoffs"]["Agent 1"]
                p2 = evaluation["payoffs"]["Agent 2"]
                U_A = p1 if negotiation_role == 1 else p2
                U_B = p2 if negotiation_role == 1 else p1

                ratio_self = U_A / max_metrics["max_U_A"] if max_metrics["max_U_A"] > 0 else 0.0
                sw = U_A + U_B
                ratio_welfare = sw / max_metrics["max_social_welfare"] if max_metrics["max_social_welfare"] > 0 else 0.0
                np_val = U_A * U_B
                ratio_nash = np_val / max_metrics["max_nash_product"] if max_metrics["max_nash_product"] > 0 else 0.0

                R_coop = (env.lambda_self * ratio_self
                          + env.lambda_welfare * ratio_welfare
                          + env.lambda_fair * ratio_nash)
                ratio_rcoop = R_coop / max_metrics["max_r_coop"] if max_metrics["max_r_coop"] > 0 else 0.0
                agreed = U_A > 0 or U_B > 0

            all_U_A.append(float(U_A))
            all_U_B.append(float(U_B))
            all_ratio_self.append(float(ratio_self))
            all_ratio_welfare.append(float(ratio_welfare))
            all_ratio_nash.append(float(ratio_nash))
            all_ratio_rcoop.append(float(ratio_rcoop))
            all_agreed.append(agreed)

            archetype_data[archetype]["U_A"].append(float(U_A))
            archetype_data[archetype]["U_B"].append(float(U_B))
            archetype_data[archetype]["ratio_self"].append(float(ratio_self))
            archetype_data[archetype]["ratio_welfare"].append(float(ratio_welfare))
            archetype_data[archetype]["ratio_nash"].append(float(ratio_nash))
            archetype_data[archetype]["ratio_rcoop"].append(float(ratio_rcoop))
            archetype_data[archetype]["agreed"].append(agreed)

            status = "OK" if agreed else "FAIL"
            game_counter += 1
            print(f"  Game {game_counter:3d}/{total} (rep {rep+1}/{repetitions}): "
                  f"U_A={U_A:.0f} U_B={U_B:.0f} "
                  f"r_self={ratio_self:.2f} r_nash={ratio_nash:.2f} [{status}] "
                  f"({gen_time:.1f}s) [{archetype}]")

            all_results.append({
                "game_idx": game_counter - 1,
                "repetition": rep,
                "config_idx": i,
                "archetype": archetype,
                "negotiation_role": negotiation_role,
                "agent_starts": agent_starts,
                "U_A": U_A, "U_B": U_B,
                "ratio_self": ratio_self, "ratio_welfare": ratio_welfare,
                "ratio_nash": ratio_nash, "ratio_rcoop": ratio_rcoop,
                "agreed": agreed,
                "conversation": conversation,
            })

    # Aggregate
    n = len(all_U_A) or 1
    agreements = sum(all_agreed)
    metrics = {
        "n_games": total,
        # NOTE: "n_configs" is preserved for backward compatibility with the
        # original v2 JSONs but is actually the number of *samples* in the slice
        # (each scenario contributes 2 samples, role-1 + role-2). Use n_samples
        # for the unambiguous reading.
        "n_configs": n_samples,
        "n_samples": n_samples,
        "sample_start_idx": sample_start_idx,
        "sample_end_idx": end_idx - 1,
        "repetitions": repetitions,
        "agreement_rate": agreements / n,
        "U_A_mean": mean([float(x) for x in all_U_A]),
        "U_A_std": stdev([float(x) for x in all_U_A]) if n > 1 else 0,
        "U_B_mean": mean([float(x) for x in all_U_B]),
        "U_B_std": stdev([float(x) for x in all_U_B]) if n > 1 else 0,
        "social_welfare_mean": mean([float(a + b) for a, b in zip(all_U_A, all_U_B)]),
        "ratio_self_mean": mean([float(x) for x in all_ratio_self]),
        "ratio_welfare_mean": mean([float(x) for x in all_ratio_welfare]),
        "ratio_nash_mean": mean([float(x) for x in all_ratio_nash]),
        "ratio_rcoop_mean": mean([float(x) for x in all_ratio_rcoop]),
    }

    # Agreed-only metrics
    if agreements > 0:
        ag_rs = [v for v, a in zip(all_ratio_self, all_agreed) if a]
        ag_rw = [v for v, a in zip(all_ratio_welfare, all_agreed) if a]
        ag_rn = [v for v, a in zip(all_ratio_nash, all_agreed) if a]
        ag_rc = [v for v, a in zip(all_ratio_rcoop, all_agreed) if a]
        metrics["agreed_ratio_self_mean"] = mean(ag_rs)
        metrics["agreed_ratio_welfare_mean"] = mean(ag_rw)
        metrics["agreed_ratio_nash_mean"] = mean(ag_rn)
        metrics["agreed_ratio_rcoop_mean"] = mean(ag_rc)

    # Per-archetype
    metrics["per_archetype"] = {}
    for arch, vals in archetype_data.items():
        m = len(vals["U_A"])
        metrics["per_archetype"][arch] = {
            "count": m,
            "U_A_mean": mean(vals["U_A"]),
            "U_B_mean": mean(vals["U_B"]),
            "agreement_rate": sum(vals["agreed"]) / m,
            "ratio_self_mean": mean(vals["ratio_self"]),
            "ratio_welfare_mean": mean(vals["ratio_welfare"]),
            "ratio_nash_mean": mean(vals["ratio_nash"]),
            "ratio_rcoop_mean": mean(vals["ratio_rcoop"]),
        }

    return metrics, all_results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Negotiation eval for GRPO checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single checkpoint to evaluate (overrides default list)")
    parser.add_argument("--model-name", type=str, default="OpenPipe/Qwen3-14B-Instruct")
    parser.add_argument("--game-type", type=str, default="multi-game")
    parser.add_argument("--num-games", type=int, default=10,
                        help="Number of game configs to use (max = eval dataset size)")
    parser.add_argument("--sample-start-idx", "--config-start-idx",
                        dest="sample_start_idx", type=int, default=0,
                        help="Index of first SAMPLE in the curated flat dataset (default 0). "
                             "The flat dataset has 2 samples per scenario (role 1 + role 2), so "
                             "for the multi-game set there are 28 samples (0..27) covering 14 "
                             "scenarios. Use --sample-start-idx 10 --num-games 6 to evaluate "
                             "samples 10..15 (= scenarios 6,7,8 in both roles).")
    parser.add_argument("--repetitions", type=int, default=1,
                        help="Times to repeat each game config (different seeds for statistical significance)")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="evaluations/results/negotiation")
    # Lambda params for R_coop calculation (should match training for fair comparison)
    parser.add_argument("--lambda-self", type=float, default=1.0)
    parser.add_argument("--lambda-welfare", type=float, default=0.0)
    parser.add_argument("--lambda-fair", type=float, default=0.0)
    parser.add_argument("--opponent-persona", type=str, default="cooperative",
                        choices=list(PERSONAS.keys()),
                        help="Adversarial persona injected into opponent system prompt (RQ4)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Default checkpoints if none specified
    if args.checkpoint:
        checkpoints = [
            (args.checkpoint, os.path.basename(args.checkpoint.rstrip("/")))
        ]
    else:
        checkpoints = [
            ("none", "base_model"),
            ("output/grpo-multigame-self-only/checkpoint-560", "self_only_560"),
            ("output/grpo-multigame-fair-only/checkpoint-560", "fair_only_560"),
            ("output/grpo-multigame-all-equal/checkpoint-620", "all_equal_620"),
            ("output/grpo-multigame-self-fair-equal/checkpoint-820", "self_fair_equal_820"),
        ]

    all_csv_rows = []

    for ckpt_path, name in checkpoints:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name} ({ckpt_path})")
        print(f"{'='*60}")

        model, tokenizer, has_lora = load_model_and_tokenizer(
            ckpt_path if ckpt_path != "none" else None,
            model_name=args.model_name,
        )

        env = NegotiationEnv(
            game_type=args.game_type,
            lambda_self=args.lambda_self,
            lambda_welfare=args.lambda_welfare,
            lambda_fair=args.lambda_fair,
        )

        metrics, results = evaluate_checkpoint(
            model, tokenizer, has_lora, env,
            num_games=args.num_games,
            max_rounds=args.max_rounds,
            temperature=args.temperature,
            seed=args.seed,
            repetitions=args.repetitions,
            opponent_persona=args.opponent_persona,
            sample_start_idx=args.sample_start_idx,
        )

        # Print summary
        print(f"\n--- {name} Summary ---")
        print(f"  Agreement rate:     {metrics['agreement_rate']*100:.1f}%")
        print(f"  U_A mean:           {metrics['U_A_mean']:.1f} (std {metrics['U_A_std']:.1f})")
        print(f"  Social welfare:     {metrics['social_welfare_mean']:.1f}")
        print(f"  Ratio self:         {metrics['ratio_self_mean']:.3f}")
        print(f"  Ratio welfare:      {metrics['ratio_welfare_mean']:.3f}")
        print(f"  Ratio nash:         {metrics['ratio_nash_mean']:.3f}")
        print(f"  Ratio R_coop:       {metrics['ratio_rcoop_mean']:.3f}")
        if "agreed_ratio_rcoop_mean" in metrics:
            print(f"  Agreed ratio R_coop:{metrics['agreed_ratio_rcoop_mean']:.3f}")

        # Save JSON (tagged with persona + slice so runs don't overwrite each other).
        # Slice tag uses sample-index range (0..27), NOT scenario-index range (1..14).
        # For multi-game: samples 10..15 == scenarios 6,7,8 in both roles.
        suffix = f"_{args.opponent_persona}" if args.opponent_persona != "cooperative" else ""
        if args.sample_start_idx != 0:
            end = args.sample_start_idx + args.num_games - 1
            suffix += f"_samples{args.sample_start_idx}-{end}"
        output_path = os.path.join(args.output_dir, f"{name}{suffix}.json")
        with open(output_path, "w") as f:
            json.dump({"args": vars(args), "metrics": metrics, "games": results},
                      f, indent=2, default=str)
        print(f"  Saved to: {output_path}")

        # CSV row (comparable to Luca's format)
        step = 0
        if ckpt_path != "none":
            try:
                step = int(ckpt_path.split("-")[-1])
            except ValueError:
                pass

        all_csv_rows.append({
            "method": name.rsplit("_", 1)[0] if "_" in name else name,
            "opponent_persona": args.opponent_persona,
            "number": step,
            "mean_reward": metrics["U_A_mean"],  # Using U_A as reward proxy
            "std_reward": metrics["U_A_std"],
            "agreement_rate": metrics["agreement_rate"],
            "ratio_self_mean": metrics["ratio_self_mean"],
            "ratio_welfare_mean": metrics["ratio_welfare_mean"],
            "ratio_nash_mean": metrics["ratio_nash_mean"],
            "ratio_rcoop_mean": metrics["ratio_rcoop_mean"],
        })

        # Free memory before loading next model
        del model
        torch.cuda.empty_cache()

    # Save CSV summary (tagged with persona)
    import csv
    csv_suffix = f"_{args.opponent_persona}" if args.opponent_persona != "cooperative" else ""
    csv_path = os.path.join(args.output_dir, f"evaluation_results_grpo{csv_suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_csv_rows)
    print(f"\nCSV summary saved to: {csv_path}")


if __name__ == "__main__":
    main()
