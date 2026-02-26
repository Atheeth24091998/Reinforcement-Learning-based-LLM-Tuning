import argparse
import random
import torch

import logging

from src.dataset.io import load_json
from src.dataset.normalize import normalize_record
from src.kg.build_triples import build_triples_from_record
from src.kg.graph import KG
from src.kg.reward import compute_reward
from src.common.prompts import SYSTEM, build_user_prompt
from src.kg.candidates import CandidateIndex, get_candidates

import os
import csv

from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model


def build_prompt(tokenizer, symptoms, candidates=None):
    user = build_user_prompt(symptoms, candidates=candidates)
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return SYSTEM + "\n\n" + user + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--train_path", type=str, default="data/clean/train.json")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ----- Load dataset -----
    recs = [normalize_record(r) for r in load_json(args.train_path)]
    if len(recs) < 10:
        raise ValueError("Train set too small for PPO debugging.")

    # ----- Build KG from ALL training records -----
    triples = []
    for r in recs:
        triples.extend(build_triples_from_record(r))
    
    kg = KG(triples)
    print(f"KG nodes={len(kg.nodes)} triples={len(triples)}")

    idx = CandidateIndex(triples)

    # ----- Tokenizer + Model (Value Head) -----
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # ----- LoRA on the underlying LM -----
    # For Llama, typical target modules: q_proj, k_proj, v_proj, o_proj (+ sometimes up/down/gate)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model.pretrained_model = get_peft_model(model.pretrained_model, lora_cfg)
    model.pretrained_model.print_trainable_parameters()

    # ----- PPO Trainer -----
    ppo_cfg = PPOConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
    )

    # TRL experimental PPOTrainer signature differs across versions,
    # so try the common patterns.
    trainer = PPOTrainer(config=ppo_cfg, model=model, ref_model=ref_model, tokenizer=tok)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    os.makedirs("outputs/logs", exist_ok=True)
    csv_path = "outputs/logs/reward_log.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "avg_reward"])

    # ----- Training loop -----
    model.train()
    for step in range(args.steps):
        #batch = random.sample(recs, k=args.batch_size)
        batch = [recs[step]]

        # prompts = [build_prompt(tok, r["symptoms"]) for r in batch]
        cand_list = [get_candidates(idx, r["symptoms"], k_problems=30, k_causes=60, k_actions=60) for r in batch]
        prompts = [build_prompt(tok, r["symptoms"], candidates=c) for r, c in zip(batch, cand_list)]

        #query_tensors = [tok(p, return_tensors="pt").input_ids[0].to(trainer.accelerator.device) for p in prompts]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(trainer.accelerator.device) for k, v in enc.items()}
        query_tensors = [enc["input_ids"][i] for i in range(enc["input_ids"].shape[0])]
        attn_masks    = [enc["attention_mask"][i] for i in range(enc["attention_mask"].shape[0])]
        # Generate responses
        response_tensors = []
        responses_text = []
        #for q in query_tensors:
        #    out = trainer.model.generate(q.unsqueeze(0), **gen_kwargs)[0]
        for q, m in zip(query_tensors, attn_masks):
            out = trainer.model.generate(q.unsqueeze(0), attention_mask=m.unsqueeze(0), **gen_kwargs)[0]
            resp = out[q.shape[0]:]  # only newly generated
            response_tensors.append(resp)
            responses_text.append(tok.decode(resp, skip_special_tokens=True))

        # Compute KG reward per sample
        rewards = []
        for r, txt in zip(batch, responses_text):
            rew = compute_reward(kg, r, txt, max_hops=3)
            rewards.append(torch.tensor(rew, device=trainer.accelerator.device))
        rewards_t = rewards

        # 2. CALCULATE average reward for this step
        avg_r = sum([x.item() for x in rewards_t]) / len(rewards_t)

        # PPO update
        stats = trainer.step(query_tensors, response_tensors, rewards_t)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([step, avg_r])

        if step % 10 == 0:
            avg_r = sum([x.item() for x in rewards_t]) / len(rewards_t)
            print(f"[step {step}] avg_reward={avg_r:.3f}")
            # print one sample for sanity
            print("  sample_response:\n", responses_text[0][:400].replace("\n", "\\n"))

    # Save LoRA adapters
    out_dir = "outputs/checkpoints/rlkgf_lora"
    model.pretrained_model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print("Saved adapters to:", out_dir)

if __name__ == "__main__":
    main()