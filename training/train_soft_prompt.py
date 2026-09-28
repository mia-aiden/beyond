from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import PeftModel

CUR = Path(__file__).resolve().parent
sys.path.insert(0, str(CUR))
from prepare_sft_data import EKMAN_LABELS, NARRATIVE_INSTRUCTION, NARRATIVE_SYSTEM_PROMPT  # noqa: E402

DEV = torch.device("cuda:0")


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def encode(tok, source, target, cutoff, max_src):
    ids = tok.encode(source, add_special_tokens=False)
    if len(ids) > max_src:
        source = tok.decode(ids[:max_src], skip_special_tokens=True)
    msgs = [
        {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
        {"role": "user", "content": f"{NARRATIVE_INSTRUCTION}\n{source}"},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                     return_dict=True)["input_ids"]
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": target}],
                                   tokenize=True, add_generation_prompt=False,
                                   return_dict=True)["input_ids"]
    input_ids = full[:cutoff]
    n_prompt = min(len(prompt), len(input_ids))
    labels = [-100] * n_prompt + input_ids[n_prompt:]
    return input_ids, labels


class DS(Dataset):
    def __init__(self, rows, tok, variant, cutoff, max_src):
        self.items = []
        for r in rows:
            ids, labels = encode(tok, r["source_text"], r["reference_narrative"], cutoff, max_src)
            if sum(x != -100 for x in labels) == 0:
                continue
            emo = 0 if variant == "shared" else EKMAN_LABELS.index(r["emotion_label"])
            self.items.append((ids, labels, emo))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id):
    m = max(len(x[0]) for x in batch)
    ii, am, lb, em = [], [], [], []
    for ids, labels, emo in batch:
        p = m - len(ids)
        ii.append(ids + [pad_id] * p)
        am.append([1] * len(ids) + [0] * p)
        lb.append(labels + [-100] * p)
        em.append(emo)
    return (torch.tensor(ii), torch.tensor(am), torch.tensor(lb), torch.tensor(em))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["emotion", "shared"], required=True)
    ap.add_argument("--base-model", default="/root/autodl-tmp/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--adapter", default="/root/autodl-fs/sft_experiments/models/narrative_lora_r8")
    ap.add_argument("--train", default="/root/autodl-fs/sft_experiments/data/joint_train.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--cutoff", type=int, default=1536)
    ap.add_argument("--max-src", type=int, default=1300)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True, use_fast=True)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", low_cpu_mem_usage=True, local_files_only=True)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False, local_files_only=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in model.parameters():
        p.requires_grad_(False)
    model.train()
    embed = model.get_input_embeddings()
    hidden = model.config.hidden_size

    n = 7 if args.variant == "emotion" else 1
    init = torch.zeros(n, hidden, dtype=torch.float32)
    with torch.no_grad():
        words = EKMAN_LABELS if args.variant == "emotion" else ["context"]
        for i, w in enumerate(words):
            wid = tok.encode(w, add_special_tokens=False)
            init[i] = embed.weight[torch.tensor(wid, device=DEV)].mean(0).float().cpu()
    soft = nn.Parameter(init.to(DEV))

    rows = read_jsonl(args.train)
    if args.limit:
        rows = rows[: args.limit]
    ds = DS(rows, tok, args.variant, args.cutoff, args.max_src)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    total = args.max_steps if args.max_steps > 0 else int(len(dl) // args.grad_accum * args.epochs)
    opt = torch.optim.AdamW([soft], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.1 * total)), total)
    print(f"variant={args.variant} soft={tuple(soft.shape)} examples={len(ds)} steps={total}", flush=True)

    step = 0
    accum = 0
    opt.zero_grad()
    done = False
    for _ in range(1000):
        for ii, am, lb, em in dl:
            ii, am, lb, em = ii.to(DEV), am.to(DEV), lb.to(DEV), em.to(DEV)
            tok_emb = embed(ii)                              # [B,T,H] frozen
            pref = soft[em].to(torch.bfloat16).unsqueeze(1)  # [B,1,H] trainable
            inputs_embeds = torch.cat([pref, tok_emb], dim=1)
            am2 = torch.cat([torch.ones(am.size(0), 1, dtype=am.dtype, device=DEV), am], dim=1)
            lb2 = torch.cat([torch.full((lb.size(0), 1), -100, dtype=lb.dtype, device=DEV), lb], dim=1)
            out = model(inputs_embeds=inputs_embeds, attention_mask=am2, labels=lb2)
            (out.loss / args.grad_accum).backward()
            accum += 1
            if accum % args.grad_accum == 0:
                gnorm = float(torch.nn.utils.clip_grad_norm_([soft], 1.0))
                before = soft.detach().norm().item()
                opt.step()
                sched.step()
                delta = (soft.detach().norm().item() - before)
                opt.zero_grad()
                step += 1
                if step % 10 == 0 or step <= 3:
                    print(f"step {step}/{total} loss {out.loss.item():.4f} gnorm {gnorm:.4f} d|soft| {delta:+.5f}", flush=True)
                if step >= total:
                    done = True
                    break
        if done:
            break

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"soft": soft.detach().cpu().float(), "variant": args.variant,
                "ekman": list(EKMAN_LABELS)}, out_dir / "soft_prompt.pt")
    print("SAVED", out_dir / "soft_prompt.pt", "shape", tuple(soft.shape), flush=True)


if __name__ == "__main__":
    main()
