from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

CUR = Path(__file__).resolve().parent
sys.path.insert(0, str(CUR))
sys.path.insert(0, str(CUR.parent / "evaluation" / "src"))
from prepare_sft_data import EKMAN_LABELS, NARRATIVE_INSTRUCTION, NARRATIVE_SYSTEM_PROMPT  # noqa: E402
from validate_inputs import load_gold_with_sample_ids  # noqa: E402

DEV = torch.device("cuda:0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["emotion", "shared"], required=True)
    ap.add_argument("--soft", required=True, help="path to soft_prompt.pt")
    ap.add_argument("--adapter", default="/root/autodl-fs/sft_experiments/models/narrative_lora_r8")
    ap.add_argument("--base-model", default="/root/autodl-tmp/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-src", type=int, default=1300)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True, use_fast=True,
                                        clean_up_tokenization_spaces=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", low_cpu_mem_usage=True, local_files_only=True)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False, local_files_only=True)
    model.eval()
    model.generation_config.max_length = None
    embed = model.get_input_embeddings()

    soft = torch.load(args.soft, map_location="cpu")["soft"].to(DEV, torch.bfloat16)  # [n, H]

    gold = load_gold_with_sample_ids(args.gold)
    preds = []
    for row in gold.itertuples(index=False):
        source = str(row.source_text)
        ids = tok.encode(source, add_special_tokens=False)
        if len(ids) > args.max_src:
            source = tok.decode(ids[: args.max_src], skip_special_tokens=True)
        msgs = [
            {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
            {"role": "user", "content": f"{NARRATIVE_INSTRUCTION}\n{source}"},
        ]
        prompt_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                             return_dict=True)["input_ids"]
        emo = 0 if args.variant == "shared" else EKMAN_LABELS.index(str(row.emotion_label))
        tok_emb = embed(torch.tensor([prompt_ids], device=DEV))          # [1,T,H]
        pref = soft[emo].view(1, 1, -1)                                   # [1,1,H]
        inputs_embeds = torch.cat([pref, tok_emb], dim=1)
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=DEV)
        with torch.inference_mode():
            out = model.generate(inputs_embeds=inputs_embeds, attention_mask=attn,
                                 max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                                 use_cache=True)
        text = tok.decode(out[0], skip_special_tokens=True).strip()
        preds.append({
            "sample_id": int(row.sample_id),
            "predicted_narrative": text,
            "predicted_emotion_label": "",
            "raw_generation": text,
            "parse_success": bool(text),
            "request_error": "",
            "finish_reason": "stop",
            "model_name": Path(args.soft).parent.name,
            "run_name": f"soft_{args.variant}",
        })
        if len(preds) % 100 == 0:
            print(f"generated {len(preds)}/{len(gold)}", flush=True)

    df = pd.DataFrame(preds).sort_values("sample_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(df)} to {args.output}; empty={(~df['parse_success']).sum()}", flush=True)


if __name__ == "__main__":
    main()
