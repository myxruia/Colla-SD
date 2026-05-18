#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import sys
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re

# ======================
# Optional: lmdeploy support
# ======================
try:
    from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
    LMDeploy_AVAILABLE = True
except ImportError:
    LMDeploy_AVAILABLE = False

# ======================
# Transformers Runner (for Qwen / Llama)
# ======================
class LLM_Runner:
    def __init__(self, model_path, dtype=torch.bfloat16, QInt4=False):
        self.model_path = model_path
        configs = {
            "torch_dtype": dtype,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if QInt4:
            configs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=False
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **configs).eval()

    def Response(self, input_str, **gen_kwargs):
        torch.cuda.empty_cache()
        with torch.inference_mode():
            inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": input_str}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True
            ).to(self.model.device)
            if not gen_kwargs.get("do_sample", False):
                for key in ["temperature", "top_p", "top_k"]:
                    gen_kwargs.pop(key, None)
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs.input_ids.shape[1]:]
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# ======================
# LMDeploy Runner (for InternLM)
# ======================
class LMDeploy_Runner:
    def __init__(self, model_path):
        if not LMDeploy_AVAILABLE:
            raise ImportError("lmdeploy is not installed.")
        backend_config = TurbomindEngineConfig(
            model_format='hf',
            session_len=4096,
            cache_max_entry_count=0.8,
            tp=1
        )
        self.pipe = pipeline(model_path, backend_config=backend_config)

    def Response(self, input_str, **gen_kwargs):
        messages = [{"role": "user", "content": input_str}]
        gen_config = GenerationConfig(
            max_new_tokens=gen_kwargs.get("max_new_tokens", 512),
            temperature=gen_kwargs.get("temperature", 0.0),
            top_p=gen_kwargs.get("top_p", 1.0),
            top_k=gen_kwargs.get("top_k", 1),
            repetition_penalty=1.0,
            do_sample=gen_kwargs.get("do_sample", False)
        )
        response = self.pipe(messages, gen_config=gen_config)
        if isinstance(response, list):
            return response[0].text
        else:
            return response.text

# ======================
# Parsing & Prompt (unchanged)
# ======================

import ast

def parse_model_output(text: str):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    blocks = re.split(r'\n\s*\n', text.strip())
    first_block = ""
    for block in blocks:
        b = block.strip()
        if ('intents:' in b and 'slot label-value pairs:' in b):
            first_block = b
            break
    if not first_block:
        first_block = text

    intents = []
    slot_pairs = []

    intent_match = re.search(r'intents\s*:\s*(\[[^\n]*\])', first_block, re.IGNORECASE)
    if intent_match:
        try:
            intents = ast.literal_eval(intent_match.group(1))
            if not isinstance(intents, list):
                intents = []
            intents = [str(x).strip() for x in intents if isinstance(x, str)]
        except Exception as e:
            print(f"⚠️ Intent parse error: {e}")

    slot_match = re.search(r'slot label-value pairs\s*:\s*(\[[^\n]*\])', first_block, re.IGNORECASE)
    if slot_match:
        try:
            raw_list = ast.literal_eval(slot_match.group(1))
            if not isinstance(raw_list, list):
                raw_list = []
            for item in raw_list:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    slot_label = str(item[0]).strip()
                    slot_value = str(item[1]).strip()
                    slot_pairs.append({"slot_label": slot_label, "slot_value": slot_value})
        except Exception as e:
            print(f"⚠️ Slot parse error: {e}")

    return intents, slot_pairs

def build_parsing_prompt(utterance, intent_descriptions, slot_names):
    intent_lines = [f"- {intent}: {desc}" for intent, desc in intent_descriptions.items()]
    slot_lines = [f"- {slot}" for slot in slot_names]
    return f"""You are an expert in aviation-domain spoken language understanding. Your task is to analyze the user utterance and output the expressed intents and the included slot label-value pairs:

1. **Intents**: Choose ONLY from the following labeled intents:
{chr(10).join(intent_lines)}

2. **Slots**: Extract explicit slot-value pairs where slot labels are ONLY from the following list:
{chr(10).join(slot_lines)}

### Rules:
- Only include intents/slots that are clearly expressed in the utterance.
- Slot values must be **exact substrings** from the utterance (case-sensitive, no normalization).
- Do NOT infer or guess missing information.
- **At most 3 intents** may be selected. If more than 3 are expressed, choose the 3 most prominent.
- **At most 4 slot-value pairs** may be extracted. If more than 4 are present, choose the 4 most relevant.

Please Generate strictly according to the following format. No explanations. No extra text.
format:

intents: ["intent1", "intent2", ...]
slot label-value pairs: [["slot label 1", "slot value 1"], ["slot label 2", "slot value 2"],...]
Utterance: "{utterance}"
"""

# ======================
# Main
# ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["transformers", "lmdeploy"], required=True)
    parser.add_argument("--input_path", type=str, default="/home/myxruia/SLU/generate_pipeline/stage2/unique_scored.json")
    parser.add_argument("--cache_dir", type=str, default="/home/myxruia/SLU/generate_pipeline/stage3/")
    parser.add_argument("--use_quant", action="store_true", help="Enable 4-bit quantization (ignored for lmdeploy)")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    final_output = os.path.join(args.cache_dir, f"{args.model_name}_predictions.json")
    tmp_output = final_output + ".progress.jsonl"

    # Load descriptions
    with open("/home/myxruia/SLU/des/intent_des.txt", "r", encoding="utf-8") as f:
        intent_descriptions = {}
        for line in f:
            line = line.strip()
            if line and ":" in line:
                intent, desc = line.split(":", 1)
                intent_descriptions[intent.strip()] = desc.strip()
    all_intents_list = list(intent_descriptions.keys())

    with open("/home/myxruia/SLU/des/slots.txt", "r", encoding="utf-8") as f:
        all_slots = [line.strip() for line in f if line.strip()]

    with open(args.input_path, "r", encoding="utf-8") as f:
        original_samples = json.load(f)
    total = len(original_samples)
    print(f"🔍 Total samples: {total}")

    done_indices = set()
    if os.path.exists(tmp_output):
        with open(tmp_output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        done_indices.add(item["__idx__"])
                    except:
                        continue
        print(f"⏭️  Already processed: {len(done_indices)} samples")

    # Load model based on type
    print(f"🚀 Loading {args.model_name} via {args.model_type}")
    if args.model_type == "lmdeploy":
        if not LMDeploy_AVAILABLE:
            raise RuntimeError("lmdeploy not installed but required for InternLM.")
        runner = LMDeploy_Runner(args.model_path)
    elif args.model_type == "transformers":
        runner = LLM_Runner(args.model_path, QInt4=args.use_quant)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    with open(tmp_output, "a", encoding="utf-8") as fout:
        for idx, sample in enumerate(tqdm(original_samples, desc=f"Running {args.model_name}")):
            if idx in done_indices:
                continue

            utterance = sample["generated_utterance"]
            gold_intents = sample.get("intents", [])
            gold_slot_label_value_pairs = sample.get("slot label_value_pairs", [])

            prompt = build_parsing_prompt(utterance, intent_descriptions, all_slots)
            try:
                response = runner.Response(prompt, max_new_tokens=512, do_sample=False)
                intents, pred_slot_label_value_pairs = parse_model_output(response)

                intents = [i for i in intents if i in all_intents_list]
                pred_slot_label_value_pairs = [
                    p for p in pred_slot_label_value_pairs if p["slot_label"] in all_slots
                ]
            except Exception as e:
                print(f"⚠️ Error on sample {idx}: {e}")
                intents, pred_slot_label_value_pairs, response = [], [], str(e)

            intent_em = set(gold_intents) == set(intents)
            slot_em = set((p["slot_label"], p["slot_value"]) for p in gold_slot_label_value_pairs) == \
                      set((p["slot_label"], p["slot_value"]) for p in pred_slot_label_value_pairs)

            result = {
                "__idx__": idx,
                "generated_utterance": utterance,
                "model_used": sample.get("model_used"),
                "gold_intents": gold_intents,
                "gold_slot label_value_pairs": gold_slot_label_value_pairs,
                "intents": sorted(intents),
                "slot label_value_pairs": pred_slot_label_value_pairs,
                "intent_exact_match": intent_em,
                "slot_exact_match": slot_em,
                "stage3_raw_output": response
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

    # Finalize
    results = []
    with open(tmp_output, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    results.sort(key=lambda x: x["__idx__"])
    for r in results:
        del r["__idx__"]

    with open(final_output, "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=4, ensure_ascii=False)

    os.remove(tmp_output)
    print(f"✅ Final predictions saved to: {final_output}")

if __name__ == "__main__":
    main()