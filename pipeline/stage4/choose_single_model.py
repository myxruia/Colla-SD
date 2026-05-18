#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import re
import argparse
from tqdm import tqdm

# ======================
# Optional: lmdeploy
# ======================
try:
    from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
    LMDeploy_AVAILABLE = True
except ImportError:
    LMDeploy_AVAILABLE = False

# ======================
# Transformers Runner (NO quantization)
# ======================
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class TransformersRunner:
    def __init__(self, model_path):
        self.model_path = model_path
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        ).eval()

    def generate(self, prompt, max_new_tokens=512):
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        output = self.tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        return output

# ======================
# LMDeploy Runner
# ======================
class LMDeployRunner:
    def __init__(self, model_path):
        if not LMDeploy_AVAILABLE:
            raise ImportError("lmdeploy not installed.")
        backend_config = TurbomindEngineConfig(
            model_format='hf',
            session_len=4096,
            cache_max_entry_count=0.8,
            tp=1
        )
        self.pipe = pipeline(model_path, backend_config=backend_config)

    def generate(self, prompt, max_new_tokens=512):
        messages = [{"role": "user", "content": prompt}]
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            repetition_penalty=1.0,
            do_sample=False
        )
        response = self.pipe(messages, gen_config=gen_config)
        if isinstance(response, list):
            return response[0].text
        else:
            return response.text

# ======================
# Shared utilities
# ======================
SLOT_VOCAB_PATH = ""
INTENT_DESC_PATH = ""

def load_vocab_set(path):
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}

def load_intent_descriptions_and_set(path):
    intent_to_desc = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, desc = line.split(":", 1)
                key = key.strip()
                desc = desc.strip()
                if key and desc:
                    intent_to_desc[key] = desc
    return set(intent_to_desc.keys()), intent_to_desc

VALID_SLOTS = load_vocab_set(SLOT_VOCAB_PATH)
VALID_INTENTS, _ = load_intent_descriptions_and_set(INTENT_DESC_PATH)

def parse_model_output(raw_text, valid_intents, valid_slots):
    try:
        lines = raw_text.split('\n')
        intent_str = None
        slot_str = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if re.match(r'intents\s*:', line, re.IGNORECASE):
                intent_str = line
            elif re.match(r'slot\s+label-value\s+pairs\s*:', line, re.IGNORECASE):
                slot_str = line

        intents = []
        if intent_str:
            m = re.search(r'\[(.*)\]', intent_str)
            if m:
                try:
                    arr = json.loads("[" + m.group(1).strip() + "]") if m.group(1).strip() else []
                    intents = sorted(set(it for it in arr if isinstance(it, str) and it in valid_intents))
                except:
                    pass

        slots = []
        if slot_str:
            m = re.search(r'\[(.*)\]', slot_str)
            if m:
                try:
                    json_part = "[" + m.group(1).strip() + "]"
                    arr = json.loads(json_part)
                    if isinstance(arr, list):
                        for item in arr:
                            if isinstance(item, list) and len(item) == 2:
                                label, value = item[0], item[1]
                                if isinstance(label, str) and isinstance(value, str) and label in valid_slots:
                                    slots.append({"slot_label": label, "slot_value": value})
                except:
                    pass

        return {"intents": intents, "slot label_value_pairs": slots}
    except:
        return None

# ======================
# Main
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_type", choices=["transformers", "lmdeploy"], required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    print(f"🚀 Loading {args.model_name} via {args.model_type} (NO quantization)")

    if args.model_type == "lmdeploy":
        if not LMDeploy_AVAILABLE:
            raise RuntimeError("lmdeploy not available for InternLM.")
        runner = LMDeployRunner(args.model_path)
    elif args.model_type == "transformers":
        runner = TransformersRunner(args.model_path)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    with open(args.input_file, encoding="utf-8") as f:
        tasks = json.load(f)

    results = []

    for task in tqdm(tasks, desc=f"Adjudicating ({args.model_name})", unit="sample"):
        try:
            raw_output = runner.generate(task["prompt"], max_new_tokens=512)
            parsed = parse_model_output(raw_output, VALID_INTENTS, VALID_SLOTS)
        except Exception as e:
            raw_output = str(e)
            parsed = None

        if parsed is None:
            # Fallback to pred_a
            parsed = {
                "intents": task["pred_a"]["intents"],
                "slot label_value_pairs": task["pred_a"]["slot label_value_pairs"]
            }

        results.append({
            "id": task["id"],
            "utterance": task["utterance"],
            "gold": task["gold"],
            "pred_a": task["pred_a"],
            "pred_b": task["pred_b"],
            "adjudicated": parsed,
            "model_raw_output": raw_output
        })

    model_short = "internlm" if "internlm" in args.model_name.lower() else \
                  "llama" if "llama" in args.model_name.lower() else "qwen"

    output_path = os.path.join(args.output_dir, f"{model_short}_adjudicated.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Results saved to: {output_path}")

if __name__ == "__main__":
    main()