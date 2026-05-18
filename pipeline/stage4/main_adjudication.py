#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import sys
import random
import subprocess
from pathlib import Path

# ===== 配置：三种模型组合（全部为 small models → NO quantization）=====
COMBINATIONS = {
    "qwen_llama": [
        ("Qwen2.5-7B", "", "transformers"),
        ("Llama3-8B", "", "transformers")
    ],
    "qwen_internlm": [
        ("Qwen2.5-7B", "", "transformers"),
        ("internlm3-8b", "", "lmdeploy")
    ],
    "llama_internlm": [
        ("Llama3-8B", "", "transformers"),
        ("internlm3-8b", "", "lmdeploy")
    ]
}

COMBO_NAME = os.getenv("SLU_COMBO", "qwen_llama")
if COMBO_NAME not in COMBINATIONS:
    print(f"❌ Invalid SLU_COMBO: {COMBO_NAME}. Choose from: {list(COMBINATIONS.keys())}")
    sys.exit(1)

MODELS = COMBINATIONS[COMBO_NAME]

INPUT_DISAGREEMENT_FILE = f""
OUTPUT_DIR = f""

SLOT_VOCAB_PATH = ""
INTENT_DESC_PATH = ""

CHUNK_SIZE = 500

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
VALID_INTENTS, INTENT_DESCRIPTIONS = load_intent_descriptions_and_set(INTENT_DESC_PATH)

intent_defs_lines = [f"- {intent}: {desc}" for intent, desc in sorted(INTENT_DESCRIPTIONS.items())]
INTENT_DEFINITIONS_BLOCK = "## Intent Definitions:\n" + "\n".join(intent_defs_lines)
valid_slots_str = ", ".join(f'"{sl}"' for sl in sorted(VALID_SLOTS))

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def run_cmd(cmd):
    print(f"▶️ Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("❌ Failed!")
        sys.exit(1)

def format_parse(p):
    intents = ", ".join(sorted(p.get("intents", [])))
    slot_list = p.get("slot label_value_pairs", [])
    slots = "; ".join(f"{item['slot_label']}: {item['slot_value']}" for item in slot_list) if slot_list else ""
    return f"[Intents: {intents}] [Slots: {slots}]"

def main():
    if not os.path.exists(INPUT_DISAGREEMENT_FILE):
        print(f"⚠️ Disagreement file not found: {INPUT_DISAGREEMENT_FILE}")
        print("👉 Run Stage 3 for this combo first.")
        sys.exit(1)

    with open(INPUT_DISAGREEMENT_FILE, encoding="utf-8") as f:
        all_samples = json.load(f)
    print(f"🔍 Loaded {len(all_samples)} disagreement samples (NO quantization)")
    random.shuffle(all_samples)

    for i in range(0, len(all_samples), CHUNK_SIZE):
        chunk = all_samples[i:i + CHUNK_SIZE]
        chunk_id = f"{i:05d}_{min(i + CHUNK_SIZE - 1, len(all_samples) - 1):05d}"
        chunk_dir = os.path.join(OUTPUT_DIR, f"chunk_{chunk_id}")
        os.makedirs(chunk_dir, exist_ok=True)

        mid = len(chunk) // 2
        group_a = chunk[:mid]
        group_b = chunk[mid:]

        for idx, (group_name, samples, (model_name, model_path, model_type)) in enumerate([
            ("model_a", group_a, MODELS[0]),
            ("model_b", group_b, MODELS[1])
        ]):
            if not samples:
                continue

            tasks = []
            for s in samples:
                prompt = f"""You are an expert in aviation-domain spoken language understanding.
Your task is to produce the best possible semantic parse for the following user utterance.

## Constraints:
- You MUST ONLY use intents from this predefined set: [{", ".join(f'"{it}"' for it in sorted(VALID_INTENTS))}]
- You MUST ONLY use slot labels from this predefined set: [{valid_slots_str}]
- Slot values MUST be exact, case-sensitive substrings of the utterance.
- Do NOT invent new intents or slots outside the sets above.
- If none of the candidate parses are fully correct, synthesize a new result using only allowed intents and slots that appear in the utterance.

{INTENT_DEFINITIONS_BLOCK}

## User utterance:
"{s['utterance']}"

## Candidate parses (may contain errors or illegal labels):
- Gold reference: {format_parse(s['gold'])}
- Model A prediction: {format_parse(s['pred_a'])}
- Model B prediction: {format_parse(s['pred_b'])}

## Instructions:
1. Carefully check each candidate against the constraints.
2. If all candidates violate the intent/slot vocabulary or extract wrong values, IGNORE them and generate a correct parse from scratch.
3. Only include intents/slots that are explicitly supported by words in the utterance.
4.**At most 3 intents** may be selected. If more than 3 are expressed, choose the 3 most prominent.
5.**At most 4 slot-value pairs** may be extracted. If more than 4 are present, choose the 4 most relevant.

## Output format (STRICTLY follow this syntax, no explanations, no extra text):
intents: ["intent1", "intent2", ...]
slot label-value pairs: [["slot label 1", "slot value 1"], ["slot label 2", "slot value 2"],...]
"""
                tasks.append({
                    "id": s.get("id", s["utterance"][:20]),
                    "utterance": s["utterance"],
                    "gold": s["gold"],
                    "pred_a": s["pred_a"],
                    "pred_b": s["pred_b"],
                    "prompt": prompt
                })

            input_file = os.path.join(chunk_dir, f"adjud_tasks_{group_name}.json")
            with open(input_file, "w", encoding="utf-8") as f:
                json.dump(tasks, f, indent=2, ensure_ascii=False)

            # 调用推理脚本 —— 不传量化参数（默认关闭）
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "choose_single_model.py"),
                "--model_name", model_name,
                "--model_path", model_path,
                "--model_type", model_type,
                "--input_file", input_file,
                "--output_dir", chunk_dir
            ]
            run_cmd(cmd)

    print(f"✅ Adjudication completed for combo: {COMBO_NAME}")

if __name__ == "__main__":
    main()