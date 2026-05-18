#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Run ONLY small models (NO quantization)
# Supports three model combinations via SLU_COMBO env var

import subprocess
import sys
import os
import json
from collections import Counter

BASE_DIR = ""
CODE_DIR = ""

# === Model Combinations ===
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

# Select combo via environment variable
COMBO_NAME = os.getenv("SLU_COMBO", "qwen_llama")
if COMBO_NAME not in COMBINATIONS:
    print(f"❌ Invalid SLU_COMBO: {COMBO_NAME}. Choose from: {list(COMBINATIONS.keys())}")
    sys.exit(1)

MODELS = COMBINATIONS[COMBO_NAME]
CACHE_DIR = f"{BASE_DIR}/stage3/qwen_llama_22000/{COMBO_NAME}"  # ← 分组合缓存
INPUT_PATH = f"{BASE_DIR}/stage2/qwen_llama_22000/unique_filtered.json"


CHUNK_SIZE = 500

os.makedirs(CACHE_DIR, exist_ok=True)

# --- 工具函数 ---
def run_command(cmd):
    print(f"▶️  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("❌ Command failed!")
        sys.exit(1)

def normalize_slot_pairs(slot_label_value_pairs):
    return frozenset(
        (item["slot_label"], item["slot_value"])
        for item in slot_label_value_pairs
    )

def process_chunk(chunk_data, start_idx, end_idx):
    chunk_input = os.path.join(CACHE_DIR, f"chunk_{start_idx:05d}_{end_idx:05d}.json")
    chunk_cache_dir = os.path.join(CACHE_DIR, f"chunk_{start_idx:05d}_{end_idx:05d}")
    os.makedirs(chunk_cache_dir, exist_ok=True)

    with open(chunk_input, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, ensure_ascii=False)

    # Run each model in the current combo
    for model_name, model_path, model_type in MODELS:
        cmd = [
            sys.executable,
            os.path.join(CODE_DIR, "run_single_model.py"),
            "--model_name", model_name,
            "--model_path", model_path,
            "--model_type", model_type,
            "--input_path", chunk_input,
            "--cache_dir", chunk_cache_dir
            # NO --use_quant for small models
        ]
        run_command(cmd)

    # Aggregate predictions (same logic)
    model_names = [name for name, _, _ in MODELS]
    all_preds = []
    for model in model_names:
        pred_file = os.path.join(chunk_cache_dir, f"{model}_predictions.json")
        with open(pred_file, "r", encoding="utf-8") as f:
            all_preds.append(json.load(f))

    aligned_samples = list(zip(*all_preds))
    agreement_samples = []
    disagreement_samples = []

    for model_preds in aligned_samples:
        raw_utterance = model_preds[0]["generated_utterance"]
        gold = {
            "intents": model_preds[0]["gold_intents"],
            "slot label_value_pairs": model_preds[0]["gold_slot label_value_pairs"]
        }

        pred_contents = [
            {
                "intents": p["intents"],
                "slot label_value_pairs": p["slot label_value_pairs"]
            }
            for p in model_preds
        ]

        normalized = [
            (
                frozenset(p["intents"]),
                normalize_slot_pairs(p["slot label_value_pairs"])
            )
            for p in pred_contents
        ]

        counter = Counter(normalized)
        most_common_count = counter.most_common(1)[0][1]

        sample_record = {
            "utterance": raw_utterance,
            "gold": gold,
            "pred_a": pred_contents[0],
            "pred_b": pred_contents[1]
        }

        if most_common_count >= 2:
            agreement_samples.append(sample_record)
        else:
            disagreement_samples.append(sample_record)

    agreement_file = os.path.join(CACHE_DIR, f"agreement_samples_batch_{start_idx:05d}-{end_idx:05d}.json")
    with open(agreement_file, "w", encoding="utf-8") as f:
        json.dump(agreement_samples, f, indent=4, ensure_ascii=False)

    disagreement_file = os.path.join(CACHE_DIR, f"disagreement_samples_batch_{start_idx:05d}-{end_idx:05d}.json")
    with open(disagreement_file, "w", encoding="utf-8") as f:
        json.dump(disagreement_samples, f, indent=4, ensure_ascii=False)

    print(f"✅ Chunk {start_idx}-{end_idx}: "
          f"{len(agreement_samples)} agreement, "
          f"{len(disagreement_samples)} disagreement → saved")


if __name__ == "__main__":
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    total = len(original_data)
    print(f"🔍 Total samples: {total}, splitting into chunks of {CHUNK_SIZE}")
    print(f"🎯 Running combo: {COMBO_NAME}")

    for i in range(0, total, CHUNK_SIZE):
        chunk = original_data[i:i + CHUNK_SIZE]
        start_idx = i
        end_idx = min(i + CHUNK_SIZE - 1, total - 1)
        print(f"\n📦 Processing chunk {start_idx} to {end_idx} ({len(chunk)} samples)")
        process_chunk(chunk, start_idx, end_idx)

    print("\n🎉 All chunks processed successfully for combo:", COMBO_NAME)