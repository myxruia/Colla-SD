# stage1_main.py
import os
import json
import random
import subprocess
import sys

# Load descriptions
with open("", "r", encoding="utf-8") as f:
    intent_descriptions = {}
    for line in f:
        line = line.strip()
        if line and ":" in line:
            intent, desc = line.split(":", 1)
            intent_descriptions[intent.strip()] = desc.strip()

with open("", "r", encoding="utf-8") as f:
    slot_descriptions = {}
    for line in f:
        line = line.strip()
        if line and ":" in line:
            slot, desc = line.split(":", 1)
            slot_descriptions[slot.strip()] = desc.strip()

all_intents = list(intent_descriptions.keys())
all_slots = list(slot_descriptions.keys())

N = 
output_base_dir = ""
model_configs = [
    ("Llama3-8B", ""),
    ("Qwen2.5-7B", ""),
]

# Generate or load IS pairs
is_pairs_path = os.path.join(output_base_dir, "IS_pairs.json")
if os.path.exists(is_pairs_path):
    with open(is_pairs_path, "r", encoding="utf-8") as f:
        stage1_samples = json.load(f)
else:
    random.seed(42)
    stage1_samples = []
    for i in range(N):
        num_intents = random.randint(1, 3)
        num_slots = random.randint(1, 4)
        intents = random.sample(all_intents, num_intents)
        slots = random.sample(all_slots, num_slots)
        stage1_samples.append({"id": i, "intents": intents, "slots": slots})
    os.makedirs(output_base_dir, exist_ok=True)
    with open(is_pairs_path, "w", encoding="utf-8") as f:
        json.dump(stage1_samples, f, indent=4, ensure_ascii=False)

print(f"✅ Loaded {len(stage1_samples)} (I,S) pairs.")

# Distribute samples
num_models = len(model_configs)
samples_per_model = len(stage1_samples) // num_models
remainder = len(stage1_samples) % num_models

start = 0
all_final_samples = []

for idx, (model_name, model_path) in enumerate(model_configs):
    end = start + samples_per_model + (1 if idx < remainder else 0)
    model_samples = stage1_samples[start:end]
    start = end

    model_output_dir = os.path.join(output_base_dir, model_name)
    temp_input = os.path.join(model_output_dir, "input_samples.json")
    os.makedirs(model_output_dir, exist_ok=True)
    with open(temp_input, "w", encoding="utf-8") as f:
        json.dump(model_samples, f, indent=4, ensure_ascii=False)

    cmd = [
        sys.executable, "-c",
        f"""
import json, os
from stage1_worker_new import process_samples
with open('{temp_input}', 'r', encoding='utf-8') as f:
    samples = json.load(f)
process_samples(
    model_name='{model_name}',
    model_path='{model_path}',
    samples=samples,
    output_dir='{model_output_dir}',
    use_qint4=False
)
"""
    ]

    print(f"\n▶️ Starting {model_name} in subprocess...")
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        raise RuntimeError(f"❌ {model_name} failed with exit code {result.returncode}")

    result_file = os.path.join(model_output_dir, "final_samples.json")
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            all_final_samples.extend(json.load(f))

combined_file = os.path.join(output_base_dir, "combined_small.json")
with open(combined_file, "w", encoding="utf-8") as f:
    json.dump(all_final_samples, f, indent=4, ensure_ascii=False)

print(f"\n🎉 Small models completed! Results saved to {combined_file}")
