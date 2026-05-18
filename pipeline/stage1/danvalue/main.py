# stage1_main.py
import os
import json
import random
import subprocess
import sys

# ======================
# Load descriptions for Stage 1
# ======================
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

print(f"Loaded {len(all_intents)} intents and {len(all_slots)} slots.")

# ======================
# Config
# ======================
N =n
output_base_dir = ""
model_configs = [

    ("Internlm3-8B","")

]


# ======================
# Stage 1: Random (I,S) pairs
# ======================
random.seed(42)
stage1_samples = []
for i in range(N):
    num_intents = random.randint(1, 3)
    num_slots = random.randint(1, 4)
    intents = random.sample(all_intents, num_intents)
    slots = random.sample(all_slots, num_slots)
    stage1_samples.append({"id": i, "intents": intents, "slots": slots})

os.makedirs(output_base_dir, exist_ok=True)
with open(os.path.join(output_base_dir, "IS_pairs.json"), "w", encoding="utf-8") as f:
    json.dump(stage1_samples, f, indent=4, ensure_ascii=False)

print(f"✅ Stage 1 completed. Saved {N} random (I,S) pairs to {output_base_dir}/IS_pairs.json")

# ======================
# Distribute samples
# ======================
num_models = len(model_configs)
samples_per_model = N // num_models
remainder = N % num_models

start = 0
all_final_samples = []

for idx, (model_name, model_path) in enumerate(model_configs):
    end = start + samples_per_model + (1 if idx < remainder else 0)
    model_samples = stage1_samples[start:end]
    start = end

    model_output_dir = os.path.join(output_base_dir, model_name)
       # Save samples to temp file for subprocess
    temp_input = os.path.join(model_output_dir, "input_samples.json")
    os.makedirs(model_output_dir, exist_ok=True)
    with open(temp_input, "w", encoding="utf-8") as f:
        json.dump(model_samples, f, indent=4, ensure_ascii=False)

    # === 调用 worker，不传 use_qint4 ===
    cmd = [
        sys.executable, "-c",
        f"""
import json, os
from worker import process_samples
with open('{temp_input}', 'r', encoding='utf-8') as f:
    samples = json.load(f)
process_samples(
    model_name='{model_name}',
    model_path='{model_path}',
    samples=samples,
    output_dir='{model_output_dir}'
)
"""
    ]

    print(f"\n▶️ Starting {model_name} in a fresh subprocess...")
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        raise RuntimeError(f"❌ {model_name} failed with exit code {result.returncode}")

    # Load results
    result_file = os.path.join(model_output_dir, "final_samples.json")
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            all_final_samples.extend(json.load(f))

# Save combined
combined_file = os.path.join(output_base_dir, "combined_final_samples.json")
with open(combined_file, "w", encoding="utf-8") as f:
    json.dump(all_final_samples, f, indent=4, ensure_ascii=False)

print(f"\n🎉 All models completed sequentially! Results saved to {combined_file}")