import json
import os
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch

# === 配置路径 ===
input_path = ""
deduped_output_dir = ""

os.makedirs(deduped_output_dir, exist_ok=True)

# 1. 加载数据
print(f"📂 Loading data from: {input_path}")
with open(input_path, "r", encoding="utf-8") as f:
    samples = json.load(f)
print(f"✅ Loaded {len(samples)} samples.")

# 2. 去重（保留原始逻辑 + 添加进度条）
print("\n🔍 Deduplicating with Sentence-BERT (exact, O(N^2))...")
try:
    local_sbert_path = "/data/ha/all-MiniLM-L6-v2/AI-ModelScope/all-MiniLM-L6-v2"
    sbert_model = SentenceTransformer(local_sbert_path)

    utterances = [s["generated_utterance"] for s in samples]
    print("  → Encoding utterances...")
    embeddings = sbert_model.encode(utterances, convert_to_tensor=True, show_progress_bar=True,batch_size=1024)
    cos_sim_matrix = cos_sim(embeddings, embeddings)

    keep_indices = []
    duplicate_indices = []

    print("  → Comparing pairwise similarities...")
    from tqdm import tqdm
    for i in tqdm(range(len(utterances)), desc="Dedup progress"):
        is_duplicate = False
        for j in keep_indices:
            if cos_sim_matrix[i][j] > 0.90:
                is_duplicate = True
                duplicate_indices.append(i)
                break
        if not is_duplicate:
            keep_indices.append(i)

    deduped_samples = [samples[i] for i in keep_indices]
    duplicate_samples = [samples[i] for i in duplicate_indices]

    print(f"✅ Deduplication: {len(samples)} → {len(deduped_samples)} unique")

except Exception as e:
    print(f"⚠️ Dedup failed: {e}. Using original data.")
    deduped_samples = samples
    duplicate_samples = []


# 3. 保存全局去重结果（可选）
global_dedup_path = os.path.join(deduped_output_dir, "unique.json")
with open(global_dedup_path, "w", encoding="utf-8") as f:
    json.dump(deduped_samples, f, indent=4, ensure_ascii=False)
print(f"💾 Global unique saved to: {global_dedup_path}")

duplicates_path = os.path.join(deduped_output_dir, "duplicates.json")
with open(duplicates_path, "w", encoding="utf-8") as f:
    json.dump(duplicate_samples, f, indent=4, ensure_ascii=False)

# 4. ✅ 按 model_used 拆分去重后的样本
print("\n🔀 Splitting deduplicated samples by 'model_used'...")

# 标准化模型名映射（匹配 COMBINATIONS 中的名字）
MODEL_NAME_MAP = {
    "Qwen2.5-7B": "qwen2_5_7b",
    "Llama3-8B": "llama3_8b",
    "internlm3-8b": "internlm3_8b",
    # 如果原始数据中 model_used 是其他写法，加在这里
    "Internlm3-8B": "internlm3_8b",  # 注意大小写
    "Qwen": "qwen2_5_7b",
    "Llama": "llama3_8b",
}

model_groups = defaultdict(list)
for sample in deduped_samples:
    raw_model = sample.get("model_used", "unknown")
    # 标准化
    clean_model = MODEL_NAME_MAP.get(raw_model, raw_model.lower().replace('-', '_').replace('.', '_'))
    model_groups[clean_model].append(sample)

# 保存每个模型的去重结果
for model_name, group in model_groups.items():
    output_file = os.path.join(deduped_output_dir, f"{model_name}_unique.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(group, f, indent=4, ensure_ascii=False)
    print(f"📁 Saved {len(group)} samples for model '{model_name}' to: {output_file}")

print("\n✅ Deduplication and splitting completed.")