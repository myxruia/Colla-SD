import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import re

# === 配置路径 ===
INPUT_PATH = ""
OUTPUT_PATH = ""
QWEN_MODEL_PATH = ""  # 请确认路径正确

# === 加载 4-bit 量化模型 ===
print(f"Loading Qwen2.5-72B-Instruct in 4-bit quantization from {QWEN_MODEL_PATH}...")

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",  # 自动分配到可见 GPU（CUDA_VISIBLE_DEVICES=2,3）
    trust_remote_code=True
)
model.eval()
print("✅ Model loaded with 4-bit quantization.")

# === 构造评分提示 ===
def build_prompt(sample):
    slot_values = sample.get("slot_values", {})
    intents = sample.get("intents", [])
    utterance = sample["generated_utterance"].strip()

    slots_str = ", ".join([f'"{k}": "{v}"' for k, v in slot_values.items()]) if slot_values else "none"
    intents_str = ", ".join([f'"{i}"' for i in intents]) if intents else "none"

    prompt = f"""You are an expert evaluator for spoken language understanding (SLU) data in the aviation domain.  
Please score the following user utterance on four dimensions from 1 to 5, based **solely on the quality of the natural language**, not on how well it matches predefined slots or intents.

**Scoring Criteria**:
1. **Fluency**: Is the utterance grammatically correct, smoothly phrased, and free of awkward or unnatural constructions?
2. **Naturalness**: Does it sound like something a real passenger would actually say in a conversational or query context?
3. **Semantic Richness**: Does the utterance convey meaningful, specific, and non-trivial information (e.g., includes concrete details, avoids vagueness)?
4. **Logical Coherence**: Is the content internally consistent and free of contradictions or implausible combinations?

**Input**:
- Utterance: "{utterance}"

Output exactly in this format (no extra text, no JSON braces, no code blocks):
"fluency": X, "naturalness": Y, "semantic_richness": Z, "logical_coherence": W 
- Example: "fluency": 4, "naturalness": 5, "semantic_richness": 4, "logical_coherence": 5
"""
    return prompt


def get_score_from_qwen(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            pattern = r'"(\w+)":\s*(\d+)'
            matches = re.findall(pattern, response)
            score_dict = {key: int(value) for key, value in matches}

            required = ["fluency", "naturalness", "semantic_richness", "logical_coherence"]
            if not all(k in score_dict for k in required):
                raise ValueError(f"Missing keys. Got: {list(score_dict.keys())}")

            for k in required:
                if not (1 <= score_dict[k] <= 5):
                    raise ValueError(f"Score out of range for {k}: {score_dict[k]}")

            return {k: score_dict[k] for k in required}

        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}. Response: '{response}'")
            if attempt == max_retries - 1:
                return {
                    "fluency": 3,
                    "naturalness": 3,
                    "semantic_richness": 3,
                    "logical_coherence": 3
                }
    return {
        "fluency": 3,
        "naturalness": 3,
        "semantic_richness": 3,
        "logical_coherence": 3
    }


# === 主流程 ===
if __name__ == "__main__":
    # 加载数据
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples.")

    kept_samples = []
    for sample in tqdm(data, desc="Scoring with Qwen-72B (4-bit)"):
        prompt = build_prompt(sample)
        scores = get_score_from_qwen(prompt)

        # ✅ 改为：计算四个维度的简单平均分
        avg_score = sum(scores.values()) / len(scores)
        scores["average_score"] = round(avg_score, 2)

        # ✅ 只保留 average_score >= 3.0 的样本
        if avg_score >= 3.0:
            sample["quality_scores"] = scores
            kept_samples.append(sample)

    # 保存过滤后的结果
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(kept_samples, f, indent=4, ensure_ascii=False)

    print(f"✅ Kept {len(kept_samples)} / {len(data)} samples (avg_score >= 3.0).")
    print(f"✅ Saved filtered results to {OUTPUT_PATH}")

    del model
    torch.cuda.empty_cache()
    print("✅ Model released and memory cleared.")