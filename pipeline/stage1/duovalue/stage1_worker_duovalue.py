# stage1_worker_new.py
import os
import json
import re
import ast
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import random 
random.seed(42)

# ======================
# Configurable paths
# ======================
INTENT_DESC_PATH = ""
SLOT_DESC_PATH = ""


# ======================
# Load descriptions (shared)
# ======================
def _load_descriptions():
    intent_descriptions = {}
    with open(INTENT_DESC_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and ":" in line:
                intent, desc = line.split(":", 1)
                intent_descriptions[intent.strip()] = desc.strip()

    slot_descriptions = {}
    with open(SLOT_DESC_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and ":" in line:
                slot, desc = line.split(":", 1)
                slot_descriptions[slot.strip()] = desc.strip()

    return intent_descriptions, slot_descriptions

INTENT_DESC, SLOT_DESC = _load_descriptions()
ALL_INTENTS = list(INTENT_DESC.keys())
ALL_SLOTS = list(SLOT_DESC.keys())

def load_slot_pool(pool_dir=""):
    """
    从 pool/ 目录加载所有 slot label 对应的候选值。
    返回字典：{slot_label: [value1, value2, ...]}
    """
    slot_pool = {}
    for filename in os.listdir(pool_dir):
        if filename.endswith(".txt"):
            slot_label = os.path.splitext(filename)[0]  # e.g., "city" from "city.txt"
            file_path = os.path.join(pool_dir, filename)
            values = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    value = line.strip()
                    if value:
                        values.append(value)
            if values:
                slot_pool[slot_label] = values
    return slot_pool

# ======================
# 全局变量：只加载一次
# ======================
SLOT_POOL = load_slot_pool()

# ======================
# Helper: Build description block
# ======================
def build_desc_block(desc_dict, keys, default="No description available."):
    lines = []
    for key in keys:
        desc = desc_dict.get(key, default)
        lines.append(f"- {key}: {desc}")
    return "\n".join(lines)

def build_intent_desc_str(intents):
    return build_desc_block(INTENT_DESC, intents)

def build_slot_desc_str(slots):
    return build_desc_block(SLOT_DESC, slots)


# ======================
# Prompts (unchanged, already correct)
# ======================
def get_prompt_stage2(intents, slots):
    intent_desc_block = build_intent_desc_str(intents)
    slot_desc_block = build_slot_desc_str(slots)

    # 定义允许多值的 slot 白名单
    MULTIVALUE_SLOTS = {
        "aircraft_code",
        "airline_code",
        "airline_name",
        "airport_name",
        "arrive_date.day_name",
        "arrive_time.time",
        "arrive_time.time_relative",
        "city_name",
        "class_type",
        "cost_relative",
        "day_name",
        "day_number",
        "depart_date.day_name",
        "depart_date.day_number",
        "depart_date.month_name",
        "depart_date.today_relative",
        "depart_time.period_mod",
        "depart_time.period_of_day",
        "depart_time.time",
        "depart_time.time_relative",
        "fare_amount",
        "fare_basis_code",
        "flight_mod",
        "flight_number",
        "flight_time",
        "fromloc.airport_name",
        "fromloc.city_name",
        "meal",
        "mod",
        "restriction_code",
        "round_trip",
        "stoploc.city_name",
        "toloc.airport_code",
        "toloc.city_name",
        "transport_type"
    }
    # 找出当前 slots 中属于多值白名单的
    candidate_multivalue = [s for s in slots if s in MULTIVALUE_SLOTS]

    # 如果没有多值 slot，则随机替换一个普通 slot 为多值 slot
    if not candidate_multivalue and slots:
        # 可用于替换的多值 slots（必须也在 ALL_SLOTS 中）
        available_multivalue = list(MULTIVALUE_SLOTS & set(ALL_SLOTS))
        if available_multivalue:
            # 随机选一个要替换的位置（只替换非多值的，但此时全都是非多值）
            replace_idx = random.randrange(len(slots))
            new_slot = random.choice(available_multivalue)
            old_slot = slots[replace_idx]
            # 创建新 slots 列表（避免修改原始输入）
            slots = slots.copy()
            slots[replace_idx] = new_slot
            print(f"⚠️ No multivalue slot found. Replaced '{old_slot}' with '{new_slot}' to satisfy constraint.")
            candidate_multivalue = [new_slot]  # 更新候选
        else:
            # 极端情况：没有可用的多值 slot（理论上不会发生）
            candidate_multivalue = [slots[0]]  # fallback

    # 从 candidate_multivalue 中选择 1 或 2 个作为真正多值的
    num_multivalue = min(1, len(candidate_multivalue))
    selected_multivalue = random.sample(candidate_multivalue, num_multivalue)

    # 构建 slot_label_value_pairs
    slot_label_value_pairs = []

    for slot in slots:
        if slot in SLOT_POOL and SLOT_POOL[slot]:
            if slot in selected_multivalue:
                # 抽取两个不同的值（如果 pool 足够）
                pool_vals = SLOT_POOL[slot]
                if len(pool_vals) >= 2:
                    val1, val2 = random.sample(pool_vals, 2)
                    slot_label_value_pairs.append({"slot_label": slot, "slot_value": val1})
                    slot_label_value_pairs.append({"slot_label": slot, "slot_value": val2})
                else:
                    # 只有一个值，重复使用（或只加一次？这里加两次以保持语义“多值”）
                    val = pool_vals[0]
                    slot_label_value_pairs.append({"slot_label": slot, "slot_value": val})
                    slot_label_value_pairs.append({"slot_label": slot, "slot_value": val})
            else:
                # 单值
                value = random.choice(SLOT_POOL[slot])
                slot_label_value_pairs.append({"slot_label": slot, "slot_value": value})
        else:
            print(f"⚠️ Slot '{slot}' not found in pool or has no values. Using placeholder.")
            slot_label_value_pairs.append({"slot_label": slot, "slot_value": "N/A"})

    # 构造 pairs_desc
    pairs_desc = "\n".join(
        f"-({p['slot_label']}, {p['slot_value']})"
        for p in slot_label_value_pairs
    )

    return f"""You are an expert in Aviation domain.


Your tasks are:
1. Create a semantic DRAFT for synthesizing a natural user utterance that reflects all given intents and includes all assigned slot label-value pairs.


Given intents and descriptions: {intent_desc_block}
Given slot label-value pairs and slot descriptions:
{pairs_desc}
{slot_desc_block}

Generation Rules:
- The draft must include each slot value exactly and directly as given.
- The draft must clearly reflect **all given intents** through fluent, conversational phrasing.
- The draft should be as simple as possible.
- The draft can be a phrase, question, or command fragment — but MUST contain all provided slot label-value pairs and convey all the given intents.
- Do **not introduce** any new intent,slot or slot value beyond those provided.
- Output ONLY in the exact format below. No explanations. No extra text.



Required output format:
intents: ["intent1", "intent2", ...]
slot label-value pairs: [["slot label 1", "slot value 1"], ["slot label 2", "slot value 2"],...]
utterance_overview:[your concise natural-language draft ]"""

def get_prompt_generate(intents, utterance_overview, slot_pairs):
    intent_desc_block = build_intent_desc_str(intents)
    slot_desc_block = build_slot_desc_str([p["slot_label"] for p in slot_pairs]) 
    pairs_desc = "\n".join(
        f"-({p['slot_label']}, {p['slot_value']})"
        for p in slot_pairs
    )
    return f"""You are an expert in Aviation domain.

Your task is to generate a fluent, natural-sounding **user query** (e.g., a question, request, or command) based on the given draft and slot value-label pairs.

Draft (for phrasing inspiration only): "{utterance_overview}"

Given intents and descriptions: {intent_desc_block}
Given slot label-value pairs and slot descriptions:
{pairs_desc}
{slot_desc_block}


Rules:
1. You must include each slot value exactly as given—no paraphrasing, no omission.
2. DO NOT introduce any new intents or slot label-value pairs beyond those provided.
3. Use the draft only for tone and structure—not as a complete guide.
4. Make the utterance sound realistic and conversational, like something a real user would say to a smart speaker.
5. The output must be a **realistic user query**: typically a **question**, **command**, or **statement**.
6. Do NOT use: "Can you...", "Could you...", "I need...", "I am looking for..."
7. **Ensure the utterance is clear and concise**: Avoid redundant phrases.

Output ONLY the utterance. No explanations. No extra text."""


# ======================
# Parser (improved: optional debug)
# ======================
def parse_intents_and_slots(text: str, debug=False):
    if debug:
        print(f"❌ Raw Stage2 output:\n{text}\n{'='*50}")

    intents = []
    slot_pairs = []
    utterance_overview = ""

    intent_match = re.search(r'intents\s*:\s*(\[[^\n]*\])', text, re.IGNORECASE)
    if intent_match:
        try:
            intents = ast.literal_eval(intent_match.group(1))
        except Exception as e:
            print(f"⚠️ Failed to parse intents: {e}")

    slot_match = re.search(r'slot label-value pairs\s*:\s*(\[[^\n]*\])', text, re.IGNORECASE)
    if slot_match:
        try:
            raw_list = ast.literal_eval(slot_match.group(1))
            slot_pairs = []
            for item in raw_list:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    slot_label = str(item[0]).strip()
                    slot_value = str(item[1]).strip()
                    if slot_label in ALL_SLOTS:
                        slot_pairs.append({"slot_label": slot_label, "slot_value": slot_value})
        except Exception as e:
            print(f"⚠️ Failed to parse slot pairs: {e}")

    overview_match = re.search(r'utterance_overview\s*:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if overview_match:
        utterance_overview = overview_match.group(1).strip()

    return intents, slot_pairs, utterance_overview


# ======================
# LLM Runner with fallback for non-chat models
# ======================
class LLM_Runner:
    def __init__(self, model_name, model_path, use_qint4=False):
        self.model_name = model_name
        self.model_path = model_path
        self.use_qint4 = use_qint4

        print(f"[Transformers] Loading {self.model_name} (int4={use_qint4})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if use_qint4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )

        self.model.eval()

    def Response(self, input_str, max_new_tokens=256, temperature=1.0, top_p=0.9):
        # 尝试使用 chat template；若失败，回退到 plain prompt
        try:
            messages = [{"role": "user", "content": input_str}]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True
            )
        except Exception as e:
            # Fallback: plain concatenation (e.g., for base models without chat template)
            print(f"⚠️ Chat template not available, using plain prompt. Error: {e}")
            prompt = f"<|user|>\n{input_str}<|end|>\n<|assistant|>"
            if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is None:
                pass  # expected
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                return_token_type_ids=False
            )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id
            )
            response = outputs[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(response, skip_special_tokens=True).strip()


# ======================
# Main processing function
# ======================
def process_samples(model_name, model_path, samples, output_dir, use_qint4=False):
    print(f"\n [PID {os.getpid()}] Processing {len(samples)} samples with {model_name} (int4={use_qint4})...")

    llm = LLM_Runner(model_name=model_name, model_path=model_path, use_qint4=use_qint4)
    model_results = []

    for sample in tqdm(samples, desc=f"Stage 2+3 with {model_name}"):
        max_retries = 3
        success = False
        utterance_overview = ""
        slot_label_value_pairs = []  # ✅ FIXED: was "" → now []
        final_resp2 = ""

        # === Stage 2: Slot assignment + draft ===
        user_prompt2 = get_prompt_stage2(sample['intents'], sample['slots'])

        for attempt in range(max_retries):
            print(f"\n[DEBUG] 🟦 Stage 2 Prompt for sample {sample['id']}:\n{user_prompt2}\n{'='*80}")
            resp2 = llm.Response(
                input_str=user_prompt2,
                max_new_tokens=256,
                temperature=1.2,
                top_p=0.9
            )

            parsed_intents, slot_label_value_pairs, utterance_overview = parse_intents_and_slots(
                resp2, debug=(attempt == max_retries - 1)  # 仅最后一次重试时打印
            )

            generated_labels = {p["slot_label"] for p in slot_label_value_pairs}
            required_labels = set(sample["slots"])
            if utterance_overview and generated_labels == required_labels:
                success = True
                final_resp2 = resp2
                break

        if not success:
            print(f"⚠️ Stage 2 failed after {max_retries} retries for {sample['id']}. Skipping.")
            continue

        # === Stage 3: Final utterance generation ===
        user_prompt_gen = get_prompt_generate(
            intents=sample['intents'],
            utterance_overview=utterance_overview,
            slot_pairs=slot_label_value_pairs
        )
        print(f"\n[DEBUG] 🟩 Stage 3 Prompt for sample {sample['id']}:\n{user_prompt_gen}\n{'='*80}")
        final_utterance = llm.Response(
            input_str=user_prompt_gen,
            max_new_tokens=256,
            temperature=1.0,
            top_p=0.9
        ).strip()

        result_sample = {
            "id": sample["id"],
            "model_used": model_name,
            "intents": sample["intents"],
            "slots": sample["slots"],
            "slot label_value_pairs": slot_label_value_pairs,
            "utterance_overview": utterance_overview,
            "generated_utterance": final_utterance,
            "stage2_raw_response": final_resp2
        }
        model_results.append(result_sample)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "final_samples.json"), "w", encoding="utf-8") as f:
        json.dump(model_results, f, indent=4, ensure_ascii=False)

    print(f"✅ [PID {os.getpid()}] Saved {len(model_results)} samples for {model_name}")