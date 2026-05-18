# stage1_worker_new.py
import os
import json
import re
import ast
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ======================
# Load descriptions (shared)
# ======================
def _load_descriptions():
    intent_file = ""
    slot_file = ""

    with open(intent_file, "r", encoding="utf-8") as f:
        intent_descriptions = {}
        for line in f:
            line = line.strip()
            if line and ":" in line:
                intent, desc = line.split(":", 1)
                intent_descriptions[intent.strip()] = desc.strip()

    with open(slot_file, "r", encoding="utf-8") as f:
        slot_descriptions = {}
        for line in f:
            line = line.strip()
            if line and ":" in line:
                slot, desc = line.split(":", 1)
                slot_descriptions[slot.strip()] = desc.strip()

    return intent_descriptions, slot_descriptions

INTENT_DESC, SLOT_DESC = _load_descriptions()
ALL_INTENTS = list(INTENT_DESC.keys())
ALL_SLOTS = list(SLOT_DESC.keys())

def build_intent_desc_str(intents):
    lines = []
    for intent in intents:
        desc = INTENT_DESC.get(intent, "No description")
        lines.append(f"- {intent}: {desc}")
    return "\n".join(lines)

# ======================
# System prompts
# ======================
def get_prompt_stage2(intents, slots):
    intent_desc_block = build_intent_desc_str(intents)
    return f"""You are an expert in Aviation domain.

Available Intent Descriptions:
{intent_desc_block}

Your tasks are:
1. Assign a realistic slot value to each given slot label (keep values consistent with real-world usage).
2. Create a semantic DRAFT for synthesizing a user query that reflects all given intents and includes all slot value-label pairs.

Given intents: {', '.join(intents)}
Given slot labels: {', '.join(slots)}

Generation Rules:
- The draft must clearly reflect **all given intents** through natural phrasing.
- Express each slot value naturally in the draft.
- The draft can be a phrase or question fragment — but MUST contain all slot label-value pairs and convey all the given intents.
- Do **not introduce** any new intent or slot beyond those provided.
- Output ONLY in the exact format below. No explanations. No extra text.

Examples of GOOD drafts:
- "non-refundable (NR) ticket on American Airlines (AA)"
- "price of the cheapest non-refundable fare on American Airlines from JFK to LAX"
- "which airline flies from Boston to Denver on a 737?"
- "cheapest flight from NYC to Denver on Monday morning"
- "a vegetarian meal on the United flight to Chicago"

Required output format:
intents: ["intent1", "intent2", ...]
slot label-value pairs: [["slot label 1", "slot value 1"], ["slot label 2", "slot value 2"],...]
utterance_overview:[your concise draft with all slot values]"""

def get_prompt_generate(intents, utterance_overview, slot_pairs):
    intent_desc_block = build_intent_desc_str(intents)
    pairs_desc = "\n".join(
        f"- {p['slot_label']}: ({p['slot_value']}, {p['slot_label']})"
        for p in slot_pairs
    )
    return f"""You are an expert in Aviation domain.

Available Intent Descriptions:
{intent_desc_block}

Your task is to generate a fluent, natural-sounding **user query** (e.g., a question or request) based on the given draft and slot value-label pairs.

Draft (for phrasing inspiration only): "{utterance_overview}"

Given intents: {', '.join(intents)}
Given slot label-value pairs:
{pairs_desc}

Rules:
1. Include every slot value exactly as given—no paraphrasing, no omission.
2. DO NOT introduce any new intents or slot label-value pairs beyond those provided.
3. Use the draft only for tone and structure—not as a complete guide.
4. Make the utterance sound realistic and conversational, like something a real traveler would say.
5. The output must be a **realistic user query**: typically a **question**, **request for information**, or **conditional statement**.
6. Do NOT use: "Can you...", "Could you...", "I need...", "I am looking for..."
7. **Ensure the utterance is clear and concise**: Avoid redundant phrases.

Output ONLY the utterance. No explanations. No extra text."""

# ======================
# Parser
# ======================
def parse_intents_and_slots(text: str):
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
# LLM Runner with optional int4
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
            # Auto dtype for small models
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )

        self.model.eval()

    def Response(self, input_str, max_new_tokens=256, temperature=1.0, top_p=0.9):
        messages = [{"role": "user", "content": input_str}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
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
        slot_label_value_pairs = ""
        final_resp2 = ""

        # === Stage 2: Slot assignment + draft ===
        user_prompt2 = get_prompt_stage2(sample['intents'], sample['slots'])

        for attempt in range(max_retries):
            resp2 = llm.Response(
                input_str=user_prompt2,
                max_new_tokens=256,
                temperature=1.2,
                top_p=0.9
            )

            parsed_intents, slot_label_value_pairs, utterance_overview = parse_intents_and_slots(resp2)

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