# Zero-Shot SLU Data Generation Framework

This repository contains the official implementation of our Zero-Shot Spoken Language Understanding (SLU) data generation framework. Our pipeline leverages a combination of Small Large Language Models (LLMs) to generate, filter, and adjudicate high-quality domain-specific (e.g., Aviation) SLU datasets, effectively mitigating hallucinations.

## 🌟 Key Features
- **Multi-Model Generator Combinations**: Utilizes three robust small models (`Qwen2.5-7B`, `Llama3.1-8B`, and `InternLM-3-8B`) for diverse utterance generation and cross-validation.
- **Duo-Value Support**: Handles both single-value (`danvalue`) and complex multi-value (`duovalue`) slot generation (e.g., round trips, multiple cities).
- **LLM-as-a-Judge Filtering**: Adopts `Qwen-2.5-72B (int4-quant)` to score and filter utterances based on fluency, naturalness, semantic richness, and logical coherence.
- **Cross-Model Adjudication**: Automatically identifies disagreements between model predictions and re-adjudicates them to construct a highly reliable dataset.

## 📁 Repository Structure

```text
.
├── stage1/                     # Data Generation Stage
│   ├── danvalue/               # Single-value slot generation
│   │   ├── main.py
│   │   └── stage1_worker_new.py
│   └── duovalue/               # Multi-value slot generation
│       ├── stage1_main_duovalue.py
│       └── stage1_worker_duovalue.py
├── stage2/                     # Quality Filtering & Deduplication Stage
│   ├── cha.py                  # Deduplication using Sentence-BERT
│   └── judge.py                # LLM-as-a-Judge (Qwen2.5-72B) filtering
├── stage3/                     # Consistency Evaluation Stage
│   ├── main_pipeline.py        # Pipeline orchestrator for model pairs
│   └── run_single_model.py     # Inference script (Transformers / LMDeploy)
└── stage4/                     # Adjudication Stage
    ├── choose_single_model.py  # Zero-shot re-parsing for adjudication
    └── main_adjudication.py    # Disagreement resolution pipeline
