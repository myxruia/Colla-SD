# Zero-Shot SLU Data Generation Framework

This repository contains the official implementation and generated training data for our zero-shot Spoken Language Understanding (SLU) data generation framework.

The framework constructs synthetic multi-intent SLU training data without using utterance-level human annotations. It combines schema-guided utterance generation, semantic deduplication, LLM-based quality filtering, cross-model consensus, and disagreement adjudication to improve the reliability of generated intent-slot annotations.

## 🌟 Key Features

- **Multiple Model Combinations**  
  Uses three open-weight LLMs—`Qwen2.5-7B`, `Llama-3.1-8B`, and `InternLM3-8B`—in different pairwise combinations for data generation, semantic parsing, consensus checking, and adjudication.

- **Single-Value and Multi-Value Slot Generation**  
  Supports both single-value (`danvalue`) and multi-value (`duovalue`) slot generation. The multi-value setting is designed for utterances containing multiple values of the same slot type, such as multiple cities, artists, or destinations.

- **LLM-as-a-Judge Quality Filtering**  
  Uses an int4-quantized `Qwen2.5-72B-Instruct` model to evaluate generated utterances in terms of fluency, naturalness, semantic richness, and logical coherence.

- **Cross-Model Consensus**  
  Two LLMs independently parse the same generated utterance. Samples with identical intent sets and slot label-value pairs are retained as consensus samples.

- **Disagreement Adjudication**  
  Samples with inconsistent model predictions are sent to an adjudication stage, where the utterance, task schema, initial semantic specification, and candidate parses are jointly examined to produce the final annotation.

- **Released Main-Experiment Training Data**  
  The generated training data used in the main experiments are provided for both MixATIS and MixSNIPS under three model combinations.

## 📁 Repository Structure

```text
.
├── stage1/                         # Schema-guided data generation
│   ├── danvalue/                   # Single-value slot generation
│   │   ├── main.py
│   │   └── stage1_worker_new.py
│   └── duovalue/                   # Multi-value slot generation
│       ├── stage1_main_duovalue.py
│       └── stage1_worker_duovalue.py
│
├── stage2/                         # Deduplication and quality filtering
│   ├── cha.py                      # Sentence-BERT-based deduplication
│   └── judge.py                    # Qwen2.5-72B quality filtering
│
├── stage3/                         # Cross-model semantic parsing
│   ├── main_pipeline.py            # Model-pair pipeline controller
│   └── run_single_model.py         # Single-model parsing inference
│
├── stage4/                         # Disagreement adjudication
│   ├── choose_single_model.py      # Adjudicator inference
│   └── main_adjudication.py        # Adjudication pipeline controller
│
└── data/                           # Released main-experiment training data
    ├── MixATIS/
    │   ├── qwen+llama/
    │   ├── qwen+internlm/
    │   └── llama+internlm/
    │
    └── MixSNIPS/
        ├── qwen+llama/
        ├── qwen+internlm/
        └── llama+internlm/
