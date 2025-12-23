# LLM Attack Arena

A framework for testing jailbreak attacks on large language models.

## Overview

This project implements and evaluates multiple jailbreak attack methods against various LLMs to assess their robustness and safety alignment.

## Attack Methods

- **DeepInception**: Nested scenario-based jailbreaking
- **FFA**: Few-shot attack prompting
- **Parameter**: Parameter manipulation attacks
- **TemplateJailbreak**: Template-based prompt injection

## Tested Models

| Model Key | Model Path |
|-----------|------------|
| `phi2` | microsoft/phi-2 |
| `llama` | meta-llama/Llama-3.2-1B |
| `deepseek` | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B |
| `qwen` | Qwen/Qwen3-0.6B |
| `gemma` | google/gemma-3n-E4B-it |

## Usage

### Setup

```bash
pip install -r requirements.txt
```

### Run Attacks

```bash
# Run any attack using main.py
python main.py --model <model_key> --type <attack_type>

# Examples:
python main.py --model phi2 --type TemplateJailbreak
python main.py --model llama --type DeepInception
python main.py --model deepseek --type FFA
python main.py --model qwen --type Parameters
```

Available attack types: `DeepInception`, `Jailbroken`, `TemplateJailbreak`, `Parameters`, `FFA`

## Structure

```
├── Attacks/          # Attack implementations
├── Data/             # Test datasets
├── Evaluation/       # Evaluation results
├── main.py           # Main entry point
├── models.py         # Model implementations
└── utils.py          # Utility functions
```

## Results

Attack success rates and evaluations are stored in:
- `Attacks/*/bpr_results/` - Attack results per method
- `Evaluation/` - Evaluation metrics and analysis
