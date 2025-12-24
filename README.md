# LLM Attack Arena

A framework for testing jailbreak attacks on large language models.

## Overview

This project implements and evaluates multiple jailbreak attack methods against various LLMs to assess their robustness and safety alignment.

## Tested Models

| Model Key | Model Path |
|-----------|------------|
| `phi2` | microsoft/phi-2 |
| `llama` | meta-llama/Llama-3.2-1B |
| `deepseek` | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B |
| `qwen` | Qwen/Qwen3-0.6B |
| `gemma` | google/gemma-3n-E4B-it |

## Attack Methods

- **DeepInception**: Nested scenario-based jailbreaking
- **FFA**: Few-shot attack prompting
- **Parameter**: Parameter manipulation attacks
- **TemplateJailbreak**: Template-based prompt injection

## Dataset

- `Data/data.csv`: 60 malicious prompts obtained from [llmcomprehensive](https://drive.google.com/drive/u/3/folders/15QKEpWMn8H97YIBoyD-fF1gwAorYGYmz)
- `Data/jailbrea-prompt.xlsx`: 78 Jailbreak templates obtained from [llm-jailbreak-study](https://sites.google.com/view/llm-jailbreak-study/taxonomy?authuser=0)


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

Available models: `phi2`, `llama`, `deepseek`, `qwen`, `gemma`

Available attack types: `DeepInception`, `Jailbroken`, `TemplateJailbreak`, `Parameters`, `FFA`

## Structure

```
├── Attacks/          # Attack implementations
├── Data/             # Test datasets
├── Evaluation/       # Evaluation results
├── main.py           # Main entry point
├── models.py         # Model implementations using vLLM
├── requirements.txt  # Required packages
└── utils.py          # Utility functions
```

## Results

Evaluations are stored in:
- `Attacks/*/bpr_results/` - BPR results per attack type
- `Attacks/*/Evaluation/` - ASR and efficiency per attack type
- `Evaluation` - ASR and effiency combined for all attack types
- `bpr_results.json` - Quick access to BPR results for all attack types
