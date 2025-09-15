# TempAnswerQA

This repository contains the code and datasets for evaluating large language models (LLMs) on the numeric temporal reasoning dataset *TempAnswerQA*, sampled from [**Test of Time (ToT)**](https://huggingface.co/datasets/baharef/ToT) and [**TempTabQA (TTQA)**](https://github.com/clear-temptabqa/clear_temptabqa). This projects implements a comprehensive evaluation framework using regression-like metrics.

The corresponding paper titled "*Time* to Rethink Exact Match" has been accepted to 2025 EMNLP Findings.

## Overview

The codebase provides tools for:

- Running inference on *TempAnswerQA* using Hugging Face transformers
- Parsing model responses into numeric, time-aware objects
- Evaluating model responses with symmetric mean absolute percentage error (sMAPE) and mean absolute scaled error (MASE)

## Installation

### Requirements

- Python ≥ 3.11
- CUDA-compatible GPU (recommended for model inference)
- Hugging Face account with access token

### Setup

1. Clone the repository:

2. Install dependencies using uv (recommended):

```bash
uv sync
```

Or using pip:

```bash
pip install -e .
```

3. Set up environment variables:

You are expected to set your Hugging Face token in an `.env` file since our experiments used access-restricted Llama models.

```bash
# Create a .env file with your Hugging Face token
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

## Datasets

Since *TempAnswerQA* consists of ToT and TTQA, we still refer to both datasets and their splits by theirs names.

### ToT

A synthetic temporal reasoning dataset with two categories:

- **Arithmetic**: Date calculations, duration computations, and temporal arithmetic
- **Semantic**: Temporal logic with graph as context

### TTQA

A dataset on semi-structured Wikipedia tables with temporal, entity-based questions based with two splits:

- **Head**: Questions about more prominent entities
- **Tail**: Questions about less-frequented entities

## Usage

The main interface is through the CLI using `main.py`:

### Running Inference

#### ToT Dataset

```bash
# Few-shot prompting on arithmetic split
python main.py inference-tot "meta-llama/Llama-3.1-8B-Instruct" add_generation_prompt few-shot arithmetic

# Zero-shot prompting on semantic split  
python main.py inference-tot "meta-llama/Llama-3.1-8B-Instruct" continue_final_message zero-shot semantic
```

#### TTQA Dataset

```bash
# Few-shot prompting on head split
python main.py inference-ttqa "meta-llama/Llama-3.1-8B-Instruct" add_generation_prompt few-shot head

# Zero-shot prompting on tail split
python main.py inference-ttqa "meta-llama/Llama-3.1-8B-Instruct" continue_final_message zero-shot tail
```

### Evaluation

These scripts will calculate sMAPE, MASE and EM for all model responses generated in the above step.

#### ToT Evaluation

```bash
python main.py evaluate-tot data/responses/ continue_final_message
```

#### TTQA Evaluation  

```bash
python main.py evaluate-ttqa data/responses/ add_generation_prompt 
```

### Parameters

- **model_name**: Hugging Face model identifier (e.g., `meta-llama/Llama-3.1-8B-Instruct`)
- **last_token**: Token handling strategy
  - `add_generation_prompt`: Adds generation prompt to the chat template
  - `continue_final_message`: Continues from the final message
- **prompting**: Prompting strategy
  - `few-shot`: Uses example demonstrations
  - `zero-shot`: No examples provided
- **split**: Dataset split
  - ToT: `arithmetic` or `semantic`
  - TTQA: `head` or `tail`
- **test_mode**: Boolean flag for testing with a small subset of data

## Project Structure

```
temp-answer-qa/
├── main.py                          # CLI interface
├── temp_answer_qa/                  # Main package
│   ├── __init__.py                  # Core enums and constants
│   ├── chat_builder.py              # Chat template builders
│   ├── data_loader.py               # Dataset loading utilities
│   ├── evaluate.py                  # Evaluation pipeline
│   ├── inference.py                 # Model inference
│   ├── measure_error.py             # Parsing and metric application
│   ├── metrics.py                   # Evaluation metrics
│   ├── models.py                    # Hugging Face model wrapper
│   └── response_processing.py       # Response parsing and processing
├── data/
│   ├── prompts/                     # Few-shot examples and system prompts
│   ├── questions/                   # Dataset files (tot.csv, ttqa.csv)
│   ├── responses/                   # Generated model responses
│   └── responses_evaluated/         # Evaluation results
└── tests/                           # Unit tests
```

## Data Format

### ToT Dataset

- `question`: Full question with formatting instructions
- `label`: Ground truth answer as dictionary
- `question_wo_instruct`: Question without formatting instructions
- `instruction`: JSON formatting instructions
- `answer_format`: Expected answer format
- `answer_temporal_unit`: Type of temporal unit (date, days, months, etc.)
- `split`: Dataset split (arithmetic/semantic)

### TTQA Dataset

- `question`: Question about the table
- `label`: Ground truth answer
- `table_context`: Structured table data
- `answer_format`: Expected answer format
- `answer_temporal_unit`: Type of temporal unit
- `split`: Dataset split (head/tail)

## Differences between repo and paper

This repo underwent refactoring after submission. During that process, we found a few issues.

Clustering depends on the order of the data, which we did not adequately control for during experiments. Therefore, we need to apply the same data ordering as done for the paper to reproduce results (applied in `evaluate.py`, using a `DataFrame`'s index).

Despite using the same library versions and getting the same clusters using the same values, this new version of the code exhibits small differences in MASE scores for the ToT dataset. We suspect numeric instabilities when calculating the centroid to be the reason.

After refactoring we also found a mistake when calculating MASE in TTQA for 137 examples. Instead of using an error based on the timestamp of the date, we used the number of days. The difference in scores is, however, low.

The uploaded [results](data/responses_evaluated) are the ones we generated and used for the paper.

## License

The code in this repository is licensed under the MIT License. See the `LICENSE` file for details.

The datasets in `data/questions/` are licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. See `data/questions/LICENSE` for details.

Other files under `data/` may include artifacts or evaluation outputs; they retain the licenses of their respective sources unless otherwise noted.
