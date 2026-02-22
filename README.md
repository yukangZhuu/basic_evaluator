# LLM Evaluator

A comprehensive LLM evaluation framework with vLLM acceleration for benchmarking language models on mathematical reasoning tasks.

## Features

- **Local Model Loading**: Load and evaluate models from local file system
- **Multiple Benchmark Support**: Built-in support for GSM8K, Math-500, Teacher Traces 12K, MMLU, and extensible to other benchmarks
- **vLLM Acceleration**: Leverages vLLM for fast, parallel inference
- **Comprehensive Reporting**: Generates detailed reports including accuracy, inference speed, and error analysis
- **Modular Architecture**: Easy to extend with new benchmarks and evaluation metrics
- **Thinking Mode**: Support for chain-of-thought style reasoning
- **Centralized Configuration**: All configurations managed in main.py

## Project Structure

```
basic_evaluator/
├── core/
│   ├── __init__.py
│   ├── model_inference.py      # Basic model inference
│   ├── parallel_inference.py   # Parallel inference with vLLM
│   └── evaluator.py            # Main evaluation logic
├── adaptors/
│   ├── __init__.py
│   ├── base_adaptor.py         # Base adaptor interface
│   ├── gsm8k_adaptor.py        # GSM8K benchmark adaptor
│   ├── math500_adaptor.py      # Math-500 benchmark adaptor
│   ├── teacher_traces_adaptor.py  # Teacher Traces 12K benchmark adaptor
│   ├── mmlu_adaptor.py         # MMLU benchmark adaptor
│   └── adaptor_factory.py      # Adaptor factory
├── data/
│   ├── gsm8k_sample.jsonl
│   ├── math500_sample.jsonl
│   ├── mmlu_sample.jsonl
│   └── teacher_traces_12k.jsonl
├── main.py                     # Entry point and configuration
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

All configurations are centralized in the `Config` class in `main.py`. Edit the following parameters to configure your evaluation:

```python
class Config:
    
    MODEL_PATH = "/path/to/your/model"
    
    BENCHMARK_DATA_PATH = "./data/gsm8k_sample.jsonl"
    BENCHMARK_TYPE = "gsm8k"
    
    THINKING_MODE = False
    
    OUTPUT_DIR = "./outputs"
    
    TENSOR_PARALLEL_SIZE = 1
    GPU_MEMORY_UTILIZATION = 0.9
    MAX_MODEL_LEN = 4096
    
    USE_PARALLEL = True
    BATCH_SIZE = 32
    
    MAX_TOKENS = 2048
    TEMPERATURE = 0.0
    TOP_P = 1.0
    STOP_TOKENS = None
```

### Configuration Parameters

- **MODEL_PATH**: Path to the local model directory
- **BENCHMARK_DATA_PATH**: Path to the benchmark data file (JSONL format)
- **BENCHMARK_TYPE**: Type of benchmark (gsm8k, math-500, teacher_traces_12k, mmlu)
- **THINKING_MODE**: Enable chain-of-thought reasoning
- **OUTPUT_DIR**: Directory to save evaluation results
- **TENSOR_PARALLEL_SIZE**: Number of GPUs for tensor parallelism
- **GPU_MEMORY_UTILIZATION**: GPU memory utilization (0.0-1.0)
- **MAX_MODEL_LEN**: Maximum model context length
- **USE_PARALLEL**: Enable parallel inference with vLLM
- **BATCH_SIZE**: Batch size for parallel inference
- **MAX_TOKENS**: Maximum tokens to generate
- **TEMPERATURE**: Sampling temperature
- **TOP_P**: Nucleus sampling parameter
- **STOP_TOKENS**: Stop tokens for generation

## Benchmark Data Format

Benchmarks should be in JSONL format with the following structure:

### GSM8K Format
```json
{"question": "What is 2 + 2?", "answer": "4"}
```

### Math-500 Format
```json
{"question": "Solve for x: 2x + 3 = 7", "answer": "2"}
```

### Teacher Traces 12K Format
```json
{"question": "Two is $10 \\%$ of $x$ and $20 \\%$ of $y$. What is $x - y$?", "ground_truth": "10"}
```

### MMLU Format
```json
{"question": "What is the capital of France?", "choices": ["London", "Berlin", "Paris", "Madrid"], "answer": "C"}
```

## Usage

1. Edit the `Config` class in `main.py` with your desired settings
2. Run the evaluation:

```bash
python main.py
```

## Adding New Benchmarks

To add a new benchmark:

1. Create a new adaptor class in `adaptors/` that inherits from `BaseAdaptor`
2. Implement the required methods:
   - `_load_data()`: Load benchmark data
   - `_get_system_prompt()`: Return the system prompt
   - `format_prompt()`: Format the input prompt
   - `extract_answer()`: Extract the answer from model output
   - `verify_answer()`: Verify if the answer is correct
3. Register the adaptor in `adaptor_factory.py`
4. Update the configuration in `main.py` by setting `BENCHMARK_TYPE` to your new benchmark type

## Output

The evaluator generates two output files in the `OUTPUT_DIR`:

1. **Report JSON** (`report_*.json`): Contains summary statistics and metrics
2. **Results JSONL** (`results_*.jsonl`): Contains detailed results for each sample

## Example Output

```
============================================================
LLM Evaluator - Starting Evaluation
============================================================

Configuration:
  Model Path: /path/to/model
  Benchmark Data Path: /path/to/benchmark.jsonl
  Benchmark Type: gsm8k
  Thinking Mode: False
  Use Parallel: True
  Batch Size: 32

============================================================
Evaluation Results
============================================================

Total Samples: 100
Correct Samples: 85
Incorrect Samples: 15
Accuracy: 85.00%

Inference Metrics:
  Total Time: 45.23s
  Average Time per Request: 0.45s
  Total Tokens: 25000
  Tokens per Second: 552.89
  Requests per Second: 2.21
```

For detailed usage instructions and examples, see [USAGE.md](USAGE.md).
