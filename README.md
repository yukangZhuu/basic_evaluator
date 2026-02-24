# LLM Evaluator

A comprehensive LLM evaluation framework with vLLM acceleration for benchmarking language models on mathematical reasoning tasks.

## Features

- **Local Model Loading**: Load and evaluate models from local file system
- **Multiple Benchmark Support**: Built-in support for Math-500, AIME24, and extensible to other benchmarks
- **vLLM Acceleration**: Leverages vLLM for fast, parallel inference
- **Comprehensive Reporting**: Generates detailed reports including accuracy, inference speed, and error analysis
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
│   ├── math500_adaptor.py      # Math-500 benchmark adaptor
│   ├── aime24_adaptor.py       # AIME24 benchmark adaptor
│   └── adaptor_factory.py      # Adaptor factory
├── data/
│   ├── math500.jsonl
│   └── aime24.jsonl
├── main.py                     # Entry point and configuration
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

All configurations are centralized in a `Config` class in `main.py`. Edit the following parameters to configure your evaluation:

```python
class Config:
    
    MODEL_PATH = "/path/to/your/model"
    
    BENCHMARK_DATA_PATH = "./data/math500.jsonl"
    BENCHMARK_TYPE = "math500"
    
    THINKING_MODE = True
    
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
- **BENCHMARK_TYPE**: Type of benchmark (math500, aime24, etc.)
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

### Math-500 Format
```json
{"question": "Solve for x: 2x + 3 = 7", "answer": "2"}
```

### AIME24 Format
```json
{"question": "Alice chooses a set $A$ of positive integers...", "answer": "55"}
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
   - `_get_system_prompt()`: Return system prompt
   - `format_prompt()`: Format input prompt
   - `extract_answer()`: Extract answer from model output
   - `verify_answer()`: Verify if the answer is correct
3. Register the adaptor in `adaptor_factory.py` by setting `BENCHMARK_TYPE` to your new benchmark type
4. Update the configuration in `main.py` by setting `BENCHMARK_DATA_PATH` and `BENCHMARK_TYPE`

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
  Benchmark Type: math500
  Thinking Mode: True
  Use Parallel: True
  Batch Size: 32

============================================================
Evaluation Results
============================================================

Total Samples: 30
Correct Samples: 25
Incorrect Samples: 5
Accuracy: 83.33%

Inference Metrics:
  Total Time: 45.23s
  Average Time per Request: 1.51s
  Total Tokens: 25000
  Tokens per Second: 552.89
  Requests per Second: 2.21

Error Analysis:
  Total Errors: 5
  Error Types: {'incorrect_answer': 3, 'no_answer': 2}
```

For detailed usage instructions and examples, see [USAGE.md](USAGE.md).
