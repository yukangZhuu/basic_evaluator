# LLM Evaluator - 使用指南

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置评测参数

所有配置都集中在 `main.py` 的 `Config` 类中。编辑 `main.py` 文件，修改 `Config` 类的参数：

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

### 3. 运行评测

```bash
python main.py
```

## 支持的 Benchmark

### GSM8K
- **数据格式**: JSONL
- **字段**: `question`, `answer`
- **示例**:
```json
{"question": "What is 2 + 2?", "answer": "4"}
```
- **配置建议**:
  - `THINKING_MODE`: true（推荐启用）
  - `MAX_TOKENS`: 2048

### Math-500
- **数据格式**: JSONL
- **字段**: `question`, `answer`
- **示例**:
```json
{"question": "Solve for x: 2x + 3 = 7", "answer": "2"}
```
- **配置建议**:
  - `THINKING_MODE`: true（推荐启用）
  - `MAX_TOKENS`: 4096

### Teacher Traces 12K
- **数据格式**: JSONL
- **字段**: `question`, `ground_truth`
- **特点**: 
  - 使用 math-verify 库进行智能答案验证
  - 支持多种答案格式（LaTeX、表达式、字符串、数字）
  - 要求模型答案在 `\boxed{}` 中
- **示例**:
```json
{"question": "Two is $10 \\%$ of $x$ and $20 \\%$ of $y$. What is $x - y$?", "ground_truth": "10"}
```
- **配置建议**:
  - `THINKING_MODE`: true（推荐启用）
  - `MAX_TOKENS`: 4096（数学问题通常需要较长回答）
  - `MAX_MODEL_LEN`: 8192（支持更长的上下文）

### MMLU
- **数据格式**: JSONL
- **字段**: `question`, `choices`, `answer`
- **示例**:
```json
{"question": "What is the capital of France?", "choices": ["London", "Berlin", "Paris", "Madrid"], "answer": "C"}
```
- **配置建议**:
  - `THINKING_MODE`: false
  - `MAX_TOKENS`: 512
  - `BATCH_SIZE`: 128（可以更大）

## 添加新的 Benchmark

### 步骤 1: 创建 Adaptor 类

在 `adaptors/` 目录下创建新的 adaptor 文件，例如 `custom_adaptor.py`：

```python
import json
from typing import Dict, Any, List
from .base_adaptor import BaseAdaptor

class CustomAdaptor(BaseAdaptor):
    def _load_data(self) -> List[Dict[str, Any]]:
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def _get_system_prompt(self) -> str:
        if self.thinking_mode:
            return "Your system prompt for thinking mode"
        else:
            return "Your system prompt"

    def format_prompt(self, item: Dict[str, Any]) -> str:
        question = item.get('question', '')
        return f"{self.system_prompt}\n\nQuestion: {question}\n\nAnswer:"

    def extract_answer(self, model_output: str) -> str:
        return model_output.strip()

    def verify_answer(self, model_answer: str, ground_truth: str) -> bool:
        return model_answer.strip() == ground_truth.strip()
```

### 步骤 2: 注册 Adaptor

在 `adaptors/adaptor_factory.py` 中注册新的 adaptor：

```python
from .custom_adaptor import CustomAdaptor

class AdaptorFactory:
    @staticmethod
    def create_adaptor(benchmark_type: str, data_path: str, thinking_mode: bool = False) -> BaseAdaptor:
        adaptor_map = {
            'gsm8k': GSM8KAdaptor,
            'math-500': Math500Adaptor,
            'math500': Math500Adaptor,
            'mmlu': MMLUAdaptor,
            'teacher_traces_12k': TeacherTracesAdaptor,
            'custom': CustomAdaptor,  # 添加新的 benchmark
        }
        # ...
```

### 步骤 3: 准备数据文件

创建符合新 benchmark 格的 JSONL 数据文件，例如 `data/custom_benchmark.jsonl`：

```json
{"question": "Your question here", "answer": "Expected answer"}
```

### 步骤 4: 更新配置

在 `main.py` 的 `Config` 类中更新配置：

```python
class Config:
    BENCHMARK_TYPE = "custom"
    BENCHMARK_DATA_PATH = "./data/custom_benchmark.jsonl"
    # ... 其他配置
```

## 输出说明

评测完成后，会在 `OUTPUT_DIR` 目录下生成两个文件：

### 1. Report JSON (`report_*.json`)
包含评测的汇总统计信息：

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "total_samples": 100,
  "correct_samples": 85,
  "incorrect_samples": 15,
  "accuracy": 0.85,
  "accuracy_percentage": 85.0,
  "inference_metrics": {
    "total_time": 45.23,
    "average_time_per_request": 0.45,
    "total_tokens": 25000,
    "tokens_per_second": 552.89,
    "requests_per_second": 2.21
  },
  "error_analysis": {
    "total_errors": 15,
    "error_types": {
      "no_answer": 2,
      "format_mismatch": 3,
      "incorrect_answer": 10
    },
    "sample_errors": [...]
  }
}
```

### 2. Results JSONL (`results_*.jsonl`)
包含每个样本的详细结果，每行一个 JSON 对象：

```json
{"question": "What is 2 + 2?", "model_output": "The answer is 4.", "model_answer": "4", "ground_truth": "4", "is_correct": true}
```

## 性能优化建议

1. **调整 Batch Size**: 根据 GPU 内存调整 `BATCH_SIZE`，通常 32-128 是合理的范围
2. **GPU 内存利用率**: 调整 `GPU_MEMORY_UTILIZATION`（0.7-0.9）以避免 OOM
3. **并行推理**: 确保 `USE_PARALLEL=true` 以获得最佳性能
4. **Tensor Parallel**: 对于多 GPU 设置，增加 `TENSOR_PARALLEL_SIZE`

## 常见问题

### Q: 如何使用多张 GPU？
A: 设置 `TENSOR_PARALLEL_SIZE` 为 GPU 数量。

### Q: 如何启用思考模式（Chain-of-Thought）？
A: 在 `main.py` 的 `Config` 类中设置 `THINKING_MODE=true`。

### Q: 如何自定义停止词？
A: 在 `Config` 类中设置 `STOP_TOKENS` 为字符串列表，例如 `["####", "Answer:"]`。

### Q: 如何调整生成参数？
A: 在 `Config` 类中修改 `MAX_TOKENS`, `TEMPERATURE`, `TOP_P` 等参数。

## 示例配置

### GSM8K 评测配置
在 `main.py` 中设置：
```python
class Config:
    MODEL_PATH = "/models/Qwen2.5-7B-Instruct"
    BENCHMARK_DATA_PATH = "./data/gsm8k_test.jsonl"
    BENCHMARK_TYPE = "gsm8k"
    THINKING_MODE = True
    BATCH_SIZE = 64
    MAX_TOKENS = 2048
```

### Math-500 评测配置
在 `main.py` 中设置：
```python
class Config:
    MODEL_PATH = "/models/Qwen2.5-7B-Instruct"
    BENCHMARK_DATA_PATH = "./data/math500_test.jsonl"
    BENCHMARK_TYPE = "math-500"
    THINKING_MODE = True
    BATCH_SIZE = 32
    MAX_TOKENS = 4096
```

### Teacher Traces 12K 评测配置
在 `main.py` 中设置：
```python
class Config:
    MODEL_PATH = "/models/Qwen2.5-7B-Instruct"
    BENCHMARK_DATA_PATH = "./data/teacher_traces_12k.jsonl"
    BENCHMARK_TYPE = "teacher_traces_12k"
    THINKING_MODE = True
    BATCH_SIZE = 16
    MAX_TOKENS = 4096
    MAX_MODEL_LEN = 8192
```

### MMLU 评测配置
在 `main.py` 中设置：
```python
class Config:
    MODEL_PATH = "/models/Qwen2.5-7B-Instruct"
    BENCHMARK_DATA_PATH = "./data/mmlu_test.jsonl"
    BENCHMARK_TYPE = "mmlu"
    THINKING_MODE = False
    BATCH_SIZE = 128
    MAX_TOKENS = 512
```
