import os
from core.evaluator import Evaluator
from adaptors.adaptor_factory import AdaptorFactory


class Config:
    
    # MODEL_PATH = "../models/verl_cgrpo_gsm8k_qwen3_0.6b_gsm8k/verl_cgrpo_gsm8k_qwen3_0.6b_gsm8k_2.14_2"
    # OUTPUT_DIR = "./outputs/teacher_traces/verl_cgrpo_gsm8k_qwen3_0.6b_gsm8k_2.14_2"

    MODEL_PATH = "../models/Qwen3-0.6B"
    OUTPUT_DIR = "./outputs/math500/Qwen3-0.6B"


    BENCHMARK_DATA_PATH = "./data/math500.jsonl"
    BENCHMARK_TYPE = "math500"
    
    THINKING_MODE = True
    
    TENSOR_PARALLEL_SIZE = 1
    GPU_MEMORY_UTILIZATION = 0.9
    MAX_MODEL_LEN = 16384
    
    USE_PARALLEL = True
    BATCH_SIZE = 20
    
    MAX_TOKENS = 16384
    TEMPERATURE = 0.0
    TOP_P = 1.0
    STOP_TOKENS = None
    
    MAX_SAMPLE =  100 # Maximum number of samples to evaluate, None means evaluate all


def main():
    print("=" * 60)
    print("LLM Evaluator - Starting Evaluation")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  Model Path: {Config.MODEL_PATH}")
    print(f"  Benchmark Data Path: {Config.BENCHMARK_DATA_PATH}")
    print(f"  Benchmark Type: {Config.BENCHMARK_TYPE}")
    print(f"  Thinking Mode: {Config.THINKING_MODE}")
    print(f"  Use Parallel: {Config.USE_PARALLEL}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Max Samples: {Config.MAX_SAMPLE if Config.MAX_SAMPLE is not None else 'All'}")
    print(f"  Output Directory: {Config.OUTPUT_DIR}")
    
    if not os.path.exists(Config.MODEL_PATH):
        print(f"\nError: Model path does not exist: {Config.MODEL_PATH}")
        return
    
    if not os.path.exists(Config.BENCHMARK_DATA_PATH):
        print(f"\nError: Benchmark data path does not exist: {Config.BENCHMARK_DATA_PATH}")
        return
    
    print("\nInitializing adaptor...")
    adaptor = AdaptorFactory.create_adaptor(
        benchmark_type=Config.BENCHMARK_TYPE,
        data_path=Config.BENCHMARK_DATA_PATH,
        thinking_mode=Config.THINKING_MODE
    )
    
    print("Initializing evaluator...")
    evaluator = Evaluator(
        model_path=Config.MODEL_PATH,
        tensor_parallel_size=Config.TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=Config.GPU_MEMORY_UTILIZATION,
        max_model_len=Config.MAX_MODEL_LEN,
        use_parallel=Config.USE_PARALLEL,
        batch_size=Config.BATCH_SIZE,
        enable_thinking=Config.THINKING_MODE
    )
    
    print("\nRunning evaluation...")
    evaluation_result = evaluator.evaluate(
        adaptor=adaptor,
        max_tokens=Config.MAX_TOKENS,
        temperature=Config.TEMPERATURE,
        top_p=Config.TOP_P,
        stop=Config.STOP_TOKENS,
        max_samples=Config.MAX_SAMPLE
    )
    
    report = evaluation_result['report']
    results = evaluation_result['results']
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"\nTotal Samples: {report['total_samples']}")
    print(f"Correct Samples: {report['correct_samples']}")
    print(f"Incorrect Samples: {report['incorrect_samples']}")
    print(f"Accuracy: {report['accuracy_percentage']:.2f}%")
    
    print("\nInference Metrics:")
    print(f"  Total Time: {report['inference_metrics']['total_time']:.2f}s")
    print(f"  Average Time per Request: {report['inference_metrics']['average_time_per_request']:.2f}s")
    print(f"  Total Tokens: {report['inference_metrics']['total_tokens']}")
    print(f"  Tokens per Second: {report['inference_metrics']['tokens_per_second']:.2f}")
    print(f"  Requests per Second: {report['inference_metrics']['requests_per_second']:.2f}")
    
    print("\nError Analysis:")
    print(f"  Total Errors: {report['error_analysis']['total_errors']}")
    print(f"  Error Types: {report['error_analysis']['error_types']}")
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    timestamp = report['timestamp'].replace(':', '-').replace('.', '-')
    report_path = os.path.join(Config.OUTPUT_DIR, f"report_{timestamp}.json")
    results_path = os.path.join(Config.OUTPUT_DIR, f"results_{timestamp}.jsonl")
    
    evaluator.save_report(report, report_path)
    evaluator.save_detailed_results(results, results_path)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    
    evaluator.cleanup()


if __name__ == "__main__":
    main()
