import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaptors.adaptor_factory import AdaptorFactory


def test_teacher_traces_adaptor():
    print("=" * 60)
    print("Testing Teacher Traces 12K Adaptor")
    print("=" * 60)
    
    data_path = "./data/teacher_traces_12k.jsonl"
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        return False
    
    try:
        adaptor = AdaptorFactory.create_adaptor(
            benchmark_type="teacher_traces_12k",
            data_path=data_path,
            thinking_mode=True
        )
        
        print(f"\n✓ Adaptor created successfully")
        print(f"✓ System prompt: {adaptor.system_prompt[:100]}...")
        
        data = adaptor.load_benchmark_data()
        print(f"✓ Loaded {len(data)} samples")
        
        if len(data) > 0:
            sample = data[0]
            print(f"\nSample question (first 100 chars): {sample.get('question', '')[:100]}...")
            print(f"Ground truth: {sample.get('ground_truth', '')}")
            
            prompt = adaptor.format_prompt(sample)
            print(f"\n✓ Prompt formatted successfully (length: {len(prompt)})")
            
            test_outputs = [
                "The answer is \\boxed{10}.",
                "After solving, we get \\boxed{10}",
                "Answer: 10",
                "The final answer is 10."
            ]
            
            for i, output in enumerate(test_outputs):
                extracted = adaptor.extract_answer(output)
                is_correct = adaptor.verify_answer(extracted, sample.get('ground_truth', ''))
                print(f"\nTest {i+1}:")
                print(f"  Output: {output}")
                print(f"  Extracted: {extracted}")
                print(f"  Verified: {'✓ Correct' if is_correct else '✗ Incorrect'}")
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_teacher_traces_adaptor()
    sys.exit(0 if success else 1)
