import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaptors.teacher_traces_adaptor import TeacherTracesAdaptor


def test_extract_answer():
    print("=" * 60)
    print("Testing extract_answer() Function")
    print("=" * 60)
    
    adaptor = TeacherTracesAdaptor.__new__(TeacherTracesAdaptor)
    
    test_cases = [
        ("The answer is \\boxed{10}.", "10", True),
        ("After solving, we get \\boxed{10}", "10", True),
        ("\\boxed{42}", "42", True),
        ("\\boxed{x + y}", "x + y", True),
        ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}", True),
        ("Answer: 10", "", False),
        ("The final answer is 10.", "", False),
        ("10", "", False),
        ("\\boxed{10} and \\boxed{20}", "20", True),
        ("Some text \\boxed{answer} more text", "answer", True),
    ]
    
    print("\nTest Cases:")
    print("-" * 60)
    
    all_passed = True
    for output, expected, should_extract in test_cases:
        extracted = adaptor.extract_answer(output)
        passed = extracted == expected
        status = "✓" if passed else "✗"
        
        if not passed:
            all_passed = False
        
        extract_status = "Extracted" if extracted else "Failed to extract"
        print(f"{status} Output: '{output}'")
        print(f"   Expected: '{expected}' | Got: '{extracted}' | {extract_status}")
        print()
    
    print("=" * 60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed!")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = test_extract_answer()
    sys.exit(0 if success else 1)
