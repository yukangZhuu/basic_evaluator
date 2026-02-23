import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaptors.teacher_traces_adaptor import TeacherTracesAdaptor


def test_latex_wrapping():
    print("=" * 60)
    print("Testing LaTeX Wrapping Functions")
    print("=" * 60)
    
    adaptor = TeacherTracesAdaptor.__new__(TeacherTracesAdaptor)
    
    test_cases = [
        ("10", False),
        ("\\boxed{10}", True),
        ("$10$", True),
        ("$$10$$", True),
        ("\\[10\\]", True),
        ("\\(10\\)", True),
        ("[10]", True),
        ("\\boxed{x + y}", True),
        ("$x + y$", True),
        ("$$\\frac{1}{2}$$", True),
        ("\\[\\sqrt{2}\\]", True),
        ("42", False),
        ("\\frac{1}{2}", False),
    ]
    
    print("\nTesting _is_latex_wrapped():")
    print("-" * 60)
    for text, expected in test_cases:
        result = adaptor._is_latex_wrapped(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{text}' -> {result} (expected: {expected})")
    
    print("\n\nTesting _wrap_latex():")
    print("-" * 60)
    wrap_test_cases = [
        ("10", "\\boxed{10}"),
        ("\\boxed{10}", "\\boxed{10}"),
        ("$10$", "$10$"),
        ("$$10$$", "$$10$$"),
        ("\\[10\\]", "\\[10\\]"),
        ("\\(10\\)", "\\(10\\)"),
        ("[10]", "[10]"),
        ("42", "\\boxed{42}"),
        ("\\frac{1}{2}", "\\boxed{\\frac{1}{2}}"),
        ("x + y", "\\boxed{x + y}"),
    ]
    
    all_passed = True
    for text, expected in wrap_test_cases:
        result = adaptor._wrap_latex(text)
        passed = result == expected
        status = "✓" if passed else "✗"
        if not passed:
            all_passed = False
        print(f"{status} '{text}' -> '{result}' (expected: '{expected}')")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed!")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = test_latex_wrapping()
    sys.exit(0 if success else 1)
