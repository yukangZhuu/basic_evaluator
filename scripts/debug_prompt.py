#!/usr/bin/env python3
"""
Diagnostic: print the EXACT prompt that would be sent to vLLM,
including chat template rendering, so you can compare with verl training prompts.

Usage:
    python scripts/debug_prompt.py [--n 3] [--model MODEL_PATH]
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import Config
from adaptors.adaptor_factory import AdaptorFactory


def main():
    n = 3
    for arg in sys.argv[1:]:
        if arg.startswith("--n"):
            n = int(sys.argv[sys.argv.index(arg) + 1])

    print("=" * 70)
    print("PROMPT DIAGNOSTIC")
    print("=" * 70)
    print(f"  BENCHMARK_TYPE:      {Config.BENCHMARK_TYPE}")
    print(f"  BENCHMARK_DATA_PATH: {Config.BENCHMARK_DATA_PATH}")
    print(f"  THINKING_MODE:       {Config.THINKING_MODE}")
    print(f"  MODEL_PATH:          {Config.MODEL_PATH}")
    print(f"  TEMPERATURE:         {Config.TEMPERATURE}")
    print()

    # 1. Adaptor-level prompt (before chat template)
    adaptor = AdaptorFactory.create_adaptor(
        Config.BENCHMARK_TYPE, Config.BENCHMARK_DATA_PATH, Config.THINKING_MODE
    )
    data = adaptor.load_benchmark_data()

    is_raw = getattr(adaptor, "raw_prompts", False)
    system_prompt = None if is_raw else getattr(adaptor, "system_prompt", None)

    print(f"  Adaptor class:       {type(adaptor).__name__}")
    print(f"  System prompt:       {system_prompt!r}")
    print(f"  raw_prompts:         {is_raw}")
    print(f"  Total data items:    {len(data)}")

    for i in range(min(n, len(data))):
        item = data[i]
        raw_prompt = adaptor.format_prompt(item)
        gt = adaptor.get_ground_truth(item)

        print(f"\n{'─' * 70}")
        print(f"  ITEM {i}  |  ground_truth = {gt!r}")
        print(f"{'─' * 70}")
        print(f"\n[Adaptor raw prompt (user content)]:\n{raw_prompt}")

    # 2. After chat template (requires model tokenizer)
    print(f"\n{'=' * 70}")
    print("RENDERED PROMPTS (after chat template)")
    print("=" * 70)

    if not os.path.exists(Config.MODEL_PATH):
        print(f"\n  Model not found at {Config.MODEL_PATH}, skipping chat template render.")
        print("  To see rendered prompts, set MODEL_PATH to a valid local model.")
        return

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH, trust_remote_code=True)

        for i in range(min(n, len(data))):
            item = data[i]
            raw_prompt = adaptor.format_prompt(item)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": raw_prompt})

            # Try with enable_thinking
            try:
                rendered_thinking = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=Config.THINKING_MODE
                )
            except TypeError:
                rendered_thinking = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )

            # Try without enable_thinking
            try:
                rendered_no_thinking = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False
                )
            except TypeError:
                rendered_no_thinking = rendered_thinking

            print(f"\n{'─' * 70}")
            print(f"  ITEM {i}  |  enable_thinking={Config.THINKING_MODE}")
            print(f"{'─' * 70}")
            print(rendered_thinking)

            if rendered_thinking != rendered_no_thinking:
                print(f"\n  [enable_thinking=False would produce different result!]")
                diff_start = next(
                    (j for j in range(min(len(rendered_thinking), len(rendered_no_thinking)))
                     if rendered_thinking[j] != rendered_no_thinking[j]),
                    min(len(rendered_thinking), len(rendered_no_thinking))
                )
                print(f"  First diff at char {diff_start}")
                print(f"  thinking:    ...{rendered_thinking[max(0,diff_start-20):diff_start+50]!r}...")
                print(f"  no_thinking: ...{rendered_no_thinking[max(0,diff_start-20):diff_start+50]!r}...")

    except Exception as e:
        print(f"\n  Error loading tokenizer: {e}")


if __name__ == "__main__":
    main()
