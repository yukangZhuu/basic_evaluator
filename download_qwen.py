#!/usr/bin/env python3
"""Download Qwen3-1.7B to local models folder."""
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-1.7B"
SAVE_DIR = "/root/autodl-tmp/models/Qwen3-1.7B"

print(f"Downloading {MODEL_ID} to {SAVE_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

print("Saving to local folder...")
tokenizer.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)
print("Done.")
