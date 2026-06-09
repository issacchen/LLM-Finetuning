import os
import csv
import json
import shutil
from typing import Optional, List, Any
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from tools import DEFAULT_SYSTEM_MSG 
# Note: We do NOT import TOOLS here anymore to avoid stale data

def authenticate_hf(token: Optional[str]) -> None:
    """Logs into the Hugging Face Hub."""
    if token:
        print("Logging into Hugging Face Hub...")
        login(token=token)
    else:
        print("Skipping Hugging Face login: HF_TOKEN not set.")

def load_model_and_tokenizer(model_name: str):
    print(f"Loading Transformer model: {model_name}")
    try:
        target_model = model_name
        if model_name.startswith("..") and not os.path.exists(model_name):
            print(f"Warning: Local path {model_name} not found. Falling back to default hub model.")
            target_model = "google/gemma-2b-it" 

        tokenizer = AutoTokenizer.from_pretrained(target_model)
        model = AutoModelForCausalLM.from_pretrained(target_model)
        print("Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading Transformer model {target_model}: {e}")
        raise e

# UPDATED: Now accepts tools_list as an argument
def create_conversation_format(sample, tools_list):
    """Formats a dataset row into the conversational format required for SFT."""
    try:
        tool_args = json.loads(sample["tool_arguments"])
    except (json.JSONDecodeError, TypeError):
        tool_args = {}
        
    return {
        "messages": [
            {"role": "developer", "content": DEFAULT_SYSTEM_MSG},
            {"role": "user", "content": sample["user_content"]},
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": sample["tool_name"], "arguments": tool_args}}]},
        ],
        "tools": tools_list # Injects the dynamic tools
    }

def parse_csv_dataset(file_path: str) -> List[List[str]]:
    """Parses an uploaded CSV file."""
    dataset = []
    if not file_path:
        return dataset
        
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            if not (header and "user_content" in header[0].lower()):
                f.seek(0)
        except StopIteration:
            return dataset

        for row in reader:
            if len(row) >= 3:
                dataset.append([s.strip() for s in row[:3]])
    return dataset

def zip_directory(source_dir: str, output_name_base: str) -> str:
    """Zips a directory."""
    return shutil.make_archive(
        base_name=output_name_base,
        format='zip',
        root_dir=source_dir,
    )
