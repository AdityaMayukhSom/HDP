import os
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Step 1: Download and save dataset
dataset_path = "./mixsub_dataset"
if not os.path.exists(dataset_path):
    dataset = load_dataset("TRnlp/MixSub")
    if isinstance(dataset, dict):  # Handle DatasetDict
        dataset = dataset["train"]
    dataset.save_to_disk(dataset_path)
else:
    dataset = load_from_disk(dataset_path)

# Step 2: Load model and tokenizer
model_name = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1,
)


# Generation function
def generate_hallucination(example):
    prompt = f"""<|im_start|>user
Generate a scientifically inaccurate or hallucinated highlight for this abstract:
{example['Abstract']}
The highlight should sound plausible but contain incorrect information.<|im_end|>
<|im_start|>assistant
"""
    response = pipe(
        prompt,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    full_text = response[0]["generated_text"]
    start_idx = full_text.find("<|im_start|>assistant\n") + len(
        "<|im_start|>assistant\n"
    )
    end_idx = full_text.find("<|im_end|>", start_idx)
    hallucination = (
        full_text[start_idx:end_idx].strip()
        if end_idx != -1
        else full_text[start_idx:].strip()
    )

    return {"hallucinated_highlight": hallucination}


# Process dataset
updated_dataset = dataset.map(generate_hallucination)

# Step 5: Save updated dataset
updated_dataset.save_to_disk("./mixsub_with_hallucinations")
print("Dataset updated and saved successfully!")
