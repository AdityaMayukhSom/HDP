from typing import Literal

import torch
from datasets import DatasetDict, load_dataset
from dotenv import dotenv_values
from loguru import logger

from unsloth import FastLanguageModel  # isort:skip
from transformers import AutoModelForCausalLM, AutoTokenizer  # isort:skip


def get_mixsub():
    ds: DatasetDict = load_dataset("TRnlp/MixSub")
    # Changing all the column names to have uniform singular forms
    # All column names are now in singular form
    ds = ds.rename_column("Highlights", "Highlight")
    return ds


instruction = """
You're instructed to generate scientifically inaccurate or hallucinated highlights of the provided passage
only without additional sentences like headings, introductions, or text before or after the generated output
as the output will be directly used as highlight in a custom dataset. The highlight should sound plausible
but contain incorrect information.Generate 3-5 concise highlight points from the provided research paper abstract,
covering key contributions, methods, and outcomes. Each point should contain 10 to 15 words only. Return the
points in plain text format without bullets.

No Additional Commentary: Exclude lines like "Here are 3-5 concise highlight points".
"""


def generate_hallucinated_highlights(
    row,
    *args,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    **kwargs,
) -> dict[Literal["Hallucination"], str]:
    init_identifier = "<|start_header_id|>assistant<|end_header_id|>"
    term_identifier = "<|eot_id|>"

    abstract = row["Abstract"]

    row_json = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": abstract},
    ]

    model_prompt = tokenizer.apply_chat_template(
        row_json,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(
        model_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    model_outputs = model.generate(
        **model_inputs,
        max_new_tokens=150,
        num_return_sequences=1,
    )

    decoded_text = tokenizer.batch_decode(
        model_outputs,
        skip_special_tokens=False,
    )

    # the generaed output contains the prompt, init identifier, generated highlight and term identifier
    # the following code splits the output with init identifier, takes the second section which contains
    # the generated highlight followed by term identifier, now splits the second section based on term
    # identifier, takes the first section, which contains only the generated output. Then it strips the
    # generated content to get rid of any white spaces from the beginning and the end, and replaces
    # newline character with whitespace.
    hallucination = decoded_text[0].split(init_identifier)[1].split(term_identifier)[0]
    hallucination = hallucination.strip().replace("\n", " ")

    return {
        "Hallucination": hallucination,
    }


if __name__ == "__main__":
    config = dotenv_values(".env")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        max_seq_length=2048,
    )
    FastLanguageModel.for_inference(model)

    mixsub = get_mixsub()

    # Map the existing dataset rows into hallucinated highlights, the returned
    # dictionary from the function passed to map, will automatically be appended
    # to the dataset on which the map function is being called, and a new dataset
    # will be returned, so note that mapping does not modify the dataset on which
    # it is being called on.
    # logger.info("hallucinated dataset generation started")
    # hal_ds = mixsub.map(
    #     generate_hallucinated_highlights,
    #     fn_kwargs={
    #         "model": model,
    #         "tokenizer": tokenizer,
    #         "device": torch.device(device),
    #     },
    # )
    # logger.info("hallucinated dataset generation finished")

    # logger.info("hallucinated dataset saving started")
    # hal_ds.push_to_hub("AdityaMayukhSom/MixSubV2-HaluEval", token=config["HF_TOKEN"])
    # logger.info("hallucinated dataset saved to huggingface as hf dataset")

    generate_hallucinated_highlights()
