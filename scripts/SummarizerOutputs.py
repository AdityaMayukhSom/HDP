import json
import os
import pathlib
import sys
import time

import torch
from loguru import logger

from unsloth import FastLanguageModel, is_bfloat16_supported  # isort: skip
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src.config import Config
from src.hypermixsub import (
    extract_hms_from_db,
    insert_generated_highlight_into_db,
)
from src.prompt import eval_summarizer_row_json_single_example
from src.utils import elapsed_time, get_postgresql_engine

os.environ["TZ"] = "Asia/Kolkata"
time.tzset()


def generate_eval_highlights(
    abstracts: list[str],
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
):
    # init_identifier = "<|start_header_id|>assistant<|end_header_id|>"
    # term_identifier = "<|eot_id|>"
    term_identifier = tokenizer.eos_token

    batch_data = [
        eval_summarizer_row_json_single_example(abstract) for abstract in abstracts
    ]

    inputs = tokenizer.apply_chat_template(
        batch_data,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_dict=True,
    ).to(Config.DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    highlights = tokenizer.batch_decode(
        outputs[:, inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    )

    # the generaed output contains the prompt, init identifier, generated highlight and term identifier
    # the following code splits the output with init identifier, takes the second section which contains
    # the generated highlight followed by term identifier, now splits the second section based on term
    # identifier, takes the first section, which contains only the generated output. Then it strips the
    # generated content to get rid of any white spaces from the beginning and the end, and replaces
    # newline character with whitespace.

    highlights = [
        highlight.split(term_identifier)[0].strip().replace("\n", " ")
        for highlight in highlights
    ]

    inputs.to("cpu")

    del batch_data
    del inputs

    return {
        "ModelGeneratedHighlight": highlights,
    }


def create_model_generated_highlights():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="AdityaMayukhSom/Llama-3.2-1B-HyperMixSub-Full",
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
    )

    FastLanguageModel.for_inference(model)

    # language=TEXT
    extract_query = """
    SELECT "MixSub"."PII", "MixSubView"."ArticleAbstract", "MixSubView"."CorrectHighlight" 
    FROM "MixSubView" JOIN "MixSub" ON "MixSubView"."PII" = "MixSub"."PII"
    WHERE "MixSub"."ModelGeneratedHighlight" IS NULL AND "MixSub"."Split" = :split
    """

    engine = get_postgresql_engine()
    dd = extract_hms_from_db(
        engine=engine,
        query=extract_query,
    )

    # Map the existing dataset rows into hallucinated highlights, the returned
    # dictionary from the function passed to map, will automatically be appended
    # to the dataset on which the map function is being called, and a new dataset
    # will be returned, so note that mapping does not modify the dataset on which
    # it is being called on.
    logger.info("summarizer model dataset generation started")

    start_time = time.time()

    for split, ds in dd.items():
        it = ds.iter(batch_size=1, drop_last_batch=False)
        logger.info(f"starting processing split {split}")

        for idx, rows in enumerate(it):
            d = generate_eval_highlights(
                rows["ArticleAbstract"],
                model,
                tokenizer,
            )

            p = {
                "pii": rows["PII"][0],
                "highlight": d["ModelGeneratedHighlight"][0],
            }

            logger.info(f"finished batch {idx}\n\n{json.dumps(p, indent=4)}")

            insert_generated_highlight_into_db(
                engine,
                rows["PII"],
                d["ModelGeneratedHighlight"],
            )

            logger.success(
                f"successfully processed  pii {rows['PII']}, split {split}, time elapsed: {elapsed_time(start_time)}"
            )

    logger.success("summarizer model dataset generation finished")

    # if not os.path.exists(out_dataset_dir):
    # Path(out_dataset_dir).mkdir(parents=True, exist_ok=True)
    # out_ds_path = os.path.join(out_dataset_dir, out_csv_file_name)

    # logger.info("started pushing finetuned model dataset to huggingface")
    # out_ds.push_to_hub(out_ds_hf_name)
    # logger.success("finetuned model dataset saved to huggingface as hf dataset")


def main():
    # generate_small_dataset()
    create_model_generated_highlights()


if __name__ == "__main__":
    main()
