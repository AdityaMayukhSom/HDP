import json
import os
import sys
from typing import Literal

import numpy as np
import torch
from dotenv import dotenv_values

from unsloth import FastLanguageModel, is_bfloat16_supported  # isort:skip
from unsloth.chat_templates import train_on_responses_only  # isort:skip
from transformers import DataCollatorForSeq2Seq, TrainingArguments  # isort:skip
from transformers.models.llama import LlamaForCausalLM  # isort:skip
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast  # isort:skip
from datasets import DatasetDict, load_dataset  # isort:skip
from peft import PeftModelForCausalLM  # isort:skip
from trl import SFTTrainer  # isort:skip
import evaluate  # isort:skip


class FineTuneConf:
    BASE_MODEL_REPO = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"

    TRAINED_MODEL_ACNT = "AdityaMayukhSom"
    TRAINED_MODEL_NAME = "Llama-3.2-1B-Instruct-MixSub-Remastered"
    TRAINED_MODEL_REPO = f"{TRAINED_MODEL_ACNT}/{TRAINED_MODEL_NAME}"

    SUMMARIZER_INSTRUCTION = """
    You are instructed to generate a scientifically accurate highlight of the provided passage without additional
    sentences such as headings or introductions before or after the generated text as it will be used as summary
    in a custom dataset. The highlight should sound plausible and should not contain incorrect information. Generate
    3-5 concise highlight points from the provided research paper abstract, covering key contributions, methods and
    outcomes. Each point should contain 10 to 15 words only. Return the points in plain text format without bullets.

    No Additional Commentary: Exclude lines like "Here are 3-5 concise highlight points".
    """


def prepare_prompt(
    examples: dict[str, list[str]],
    *,
    tokenizer: PreTrainedTokenizerFast,
):
    prompts: list[str] = []

    abstracts = examples["Abstract"]
    highlights = examples["Highlight"]

    for abstract, highlight in zip(abstracts, highlights):
        row_json = [
            {
                "role": "system",
                "content": FineTuneConf.SUMMARIZER_INSTRUCTION,
            },
            {
                "role": "user",
                "content": abstract,
            },
            {
                "role": "assistant",
                # Must add EOS_TOKEN, otherwise your generation will go on forever!
                "content": highlight + tokenizer.eos_token,
            },
        ]

        prompt = tokenizer.apply_chat_template(
            row_json,
            tokenize=False,
            add_generation_prompt=False,
            return_tensors="pt",
        )

        prompts.append(prompt)

    return {
        "Prompt": prompts,
    }


def get_mixsub(
    tknz: PreTrainedTokenizerFast,
    *,
    trim_ds: bool = False,
    trim_trn_len=2000,
    trim_val_len=1000,
    trim_tst_len=1000,
):
    # https://huggingface.co/docs/datasets/en/loading#hugging-face-hub
    ds: DatasetDict[Literal["train", "validation", "test"]] = load_dataset(
        "TRnlp/MixSub"
    )

    # Changing all the column names to have uniform singular forms
    # All column names are now in singular form
    ds = ds.rename_column("Highlights", "Highlight")

    ds = ds.map(
        prepare_prompt,
        num_proc=os.cpu_count(),
        batched=True,
        fn_kwargs={"tokenizer": tknz},
    )

    # execute the following lines to train the model on the entire dataset.
    trn_ds, val_ds, tst_ds = ds["train"], ds["validation"], ds["test"]

    # in case prototyping, set trim_ds to true
    if trim_ds:
        # select less number of examples for training, and even less for testing,
        # if everything goes well,  we can fine tune on a larger dataset
        trn_ds = trn_ds.select(range(trim_trn_len))
        val_ds = val_ds.select(range(trim_val_len))
        tst_ds = tst_ds.select(range(trim_tst_len))

    return trn_ds, val_ds, tst_ds


# https://huggingface.co/docs/evaluate/package_reference/loading_methods#evaluate.load.path
# Trainer.py does not have this, the following snippet is sourced from this link.
# https://github.com/huggingface/trl/issues/862#issuecomment-1896074498

# we need to compute two metrices seperately
# https://discuss.huggingface.co/t/log-multiple-metrics-while-training/8115/2
metric_bleu = evaluate.load("bleu")
metric_rouge = evaluate.load("rouge")


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds, tokenizer):
    # Here preds are all_preds and labels are label_ids/all_labels.
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 in the preds as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(pred.strip()) for pred in decoded_preds]

    decoded_labels = ["\n".join(label.strip()) for label in decoded_labels]

    scores_bleu = metric_bleu.compute(
        predictions=decoded_preds,
        references=decoded_labels,
    )
    scores_rouge = metric_rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
    )

    # https://www.freecodecamp.org/news/python-merge-dictionaries-merging-two-dicts-in-python/
    # https://www.datacamp.com/tutorial/python-dictionary-append

    scores: dict[str, str] = {}

    for key, score in scores_bleu.items():
        scores[f"bleu_{key}"] = score

    for key, score in scores_rouge.items():
        scores[f"rouge_{key}"] = score

    return scores


def main(argv: list[str]):
    config = dotenv_values(".env")

    _t = FastLanguageModel.from_pretrained(
        model_name=FineTuneConf.BASE_MODEL_REPO,
        max_seq_length=1024,
        dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        load_in_4bit=True,
    )

    flm: LlamaForCausalLM = _t[0]
    tknz: PreTrainedTokenizerFast = _t[1]

    model: PeftModelForCausalLM = FastLanguageModel.get_peft_model(
        flm,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    trn_ds, val_ds, tst_ds = get_mixsub(tknz)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tknz,
        train_dataset=trn_ds,
        eval_dataset=tst_ds,
        # The field on which to train the model, we have added the generated prompt under 'Prompt'
        dataset_text_field="Prompt",
        # max_seq_length=MAX_SEQ_LEN,
        dataset_num_proc=2,
        packing=False,
        compute_metrics=lambda preds: compute_metrics(preds, tknz),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tknz,
            model=model,
        ),
        args=TrainingArguments(
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            eval_strategy="steps",
            eval_steps=2,
            num_train_epochs=3,  # Set this to 1 for one full training run
            save_total_limit=3,
            save_steps=2,
            # max_steps = MAX_STEPS,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=FineTuneConf.TRAINED_MODEL_NAME,
            report_to="none",
            load_best_model_at_end=True,
            push_to_hub=False,
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>system<|end_header_id|>",
        response_part="<|start_header_id|>assistant<|end_header_id|>",
    )

    print(trn_ds.to_pandas().head())

    return

    # gpu_stats = torch.cuda.get_device_properties(0)
    # start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    # max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    # print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    # print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    # used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    # used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    # used_percentage = round(used_memory / max_memory * 100, 3)
    # lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    # print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    # print(
    #     f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
    # )
    # print(f"Peak reserved memory = {used_memory} GB.")
    # print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    # print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    # print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    trainer.push_to_hub(
        commit_message="first epoch fine tuning on mixsub",
        model_name=FineTuneConf.TRAINED_MODEL_NAME,
        token=config["HF_TOKEN"],
        # language="en",
        # finetuned_from=MODEL_NAME,
        # dataset=DATASET_NAME
    )

    # This is to evaluate the fine-tuned model on the eval dataset
    # it will compute the compute metrics for the model
    results = trainer.evaluate()

    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main(sys.argv)
