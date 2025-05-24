import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import sqlalchemy as sa
import torch
from dotenv import dotenv_values

from unsloth import FastLanguageModel, is_bfloat16_supported  # isort:skip
from unsloth.chat_templates import train_on_responses_only  # isort:skip
from transformers import TrainingArguments  # isort:skip
from transformers.models.llama import LlamaForCausalLM  # isort:skip
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast  # isort:skip
from datasets import DatasetDict, Dataset  # isort:skip
from peft import PeftModelForCausalLM  # isort:skip
from trl import SFTTrainer  # isort:skip
import evaluate  # isort:skip
from huggingface_hub import login


def prepare_prompt(
    examples: dict[str, list[str]],
    *,
    tknzr: PreTrainedTokenizerFast,
    sys_instr: str,
    usr_instr: str,
):
    prompts: list[str] = []

    abstracts = examples["Abstract"]
    highlights = examples["Highlight"]

    for abstract, highlight in zip(abstracts, highlights):
        row_json = [
            {
                "role": "system",
                "content": sys_instr,
            },
            {
                "role": "user",
                "content": f"{usr_instr}\n\n{abstract}",
            },
            {
                "role": "assistant",
                # Must add EOS_TOKEN, otherwise your generation will go on forever!
                "content": highlight + tknzr.eos_token,
            },
        ]

        prompt = tknzr.apply_chat_template(
            row_json,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompts.append(prompt)

    return {
        "Prompt": prompts,
    }


def prepare_mixsub(
    engine: sa.Engine,
    tknzr: PreTrainedTokenizerFast,
):
    extract_query = sa.text("""
    SELECT "Filename",
        CASE
            WHEN "BetterAbstract" IS NOT NULL
                    AND "BetterAbstract" != 'NOT_AVAILABLE' THEN "BetterAbstract"
            ELSE "Abstract"
        END AS "Abstract",
        CASE
            WHEN "BetterHighlight" IS NOT NULL
                    AND "BetterHighlight" != 'NOT_AVAILABLE' THEN "BetterHighlight"
            ELSE "Highlight"
        END AS "Highlight"
    FROM "MixSub"
    WHERE "Split" = :split; 
    """)

    with engine.connect() as conn:
        trn_df = pd.read_sql(extract_query, conn, params={"split": "TRAIN"})
        val_df = pd.read_sql(extract_query, conn, params={"split": "VALIDATION"})
        tst_df = pd.read_sql(extract_query, conn, params={"split": "TEST"})

    print("train dataframe null", trn_df.isnull().sum().sum())
    print("validation dataframe null", val_df.isnull().sum().sum())
    print("test dataframe null", tst_df.isnull().sum().sum())

    trn_ds = Dataset.from_pandas(trn_df)
    val_ds = Dataset.from_pandas(val_df)
    tst_ds = Dataset.from_pandas(tst_df)

    dd = DatasetDict(
        {
            "train": trn_ds,
            "validation": val_ds,
            "test": tst_ds,
        }
    )

    # https://huggingface.co/docs/datasets/en/loading#hugging-face-hub
    # ds: DatasetDict[Literal["train", "validation", "test"]] = load_dataset(
    #     "TRnlp/MixSub"
    # )

    # Changing all the column names to have uniform singular forms
    # All column names are now in singular form
    # ds = dd.rename_column("Highlights", "Highlight")

    with (
        open("./scripts/instruction/finetune-system.txt", "r", encoding="utf-8") as sif,
        open("./scripts/instruction/finetune-user.txt", "r", encoding="utf-8") as uif,
    ):
        sys_instr = sif.read()
        usr_instr = uif.read()

        dd = dd.map(
            prepare_prompt,
            num_proc=os.cpu_count(),
            batched=True,
            fn_kwargs={
                "tknzr": tknzr,
                "sys_instr": sys_instr,
                "usr_instr": usr_instr,
            },
        )

    # execute the following lines to train the model on the entire dataset.
    trn_ds, val_ds, tst_ds = dd["train"], dd["validation"], dd["test"]

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


def compute_metrics(eval_preds, tokenizer: PreTrainedTokenizerFast):
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
    login(token=config["HF_TOKEN"])

    conn_url = sa.URL.create(
        drivername="postgresql+psycopg2",
        username=config["PG_USERNAME"],
        password=config["PG_PASSWORD"],
        host=config["PG_HOST"],
        database=config["PG_DATABASE"],
        port=int(config["PG_PORT"]),
    )

    engine = sa.create_engine(conn_url)

    _t = FastLanguageModel.from_pretrained(
        model_name=config["HF_BASE_MODEL_REPO"],
        max_seq_length=1024,
        dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        load_in_4bit=True,
        token=config["HF_TOKEN"],
        disable_log_stats=False,
    )

    flm: LlamaForCausalLM = _t[0]
    tknzr: PreTrainedTokenizerFast = _t[1]

    trn_ds, val_ds, tst_ds = prepare_mixsub(engine, tknzr)

    model: PeftModelForCausalLM = FastLanguageModel.get_peft_model(
        flm,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        use_gradient_checkpointing="unsloth",
        use_rslora=True,
        loftq_config=None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tknzr,
        train_dataset=trn_ds,
        eval_dataset=val_ds,
        packing=True,
        # The field on which to train the model, we have added the generated prompt under 'Prompt'
        dataset_text_field="Prompt",
        # max_seq_length=MAX_SEQ_LEN,
        dataset_num_proc=os.cpu_count(),
        compute_metrics=lambda preds: compute_metrics(preds, tknzr),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # data_collator=DataCollatorForSeq2Seq(
        #     tokenizer=tknzr,
        #     model=model,
        # ),
        args=TrainingArguments(
            per_device_train_batch_size=6,
            per_device_eval_batch_size=6,
            num_train_epochs=4,  # Set this to 1 for one full training run
            warmup_steps=512,
            eval_strategy="epoch",
            save_strategy="steps",
            save_steps=512,
            learning_rate=0.0004,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            report_to="none",
            output_dir=config["MODEL_OUTPUT_DIR"],
            logging_dir=config["MODEL_LOGGING_DIR"],
            logging_steps=1,
            # load_best_model_at_end=True,
            # push_to_hub=True,
            hub_model_id=f"{config['HF_TRAINED_MODEL_ACNT']}/{config['HF_TRAINED_MODEL_NAME']}",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        num_proc=os.cpu_count(),
    )

    # gpu_stats = torch.cuda.get_device_properties(0)
    # start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    # max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    # print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    # print(f"{start_gpu_memory} GB of memory reserved.")

    if list(pathlib.Path(config["MODEL_OUTPUT_DIR"]).glob("checkpoint-*")):
        trainer_stats = trainer.train(resume_from_checkpoint=True)
    else:
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

    model.save_pretrained(
        save_directory=f"${config['HF_TRAINED_MODEL_NAME']}-SaveDirectory"
    )

    trainer.push_to_hub(
        commit_message=f"Finetuning {config['HF_BASE_MODEL_REPO']} finished with HyperMixSub",
        # token=config["HF_TOKEN"],
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
