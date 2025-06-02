import json
import os
import pathlib
import sys

import evaluate
import numpy as np
import sqlalchemy as sa
import torch
from dotenv import dotenv_values
from huggingface_hub import login
from peft import PeftModelForCausalLM
from transformers import TrainingArguments
from transformers.models.llama import LlamaForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

from src.hypermixsub import extract_hms_from_db
from src.prompt import prepare_hyper_mix_sub_prompts

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
    tokenizer: PreTrainedTokenizerFast = _t[1]

    dd = extract_hms_from_db(engine)
    dd = prepare_hyper_mix_sub_prompts(dd, tokenizer)
    trn_ds, val_ds, tst_ds = dd["TRAIN"], dd["VALIDATION"], dd["TEST"]

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
        tokenizer=tokenizer,
        train_dataset=trn_ds,
        eval_dataset=val_ds,
        packing=True,
        # The field on which to train the model, we have added the generated prompt under 'Prompt'
        dataset_text_field="Prompt",
        # max_seq_length=MAX_SEQ_LEN,
        dataset_num_proc=os.cpu_count(),
        compute_metrics=lambda preds: compute_metrics(preds, tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # data_collator=DataCollatorForSeq2Seq(
        #     tokenizer=tokenizer,
        #     model=model,
        # ),
        args=TrainingArguments(
            per_device_train_batch_size=6,
            per_device_eval_batch_size=6,
            num_train_epochs=int(
                config["MODEL_EPOCHS"]
            ),  # Set this to 1 for one full training run
            warmup_steps=256,
            eval_strategy="epoch",
            save_strategy="steps",
            save_steps=512,
            learning_rate=0.0002,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            report_to=["codecarbon", "tensorboard"],
            output_dir=config["MODEL_OUTPUT_DIR"],
            logging_dir=config["MODEL_LOGGING_DIR"],
            logging_steps=1,
            # load_best_model_at_end=True,
            # push_to_hub=True,
            hub_model_id=config["HF_TRAINED_MODEL_REPO"],
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        num_proc=os.cpu_count(),
    )

    if list(pathlib.Path(config["MODEL_OUTPUT_DIR"]).glob("checkpoint-*")):
        trainer_stats = trainer.train(resume_from_checkpoint=True)
    else:
        trainer_stats = trainer.train()

    model.save_pretrained(save_directory=f"{config['MODEL_PRETRAINED_DIR']}")

    trainer.push_to_hub(
        commit_message=f"finished fine tuning till {int(config['MODEL_EPOCHS'])} epochs",
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
