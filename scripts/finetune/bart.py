import json
import os
import pathlib
import sys

import sqlalchemy as sa
import torch
from dotenv import dotenv_values
from huggingface_hub import login
from peft import PeftModelForCausalLM

from transformers import TrainingArguments  # isort:skip
from transformers.models.llama import LlamaForCausalLM  # isort:skip
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast  # isort:skip
from trl import SFTTrainer


def get_peft_model_name_or_path(model_name_or_path, peft_config):
    return (
        f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}".replace(
            "/", "_"
        )
    )


def get_optimizer(model, lr):
    return torch.optim.AdamW(model.parameters(), lr=lr)


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

    # gpu_stats = torch.cuda.get_device_properties(0)
    # start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    # max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    # print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    # print(f"{start_gpu_memory} GB of memory reserved.")

    # try:
    #     tracker = OfflineEmissionsTracker(
    #         country_iso_code="IND",
    #         measure_power_secs=4,
    #         save_to_file=True,
    #         output_dir="./emissions",
    #         log_level="debug",
    #         tracking_mode="process",
    #     )
    #     tracker.start()
    #     if list(pathlib.Path(config["MODEL_OUTPUT_DIR"]).glob("checkpoint-*")):
    #         trainer_stats = trainer.train(resume_from_checkpoint=True)
    #     else:
    #         trainer_stats = trainer.train()
    # except Exception as e:
    #     print(e)
    # finally:
    #     emission = tracker.stop()
    #     print(f"emission is {emission}")

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

    model.save_pretrained(save_directory=f"{config['MODEL_PRETRAINED_DIR']}")

    trainer.push_to_hub(
        commit_message=f"finished finetuning till {int(config['MODEL_EPOCHS'])} epochs",
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
