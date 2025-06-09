import json
import os
import pathlib
import sys

import torch

from unsloth import FastLanguageModel, is_bfloat16_supported  # isort: skip
from unsloth.chat_templates import train_on_responses_only  # isort: skip

from huggingface_hub import login
from peft import PeftModelForCausalLM
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from transformers.models.llama import LlamaForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from trl import SFTTrainer

sys.path.append(str(pathlib.Path(__file__).parent.parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from src.hypermixsub import extract_hms_from_db
from src.prompt import (
    prepare_hms_convos_for_fine_tuning,
    prepare_hms_prompts_for_fine_tuning,
)
from src.utils import get_postgresql_engine, load_dotenv_in_config

from .metrics import compute_metrics, preprocess_logits_for_metrics


def main(argv: list[str]):
    engine = get_postgresql_engine()
    config = load_dotenv_in_config()
    login(token=config["HF_TOKEN"])

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
    dd = prepare_hms_convos_for_fine_tuning(dd)
    dd = prepare_hms_prompts_for_fine_tuning(dd, tokenizer)
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
        dataset_text_field="text",
        # max_seq_length=MAX_SEQ_LEN,
        dataset_num_proc=os.cpu_count(),
        compute_metrics=lambda preds: compute_metrics(preds, tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=TrainingArguments(
            per_device_train_batch_size=6,
            per_device_eval_batch_size=6,
            # Set this to 1 for one full training run
            num_train_epochs=int(config["MODEL_EPOCHS"]),
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
            report_to=["codecarbon", "tensorboard", "wandb"],
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
        instruction_part="<|start_header_id|>system<|end_header_id|>\n\n",
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
