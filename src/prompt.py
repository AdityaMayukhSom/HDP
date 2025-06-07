import os
from functools import lru_cache
from pathlib import Path

from datasets import Dataset, DatasetDict
from loguru import logger
from transformers import PreTrainedTokenizerFast


@lru_cache
def get_summarizer_system_instructions(
    path="./instructions/summarizer-system.txt",
) -> str:
    return Path(path).read_text()


@lru_cache
def get_summarizer_user_instructions(
    path="./instructions/summarizer-user.txt",
) -> str:
    return Path(path).read_text()


@lru_cache
def get_ner_system_instructions(
    path="./instructions/ner-system.txt",
) -> str:
    return Path(path).read_text()


@lru_cache
def get_ner_user_instructions(
    path="./instructions/ner-user.txt",
) -> str:
    return Path(path).read_text()


def create_summarizer_convo_for_fine_tuning(examples):
    abstracts = examples["ArticleAbstract"]
    highlights = examples["CorrectHighlight"]

    sys_instr = get_summarizer_system_instructions()
    usr_instr = get_summarizer_user_instructions()

    convos: list = []

    for abstract, highlight in zip(abstracts, highlights):
        convo = [
            {
                "role": "system",
                "content": sys_instr,
            },
            {
                "role": "user",
                "content": f"{usr_instr}\n{abstract}",
            },
            {
                "role": "assistant",
                # Must add EOS_TOKEN, otherwise your generation will go on forever!
                "content": highlight,
            },
        ]
        convos.append(convo)

    return {
        "conversations": convos,
    }


def create_summarizer_prompt_for_fine_tuning(
    examples: Dataset,
    *,
    tokenizer: PreTrainedTokenizerFast,
):
    convos: list = examples["conversations"]

    prompts: list[str] = []

    eos_token = (
        tokenizer.eos_token
        if tokenizer.eos_token is not None and isinstance(tokenizer.eos_token, str)
        else ""
    )

    for convo in convos:
        # assuming the last value in convo is assistant one, turn on for llama
        convo[-1]["content"] = convo[-1]["content"] + eos_token

        prompt = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        prompts.append(prompt)

    return {
        "text": prompts,
    }


def prepare_hms_convos_for_fine_tuning(dd: Dataset | DatasetDict):
    dd = dd.map(
        create_summarizer_convo_for_fine_tuning,
        num_proc=os.cpu_count(),
        batched=True,
    )
    return dd


def prepare_hms_prompts_for_fine_tuning(
    dd: Dataset | DatasetDict,
    tokenizer: PreTrainedTokenizerFast,
) -> DatasetDict:
    dd = dd.map(
        create_summarizer_prompt_for_fine_tuning,
        num_proc=os.cpu_count(),
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
        },
    )
    return dd


def eval_summarizer_row_json_single_example(abstract: str):
    sys_instr = get_summarizer_system_instructions()
    usr_instr = get_summarizer_user_instructions()

    row_json = [
        {
            "role": "system",
            "content": sys_instr,
        },
        {
            "role": "user",
            "content": f"{usr_instr}\n\n{abstract}",
        },
    ]

    return row_json


def eval_summarizer_prompt_single_example(
    abstract: str,
    tokenizer: PreTrainedTokenizerFast,
):
    row_json = eval_summarizer_row_json_single_example(abstract)

    prompt = tokenizer.apply_chat_template(
        row_json,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    return prompt


def eval_summarizer_prompt_whole_dataset(
    abstracts: list,
    tokenizer: PreTrainedTokenizerFast,
):
    prompts: list[str] = []

    for abstract in abstracts:
        prompt = eval_summarizer_prompt_single_example(abstract, tokenizer)
        prompts.append(prompt)

    return {
        "text": prompts,
    }
