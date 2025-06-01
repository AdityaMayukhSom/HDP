import os
from functools import lru_cache
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerFast


@lru_cache
def get_summarizer_system_instructions(
    instr_path="./instructions/summarizer-system.txt",
) -> str:
    data = Path(instr_path).read_text()
    return data


@lru_cache
def get_summarizer_user_instructions(
    instr_path="./instructions/summarizer-user.txt",
) -> str:
    data = Path(instr_path).read_text()
    return data


def create_summarizer_prompt_for_fine_tuning(
    examples: Dataset,
    *,
    tokenizer: PreTrainedTokenizerFast,
):
    abstracts = examples["ArticleAbstract"]
    highlights = examples["CorrectHighlight"]

    sys_instr = get_summarizer_system_instructions()
    usr_instr = get_summarizer_user_instructions()

    prompts: list[str] = []

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
                "content": highlight + tokenizer.eos_token,
            },
        ]

        prompt = tokenizer.apply_chat_template(
            row_json,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompts.append(prompt)

    return {
        "Prompt": prompts,
    }


def prepare_hyper_mix_sub_prompts(
    hms_dd: DatasetDict,
    tokenizer: PreTrainedTokenizerFast,
) -> DatasetDict:
    dd = hms_dd.map(
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
    )

    return prompt


def eval_summarizer_prompt_whole_dataset(
    abstracts: list,
    tokenizer: PreTrainedTokenizerFast,
):
    prompts: list[str] = []

    for abstract in abstracts:
        prompt = eval_summarizer_prompt_single_example(
            abstract,
            tokenizer,
        )
        prompts.append(prompt)

    return {
        "Prompt": prompts,
    }
