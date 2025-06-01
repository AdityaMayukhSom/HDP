import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast  # isort:skip
from unsloth import FastLanguageModel, is_bfloat16_supported

from src.config import Config
from src.prompt import eval_summarizer_row_json_single_example


def get_summarizer_model_tokenizer():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="AdityaMayukhSom/LLaMA-3.2-1B-Instruct-MixSubV1",
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        use_cache=True,
        device_map=Config.DEVICE,
        low_cpu_mem_usage=True,
    )

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_highlight(
    abstract: str,
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
):
    init_identifier = "<|start_header_id|>assistant<|end_header_id|>"
    term_identifier = "<|eot_id|>"

    batch_data = [eval_summarizer_row_json_single_example(abstract)]

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
        max_new_tokens=128,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded_texts = tokenizer.batch_decode(
        outputs[:, inputs.input_ids.shape[1] :],
        skip_special_tokens=False,
    )

    # the generated output contains the prompt, init identifier, generated highlight and term identifier
    # the following code splits the output with init identifier, takes the second section which contains
    # the generated highlight followed by term identifier, now splits the second section based on term
    # identifier, takes the first section, which contains only the generated output. Then it strips the
    # generated content to get rid of any white spaces from the beginning and the end, and replaces
    # newline character with whitespace.

    # print(decoded_texts)
    output = [
        decoded_text.split(term_identifier)[0].strip().replace("\n", " ")
        for decoded_text in decoded_texts
    ]

    del batch_data
    del inputs
    del outputs
    del init_identifier
    del term_identifier

    return output[0]
