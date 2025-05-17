import torch
from unsloth import FastLanguageModel, is_bfloat16_supported

from .config import Config


def get_summarizer_model_tokenizer():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="AdityaMayukhSom/Llama-3.2-1B-Instruct-bnb-4bit-MixSub",
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
    )

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_highlight(abstract: str, model, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    init_identifier = "<|start_header_id|>assistant<|end_header_id|>"
    term_identifier = "<|eot_id|>"

    batch_data = [
        [
            {
                "role": "system",
                "content": Config.SUMMARIZER_INSTRUCTION,
            },
            {
                "role": "user",
                "content": abstract,
            },
        ]
    ]

    inputs = tokenizer.apply_chat_template(
        batch_data,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded_texts = tokenizer.batch_decode(
        outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=False
    )

    # the generaed output contains the prompt, init identifier, generated highlight and term identifier
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
