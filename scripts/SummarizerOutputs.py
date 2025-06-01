import os
import sys
import time
from pathlib import Path

import torch
from loguru import logger
from unsloth import FastLanguageModel, is_bfloat16_supported

sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from src.prompt import eval_summarizer_row_json_single_example

os.environ["TZ"] = "Asia/Kolkata"
time.tzset()


def generate_eval_highlights(dataset_row, model, tokenizer):
    init_identifier = "<|start_header_id|>assistant<|end_header_id|>"
    term_identifier = "<|eot_id|>"

    abstracts = dataset_row["ArticleAbstract"]

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
        max_new_tokens=128,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded_texts = tokenizer.batch_decode(
        outputs[:, inputs.input_ids.shape[1] :],
        skip_special_tokens=False,
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

    return {
        "GeneratedHighlight": output,
    }


def get_inference_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="AdityaMayukhSom/Llama-3.2-1B-HyperMixSub-Full",
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
    )

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_finetuned_model_highlights_dataset():
    # ori_dataset_dir = "/content/drive/MyDrive/Datasets/MixSub/Original"
    # dup_dataset_dir = "/content/drive/MyDrive/Datasets/MixSub/Duplicate"
    #
    # out_ds_hf_name = "AdityaMayukhSom/MixSub-LLaMA-3.2-FineTuned-Outputs-Large"
    # out_dataset_dir = "/content/drive/MyDrive/Datasets/MixSub/Outputs"
    # out_csv_file_name = "llama-3_2-1B-outputs-mixsub.csv"

    # dd = extract_hyper_mix_sub_from_db()
    model, tokenizer = get_inference_model()

    # Map the existing dataset rows into hallucinated highlights, the returned
    # dictionary from the function passed to map, will automatically be appended
    # to the dataset on which the map function is being called, and a new dataset
    # will be returned, so note that mapping does not modify the dataset on which
    # it is being called on.
    # logger.info("finetuned model dataset generation started")

    # out_ds = dd.map(
    #     lambda row: generate_eval_highlights(
    #         row,
    #         model,
    #         tokenizer,
    #     ),
    #     batched=True,
    #     batch_size=16,
    # )

    h = generate_eval_highlights(
        {
            "ArticleAbstract": [
                """In heterogeneous catalysis, atomic layer deposition (ALD) has been developed as a tool to stabilize and reduce carbon deposition on supported nanoparticles. Here, we discuss use of high vacuum ALD to deposit alumina films on size-selected, sub-nanometer Pt/SiO2 model catalysts. Mass-selected Pt24 clusters were deposited on oxidized Si(100), to form model Pt24/SiO2 catalysts with particles shown to be just under 1 nm, with multilayer three dimensional structure. Alternating exposures to trimethyl-aluminum and water vapor in an ultra-high vacuum chamber were used to grow alumina on the samples without exposing them to air. The samples were probed in situ using X-ray photoelectron spectroscopy (XPS), low-energy ion scattering spectroscopy (ISS), and CO temperature-programmed desorption (TPD). Additional samples were prepared for ex situ experiments using grazing incidence small angle x-ray scattering spectroscopy (GISAXS). Alumina growth is found to initiate at least 60 times more efficiently at the Pt24 cluster sites, compared to bare SiO2/Si, with a single ALD cycle depositing a full alumina layer on top of the clusters, with substantial additional alumina growth initiating on SiO2 sites surrounding the clusters. As a result, the clusters were completely passivated, with no exposed Pt binding sites. Selective nucleation of alumina ALD on sub-nano Pt clusters.  One cycle deposits one monolayer on top, with additional alumina around the cluster periphery.  A single ALD cycle completely blocks CO binding sites on Pt clusters.  Not possible to recover high binding energy Pt sites by heating."""
            ],
        },
        model,
        tokenizer,
    )
    print(h)
    # logger.success("fine tuned model dataset generation finished")

    # if not os.path.exists(out_dataset_dir):
    # Path(out_dataset_dir).mkdir(parents=True, exist_ok=True)
    # out_ds_path = os.path.join(out_dataset_dir, out_csv_file_name)

    # logger.info("started saving outlucinated dataset")
    # out_ds.to_csv(out_ds_path, index=False)
    # logger.success("finetuned model dataset saved to google drive as csv")

    logger.info("started pushing finetuned model dataset to huggingface")
    # out_ds.push_to_hub(out_ds_hf_name)
    logger.success("finetuned model dataset saved to huggingface as hf dataset")


if __name__ == "__main__":
    # print_mem_stats()
    # libc = ctypes.CDLL("libc.so.6")  # clearing cache
    # libc.malloc_trim(0)
    # print_mem_stats()
    # generate_small_dataset()
    generate_finetuned_model_highlights_dataset()
