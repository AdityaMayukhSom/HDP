from unsloth import FastLanguageModel  # isort: skip
import pathlib
import sys
from pathlib import Path

from loguru import logger
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

sys.path.append(str(Path(__file__).parent.parent))

from src.memory import empty_all_memory
from src.prompt import get_ner_system_instructions, get_ner_user_instructions

# nltk.download("maxent_ne_chunker_tab")
# nltk.download("averaged_perceptron_tagger_eng")
# nltk.download("punkt_tab")
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("maxent_ne_chunker")
# nltk.download("words")

# def generate_entities( rows: list[str]):
# rows should be one column from dataset_rows
# https://spacy.io/usage/processing-pipelines#processing

# docs_list = list(nlp.pipe(rows))


# ents = [
#     [
#         {
#             "ent": ent.text,
#             "start": ent.start_char,
#             "end": ent.end_char,
#             "type": ent.label_,
#             "lemma": ent.lemma_,
#         }
#         for ent in doc.ents
#     ]
#     for doc in docs_list
# ]

# print(ents)

# ents_str = [json.dumps(ent, indent=4, separators=(",", ":")) for ent in ents]
# return ents_str


def please_work(sentence: str):
    # sentence = Sentence(val, language_code="en")
    # tagger = Classifier.load("ner-large")

    # tagger.predict(sentence)

    # print(sentence.get_labels())

    # Download necessary resources

    # Sample sentence

    # Tokenize and POS tag
    # tokens = word_tokenize(sentence)
    # pos_tags = pos_tag(tokens)
    # print(pos_tags)

    # Perform NER
    # named_entities = ne_chunk(pos_tags)
    # print(named_entities)

    # for l in sentence:
    #     print(l)
    # print(sentence)

    _t = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    model: LlamaForCausalLM = _t[0]
    tokenizer: PreTrainedTokenizerFast = _t[1]

    FastLanguageModel.for_inference(model)

    inputs = tokenizer.apply_chat_template(
        [
            [
                {"role": "system", "content": get_ner_system_instructions()},
                {
                    "role": "user",
                    "content": f"{get_ner_user_instructions()}\n\n{sentence}",
                },
            ]
        ],
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    outputs = tokenizer.batch_decode(
        outputs[:, inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    )

    highlights = [
        highlight.split(tokenizer.eos_token)[0].strip().replace("\n", " ")
        for highlight in outputs
    ]

    print(highlights[0])


# def entities_from_batch(nlp:spacy.language.Language, dataset_rows):
#     source = dataset_rows["Abstract"]
#     target = dataset_rows["Highlight"]
#     hypothesis = dataset_rows["GeneratedHighlight"]


#     source_ents_str = generate_entities(nlp,source)
#     target_ents_str = generate_entities(nlp,target)
#     hypothesis_ents_str = generate_entities(nlp,hypothesis)

#     return {
#         "AbstractEntities": source_ents_str,
#         "HighlightEntities": target_ents_str,
#         "GeneratedHighlightEntities": hypothesis_ents_str,
#     }


# def generate_entities_dataset():
#     spacy.prefer_gpu()
#     nlp = spacy.load("en_core_web_trf")
#     nlp = en_core_web_trf.load()

#     logger.info("finetuned entities dataset generation started")

#     try:

#         def process_batched_rows(rows, idxs):
#             print_every = 100

#             if idxs[0] % print_every == 0 or (
#                 ((idxs[-1] // print_every) - (idxs[0] // print_every)) >= 1
#             ):
#                 print(f"Row {idxs[0]} to Row {idxs[-1]} starting...")

#             return generate_batched_entities(nlp, rows)

#         entites_ds = dataset.map(
#             function=process_batched_rows,
#             with_indices=True,
#             batched=True,
#             batch_size=1024,
#         )

#         del process_batched_rows

#         logger.success("finetuned entities dataset generation finished")
#         logger.info("started pushing finetuned entitites dataet to huggingface")
#         entites_ds.push_to_hub(entities_ds_hf_name)
#         logger.success("finetuned entitites dataset saved to huggingface as hf dataset")

#         del entites_ds
#     except Exception as e:
#         logger.exception(str(e))
#     finally:
#         del dataset
#         del nlp


def main():
    # empty_all_memory()
    # nlp = spacy.blank("en")
    # llm_ner = nlp.add_pipe("llm_ner")
    # llm_ner.add_label("PERSON")
    # llm_ner.add_label("LOCATION")
    # nlp.initialize()
    # doc = nlp("Jack and Jill rode up the hill in Les Deux Alpes")
    # print([(ent.text, ent.label_) for ent in doc.ents])
    # empty_all_memory()

    # return

    # spacy.prefer_gpu()
    # nlp = assemble("./scripts/config.cfg")
    # nlp = spacy.load("en_core_web_md")
    # ents = generate_entities(
    #     # nlp,
    #     [
    #         "Red LED lights significantly reduced probability of RLR at signalized intersections. Red LED lights could reduce cognitive load for judgement about stop go decisions. Flashing green increases risk of rear end collisions due to inconsistent stopping. Countdown VMS motivated drivers positioned in the stopping zone to cross red light. Red LED is recommended as an innovative and effective treatment for RLR prevention.",
    #         "With the advent of digital publishing and online databases, the volume of textual data generated by scientific research has increased exponentially. This makes it increasingly difficult for academics to keep up with new breakthroughs and synthesise important information for their own work. Abstracts have long been a standard feature of scientific papers, providing a concise summary of the paper's content and main findings. In recent years, some journals have begun to provide research highlights as an additional summary of the paper. The aim of this article is to create research highlights automatically by using various sections of a research paper as input. We employ a pointer-generator network with a coverage mechanism and pretrained ELMo contextual embeddings to generate the highlights. Our experiments shows that the proposed model outperforms several competitive models in the literature in terms of ROUGE, METEOR, BERTScore, and MoverScore metrics.",
    #     ],
    # )

    prompt = pathlib.Path("./data/ner-example.txt").read_text()
    please_work(prompt)

    # empty_all_memory()


try:
    main()
except Exception as e:
    logger.exception(e)
