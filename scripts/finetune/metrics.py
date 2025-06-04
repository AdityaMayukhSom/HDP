import evaluate
import numpy as np
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

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
