import pathlib
import sys

import evaluate

sys.path.append(str(pathlib.Path(__file__).parent.parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from moverscore.moverscore_v2 import get_idf_dict, word_mover_score

rouge = evaluate.load("rouge")  # []
bleu = evaluate.load("bleu")  # []
bleurt = evaluate.load("bleurt")  # []
# gleu = evaluate.load("gleu") # []
meteor = evaluate.load("meteor")  # []
bert = evaluate.load("bertscore")  # []
sari = evaluate.load("sari")
mauve = evaluate.load("mauve")


def calculate(real: str, pred: str):
    pass


def main():
    pass


if __name__ == "__main__":
    main()
