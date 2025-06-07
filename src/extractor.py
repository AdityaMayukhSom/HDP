import torch
import torch.nn.functional as F
from pydantic import BaseModel

from unsloth import FastLanguageModel, is_bfloat16_supported  # isort:skip
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import Config


class ExtractedFeatures(BaseModel):
    MTP: float
    AVGTP: float
    MDVTP: float
    MMDVP: float


class LLMModel:
    def __init__(self, model_name, model, tokenizer):
        self.__device = Config.DEVICE
        self.__model_name = model_name
        self.__model = model
        self.__tokenizer = tokenizer

    def __del__(self):
        self.__model.to("cpu")
        del self.__model
        del self.__tokenizer

    @property
    def name(self) -> str:
        return self.__model_name

    @property
    def sanitized_name(self) -> str:
        return self.__model_name.replace("/", "__")

    def generate(self, inpt):
        pass

    # Move in future commits this method to an utils.py
    def truncate_string_by_len(self, s, truncate_len):
        words = s.split()
        truncated_words = words[:-truncate_len] if truncate_len > 0 else words
        return " ".join(truncated_words)

    # Method to get the vocabulary probabilities of the LLM for a given token on the generated text from LLM-Generator
    def get_vocab_probs_at_pos(self, pos, token_probs):
        sorted_probs, sorted_indices = torch.sort(token_probs[pos, :], descending=True)
        return sorted_probs

    def get_max_length(self):
        return self.__model.config.max_position_embeddings

    def get_diff_vocab(self, vocab_probs, tprob):
        return (vocab_probs[0] - tprob).item()

    def get_diff_maximum_with_minimum(self, vocab_probs):
        return (vocab_probs[0] - vocab_probs[-1]).item()

    def extract_features(self, *, knowledge: str, document: str, generated_text: str):
        """
        By default knowledge is the empty string. If you want to add extra knowledge
        you can do it like in the cases of the qa_data.json and dialogue_data.json

        IMPORTANT: in case of summarization, pass empty string in knowledge

        TODO: document each of the function parameter
        """
        self.__model.eval()

        total_len = len(knowledge) + len(document) + len(generated_text)
        truncate_len = min(total_len - self.__tokenizer.model_max_length, 0)

        # Truncate knowledge in case is too large
        knowledge = self.truncate_string_by_len(knowledge, truncate_len // 2)
        # Truncate text_A in case is too large
        document = self.truncate_string_by_len(
            document, truncate_len - (truncate_len // 2)
        )

        inputs = self.__tokenizer(
            [knowledge + document + generated_text],
            return_tensors="pt",
            max_length=self.get_max_length(),
            truncation=True,
        )

        for key in inputs:
            inputs[key] = inputs[key].to(self.__device)

        with torch.no_grad():
            outputs = self.__model(**inputs)
            logits = outputs.logits

        probs = F.softmax(logits, dim=-1)
        probs = probs.to(self.__device)

        tokens_generated_length = len(self.__tokenizer.tokenize(generated_text))
        start_index = logits.shape[1] - tokens_generated_length
        conditional_probs = probs[0, start_index:]

        token_ids_generated = inputs["input_ids"][0, start_index:].tolist()
        token_probs_generated = [
            conditional_probs[i, tid].item()
            for i, tid in enumerate(token_ids_generated)
        ]

        tokens_generated = self.__tokenizer.convert_ids_to_tokens(token_ids_generated)

        minimum_token_prob = min(token_probs_generated)
        average_token_prob = sum(token_probs_generated) / len(token_probs_generated)

        maximum_diff_with_vocab = -1
        minimum_vocab_extreme_diff = 100000000000

        size = len(token_probs_generated)
        for pos in range(size):
            vocab_probs = self.get_vocab_probs_at_pos(pos, conditional_probs)
            maximum_diff_with_vocab = max(
                [
                    maximum_diff_with_vocab,
                    self.get_diff_vocab(vocab_probs, token_probs_generated[pos]),
                ]
            )
            minimum_vocab_extreme_diff = min(
                [
                    minimum_vocab_extreme_diff,
                    self.get_diff_maximum_with_minimum(vocab_probs),
                ]
            )

        return ExtractedFeatures(
            MTP=minimum_token_prob,
            AVGTP=average_token_prob,
            MDVTP=maximum_diff_with_vocab,
            MMDVP=minimum_vocab_extreme_diff,
        )


class LLama(LLMModel):
    def __init__(self):
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        super().__init__(model_name, model, tokenizer)

    def generate(self, inpt):
        inputs = self.tokenizer(
            [inpt], max_length=1024, return_tensors="pt", truncation=True
        )
        summary_ids = self.model.generate(inputs["input_ids"])
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


class UnslothLLaMA(LLMModel):
    def __init__(self):
        model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
            use_cache=True,
            device_map=Config.DEVICE,
            low_cpu_mem_usage=True,
        )

        super().__init__(model_name, model, tokenizer)

    def generate(self, inpt):
        # put print statements in the LLaMA model generate function, to compare if LLaMA and Unsloth
        # LLaMA are giving interoperable inputs and outputs. tokenized input's won't make much sense
        # so rather print the summary on both the models for some given input, and compare if those
        # are semantically equivalent.

        model_inputs = self.tokenizer(
            inpt, return_tensors="pt", padding=True, truncation=True
        ).to(Config.DEVICE)
        model_outputs = self.model.generate(
            **model_inputs, max_new_tokens=2048, num_return_sequences=1
        )

        # TODO: is decoding model_output and taking decoded_text[0] same as decoding model_output[0]?
        # if yes, that can save a lot of compute, so need to test that once, easiest way to test is to
        # print decoded_text[0] and decode(model_outputs[0]) and check if those are equal.

        # OPTION - 01
        # decoded_text = self.tokenizer.batch_decode(model_outputs, skip_special_tokens=True)
        # summary = decoded_text[0]

        # OPTION - 02
        summary = self.tokenizer.decode(model_outputs[0], skip_special_token=True)

        return summary


def main():
    model = UnslothLLaMA()


if __name__ == "__main__":
    main()
