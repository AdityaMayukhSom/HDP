import json

from dotenv import dotenv_values
from google import genai

config = dotenv_values()
# client = genai.Client(api_key=config["GEMINI_API_KEY"])


def generate_hallucinated_highlights(
    instruction: str,
    abstract: str,
    highlight: str,
) -> dict[str, str]:
    dat = {"Document": abstract, "CorrectSummary": highlight}

    mes = f"{instruction}\n\n{json.dumps(dat, indent=4)}".strip()

    # res = client.models.generate_content(
    #     model="gemini-2.0-flash",
    #     contents=mes,
    # )

    return {
        # "Hallucination": res.text,
        "Message": mes,
    }


if __name__ == "__main__":
    config = dotenv_values(dotenv_path=".env", verbose=True, encoding="utf-8")

    with (
        open("./scripts/instruction/gemini.txt", "r", encoding="utf-8") as ins_file,
        open("./scripts/example/abstract.txt", "r", encoding="utf-8") as abs_file,
        open("./scripts/example/highlight.txt", "r", encoding="utf-8") as hlt_file,
    ):
        hal = generate_hallucinated_highlights(
            ins_file.read(),
            abs_file.read(),
            hlt_file.read(),
        )

    with open("./scripts/example/message.txt", "w", encoding="utf-8") as mes_file:
        mes_file.write(hal["Message"])

    # with open("./scripts/instruction/oneturn.json", "r", encoding="utf-8") as f:
    #     instruction = json.load(f)
