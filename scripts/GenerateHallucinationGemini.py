from typing import Literal

from dotenv import dotenv_values
from google import genai

config = dotenv_values()
client = genai.Client(api_key=config["GEMINI_API_KEY"])


def generate_hallucinated_highlights(row) -> dict[Literal["Hallucination"], str]:
    abstract: str = row["Abstract"]
    highlight: str = row["Highlight"]

    mes = f"""
    You're instructed to generate scientifically inaccurate highlights for a document without additional sentences like headings, introductions, or texts before or after the generated output. You must add both factual and non factual hallucinations in the output, change values, change names, make the highlight look like accurate while injecting hallucination. The output will be directly used as datapoint in a custom dataset. The highlight should sound plausible but contain incorrect information. Generate 3-6 concise highlight points from the provided research paper abstract, covering key contributions, methods, and outcomes. Each point should contain 15 to 20 words only. Return the points in plain text format without bullets. Also add enough number of factual hallucinations in the generated summary i.e. contents that can be deduced from the abstract or is general knowledge or common sense.

    {{
        "Abstract": "{abstract}", 
        "Highlight": "{highlight}"
    }},    
    """

    response = client.models.generate_content(model="gemini-2.0-flash", contents=mes)

    return {
        "Hallucination": response.text,
    }


if __name__ == "__main__":
    config = dotenv_values(".env")
    l = generate_hallucinated_highlights(
        {
            "Abstract": """
The enzyme phytases facilitates in degradation of phytate . Phytate as a natural compound serving as primary source for storing phosphate among plants . From the biotechnological prospects there has been a considerable leap in the Enzyme technology which has massively broadened the commercial aspects of phytase . Their impact in the food and feed industry has become much more quintessential in the recent times . For nearly two decades there has been a wide array of commercially available microbial phytases in market with commercial significance as it facilitates the farmers with essential . Phytases in particular can not be neglected from being a threat for human diet due to its anti nutrient activity as they served as strong chelating agent against many divalent minerals . Similar to phytases activity PA also was found to showcase a potential towards binding positively charged proteins amino acids and or multivalent cations or minerals in foods . Besides the food industry has overlooked on the very fact of phytase significance as its supplementation results in improving the net availability of the essential trace elements and minerals to humans . Similarly they serve as an essential feed source for mongastric animals.
""",
            "Highlight": """
Use of phytase enzyme in removing the environmental concern and marine eutrophication. Advantages of exploring the properties of phytic acid. Commercial aspects for the production and use of microbial phytase in food and feed industries. Biotechnological approaches for production of ideal phytase. Sources of phytic acid and phytase.
""",
        }
    )

    print(l)

    # with open("./scripts/instruction/oneturn.json", "r", encoding="utf-8") as f:
    #     instruction = json.load(f)
