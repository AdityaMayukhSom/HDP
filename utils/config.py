class Config:
    IS_DEBUG = True
    APP_PORT = 5000
    LOG_LEVEL = "debug"

    SUMMARIZER_INSTRUCTION = """
    You are instructed to generate a scientifically accurate highlight of the provided passage without additional
    sentences such as headings or introductions before or after the generated text as it will be used as summary
    in a custom dataset. The highlight should sound plausible and should not contain incorrect information. Generate
    3-5 concise highlight points from the provided research paper abstract, covering key contributions, methods and
    outcomes. Each point should contain 10 to 15 words only. Return the points in plain text format without bullets.

    No Additional Commentary: Exclude lines like "Here are 3-5 concise highlight points".
    """
