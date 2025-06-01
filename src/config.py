import torch


class Config:
    IS_DEBUG = True
    APP_PORT = 5000
    LOG_LEVEL = "debug"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
