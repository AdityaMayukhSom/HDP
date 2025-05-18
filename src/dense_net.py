import pickle

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.config import Config


class SimpleDenseNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim=1, dropout_prob=0.4):
        super(SimpleDenseNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        # x = torch.relu(x)
        # x = self.fc4(x)
        return torch.sigmoid(x)


# otherwise model.load_state_dict won't work
# https://github.com/nicofdga/DZ-FaceDetailer/issues/30
# torch.serialization.add_safe_globals([SimpleDenseNet])


def get_clf_and_std_scaler():
    with open("./data/extractor/standard-scaler.pkl", "rb") as std_file:
        std: StandardScaler = pickle.load(std_file)

    clf = SimpleDenseNet(input_dim=4, hidden_dim=2048)
    checkpoint = torch.load(
        "./data/extractor/dense-model-trained.pt",
        weights_only=True,
        map_location=Config.DEVICE,
        strict=False,
    )
    clf.load_state_dict(checkpoint)
    clf.to(Config.DEVICE)
    clf.eval()

    return clf, std
