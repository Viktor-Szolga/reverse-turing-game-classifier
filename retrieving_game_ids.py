import torch
import numpy as np
import random
from src.utils import initialize_model
from omegaconf import OmegaConf
from src.trainer import Trainer
import os
import pickle

training_information = {}
print("cuda" if torch.cuda.is_available() else "cpu")

config_name = "run20.yaml"


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Load config and set device
default_config = OmegaConf.load(os.path.join("experiments", "default.yaml"))
specific_config = OmegaConf.load(os.path.join("experiments", config_name))
config = OmegaConf.merge(default_config, specific_config)
config.device = "cuda" if torch.cuda.is_available() else "cpu"
config.name = config_name
trainer = Trainer(config)

game_ids_val = []
game_ids_train = []

for index in trainer.human_val_idx:
    game_ids_val.append(trainer.game_ids[index])

for index in trainer.human_train_idx:
    game_ids_train.append(trainer.game_ids[index])

game_ids_val = set(game_ids_val)
game_ids_train = set(game_ids_train)

print(len(set(trainer.game_ids).difference(game_ids_train)))
game_ids_diff = set(trainer.game_ids).difference(game_ids_train)
with open("data/games_not_trained_on.pkl", "wb") as f:
    pickle.dump(game_ids_diff, f)

