import torch
import numpy as np
import random
from src.utils import initialize_model
from omegaconf import OmegaConf
from src.trainer import Trainer
import os
import pickle

# Specify the run config file
config_name = "run20.yaml"

# Set seed to ensure the same random processes as in the main file
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Load config and set device
default_config = OmegaConf.load(os.path.join("experiments", "default.yaml"))
specific_config = OmegaConf.load(os.path.join("experiments", config_name))
config = OmegaConf.merge(default_config, specific_config)
config.device = "cuda" if torch.cuda.is_available() else "cpu"
config.name = config_name

# Load a trainer to get the train, eval split
trainer = Trainer(config)

game_ids_val = []
game_ids_train = []

# Extract the game_ids for each message in the train/eval set
for index in trainer.human_val_idx:
    game_ids_val.append(trainer.game_ids[index])

for index in trainer.human_train_idx:
    game_ids_train.append(trainer.game_ids[index])

# Remove multiple mentions of a game id
game_ids_val = set(game_ids_val)
game_ids_train = set(game_ids_train)

# Calculate difference and print number of games only in val
game_ids_diff = set(trainer.game_ids).difference(game_ids_train)
print(len(game_ids_diff))

# Save indices
with open("data/games_not_trained_on.pkl", "wb") as f:
    pickle.dump(game_ids_diff, f)

