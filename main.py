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

for config_name in os.listdir("experiments"):
    if config_name == "default.yaml":
        continue
    # Reset seed to ensure reproducability for single config
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
    # Initialize model
    model = initialize_model(config)

    # Train Model
    model, train_acc, validation_acc, train_loss, validation_loss = trainer.train(model)

    # Store training information
    results = {
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'validation_loss': validation_loss,
        'validation_accuracy': validation_acc
    }
    training_information[config_name[:-5]] = results
    
    # Save model
    torch.save(model.state_dict(), os.path.join("trained_models", f"{config_name[:-5]}.pth"))

    # Save training data
    with open(os.path.join("training_information", f"{config_name[:-5]}.pkl"), "wb") as f:
        pickle.dump(results, f)
with open(os.path.join("training_information", "all_runs_dict.pkl"), "wb") as f:
    pickle.dump(training_information, f)