import numpy as np
import pickle
import os
from tqdm import tqdm
import torch
import torchmetrics
from src.models import MessageClassifier
import matplotlib.pyplot as plt

def load_data(path: str):
    """
    Filters messages from database and collects labels etc.

    Args:
        path (str): path to the file scraped from the Turing Game database

    Returns:
        messages: list of messages sent in active games
        labels: labels for the messages (one hot encoded with [human, bot])
        game_ids: the game id of the game the message was sent in
    """

    with open(path, "rb") as f:
        chat_data = pickle.load(f)
    
    del chat_data[-1000]

    messages = []
    labels = []
    game_ids = []
    for game_id, game_data in chat_data.items():
        for message in game_data["messages"]:
            if message["userID"] == "GameMaster":
                if "won" in message["message"] or "surrendered" in message["message"] or "canceled" in message["message"] or "lost" in message["message"] or "timed out" in message["message"] or "disconnected" in message["message"]:
                    break
                else:
                    continue
            messages.append(message["message"])
            labels.append([int(not message["botID"]), message["botID"]])
            game_ids.append(message["gameID"])
        
    return messages, labels, game_ids


def load_pre_embedded(dir_path):
    """
    Loads embedded messages as well as labels and game_ids

    Args:
        dir_path (str): Directory including message_encodings.pkl, labels.pkl, game_ids.pkl

    Returns:
        message_encodings: List of message encodings for each message
        labels: labels for the messages (one hot encoded with [human, bot])
        game_ids: the game id of the game the message was sent in
    """
    with open(os.path.join(dir_path, 'message_encodings.pkl'), 'rb') as f:
        message_encodings = pickle.load(f)
    with open(os.path.join(dir_path, 'labels.pkl'), 'rb') as f:
        labels = pickle.load(f)
    with open(os.path.join(dir_path, 'game_ids.pkl'), 'rb') as f:
        game_ids = pickle.load(f)
    return message_encodings, labels, game_ids


def plot_curves(*args, metric="Loss", legend=None):
    """
    Plots the given number of curves

    Args:
        *args: Lists of values to be plotted as a lineplot
        metric (str): Used to label the Y axis. Defaults to "Loss"
        legend: Legend to be shown. Defaults to None

    Returns:
        None
    """
    for curve in args:
        plt.plot(np.arange(len(curve)), curve)
    if legend:
        plt.legend(legend)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.title("Single message bot prediction")
    plt.show()

def plot_curves_on_ax(ax, *args, metric="Loss", title="Plot", legend=None):
    """
    Plots the given number of curves on the given ax

    Args:
        *args: Lists of values to be plotted as a lineplot
        metric (str): Used to label the Y axis. Defaults to "Loss"
        title (str): Used as a title. Defaults to "Plot"
        legend: Legend to be shown. Defaults to None

    Returns:
        None
    """
    for curve in args:
        ax.plot(np.arange(len(curve)), curve)
    if legend:
        ax.legend(legend)
    ax.set_xlabel("Epochs")
    ax.set_ylabel(metric)
    ax.set_title(title)

def initialize_model(config):
    """
    Initializes the model

    Args:
        config: The config specified in the yaml file 

    Returns:
        model: The model corresponding to the config
    """
    match config.model.type:
        case 'MessageClassifier':
            return MessageClassifier(config.model.input_size, config.model.hidden_sizes, config.model.output_size, config.model.dropout)
        case _:
            raise NotImplementedError(f"Model Type {config.model.type} is not supported")
        