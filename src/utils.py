import numpy as np
import pickle
import os
from tqdm import tqdm
import torch
import torchmetrics
from src.models import MessageClassifier
import matplotlib.pyplot as plt

def load_data(path: str):
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
    with open(os.path.join(dir_path, 'message_encodings.pkl'), 'rb') as f:
        message_encodings = pickle.load(f)
    with open(os.path.join(dir_path, 'labels.pkl'), 'rb') as f:
        labels = pickle.load(f)
    with open(os.path.join(dir_path, 'game_ids.pkl'), 'rb') as f:
        game_ids = pickle.load(f)
    return message_encodings, labels, game_ids



def evaluate(model, validation_loader, loss_fn, device="cpu"):
    model.eval()
    task="binary"
    accuracy = torchmetrics.Accuracy(num_classes=2, task=task)
    validation_loss = 0
    with torch.no_grad():
        for features, labels in validation_loader:
            output = model(features.to(device))
            accuracy(torch.argmax(output.to("cpu"), dim=1), torch.argmax(labels, dim=1).to("cpu"))
            validation_loss += loss_fn(output.to("cpu"), labels.to("cpu")).item()
    model.train()
    return accuracy.compute(), validation_loss/len(validation_loader)


def train(model, train_loader, optimizer, loss_fn, validation_loader, epochs=30, device="cpu", scheduler=None):
    model.train()
    model.to(device)
    train_acc = []
    train_losses = []
    validation_acc = []
    validation_losses = []
    
    for epoch in tqdm(range(epochs), desc="Epochs"):
        task="binary"
        accuracy = torchmetrics.Accuracy(num_classes=2,task=task)
        train_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            output = model(features.to(device))
            loss = loss_fn(output, labels.to(device))
            loss.backward()
            optimizer.step()
            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                scheduler.step(epoch + i / len(train_loader))
            accuracy(torch.argmax(output.to("cpu"), dim=1), torch.argmax(labels.to("cpu"), dim=1))
            train_loss += loss.item()
            
        val_accuracy, val_loss = evaluate(model, validation_loader, loss_fn, device)
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                pass
            else:
                scheduler.step()
        train_acc.append(accuracy.compute())
        train_losses.append(train_loss/len(train_loader))
        validation_acc.append(val_accuracy)
        validation_losses.append(val_loss)
    model.eval()
    model.to("cpu")
    return train_acc, validation_acc, train_losses, validation_losses


def plot_curves(*args, metric="Loss", legend=None):
    for curve in args:
        plt.plot(np.arange(len(curve)), curve)
    if legend:
        plt.legend(legend)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.title("Single message bot prediction")
    plt.show()

def plot_curves_on_ax(ax, *args, metric="Loss", title="Plot", legend=None):
    for curve in args:
        ax.plot(np.arange(len(curve)), curve)
    if legend:
        ax.legend(legend)
    ax.set_xlabel("Epochs")
    ax.set_ylabel(metric)
    ax.set_title(title)

def initialize_model(config):
    match config.model.type:
        case 'MessageClassifier':
            return MessageClassifier(config.model.input_size, config.model.hidden_sizes, config.model.output_size, config.model.dropout)
        case _:
            raise NotImplementedError(f"Model Type {config.model.type} is not supported")
        