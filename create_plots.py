"""
from src.utils import plot_curves, plot_curves_on_ax
import pickle
import os
import matplotlib.pyplot as plt


model_1_name = "currently_deployed_split_by_user"
model_2_name = "current_no_regularization_by_user"

with open(os.path.join("training_information", f"{model_1_name}.pkl"), "rb") as f:
    model1_dict = pickle.load(f)

with open(os.path.join("training_information", f"{model_2_name}.pkl"), "rb") as f:
    model2_dict = pickle.load(f)

m1_train_loss = model1_dict["train_loss"]
m1_train_acc = model1_dict["train_accuracy"]
m1_validation_loss = model1_dict["validation_loss"]
m1_validation_acc = model1_dict["validation_accuracy"]


m2_train_loss = model2_dict["train_loss"]
m2_train_acc = model2_dict["train_accuracy"]
m2_validation_loss = model2_dict["validation_loss"]
m2_validation_acc = model2_dict["validation_accuracy"]

#plot_curves(m1_train_loss, m2_train_loss, metric="Loss", legend=["Empty", "Specified"])

fig, axs = plt.subplots(2, 2, figsize=(6, 6))
plot_curves_on_ax(axs[0][0], m1_train_loss, m2_train_loss, metric="Loss", title="Train Loss", legend=["Reg", "Model 2"])
plot_curves_on_ax(axs[0][1], m1_train_acc, m2_train_acc, metric="Accuracy", title="Train Accuracy", legend=["Reg", "No reg"])
plot_curves_on_ax(axs[1][0], m1_validation_loss, m2_validation_loss, metric="Loss", title="Validation Loss", legend=["Reg", "No reg"])
plot_curves_on_ax(axs[1][1], m1_validation_acc, m2_validation_acc, metric="Accuracy", title="Validation Accuracy", legend=["Reg", "No reg"])
plt.tight_layout()
plt.show()
"""

from src.utils import plot_curves_on_ax
import pickle
import os
import matplotlib.pyplot as plt

# Specify the list of run names (must match .pkl filenames)

run_names = [
    "run12",
    "run04",
    "run20"
]

legends = [
    "Model A",
    "Model B",
    "Model C",
]



# Initialize lists to collect metrics
train_losses, train_accs = [], []
val_losses, val_accs = [], []

# Load metrics from each run
for run in run_names:
    with open(os.path.join("training_information", f"{run}.pkl"), "rb") as f:
        data = pickle.load(f)
        train_losses.append(data["train_loss"])
        train_accs.append(data["train_accuracy"])
        val_losses.append(data["validation_loss"])
        val_accs.append(data["validation_accuracy"])
        if run == "run20":
            print(data["validation_accuracy"][-1])

# Plot all curves on shared subplots
fig, axs = plt.subplots(2, 2, figsize=(8, 6))
fig.suptitle("Models A, B and C compared with both dropout and weight decay (User split)")
plot_curves_on_ax(axs[0][0], *train_losses, metric="Loss", title="Train Loss", legend=legends)
plot_curves_on_ax(axs[0][1], *train_accs, metric="Accuracy", title="Train Accuracy", legend=legends)
plot_curves_on_ax(axs[1][0], *val_losses, metric="Loss", title="Validation Loss", legend=legends)
plot_curves_on_ax(axs[1][1], *val_accs, metric="Accuracy", title="Validation Accuracy", legend=legends)

plt.tight_layout()
plt.show()
