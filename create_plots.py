from src.utils import plot_curves_on_ax
import pickle
import os
import matplotlib.pyplot as plt

# Specify the list of run names (include leading 0 for 0-9)
run_names = [
    "run12",
    "run04",
    "run20"
]

# Specify legend names
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
