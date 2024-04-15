import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="GraduationProject",
    name='2024-04-05-22-03',
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    }
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = epoch
    loss = epoch
    f1 = epoch
    recall = epoch

    # log metrics to wandb
    wandb.log({
        "loss": loss,
        "acc": acc,
        "f1": f1,
        "recall": recall
    })

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()