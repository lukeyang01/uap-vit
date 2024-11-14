"""
EECS 445 - Introduction to Machine Learning
Fall 2024  - Project 2
Train ViT
    Train a ViT to classify images
    Periodically output training information, and saves model checkpoints
    Usage: python train_vit.py
"""
import torch
import numpy as np
import random
from dataset import get_train_val_test_loaders
from model.vit import ViT
from train_common import *
from utils import config
import utils
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


if not os.path.exists(config("vit.checkpoint")):
    os.makedirs(config("vit.checkpoint"))

def main():
    """Train ViT and show training plots."""
    # Data loaders
    if check_for_augmented_data("./data"):
        tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
            task="target", batch_size=config("vit.batch_size"), augment=True
        )
        use_augment = True
    else:
        tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
            task="target",
            batch_size=config("vit.batch_size"),
        )
        use_augment = False

    # TODO: Define the ViT Model according to the appendix D
    model = ViT(num_blocks=2,
                   num_heads=2,
                   num_hidden=16,
                   num_patches=16)

    # TODO: define loss function, and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=10**-3)
    #
    num_params = count_parameters(model)
    print("Number of float-valued parameters:", num_params)

    # Attempts to restore the latest checkpoint if exists
    print("Loading ViT...")
    model, start_epoch, stats = restore_checkpoint(model, config("vit.checkpoint"))

    ##Debugging
    start_epoch = 0
    stats = []
    ####

    axes = utils.make_training_plot(name="ViT Training")

    # Evaluate the randomly initialized model
    evaluate_epoch(
        axes, tr_loader, va_loader, te_loader, model, criterion, start_epoch, stats
    )

    # initial val loss for early stopping
    prev_val_loss = stats[0][1]

    # TODO: define patience for early stopping
    patience = 5
    curr_patience = 0
    #

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes, tr_loader, va_loader, te_loader, model, criterion, epoch + 1, stats
        )

        save_checkpoint(model, epoch + 1, config("vit.checkpoint"), stats)

        # update early stopping parameters
        curr_patience, prev_val_loss = early_stopping(
            stats, curr_patience, prev_val_loss
        )

        epoch += 1
    print(f"Finished Training")

    # Save figure and keep plot open; for debugging
    utils.save_vit_training_plot(patience, use_augment)
    utils.hold_training_plot()


if __name__ == "__main__":
    main()
