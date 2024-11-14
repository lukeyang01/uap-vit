"""
EECS 445 - Introduction to Machine Learning
Fall 2021  - Project 2
Utility functions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def config(attr):
    """
    Retrieves the queried attribute value from the config file. Loads the
    config file on first call.
    """
    if not hasattr(config, "config"):
        with open("config.json") as f:
            config.config = eval(f.read())
    node = config.config
    for part in attr.split("."):
        node = node[part]
    return node


def denormalize_image(image):
    """ Rescale the image's color space from (min, max) to (0, 1) """
    ptp = np.max(image, axis=(0, 1)) - np.min(image, axis=(0, 1))
    return (image - np.min(image, axis=(0, 1))) / ptp


def hold_training_plot():
    """
    Keep the program alive to display the training plot
    """
    plt.ioff()
    plt.show()


def log_training(epoch, stats):
    """Print the train, validation, test accuracy/loss/auroc.

    Each epoch in `stats` should have order
        [val_acc, val_loss, val_auc, train_acc, ...]
    Test accuracy is optional and will only be logged if stats is length 9.
    """
    include_train = len(stats[-1]) / 3 == 3
    splits = ["Validation", "Train", "Test"]
    metrics = ["Accuracy", "Loss", "AUROC"]
    print("Epoch {}".format(epoch))
    for j, split in enumerate(splits):
        for i, metric in enumerate(metrics):
            idx = len(metrics) * j + i
            if idx >= len(stats[-1]):
                continue
            print(f"\t{split} {metric}:{round(stats[-1][idx],4)}")


def make_training_plot(name="CNN Training"):
    """Set up an interactive matplotlib graph to log metrics during training."""
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    plt.suptitle(name)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUROC")

    return axes


def update_training_plot(axes, epoch, stats):
    """Update the training plot with a new data point for loss and accuracy."""
    splits = ["Validation", "Train", "Test"]
    metrics = ["Accuracy", "Loss", "AUROC"]
    colors = ["r", "b", "g"]
    for i, metric in enumerate(metrics):
        for j, split in enumerate(splits):
            idx = len(metrics) * j + i
            if idx >= len(stats[-1]):
                continue
            # __import__('pdb').set_trace()
            axes[i].plot(
                range(epoch - len(stats) + 1, epoch + 1),
                [stat[idx] for stat in stats],
                linestyle="--",
                marker="o",
                color=colors[j],
            )
        axes[i].legend(splits[: int(len(stats[-1]) / len(metrics))])
    plt.pause(0.00001)


def save_cnn_training_plot(patience,use_augment):
    """Save the training plot to a file."""
    if use_augment:
        plt.savefig(f"cnn_training_plot_patience={patience}_augmented.png", dpi=200)
    else:
        plt.savefig(f"cnn_training_plot_patience={patience}.png", dpi=200)


def save_vit_training_plot(patience,use_augment):
    """Save the training plot to a file."""
    # if save_settings_debug:
    #     if not os.path.exists("vit_train_plots"):
    #         os.makedirs("vit_train_plots",exist_ok=True)
    #     plt.savefig(f"vit_train_plots/vit_training_plot_patience={patience}_blocks={info['num_blocks']}_heads={info['num_heads']}_patches={info['num_patches']}_hidden={info['num_hidden']}_augment={use_augment}.png", dpi=200)
    if use_augment:
        plt.savefig(f"vit_training_plot_patience={patience}_augmented.png", dpi=200)
    else:
        plt.savefig(f"vit_training_plot_patience={patience}.png", dpi=200)

#########################
# Remove in final release
#########################
# def save_source_vit_training_plot(patience,info):
#     """Save the source ViT training plot to a file."""
#     if not os.path.exists("vit_source_train_plots"):
#         os.makedirs("vit_source_train_plots",exist_ok=True)
#     plt.savefig(f"vit_source_train_plots/vit_training_plot_patience={patience}_blocks={info['num_blocks']}_heads={info['num_heads']}_patches={info['num_patches']}_hidden={info['num_hidden']}_augment=False.png", dpi=200)



def save_tl_training_plot(num_layers):
    """Save the transfer learning training plot to a file."""
    if num_layers == 0:
        plt.savefig("TL_0_layers.png", dpi=200)
    elif num_layers == 1:
        plt.savefig("TL_1_layers.png", dpi=200)
    elif num_layers == 2:
        plt.savefig("TL_2_layers.png", dpi=200)
    elif num_layers == 3:
        plt.savefig("TL_3_layers.png", dpi=200)


def save_source_training_plot(patience):
    """Save the source learning training plot to a file."""
    plt.savefig(f"source_training_plot_patience={patience}.png", dpi=200)

def save_challenge_training_plot():
    """Save the challenge learning training plot to a file."""
    plt.savefig("challenge_training_plot.png", dpi=200)

def save_parameter_count(info,num_params):
    parent_save_folder = "param_counts"
    if not os.path.exists(parent_save_folder):
        os.makedirs(parent_save_folder,exist_ok=True)
    df_file_name = info['model_type'] + ".csv"
    if os.path.exists(os.path.join(parent_save_folder,df_file_name)):
        df = pd.read_csv(os.path.join(parent_save_folder,df_file_name))
    else:
        col_names = list(info.keys()) + ["param_count"]
        df = pd.DataFrame(columns=col_names)
    new_col = list(info.values()) + [num_params]
    df.loc[len(df)] = new_col
    df.drop_duplicates(inplace=True)
    df.to_csv(os.path.join(parent_save_folder,df_file_name),index=False)