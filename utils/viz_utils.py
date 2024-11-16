import matplotlib.pyplot as plt
from itertools import product
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import glob
import torch
import os

def visualize_predictions(images, masks, outputs, save_path, epoch, batch_idx):
    # Apply sigmoid to model outputs and threshold to get binary predictions
    outputs = torch.sigmoid(outputs)
    predictions = (outputs > 0.5).float()
    
    # Convert tensors to NumPy arrays for visualization
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    predictions = predictions.cpu().numpy()
    
    # Define the number of samples to display
    num_samples = min(4, images.shape[0])  # Show up to 4 samples from the batch
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))
    fig.suptitle(f"Epoch {epoch}, Batch {batch_idx}", fontsize=16)
    
    for i in range(num_samples):
        img = images[i][0]  # Extract the first channel for grayscale display
        mask = masks[i][0]
        pred = predictions[i][0]
        
        # Plot image, ground truth mask, and prediction side by side
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title("Input Image")
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title("Predicted Mask")

        # Turn off axis labels
        for ax in axes[i]:
            ax.axis('off')
    
    # Save the plot to the specified directory
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"predictions_epoch_{epoch}_batch_{batch_idx}.jpg"))
    plt.close()


def plot_train_val_history(train_loss_history, val_loss_history, plot_dir, args):
    # Create plot directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Save the plot
    filename = "loss_curve.jpg"
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()


def plot_metric(x, label, plot_dir, args, metric):
    # Create plot directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(x, label=label)
    plt.xlabel("Epochs")
    plt.ylabel(label)
    plt.title(f"{label} over Epochs")

    # Save the plot
    filename = f"{metric}_curve.jpg"
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()