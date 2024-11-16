import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from model.unet import UNet
from utils.model_utils import train_arg_parser, set_seed
from utils.data_utils import MadisonStomach
from utils.viz_utils import visualize_predictions, plot_train_val_history, plot_metric
from utils.metric_utils import compute_dice_score


def train_model(model, train_loader, val_loader, optimizer, criterion, args, save_path):
    '''
    Trains the given model over multiple epochs, tracks training and validation losses, 
    and saves model checkpoints periodically.

    Args:
    - model (torch.nn.Module): The neural network model to be trained.
    - train_loader (DataLoader): DataLoader for the training dataset.
    - val_loader (DataLoader): DataLoader for the validation dataset.
    - optimizer (torch.optim.Optimizer): The optimizer used for updating model weights.
    - criterion (torch.nn.Module): The loss function used for training.
    - args (argparse.Namespace): Parsed arguments containing training configuration (e.g., epochs, batch size, device).
    - save_path (str): Directory path to save model checkpoints and training history.

    Functionality:
    - Creates directories to save results and checkpoints.
    - Calls `train_one_epoch` to train and validate the model for each epoch.
    - Saves model checkpoints every 5 epochs.
    - Plots the training and validation loss curves and the Dice coefficient curve.
    '''
    os.makedirs(os.path.join(save_path, args.exp_id), exist_ok=True)
    os.makedirs(os.path.join(save_path, args.exp_id, 'model'), exist_ok=True)

    train_loss_history = []
    val_loss_history = []
    dice_coef_history = []
    best_val_loss = float('inf')

    for epoch in range(args.epoch):
        train_one_epoch(model, 
                        train_loader, 
                        val_loader, 
                        train_loss_history, 
                        val_loss_history, 
                        dice_coef_history, 
                        optimizer, 
                        criterion, 

                        args, 
                        epoch, 
                        save_path)
        
        if val_loss_history[-1] < best_val_loss:
            best_val_loss = val_loss_history[-1]
            torch.save(model.state_dict(), os.path.join(save_path, args.exp_id, 'model', 'best_model.pt'))

    plot_train_val_history(train_loss_history, val_loss_history, save_path, args)
    plot_metric(dice_coef_history, label="dice coeff", plot_dir=save_path, args=args, metric='dice_coeff')

def train_one_epoch(model, train_loader, val_loader, train_loss_history, val_loss_history, 
                    dice_coef_history, optimizer, criterion, args, epoch, save_path):
    
    # Set model to training mode
    model.train()
    train_loss = 0.0

    for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")):
        images, masks = images.to(args.device), masks.to(args.device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate batch loss
        train_loss += loss.item()

        # Visualize predictions periodically, e.g., every 100 batches
        if (batch_idx + 1) % 100 == 0:
            with torch.no_grad():
                # Apply sigmoid to outputs for visualization
                visualize_predictions(images, masks, outputs, save_path, epoch, batch_idx)
    
    # Average training loss for the epoch
    avg_train_loss = train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    dice_score = 0.0

    with torch.no_grad():
        for val_images, val_masks in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}"):
            val_images, val_masks = val_images.to(args.device), val_masks.to(args.device)
            
            # Forward pass
            val_outputs = model(val_images)
            val_loss += criterion(val_outputs, val_masks).item()
            
            # Apply sigmoid to outputs for Dice calculation
            prob_val_outputs = torch.sigmoid(val_outputs)

            # binarized_outputs = (prob_val_outputs > 0.5).float() # this code
            
            # Calculate Dice coefficient for the batch
            dice_score += compute_dice_score(prob_val_outputs, val_masks)
        
        # Average validation loss and Dice coefficient for the epoch
        avg_val_loss = val_loss / len(val_loader)
        avg_dice_score = dice_score / len(val_loader)
        
        val_loss_history.append(avg_val_loss)
        dice_coef_history.append(avg_dice_score)
    
    print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Dice Coefficient = {avg_dice_score:.4f}")

if __name__ == '__main__':

    args = train_arg_parser()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = "D:\\dlsg projects\\homework2\\dlsg24_hw2\\results"
    set_seed(42)

    #Define dataset
    dataset = MadisonStomach(data_path="D:\\dlsg projects\\homework2\\madison-stomach\\madison-stomach", 
                            mode=args.mode)

    # Define train and val indices
    train_indices, val_indices = train_test_split(
        list(range(len(dataset))),
        test_size=0.2,
        random_state=42
    )
    
    # Define Subsets of to create trian and validation dataset
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # Define dataloader
    train_loader = DataLoader(train_subset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.bs, shuffle=False)

    model = UNet(in_channels=1, out_channels=1).to(args.device)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_model(model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                args=args,
                save_path=save_path)
