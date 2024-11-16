import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from model.unet import UNet
from utils.model_utils import test_arg_parser, set_seed
from utils.data_utils import MadisonStomach
from utils.viz_utils import visualize_predictions, plot_train_val_history, plot_metric
from utils.metric_utils import compute_dice_score

def test_model(model, args, save_path):
    '''
    Tests the model on the test dataset and computes the average Dice score.
    
    Args:
    - model (torch.nn.Module): The PyTorch model to test.
    - args (argparse.Namespace): Parsed arguments for device, batch size, etc.
    - save_path (str): Directory where results (e.g., metrics plot) will be saved.
    
    Functionality:
    - Sets the model to evaluation mode and iterates over the test dataset.
    - Computes the Dice score for each batch and calculates the average.
    - Saves a plot of the Dice coefficient history.
    '''
    model.eval()
    model.to(args.device)
    
    dice_scores = []

    # Iterate over the test DataLoader
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_dataloader, desc="Testing")):
            images, masks = images.to(args.device), masks.to(args.device)
            
            # Forward pass
            outputs = model(images)

            # Apply sigmoid to the model outputs and threshold to get binary predictions
            prob_outputs = torch.sigmoid(outputs)
            binary_outputs = (prob_outputs > 0.5).float()
            
            # Calculate the Dice score for the current batch
            dice_score = compute_dice_score(binary_outputs, masks)
            dice_scores.append(dice_score)
            
            # Visualize some predictions every few batches
            if (batch_idx + 1) % 10 == 0:
                visualize_predictions(images, masks, outputs, save_path, epoch=None, batch_idx=batch_idx)
    
    # Calculate the average Dice score over all batches
    avg_dice_score = sum(dice_scores) / len(dice_scores)
    print(f"Average Dice Score on Test Set: {avg_dice_score:.4f}")

if __name__ == '__main__':

    args = test_arg_parser()
    save_path = "D:\\dlsg projects\\homework2\\dlsg24_hw2\\results"
    set_seed(42)

    #Define dataset
    dataset = MadisonStomach(data_path="D:\\dlsg projects\\homework2\\madison-stomach\\madison-stomach", 
                            mode="test")

    test_dataloader = DataLoader(dataset, batch_size=args.bs)

    # Define and load the model
    model = UNet(in_channels=1, out_channels=1)
    file_name = "best_model.pt"
    model_path = os.path.join("D:\\dlsg projects\\homework2\\dlsg24_hw2\\results\\exp\\0\\model", file_name)
    model = torch.load(model_path, map_location=args.device)

    test_model(model=model,
                args=args,
                save_path=save_path)