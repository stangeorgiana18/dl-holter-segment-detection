import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataset.dataloader_semseg import ToTensor
from models.OneD_CNN_Unet import Model  # Replace with your model class
import torch.nn.functional as F
import json


def load_model(model_path, device):
    """
    Load the trained model.
    """
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model


def process_csv(file_path, model, device, input_length, leads):
    """
    Process a single CSV file and return predictions.
    """
    # Read the CSV file
    data = pd.read_csv(file_path)
    predictions = {}

    for lead in leads:
        # Check if the lead exists in the file
        if lead not in data.columns:
            print(f"Lead '{lead}' not found in {file_path}. Skipping.")
            continue

        # Extract the signal
        input_signal = data[lead].values[:input_length]  # Truncate or pad to `input_length`
        if len(input_signal) < input_length:
            # Pad the signal if it's shorter than `input_length`
            input_signal = np.pad(input_signal, (0, input_length - len(input_signal)), mode='constant')

        # Prepare the input tensor
        input_tensor = torch.tensor(input_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Get predictions from the model
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]  # Extract the first (and only) batch
            final_class = np.argmax(probabilities)  # Get the class with the highest probability

        # Store the predictions
        predictions[lead] = {
            "probabilities": probabilities.tolist(),  # Convert numpy array to list
            "final_class": int(final_class)  # Convert numpy int to regular int
        }

    return predictions

 


def test_on_folder(folder_path, model_path, device, input_length, leads, output_file):
    """
    Test the model on all CSV files in a folder.
    """
    # Load the trained model
    model = load_model(model_path, device)

    # Prepare an output list
    results = []

    # Process each CSV file in the folder
    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_path}")

            # Get predictions for the file
            predictions = process_csv(file_path, model, device, input_length, leads)

            # Append results
            results.append({
                "file": file_name,
                "predictions": predictions
            })

    # Save results to a properly formatted JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Configuration
    folder_path = "./data/test_csv_signals"  # Path to the folder with .csv files
    model_path = "./model_outputs/segmentation_2024_11_28-15_00/models/model_epoch_9_val_0.344238.pth"  # Path to the trained model
    output_file = "./test_results.json"  # Output file for predictions

    # Ensure input_length matches the value used during training (e.g., 7168)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_length = 7168  # Length of the input signal
    leads = ['I', 'II', 'V1']  # Specify the leads present in your .csv files

    # Run the testing
    test_on_folder(folder_path, model_path, device, input_length, leads, output_file)
