import torch
import pandas as pd
import numpy as np
import json
from collections import Counter
from models.OneD_CNN_Unet import Model  # Replace with your model class
import torch.nn.functional as F


def load_model(model_path, device):
    """
    Load the trained model.
    """
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model


def process_csv(file_path, model, device, input_length, leads):
    """
    Process a single CSV file and return predictions for each segment.
    """
    data = pd.read_csv(file_path)
    predictions = {}

    for lead in leads:
        if lead not in data.columns:
            print(f"Lead '{lead}' not found in {file_path}. Skipping.")
            continue

        # Extract the signal
        input_signal = data[lead].values[:input_length]
        if len(input_signal) < input_length:
            input_signal = np.pad(input_signal, (0, input_length - len(input_signal)), mode='constant')

        # Prepare the input tensor
        input_tensor = torch.tensor(input_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Get predictions from the model
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]  # Take the first (and only) batch
            final_class = np.argmax(probabilities)

        predictions[lead] = {
            "probabilities": probabilities.tolist(),
            "final_class": int(final_class)
        }

    return predictions


def aggregate_predictions(predictions, method="majority"):
    """
    Aggregate predictions for all segments using the specified method.
    """
    if method == "majority":
        # Majority voting: Pick the class with the highest occurrence
        segment_classes = [np.argmax(segment) for segment in predictions]
        return Counter(segment_classes).most_common(1)[0][0]
    elif method == "weighted":
        # Weighted averaging: Sum probabilities and pick the highest class
        summed_probabilities = np.sum(predictions, axis=0)
        return int(np.argmax(summed_probabilities))
    else:
        raise ValueError("Invalid aggregation method. Choose 'majority' or 'weighted'.")


def test_single_file(file_path, model_path, device, input_length, leads):
    """
    Test the model on a single CSV file and aggregate results.
    """
    # Load the trained model
    model = load_model(model_path, device)

    # Process the file and get predictions
    predictions = process_csv(file_path, model, device, input_length, leads)

    # Print detailed predictions
    print(f"\nPredictions for file: {file_path}")
    for lead, lead_data in predictions.items():
        print(f"\nLead: {lead}")
        print(f"Probabilities for each class: {lead_data['probabilities']}")
        print(f"Predicted class for this segment: {lead_data['final_class']}")

    # Aggregate predictions for each lead
    print("\nAggregated Predictions:")
    aggregated_results = {}
    for lead, lead_data in predictions.items():
        probabilities = lead_data["probabilities"]
        aggregated_majority = aggregate_predictions([probabilities], method="majority")
        aggregated_weighted = aggregate_predictions([probabilities], method="weighted")
        aggregated_results[lead] = {
            "majority_vote": aggregated_majority,
            "weighted_average": aggregated_weighted
        }
        print(f"Lead {lead}: Majority Vote = {aggregated_majority}, Weighted Average = {aggregated_weighted}")

    # Ensure JSON-serializable types
    cleaned_aggregated_results = {
        lead: {k: int(v) for k, v in lead_data.items()}
        for lead, lead_data in aggregated_results.items()
    }

    return predictions, cleaned_aggregated_results


if __name__ == "__main__":
    # Configuration
    file_path = "./data/test_csv_signals/JS00823_signals.csv"  # Path to a single .csv file
    model_path = "./model_outputs/segmentation_2024_11_28-15_00/models/model_epoch_9_val_0.344238.pth"  # Path to the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_length = 7168  # Length of the input signal
    leads = ['I', 'II', 'V1']  # Specify the leads present in your .csv file

    # Test the single file
    predictions, aggregated_results = test_single_file(file_path, model_path, device, input_length, leads)

    # Save detailed predictions to JSON
    with open("single_file_predictions.json", "w") as f:
        json.dump({"file": file_path, "predictions": predictions}, f, indent=4)

    # Save aggregated results to JSON
    with open("single_file_aggregated_results.json", "w") as f:
        json.dump({"file": file_path, "aggregated_results": aggregated_results}, f, indent=4)

    print("\nDetailed predictions saved to 'single_file_predictions.json'")
    print("Aggregated results saved to 'single_file_aggregated_results.json'")
