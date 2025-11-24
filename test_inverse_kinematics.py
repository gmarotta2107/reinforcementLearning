import csv
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
from utils.nets import RegressionModel

class CustomDataset(Dataset):
    def __init__(self, file_path):
        """
        Custom dataset class for loading data from a CSV file.

        Args:
        file_path (str): Path to the CSV file containing the data.
        """
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            data = list(reader)

        self.targets = []
        self.inputs = []

        for line in data:
            values = list(map(float, line))
            self.targets.append(values[:11])  # First 11 columns are targets
            self.inputs.append(values[-3:])  # Last 3 columns are inputs

        self.targets = torch.tensor(self.targets, dtype=torch.float32)
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def test_model(model, dataloader_test):
    """
    Function to perform model testing and return predictions and real values.

    Args:
    model (nn.Module): The PyTorch model to be tested.
    dataloader_test (DataLoader): DataLoader for the test dataset.

    Returns:
    predictions (np.array): Array of predicted values.
    real_values (np.array): Array of real (target) values.
    """
    model.eval()
    predictions = []
    real_values = []

    with torch.no_grad():
        for inputs_test, targets_test in dataloader_test:
            outputs_test = model(inputs_test)
            predictions.extend(outputs_test.tolist())
            real_values.extend(targets_test.tolist())

    return torch.tensor(predictions), torch.tensor(real_values)


def main():
    """
    Main function to run the testing procedure.
    """
    # Path of the CSV file containing the test data
    file_path_test = 'datasets/All_positions_dataset_test.csv'

    # Create the test dataset
    dataset_test = CustomDataset(file_path_test)

    # DataLoader for the test dataset
    batch_size_test = 128
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False)

    # Initialize the model
    model = RegressionModel()

    # Load the trained model weights
    model.load_state_dict(torch.load('model0.pth'))

    # Perform model testing
    predictions, real_values = test_model(model, dataloader_test)

    # Compute Mean Squared Error (MSE) using PyTorch
    mse_loss = nn.MSELoss()
    mse = mse_loss(predictions, real_values).item()
    print(f"MSE: {mse:.4f}")


if __name__ == "__main__":
    main()
