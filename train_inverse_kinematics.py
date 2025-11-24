import torch
import torch.nn as nn  # Importing the neural network module
import torch.optim as optim  # Importing the optimization module
from torch.utils.data import Dataset, DataLoader, random_split  # Importing utilities for data handling
import csv  # Importing CSV module for reading data
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

# Define the custom dataset
class CustomDataset(Dataset):
    def __init__(self, file_path):
        # Open and read the CSV file
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            data = list(reader)

        self.targets = []
        self.inputs = []

        # Parse the data into inputs and targets
        for line in data:
            values = list(map(float, line))
            self.targets.append(values[:11])  # First 11 columns are the targets
            self.inputs.append(values[-3:])  # Last 3 columns are the inputs

        # Convert lists to PyTorch tensors
        self.targets = torch.tensor(self.targets, dtype=torch.float32)
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)  # Return the length of the dataset

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]  # Return input-target pair

# Define the regression model
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(3, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 128)  # Second fully connected layer
        self.fc3 = nn.Linear(128, 11)  # Third fully connected layer (output layer)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation after first layer
        x = torch.relu(self.fc2(x))  # Apply ReLU activation after second layer
        x = self.fc3(x)  # Output layer
        return x

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100, patience=10):
    global train_losses, val_losses
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        epoch_train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Clear gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            epoch_train_loss += loss.item()  # Accumulate training loss

        epoch_train_loss /= len(train_loader)  # Average training loss
        train_losses.append(epoch_train_loss)  # Append training loss

        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient calculation
            for inputs, targets in val_loader:
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, targets)  # Compute loss
                val_loss += loss.item()  # Accumulate validation loss

        val_loss /= len(val_loader)  # Average validation loss
        val_losses.append(val_loss)  # Append validation loss

        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Early stopping mechanism
        if val_loss < best_loss:
            best_loss = val_loss  # Update best loss
            patience_counter = 0  # Reset patience counter
            torch.save(model.state_dict(), 'model0.pth')  # Save the best model
        else:
            patience_counter += 1  # Increment patience counter

        if patience_counter >= patience:
            print('Early stopping triggered')  # Trigger early stopping
            break

# Main function
def main():
    global train_losses, val_losses
    # Path to the CSV file containing the data
    file_path = 'datasets/All_positions_dataset.csv'

    # Create the dataset and split it into train and validation sets
    train_losses = []
    val_losses = []
    dataset = CustomDataset(file_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = RegressionModel()
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adam optimizer

    # Train the model with early stopping
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100, patience=10)

    # Generate and save the plots
    plt.figure(figsize=(10, 5))

    # Plot of training losses
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Train_inverse_kinematics_graphic/train_loss.png')
    plt.show()

    plt.figure(figsize=(10, 5))

    # Plot of validation losses
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Train_inverse_kinematics_graphic/val_loss.png')
    plt.show()

    plt.figure(figsize=(10, 5))

    # Combined plot of training and validation losses
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Train_inverse_kinematics_graphic/combined_loss.png')
    plt.show()

if __name__ == "__main__":
    main()
