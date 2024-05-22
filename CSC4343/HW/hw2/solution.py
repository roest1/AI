"""
Created on Mon Apr 22 12:57:40 2024

@author: yuhon
"""
from RNNModel import create_model, train_model, load_trained_model, generate_name
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gdown


def marker_with_padding(names):
    """
    This function takes a list of names, adds a start marker '^' to each input name
    and an end marker '.' to each output name. It then pads all names with spaces
    to make them equal in length to the longest name in the list.

    Parameters:
    names (list of str): A list of names to be processed.

    Returns:
    tuple of list of str: Two lists containing the padded input names with start markers
                          and the padded output names with end markers, respectively.
    """

    # Find the maximum name length in the list of names and add 1 for the end marker
    max_length = max(len(name) for name in names) + 1

    # Initialize empty lists for storing processed input and output names
    input_names = []
    output_names = []

    # Iterate through each name in the provided list
    for name in names:
        # Add a start marker to the beginning of the input name
        input_name = '^' + name
        # Add an end marker to the end of the output name
        output_name = name + '.'

        # Pad the input and output names with spaces to make them equal to max_length
        input_name_padded = input_name.ljust(max_length)
        output_name_padded = output_name.ljust(max_length)

        # Append the padded names to the respective lists
        input_names.append(input_name_padded)
        output_names.append(output_name_padded)

    # Return the lists as a tuple
    return input_names, output_names


def name_2_vec(names):
    """
    This function takes a list of names and converts each name to a one-hot encoded tensor.
    Each character in a name is represented by a vector of length 28, where the vector is all zeros
    except for a one at the index corresponding to the character in the character set.

    Parameters:
    names (list of str): A list of names to be converted into one-hot encoded vectors.

    Returns:
    numpy.ndarray: A 3D array with dimensions (number of names, maximum name length, 28),
                    containing the one-hot encoded representation of each name.
    """

    # Determine the number of names
    num_names = len(names)

    # Determine the maximum length of a name in the list for padding purposes
    max_length = max(len(name) for name in names)

    # Initialize a 3D NumPy array with zeros to store the one-hot encoded vectors
    # The dimensions are: number of names x maximum name length x character set length (28)
    encoded_vec = np.zeros((num_names, max_length, 28), dtype=np.float32)

    # Create a mapping of characters to indices, including the start '^' and end '.' markers
    char_to_idx = {char: idx for idx, char in enumerate(
        '^abcdefghijklmnopqrstuvwxyz.')}

    # Loop over each name in the list
    for i, name in enumerate(names):
        # Loop over each character in the name
        for j, char in enumerate(name):
            # Check if the character is in our defined character set
            if char in char_to_idx:
                # Find the index of the character
                index = char_to_idx[char]
                # Set the appropriate position in the one-hot encoded vector to 1
                encoded_vec[i, j, index] = 1

    # Return the one-hot encoded representations
    return encoded_vec


# Set the device to CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the LSTM model with two LSTM layers and a fully connected layer.

        Parameters:
        input_size (int): The number of input features (size of the input vector).
        hidden_size (int): The number of features in the hidden state of the LSTM.
        output_size (int): The number of output features (size of the output vector).
        """
        super(LSTMModel, self).__init__()
        # Store the hidden layer size
        self.hidden_size = hidden_size
        # Define the LSTM layers with the specified number of features and layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=2, batch_first=True).double()
        # Define the fully connected layer that outputs the final result
        self.fc = nn.Linear(hidden_size, output_size).double()

    def forward(self, x):
        """
        Defines the forward pass of the LSTM model.

        Parameters:
        x (Tensor): The input tensor to the LSTM.

        Returns:
        Tensor: The output tensor after passing through the LSTM and fully connected layers.
        """
        # Pass the input through the LSTM layers
        lstm_out, _ = self.lstm(x)
        # Pass the output of the LSTM layers through the fully connected layer
        output = self.fc(lstm_out)
        return output


def create_model(max_length):
    """
    Create an LSTM model with the given maximum sequence length. The model will be
    suitable for predicting the next character in a sequence of characters, where
    each character is represented as a one-hot vector of size 28 (26 letters plus
    start and end markers).

    Parameters:
    max_length (int): The maximum length of the sequences that the model will process.
                      This parameter is used to define the dimensions of the input layer.

    Returns:
    LSTMModel: An instance of the LSTMModel class initialized with the defined
               architecture and moved to the appropriate device (GPU or CPU).
    """

    # Define the size of the input layer based on the one-hot encoding
    input_size = 28  # 26 characters in the alphabet, plus '^' and '.' for start and end markers
    # Define the size of the hidden layers
    hidden_size = 128  # Number of LSTM units in each LSTM layer
    # The output size is equal to the input size because each output prediction is a
    # probability distribution over all possible characters
    output_size = input_size

    # Determine the computing device (GPU or CPU) depending on availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the LSTM model with the specified architecture
    model = LSTMModel(input_size, hidden_size, output_size).to(device)

    # Return the model
    return model


def train_model(model, n_epochs):
    """
    Train an LSTM model for a specified number of epochs using a dataset of names.
    
    Parameters:
    model (LSTMModel): An instance of the LSTMModel class to be trained.
    n_epochs (int): The number of epochs to train the model.
    
    Returns:
    LSTMModel: The trained model.
    """
    # Load the dataset from the file
    with open("C:/Learning/LSU/S2/TA/yob2018.txt") as file:
        lines = file.readlines()
    # Extract names from the dataset, ensuring they contain only alphabetical characters
    names = [line.split(',')[0] for line in lines if re.match(
        "^[a-zA-Z]+$", line.split(',')[0].lower())]
    # Convert names to lowercase
    filtered_names = [name.lower() for name in names]

    # Prepare the device to use for training (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Prepare the data for training by adding markers and padding
    input_names, output_names = marker_with_padding(filtered_names)
    # Convert the character names into one-hot encoded vectors
    encoded_input = name_2_vec(input_names)
    encoded_output = name_2_vec(output_names)

    # Convert the numpy arrays to PyTorch tensors and move them to the training device
    input_tensor = torch.tensor(encoded_input, dtype=torch.float64).to(device)
    output_tensor = torch.tensor(
        encoded_output, dtype=torch.float64).to(device)

    # Define the loss function and optimizer
    # Do not reduce losses yet to handle padding
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Use Adam optimizer

    # Create DataLoader for batching the data
    train_data = TensorDataset(input_tensor, output_tensor)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # Initialize a list to track losses over epochs
    epoch_losses = []

    # Begin training loop
    for epoch in range(n_epochs):
        total_loss = 0  # Reset total loss for the epoch
        model.train()  # Set the model to training mode

        # Iterate over batches of data
        for input_batch, output_batch in train_loader:
            optimizer.zero_grad()  # Zero the parameter gradients

            # Perform forward pass through the model
            output = model(input_batch)
            # Flatten output for loss calculation
            output = output.view(-1, output.size(2))

            # Flatten the target and convert it to long tensor for loss calculation
            target = output_batch.view(-1, output_batch.size(2)).argmax(dim=1)

            # Compute the loss, masking out the contributions from padding
            loss = criterion(output, target)
            # Mask for non-padding elements
            mask = input_batch.view(-1, input_batch.size(2)).sum(dim=1) > 0
            masked_loss = loss * mask.type_as(loss)  # Apply mask to the loss
            # Compute mean loss only over non-padded elements
            loss_mean = masked_loss.sum() / mask.sum()

            # Backward pass and optimize
            loss_mean.backward()
            optimizer.step()

            # Accumulate total loss
            total_loss += masked_loss.sum().item()

        # Compute and print the average loss for the epoch
        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)  # Append to losses history
        print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.4f}')

    # After all epochs, plot the training loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, label='Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

    # Save the trained model state
    model_path = 'rnn_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Return the trained model
    return model


def load_trained_model():
    """
    Load the trained LSTM model from a file. If the model file is not present locally,
    it will be downloaded from Google Drive.

    Returns:
    LSTMModel: The loaded LSTM model if successful, or None if an error occurs.
    """

    # The filename of the saved model parameters
    fname = 'lstm_model.pth'
    # Determine whether to use GPU or CPU for loading the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if the model file exists locally
    if os.path.exists(fname):
        try:
            # Assuming max_length is available globally or is determined somehow
            # Replace `1` with the actual max_length used during model creation/training
            model = create_model(16)
            # Load the model state dictionary
            model.load_state_dict(torch.load(fname, map_location=device))
            print(f'Model loaded successfully from {fname}')
            return model
        except Exception as e:
            # If there's an error during loading, report it
            print(f'Error loading model from {fname}: {e}')
            return None
    else:
        # Google Drive URL where the model file is stored
        drive_url = 'Google Drive URL where the model file is stored'
        try:
            # Attempt to download the file from Google Drive
            gdown.download(drive_url, fname, quiet=False)
            # Again, replace `1` with the correct max_length
            model = create_model(16)
            # Load the model state dictionary
            model.load_state_dict(torch.load(fname, map_location=device))
            print(f'Model downloaded and loaded successfully from {drive_url}')
            return model
        except Exception as e:
            # If there's an error during downloading or loading, report it
            print(f'Error downloading or loading model from {drive_url}: {e}')
            return None


def generate_name(model, max_length):
    """
    Generate a name using the trained LSTM model by sampling from the probability
    distribution output by the model.

    Parameters:
    model (LSTMModel): The trained LSTM model.
    max_length (int): The maximum length of the name to generate.

    Returns:
    str: The generated name.
    """

    # Create a dictionary mapping each character to a unique index
    char_to_idx = {char: idx for idx, char in enumerate(
        '^abcdefghijklmnopqrstuvwxyz.')}
    # Create a reverse dictionary mapping each index back to the corresponding character
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    # Identify the device and data type the model is using (CPU or GPU, float32 or float64)
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    # Initialize a tensor to hold the one-hot encoded vector for the start character '^'
    input_tensor = torch.zeros(
        1, 1, len(char_to_idx), device=device, dtype=model_dtype)
    input_tensor[0, 0, char_to_idx['^']] = 1

    # String to accumulate the generated characters
    generated_name = ''

    # Loop to generate each character in the name
    for _ in range(max_length):
        # Run the model forward pass with the current input tensor
        output = model(input_tensor)
        # Apply softmax to the output to get a probability distribution
        prob_distribution = torch.softmax(
            output.squeeze(), dim=0).cpu().detach().numpy()
        # Sample a character index from the distribution
        sampled_char_idx = np.random.multinomial(1, prob_distribution).argmax()
        # Append the corresponding character to the generated name
        generated_name += idx_to_char[sampled_char_idx]

        # If the sampled character is the end-of-name marker '.', stop generating
        if sampled_char_idx == char_to_idx['.']:
            break

        # Prepare the input tensor for the next iteration with the sampled character
        input_tensor = torch.zeros(
            1, 1, len(char_to_idx), device=device, dtype=model_dtype)
        input_tensor[0, 0, sampled_char_idx] = 1

    # If the generated name ends with the end marker, remove it
    if generated_name.endswith('.'):
        generated_name = generated_name[:-1]

    # Return the generated name without the end marker
    return generated_name


# We will test your model by code similar to the following:
um = create_model(16)
train_model(um, 2)
tm = load_trained_model()
for i in range(5):
    print(generate_name(tm, 10))
