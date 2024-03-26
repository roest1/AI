'''
modeling.py
-------------
'''

from functools import partial
from bayes_opt import BayesianOptimization
from sklearn.model_selection import GroupShuffleSplit
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_squared_error, r2_score
import copy

def train_test_split(dataframe: pd.DataFrame, input_columns: list, target_columns: list,
                     train_ratio=0.65, val_ratio=0.25, seed=0, verbose=True):
    '''
    Best test results had:
    input_columns = ['DW_y', 'DW_z', 'SM_y', 'SM_z', 'CB_y', 'CB_z']
    target_columns = ['P3_y', 'P3_z']
    train_ratio = 0.65
    val_ratio = 0.25
    '''
    
    np.random.seed(seed)
    unique_combinations = dataframe[['isLaminitic', 'Shoe']].drop_duplicates()

    split_data = {
        'train': {'X': pd.DataFrame(), 'y': pd.DataFrame(), 'counts': {}},
        'validate': {'X': pd.DataFrame(), 'y': pd.DataFrame(), 'counts': {}},
        'test': {'X': pd.DataFrame(), 'y': pd.DataFrame(), 'counts': {}}
    }
    trial_lengths = {}
    health_shoe_params = []

    for _, combo in unique_combinations.iterrows():
        mask = (dataframe['isLaminitic'] == combo['isLaminitic']) & (
            dataframe['Shoe'] == combo['Shoe'])
        trials = dataframe.loc[mask, 'Trial'].unique()
        np.random.shuffle(trials)

        n_trials = len(trials)
        train_end = int(n_trials * train_ratio)
        validate_end = train_end + int(n_trials * val_ratio)

        train_trials = trials[:train_end]
        validate_trials = trials[train_end:validate_end]
        test_trials = trials[validate_end:]

        for split_name, split_trials in zip(['train', 'validate', 'test'], [train_trials, validate_trials, test_trials]):
            assert isinstance(
                split_data[split_name]['X'], pd.DataFrame), f"split_data[{split_name}]['X'] is not a DataFrame"
            split_mask = dataframe['Trial'].isin(split_trials)
            split_data[split_name]['X'] = pd.concat([split_data[split_name]['X'],
                                                     dataframe.loc[split_mask, input_columns]],
                                                    ignore_index=True)
            split_data[split_name]['y'] = pd.concat([split_data[split_name]['y'],
                                                    dataframe.loc[split_mask, target_columns]],
                                                    ignore_index=True)

            split_data[split_name]['counts'][(
                combo['isLaminitic'], combo['Shoe'])] = len(split_trials)

        for trial in test_trials:
            health_shoe_params.append(
                (trial, combo['isLaminitic'], combo['Shoe']))
            test_trial_df = dataframe[dataframe['Trial'] == trial]

            # Calculate the start index for the current trial
            start_index = len(split_data['test']['X'])

            # Append the data for the current trial to test split
            split_data['test']['X'] = pd.concat([split_data['test']['X'],
                                                test_trial_df[input_columns]],
                                                ignore_index=True)
            split_data['test']['y'] = pd.concat([split_data['test']['y'],
                                                test_trial_df[target_columns]],
                                                ignore_index=True)

            # Calculate the end index for the current trial after appending
            end_index = len(split_data['test']['X'])

            # Store the start and end indices in trial_lengths
            trial_lengths[trial] = (start_index, end_index)

    if verbose:
        print("=" * 50)
        for split_name in split_data:
            print(f"{split_name.capitalize()} set:")
            for combo, count in split_data[split_name]['counts'].items():
                print(f"  {combo}: {count} trials")
            print("=" * 50)

    # Convert to torch tensors
    tensors = {
        split_name: (
            torch.tensor(split_data[split_name]
                         ['X'].values, dtype=torch.float32),
            torch.tensor(split_data[split_name]
                         ['y'].values, dtype=torch.float32)
        )
        for split_name in ['train', 'validate', 'test']
    }

    return (*tensors['train'], *tensors['validate'], *tensors['test'], trial_lengths, health_shoe_params)


class NeuralNet(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features, hidden_neurons=32):
        super(NeuralNet, self).__init__()
        self.first_layer = torch.nn.Linear(num_input_features, hidden_neurons)
        self.relu = torch.nn.ReLU()
        self.second_layer = torch.nn.Linear(hidden_neurons, hidden_neurons)
        self.third_layer = torch.nn.Linear(hidden_neurons, num_output_features)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.relu(x)
        x = self.second_layer(x)
        x = self.third_layer(x)
        return x


def train_neural_network(model, X_train, y_train, X_val, y_val, num_epochs=250, learning_rate=0.01, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        if verbose:
            print(
                f'Epoch {epoch+1}/{num_epochs} - mse train loss: {loss:.4f} - mse val loss: {val_loss:.4f}')

    model.load_state_dict(best_model_wts)

    if verbose:
        with PdfPages("../Media/loss_vs_epochs.pdf") as pdf:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(train_losses) + 1),
                     train_losses, label='Training Loss')
            plt.plot(range(1, len(val_losses) + 1),
                     val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.xlim(0, len(train_losses) + 1)
            plt.ylabel('Loss (MSE)')
            plt.yscale('log')
            plt.title('Training and Validation Losses')
            plt.legend()
            pdf.savefig()
            plt.close()

    return model


def save_model(model, filename):
    torch.save(model.module.state_dict(), f'../Models/{filename}')


def load_model(input_features, output_features, filename):
    model = NeuralNet(input_features, output_features)
    model.load_state_dict(torch.load(f'../Models/{filename}'))
    model.eval()
    return model

def train_and_evaluate_model(df, hidden_neurons, learning_rate, num_epochs, train_ratio, remaining_for_val_ratio):
    max_combined_ratio = 0.9  # This ensures at least 10% of the data is for testing

    # Calculate the actual validation ratio based on the remaining portion of the dataset
    # after allocating for the training set, constrained by the max_combined_ratio
    allocated_for_training = train_ratio
    max_possible_for_validation = max_combined_ratio - allocated_for_training
    val_ratio = max_possible_for_validation * remaining_for_val_ratio

    # Ensure val_ratio does not exceed its maximum possible value
    val_ratio = min(val_ratio, max_possible_for_validation)

    input_columns = ['DW_y', 'DW_z', 'SM_y', 'SM_z', 'CB_y', 'CB_z']
    target_columns = ['P3_y', 'P3_z']
    print(
        f"Parameters:\nTrain ratio = {train_ratio:.4}\nVal ratio = {val_ratio:.4}\nepochs = {int(num_epochs)}\nLearning rate = {learning_rate:.4}\nHidden neurons = {2 ** int(hidden_neurons)}")
    X_train, y_train, X_val, y_val, X_test, y_test, _, _ = train_test_split(
        dataframe=df,
        input_columns=input_columns,
        target_columns=target_columns,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=0,
        verbose=False
    )
    model = NeuralNet(
        X_train.shape[1], y_train.shape[1], hidden_neurons=2 ** int(hidden_neurons))
    best_model = train_neural_network(model, X_train, y_train, X_val, y_val,
                                      num_epochs=int(num_epochs), learning_rate=learning_rate, verbose=False)
    predictions = best_model(X_test)
    y_test_global = y_test.detach().numpy()
    predictions_global = predictions.detach().numpy()
    # Return negative MSE for minimization
    mse = mean_squared_error(y_test_global, predictions_global)
    print(f"MSE = {mse:.4}")
    return -mse


def black_box_function(df, hidden_neurons, learning_rate, num_epochs, train_ratio, remaining_for_val_ratio):
    val_loss = train_and_evaluate_model(df=df,
                                        hidden_neurons=hidden_neurons,
                                        learning_rate=learning_rate,
                                        num_epochs=num_epochs,
                                        train_ratio=train_ratio,
                                        remaining_for_val_ratio=remaining_for_val_ratio)
    return val_loss


def run_bayesian_optimization(get_data_function, preprocess_data_function):
    # Load and preprocess your data outside the optimization function
    original_df = get_data_function()
    preprocessed_df, _ = preprocess_data_function(original_df)

    # Create a partial function that includes the preprocessed dataframe
    optimized_function = partial(
        black_box_function,
        df=preprocessed_df
    )

    pbounds = {
        'hidden_neurons': (4, 8),  # Powers of 2 from 2^4 to 2^8
        'learning_rate': (1e-5, 1e-1),
        'num_epochs': (200, 300),
        'train_ratio': (0.4, 0.8),
        # Fraction of the remaining dataset to allocate to validation
        'remaining_for_val_ratio': (0, 1)
    }
    optimizer = BayesianOptimization(
        f=optimized_function,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=2,  # Number of iterations with random points
        n_iter=10,  # Number of iterations with Bayesian optimization
    )

    print(optimizer.max)


def cross_validation(dataframe: pd.DataFrame, input_columns: list, target_columns: list, k_folds=5):
    """
    Perform k-fold cross-validation, ensuring that trials are split according to
    unique health and shoe conditions, utilizing the existing train_test_split function.
    """
    unique_trials = dataframe['Trial'].unique()
    gss = GroupShuffleSplit(
        n_splits=k_folds, test_size=1/k_folds, random_state=42)

    r2_scores, mse_scores = [], []

    for fold, (train_val_idx, test_idx) in enumerate(gss.split(unique_trials, groups=unique_trials), 1):
        print(f"Processing fold {fold}...")
        train_val_trials = unique_trials[train_val_idx]
        test_trials = unique_trials[test_idx]

        # Generate a mask for selecting the dataframe rows corresponding to the current fold's train and test trials
        train_val_mask = dataframe['Trial'].isin(train_val_trials)
        test_mask = dataframe['Trial'].isin(test_trials)

        # Split the dataframe into training/validation and testing dataframes based on the mask
        train_val_df = dataframe[train_val_mask]
        test_df = dataframe[test_mask]

        X_train, y_train, X_val, y_val, X_test, y_test, _, _ = train_test_split(
            dataframe=train_val_df,
            input_columns=input_columns,
            target_columns=target_columns,
            verbose=False
        )

        X_test, y_test = torch.tensor(test_df[input_columns].values, dtype=torch.float32), torch.tensor(
            test_df[target_columns].values, dtype=torch.float32)

        # Initialize and train the model
        model = NeuralNet(X_train.shape[1], y_train.shape[1])
        trained_model = train_neural_network(
            model, X_train, y_train, X_val, y_val, verbose=False)

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            y_pred = trained_model(X_test)
            r2_scores.append(r2_score(y_test.numpy(), y_pred.numpy()))
            mse_scores.append(mean_squared_error(
                y_test.numpy(), y_pred.numpy()))

        print(
            f"Fold {fold}: R² = {r2_scores[-1]:.4f}, MSE = {mse_scores[-1]:.4f}")
    print("="*50)
    print(
        f"\nAverage R²: {np.mean(r2_scores):.4f}, Average MSE: {np.mean(mse_scores):.4f}")
