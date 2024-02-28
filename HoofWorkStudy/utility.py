import pandas as pd
import numpy as np
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfWriter, PdfReader
from graphviz import Digraph

import torch
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import scipy.stats as stats

import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
plt.rcParams.update({'font.size': 14})

PATH_TO_DATA = "/Users/rileyoest/VS_Code/Hoof/Original_Data"

HEALTHY = ['10', '13', '25', '06', '12', '19', '22', '23']  # Healthy ID's

SHOE_DICT = {'EB': 'Eggbar', 'US': 'Unshod', 'HB': 'Heartbar', 'OH': 'Standard'}

COORDS = ['DW_x', 'DW_y', 'DW_z', 'SM_x', 'SM_y', 'SM_z', 'CB_x', 'CB_y', 'CB_z', 'P3_x', 'P3_y', 'P3_z']

########################################################################

'''
Data Collection
'''

def get_data():
    '''
    Reads all trials from Original_Data/* and concatenates them into a single dataframe.
    '''
    dfs = []
    trial_count = 1
    for year in os.listdir(PATH_TO_DATA):
        # year looks like "07-21"
        if "Health_Status" not in year:
            for horse in os.listdir(PATH_TO_DATA + '/' + year + '/'):
                # horse looks like "Hoof03_EB"

                # Determining if horse is laminitic
                isLaminitic = True
                if horse[-5:-3] in HEALTHY:
                    isLaminitic = False

                # Determine shoe type
                shoe = SHOE_DICT.get(horse[-2:])

                for trial in os.listdir(PATH_TO_DATA + '/' + year + '/' + horse + '/'):
                    # trial looks like "Hoof03_EB_02.csv"

                    file = pd.read_csv(PATH_TO_DATA + '/' +
                                       year + '/' + horse + '/' + trial)

                    # Renaming column titles
                    file = file.rename(columns={'Marker 09.X': 'DW_x', 'Marker 09.Y': 'DW_y', 'Marker 09.Z': 'DW_z', 'Marker 12.X': 'SM_x', 'Marker 12.Y': 'SM_y', 'Marker 12.Z': 'SM_z',
                                                'Marker 10.X': 'CB_x', 'Marker 10.Y': 'CB_y', 'Marker 10.Z': 'CB_z', 'Marker 11.X': 'P3_x', 'Marker 11.Y': 'P3_y', 'Marker 11.Z': 'P3_z'})

                    # P3 Medial coordinates will not be used in this project
                    file.drop(columns=file.columns[-3:], axis=1, inplace=True)

                    file['Trial'] = trial_count
                    file['isLaminitic'] = isLaminitic
                    file['Shoe'] = shoe
                    trial_count += 1
                    dfs.append(file)

    df = pd.concat(dfs, ignore_index=True)
    column_order = ['Trial', 'isLaminitic', 'Shoe', 'Time', 'DW_x', 'DW_y', 'DW_z',
                    'SM_x', 'SM_y', 'SM_z', 'CB_x', 'CB_y', 'CB_z', 'P3_x', 'P3_y', 'P3_z']
    df = df.reindex(columns=column_order)

    df = df.dropna()

    return df



########################################################################

'''
Data Preprocessing
'''

def interpolate(dataframe:pd.DataFrame):
    df = dataframe.copy()
    for t in df['Trial'].unique():
        trial = df[df['Trial'] == t]
        for coord in COORDS:
            values = trial[coord].values
            is_zero = (values == 0) 
            zero_indices = np.flatnonzero(is_zero)
            non_zero_indices = np.flatnonzero(~is_zero)
            for i in zero_indices:
                values[i] = (values[non_zero_indices[non_zero_indices < i][-1]] + values[non_zero_indices[non_zero_indices > i][0]]) / 2.0
            df.loc[df['Trial'] == t, coord] = values
    return df

def remove_bad_trials(dataframe:pd.DataFrame):
    df = dataframe.copy()
    # deleting bad p3 trials
    df = df[~df['Trial'].isin([174, 175, 176, 177, 178])]
    df['Trial'] = (df['Time'] == 0).cumsum()
    return df

def translate_data_to_zero(dataframe:pd.DataFrame):
    df = dataframe.copy()
    for t in df['Trial'].unique():
        trial = df['Trial'] == t
        df.loc[trial, COORDS] -= df.loc[trial & (df['Time'] == 0), COORDS].values
    return df

def normalize(dataframe:pd.DataFrame):
    df = dataframe.copy()
    for t in df['Trial'].unique():
        trial = df[df['Trial'] == t]
        for coord in COORDS:
            df.loc[df['Trial'] == t, coord] = (df[df['Trial'] == t][coord] - trial[coord].min()) / (trial[coord].max() - trial[coord].min())
        
        df.loc[df['Trial'] == t, 'Time'] = np.linspace(0, 100, len(trial['Time']))

    return df


def suggest_cutoff(trial_df, coord_sets, axis_labels, threshold_value=0.05, last_percent_of_trial=0.25):
    potential_cutoff_indices = []

    relevant_indices = trial_df.index[int(
        len(trial_df) * (1 - last_percent_of_trial)):]
    relevant_data = trial_df.loc[relevant_indices].copy()

    for coord_set in coord_sets:
        for axis_label in axis_labels:

            first_derivative = np.abs(
                np.diff(relevant_data[f'{coord_set}_{axis_label.lower()}'], prepend=np.nan))
            relevant_data[f'First_Derivative_{coord_set}'] = first_derivative
            first_derivative_diffs = np.diff(first_derivative, prepend=np.nan)
            potential_cutoffs = np.where(
                first_derivative_diffs > threshold_value)[0]
            potential_cutoffs_indices = relevant_indices[potential_cutoffs]
            potential_cutoff_indices.extend(potential_cutoffs_indices)

    if potential_cutoff_indices:
        earliest_cutoff_index = min(potential_cutoff_indices)
        if earliest_cutoff_index in trial_df.index:
            return earliest_cutoff_index - 10

    return None


def trim_trials_at_cutoff(dataframe:pd.DataFrame):
    trimmed_dataframes = []
    coord_sets = ['DW', 'SM', 'CB', 'P3']
    for trial in dataframe['Trial'].unique():
        trial_df = dataframe[dataframe['Trial'] == trial]
        cutoff = suggest_cutoff(trial_df, coord_sets, ['Y'])
        if cutoff is not None:
            trimmed_df = trial_df.loc[:cutoff]
            trimmed_dataframes.append(trimmed_df)
        else:
            trimmed_dataframes.append(trial_df)

    final_trimmed_df = pd.concat(trimmed_dataframes)

    return final_trimmed_df


def preprocess_data(dataframe: pd.DataFrame, normalize=True):
    df = dataframe.copy()
    df = interpolate(df)
    df = remove_bad_trials(df)
    df = translate_data_to_zero(df)
    if normalize:
        # Get min/max values after translation
        p3_y_min, p3_y_max = df['P3_y'].min(), df['P3_y'].max()
        p3_z_min, p3_z_max = df['P3_z'].min(), df['P3_z'].max()
        df = normalize(df)
        df = trim_trials_at_cutoff(df)
        return df, p3_y_min, p3_y_max, p3_z_min, p3_z_max
    else:
        df = trim_trials_at_cutoff(df)
        return df

 
########################################################################
'''
Machine Learning
'''

def train_test_split(dataframe:pd.DataFrame, train_ratio=0.65, val_ratio=0.25, verbose=True):
    df = dataframe.copy()
    unique_combinations = df[['isLaminitic', 'Shoe']
                             ].drop_duplicates().reset_index(drop=True)

    X_train, X_val, X_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    y_train, y_val, y_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    non_train_columns = ['Trial', 'Time', 'DW_x', 'SM_x',
                         'CB_x', 'P3_x', 'P3_y', 'P3_z', 'isLaminitic', 'Shoe']

    combo_map = {}

    combo_counts = {
        'Training': [0] * len(unique_combinations),
        'Validation': [0] * len(unique_combinations),
        'Testing': [0] * len(unique_combinations)
    }

    trial_lengths = {}
    health_shoe_params = []

    for i, combo in unique_combinations.iterrows():
        combo_df = df[(df['isLaminitic'] == combo['isLaminitic'])
                      & (df['Shoe'] == combo['Shoe'])]

        combo_map[i] = combo.to_dict()

        total_trials = len(combo_df['Trial'].unique())
        train_trials = int(total_trials * train_ratio)
        val_trials = int(total_trials * val_ratio)
        test_trials = total_trials - (train_trials + val_trials)

        combo_counts['Training'][i] += train_trials
        combo_counts['Validation'][i] += val_trials
        combo_counts['Testing'][i] += test_trials

        shuffled_trials = combo_df['Trial'].unique()
        np.random.shuffle(shuffled_trials)

        train_indices = shuffled_trials[:train_trials]
        val_indices = shuffled_trials[train_trials:train_trials+val_trials]
        test_indices = shuffled_trials[train_trials+val_trials:]

        X_train = pd.concat(
            [X_train, combo_df[combo_df['Trial'].isin(
                train_indices)].drop(columns=non_train_columns)],
            ignore_index=True
        )

        X_val = pd.concat(
            [X_val, combo_df[combo_df['Trial'].isin(val_indices)].drop(
                columns=non_train_columns)],
            ignore_index=True

        )

        y_train = pd.concat(
            [y_train, combo_df[combo_df['Trial'].isin(train_indices)][[
                'P3_y', 'P3_z']]],
            ignore_index=True
        )

        y_val = pd.concat(
            [y_val, combo_df[combo_df['Trial'].isin(
                val_indices)][['P3_y', 'P3_z']]],
            ignore_index=True
        )

        for test_trial in test_indices:
            test_trial_df = combo_df[combo_df['Trial'] == test_trial]
            start_index = len(X_test)

            X_test = pd.concat([X_test, test_trial_df.drop(
                columns=non_train_columns)], ignore_index=True)
            y_test = pd.concat([y_test, combo_df[combo_df['Trial'] == test_trial][[
                               'P3_y', 'P3_z']]], ignore_index=True)

            end_index = len(X_test)

            trial_lengths[test_trial] = (start_index, end_index)
            health_shoe_params.append(
                (test_trial, combo.isLaminitic, combo.Shoe))

    if verbose:
        print("="*50, "\nNumber of (isLaminitic, Shoe) combinations per set:\n", "="*50)
        for col in ['Training', 'Validation', 'Testing']:
            print(f"{col} Set:", "-"*50)
            for i, count in enumerate(combo_counts[col]):
                print(f"({combo_map[i]}) : {count}")

        print("="*50)

    return torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(X_val.values, dtype=torch.float32), torch.tensor(X_test.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32), trial_lengths, health_shoe_params


class NeuralNet(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features, num_layers=32):
        super(NeuralNet, self).__init__()
        self.first_layer = torch.nn.Linear(num_input_features, num_layers)
        self.relu = torch.nn.ReLU()
        self.second_layer = torch.nn.Linear(num_layers, num_layers)
        self.third_layer = torch.nn.Linear(num_layers, num_output_features)
    
    def forward(self, x):
        x = self.first_layer(x)
        x = self.relu(x)
        x = self.second_layer(x)
        x = self.third_layer(x)
        return x

    
def train_neural_network(model, X_train, y_train, X_val, y_val, num_epochs=250, learning_rate=0.01, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses, val_losses = [], []

    for i in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = torch.nn.functional.mse_loss(y_pred, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = torch.nn.functional.mse_loss(val_pred, y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        if verbose:
            print(
                f'Epoch {i+1}/{num_epochs} - mse train loss: {loss:.4f} - mse val loss: {val_loss:.4f}')
    
    if verbose:
        with PdfPages("loss_vs_epochs.pdf") as pdf:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
            plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.xlim(0, len(train_losses) + 1)
            plt.ylabel('Loss (MSE)')
            plt.yscale('log')
            plt.title('Training and Validation Losses')
            plt.legend()
            pdf.savefig()
            plt.close()
    

def cross_validation(dataframe: pd.DataFrame, k_folds=5):
    print(f"Performing {k_folds} Cross Validation\n", "="*50)
    df = dataframe.copy()
    non_train_columns = ['Trial', 'Time', 'DW_x', 'SM_x',
                         'CB_x', 'P3_x', 'P3_y', 'P3_z', 'isLaminitic', 'Shoe']
    kf = KFold(n_splits=k_folds, shuffle=True)

    r2 = []
    mse = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df['Trial'].unique())):
        print(f"Processing fold {fold + 1}...")

        train_trials = df['Trial'].unique()[train_idx]
        test_trials = df['Trial'].unique()[test_idx]

        train_df = df[df['Trial'].isin(train_trials)]
        test_df = df[df['Trial'].isin(test_trials)]

        X_train, X_val, _, y_train, y_val, _, _, _ = train_test_split(
            train_df, verbose=False)
        
        X_test = test_df.drop(columns=non_train_columns)
        y_test = test_df[['P3_y', 'P3_z']]

        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32)

        model = NeuralNet(X_train.shape[1], y_train.shape[1])
        model = torch.nn.DataParallel(model)
        train_neural_network(model, X_train, y_train, X_val, y_val, verbose=False)

        predictions = model(X_test)
        y_test_np = y_test.detach().numpy()
        predictions_np = predictions.detach().numpy()

        r2.append(r2_score(y_test_np, predictions_np))
        mse.append(mean_squared_error(y_test_np, predictions_np))
        print(f"Fold {fold + 1}: R2 = {r2[-1]}, MSE = {mse[-1]}")

    print(f"\nAverage R²: {np.mean(r2):.4f}")
    print(f"Average MSE: {np.mean(mse):.4f}")



########################################################################

'''
Visualization
'''
def plot_model():
    model = NeuralNet(6, 2, num_layers=32)
    dot = Digraph()
    dot.node('X', label='Inputs = [DW_y, DW_z, SM_y, SM_z, CB_y, CB_z]', shape='box')
    prev_node = 'X'
    for layer in model.children():
        dot.node(str(layer), label=str(layer))
        dot.edge(prev_node, str(layer))
        prev_node = str(layer)
    dot.node('Ouput', label='Outputs = [P3_y, P3_z]', shape='box')
    dot.edge(prev_node, 'output')
    dot.format = 'png'
    dot.render('neural_net_graph', view=False)


def plot_all_coords_to_pdf(dataframe: pd.DataFrame, coord_sets=['DW', 'SM', 'CB', 'P3'], colors=['red', 'green', 'blue', 'purple'], dimensions=['Y', 'Z'], use_suggest_cutoff=True, pdf_name='all_coords.pdf'):
    assert len(coord_sets) == len(colors), "Number of coord_sets and colors must be equal"
    
    df = dataframe.copy()
    unique_trials = df['Trial'].unique()

    with PdfPages(pdf_name) as pdf:
        for trial in unique_trials:
            fig, axes = plt.subplots(1, len(dimensions), figsize=(15, 5))
            trial_df = df[df['Trial'] == trial]
            is_laminitic = 'Laminitic' if trial_df['isLaminitic'].iloc[0] else 'Healthy'
            shoe = trial_df['Shoe'].iloc[0]
            title = f'Trial {trial}, {is_laminitic}, Shoe: {shoe}'

            if use_suggest_cutoff:
                cutoff = suggest_cutoff(trial_df, coord_sets, ['Y'])

                shading_start = trial_df['Time'].iloc[int(
                    len(trial_df) * (1 - 25/100))] # using last 25 % 
            
            for i, axis_label in enumerate(dimensions):
                for coord_set, color in zip(coord_sets, colors):
                    axes[i].plot(trial_df['Time'], trial_df[f'{coord_set}_{axis_label.lower()}'],
                                label=f'{coord_set}_{axis_label}', color=color)
                    axes[i].set_xlabel('Stride %')
                    axes[i].set_ylabel(f'{axis_label}')
                    if use_suggest_cutoff:
                        axes[i].axvspan(
                            shading_start, trial_df['Time'].iloc[-1], color='grey', alpha=0.2)
                    axes[i].legend()
                if use_suggest_cutoff:
                    if cutoff is not None and cutoff in trial_df.index:
                        for coord_set in coord_sets:
                            axes[i].plot(trial_df.loc[cutoff, 'Time'], trial_df.loc[cutoff, f'{coord_set}_{axis_label.lower()}'],
                                        'ko', label='Suggested Cutoff', markersize=10)

            fig.suptitle(title)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"All coordinates figure saved as {pdf_name}")


def plot_derivative_with_threshold(trial_df, coord_sets, axis_label, threshold_value, last_percent_of_trial=0.25):
    '''
    Usage:
    trial_number = 54
    axis_label = 'Y'
    threshold_value = 0.06
    coord_sets = ['DW', 'SM', 'CB', 'P3']


    trial_df = df[df['Trial'] == trial_number]
    earliest_cutoff = plot_derivative_with_threshold(
        trial_df, coord_sets, axis_label, threshold_value
    )
    '''
    plt.figure(figsize=(10, 4))

    relevant_data = trial_df[int(
        len(trial_df) * (1 - last_percent_of_trial)):].copy()
    potential_cutoffs_all = []

    for coord_set in coord_sets:

        relevant_data[f'First_Derivative_{coord_set}'] = np.abs(
            np.diff(relevant_data[f'{coord_set}_{axis_label.lower()}'], prepend=np.nan))

        first_derivative_diffs = np.diff(
            relevant_data[f'First_Derivative_{coord_set}'], prepend=0)

        potential_cutoffs = np.where(
            first_derivative_diffs > threshold_value)[0]

        potential_cutoffs_all.extend(potential_cutoffs - 1)

        plt.plot(relevant_data['Time'], relevant_data[f'First_Derivative_{coord_set}'],
                 label=f'First Derivative {coord_set}')

    if potential_cutoffs_all:
        earliest_cutoff_index = min(potential_cutoffs_all)
        cutoff_time = relevant_data['Time'].iloc[earliest_cutoff_index]

        first_derivative_column = f'First_Derivative_{coord_sets[0]}'
        plt.scatter(cutoff_time, relevant_data[first_derivative_column].iloc[earliest_cutoff_index],
                    color='red', label='Earliest Suggested Cutoff', zorder=5)

    plt.title(f'First Derivative and Suggested Cutoff for {axis_label}')
    plt.xlabel('Stride %')
    plt.ylabel('First Derivative')
    plt.legend()
    plt.show()

    return cutoff_time if potential_cutoffs_all else None


def plot_histogram(intervals=[0.2, 0.4, 0.6, 0.8], coord='P3_y'):
    df = get_data()
    df = interpolate(df)
    df = remove_bad_trials(df)
    df = translate_data_to_zero(df)
    for t in df['Trial'].unique():
        trial = df[df['Trial'] == t]
        df.loc[df['Trial'] == t, 'Time'] = np.linspace(0, 100, len(trial['Time']))
    df = trim_trials_at_cutoff(df)

    interval_values = {i: [] for i in intervals}

    for t in df['Trial'].unique():
        trial = df[df['Trial'] == t]
        for i in intervals:
            interval_values[i].append(trial[coord].iloc[max(
                0, min(int(len(trial) * i) - 1, len(trial) - 1))])

    interval_averages = {i: sum(values) / len(values)
                         for i, values in interval_values.items() if values}

    fig, ax = plt.subplots()
    intervals_scaled = [i * 100 for i in intervals] 
    averages = [interval_averages[i]
                for i in intervals]
    
    ax.bar(intervals_scaled, averages, width=10)
    ax.set_xlabel('Intervals (%)')
    ax.set_ylabel(coord)
    ax.set_title(f'Average {coord} by Interval')
    ax.set_xticks(intervals_scaled)
    ax.set_xticklabels([f'{int(i)}%' for i in intervals_scaled])
    plt.show()

    print(f"Shapiro-Wilk Test for Normality of {coord} Data")

    for k, v in interval_values.items():
        print(f'{int(100*k)}%:')
        w, p = stats.shapiro(v)
        print(f"W = {w:.4f}\np = {p:.4f}")
        print("="*50)

    return interval_values

def plot_preprocessing_steps_to_pdf(df, pdf_filename='preprocessing_steps.pdf'):
    preprocessing_steps = [
        ('Raw', lambda x: x),
        ('Interpolation', interpolate),
        ('Removing Bad Trials', remove_bad_trials),
        ('Translation', translate_data_to_zero),
        ('Normalization', normalize),
        ('Trimming', trim_trials_at_cutoff)
    ]

    fig, axes = plt.subplots(
        nrows=len(preprocessing_steps), ncols=3, figsize=(20, 15))

    processed_df = df
    for i, (step_name, func) in enumerate(preprocessing_steps):
        processed_df = func(processed_df.copy()) if func else processed_df

        for j, coord in enumerate(['P3_x', 'P3_y', 'P3_z']):
            ax = axes[i, j]  
            ax.plot(processed_df['Time'], processed_df[coord])
            
            ymin, ymax = ax.get_ylim()
            yrange = ymax - ymin
            
            ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)

            if i == 0:  
                ax.set_title(coord)

            if j == 0:
                ax.set_ylabel(f'{step_name}')

            if i == len(preprocessing_steps) - 1:  
                ax.set_xlabel('Time')

    plt.tight_layout()
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Preprocessing steps figure saved as {pdf_filename}")


def get_confidence_intervals():
    df = get_data()
    df = preprocess_data(df, normalize=False)
    

def plot_model_results_to_pdf(model, X_test, y_test, test_args, trial_lengths, p3_y_min, p3_y_max, p3_z_min, p3_z_max, pdf_filename='singular_model_results.pdf'):

    def get_yz(data, p3_y_min, p3_y_max, p3_z_min, p3_z_max):
        Y, Z = [], []

        for y, z in data:
            Y.append(y * (p3_y_max - p3_y_min) + p3_y_min)
            Z.append(z * (p3_z_max - p3_z_min) + p3_z_min)

        return np.array(Y), np.array(Z)
    shoe_map = {1: 'Standard', 2: 'Unshod', 3: 'Heartbar', 4: 'Eggbar'}
    health_map = {False: 'Healthy', True: 'Laminitic'}
    predictions = model(X_test)
    y_test = y_test.detach().numpy()
    predictions = predictions.detach().numpy()

    print(f"Accuracy of Predictions for P3 Model")
    print("Inputs = dwy, dwz, smy, smz, cby, cbz")
    print("="*50)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    print(f"R² = {r2:.4f}")
    print(f"MSE = {mse:.4f}")
    print("="*50)

    y_test = get_yz(y_test, p3_y_min, p3_y_max, p3_z_min, p3_z_max)
    predictions = get_yz(predictions, p3_y_min, p3_y_max, p3_z_min, p3_z_max)

    r2_scores_y = defaultdict(list)
    mse_scores_y = defaultdict(list)
    r2_scores_z = defaultdict(list)
    mse_scores_z = defaultdict(list)

    with PdfPages(pdf_filename) as pdf:
        for i, (trial, isLaminitic, shoe) in enumerate(test_args):
            start_index, end_index = trial_lengths[trial]
            laminitic = health_map[isLaminitic]
            category = f"{shoe}_{laminitic}"
            title = f"Prediction vs Actual P3_y/z Coordinates with 95% Confidence Interval for trial {trial}\n{laminitic}, {shoe} shoe"

            y_pred, z_pred = predictions
            ytest, ztest = y_test

            ytest = ytest[start_index:end_index]
            ztest = ztest[start_index:end_index]
            y_pred = y_pred[start_index:end_index]
            z_pred = z_pred[start_index:end_index]

            r2_scores_y[category].append(r2_score(ytest, y_pred))
            mse_scores_y[category].append(mean_squared_error(ytest, y_pred))
            r2_scores_z[category].append(r2_score(ztest, z_pred))
            mse_scores_z[category].append(mean_squared_error(ztest, z_pred))

            _, ax = plt.subplots(1, 2, figsize=(10, 5))
            residuals = sorted([x - y for x, y in zip(y_pred, ytest)])
            RMSFE = np.sqrt(sum([x**2 for x in residuals]) / len(residuals))
            band_size = 1.96*RMSFE

            ax[0].plot(np.linspace(0, 100, len(ytest)), ytest,
                       color='#fc7d0b', label='True')
            ax[0].plot(np.linspace(0, 100, len(y_pred)), y_pred,
                       color='r', alpha=1, label='Predicted')
            ax[0].fill_between(np.linspace(0, 100, len(ytest)), (ytest-band_size), (ytest+band_size),
                               color='lightblue', alpha=.7, label='95% Confidence Interval')

            ax[0].set_xlabel('Stride (%)', fontsize=14)
            ax[0].set_ylabel('P3_y (mm)', fontsize=14)
            ax[0].legend()

            residuals = sorted([x - y for x, y in zip(z_pred, ztest)])
            RMSFE = np.sqrt(sum([x**2 for x in residuals]) / len(residuals))
            band_size = 1.96*RMSFE

            ax[1].plot(np.linspace(0, 100, len(ztest)), ztest,
                       color='#fc7d0b', label='True')
            ax[1].plot(np.linspace(0, 100, len(z_pred)), z_pred,
                       color='r', alpha=1, label='Predicted')
            ax[1].fill_between(np.linspace(0, 100, len(ztest)), (ztest-band_size), (ztest+band_size),
                               color='lightblue', alpha=.7, label='95% Confidence Interval')

            ax[1].set_xlabel('Stride (%)', fontsize=14)
            ax[1].set_ylabel('P3_z (mm)', fontsize=14)
            ax[1].legend()

            plt.suptitle(title)
            plt.tight_layout()
            pdf.savefig()  
            plt.close()    

    label_mapping = {
        'Standard_Healthy': 'SD_H',
        'Unshod_Healthy': 'US_H',
        'Heartbar_Healthy': 'HB_H',
        'Eggbar_Healthy': 'EB_H',
        'Standard_Laminitic': 'SD_L',
        'Unshod_Laminitic': 'US_L',
        'Heartbar_Laminitic': 'HB_L',
        'Eggbar_Laminitic': 'EB_L'
    }

    with PdfPages(f"{pdf_filename[:-4]}_boxplot.pdf") as pdf:

        r2_scores_y_lists = [r2_scores_y[key] for key in r2_scores_y]
        mse_scores_y_lists = [mse_scores_y[key] for key in mse_scores_y]
        r2_scores_z_lists = [r2_scores_z[key] for key in r2_scores_z]
        mse_scores_z_lists = [mse_scores_z[key] for key in mse_scores_z]

        all_categories = [f"{shoe}_{health}" for shoe in ['Eggbar', 'Unshod', 'Heartbar', 'Standard']
                          for health in health_map.values()]

        new_labels = [label_mapping.get(category, category)
                      for category in all_categories]

        whiskerprops = dict(linestyle='-', linewidth=1.5, color='darkorange')
        boxprops = dict(facecolor='skyblue', color='darkblue')
        medianprops = dict(linestyle='-', linewidth=1.5, color='darkred')
        flierprops = dict(marker='o', markerfacecolor='green',
                          markersize=5, linestyle='none')

        fontsize_title = 16
        fontsize_labels = 16
        fontsize_ticks = 14

        fig, axs = plt.subplots(2, 2, figsize=(14, 12))

        axs[0, 0].boxplot(r2_scores_y_lists, labels=new_labels, patch_artist=True,
                          whiskerprops=whiskerprops, boxprops=boxprops,
                          medianprops=medianprops, flierprops=flierprops)
        axs[0, 0].set_title('R² Scores for P3_y', fontsize=fontsize_title)
        axs[0, 0].set_ylabel('R² Score', fontsize=fontsize_labels)
        axs[0, 0].tick_params(labelrotation=45, labelsize=fontsize_ticks)

        axs[0, 1].boxplot(r2_scores_z_lists, labels=new_labels, patch_artist=True,
                          whiskerprops=whiskerprops, boxprops=boxprops,
                          medianprops=medianprops, flierprops=flierprops)
        axs[0, 1].set_title('R² Scores for P3_z', fontsize=fontsize_title)
        axs[0, 1].set_ylabel('R² Score', fontsize=fontsize_labels)
        axs[0, 1].tick_params(labelrotation=45, labelsize=fontsize_ticks)

        axs[1, 0].boxplot(mse_scores_y_lists, labels=new_labels, patch_artist=True,
                          whiskerprops=whiskerprops, boxprops=boxprops,
                          medianprops=medianprops, flierprops=flierprops)
        axs[1, 0].set_title('MSE Scores for P3_y', fontsize=fontsize_title)
        axs[1, 0].set_ylabel('MSE Score', fontsize=fontsize_labels)
        axs[1, 0].tick_params(labelrotation=45, labelsize=fontsize_ticks)

        axs[1, 1].boxplot(mse_scores_z_lists, labels=new_labels, patch_artist=True,
                          whiskerprops=whiskerprops, boxprops=boxprops,
                          medianprops=medianprops, flierprops=flierprops)
        axs[1, 1].set_title('MSE Scores for P3_z', fontsize=fontsize_title)
        axs[1, 1].set_ylabel('MSE Score', fontsize=fontsize_labels)
        axs[1, 1].tick_params(labelrotation=45, labelsize=fontsize_ticks)

        plt.tight_layout()
        pdf.savefig()
        plt.close()
#########################################################################

