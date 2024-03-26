import json
from typing import List, Optional
import pickle
import scipy.stats as stats
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import hashlib
import os
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')


PATH_TO_DATA = '/Users/rileyoest/VS_Code/HoofStudy/Data/Original_Data'
PATH_TO_CACHE = '/Users/rileyoest/VS_Code/HoofStudy/Data/Cache/'

HEALTHY = ['10', '13', '25', '06', '12', '19', '22', '23']  # Healthy ID's

SHOE_DICT = {'EB': 'Eggbar', 'US': 'Unshod',
             'HB': 'Heartbar', 'OH': 'Standard'}

COORDS = ['DW_x', 'DW_y', 'DW_z', 'SM_x', 'SM_y', 'SM_z',
          'CB_x', 'CB_y', 'CB_z', 'P3_x', 'P3_y', 'P3_z']

# Get Data 
def hash_file(filepath):
    hasher = hashlib.blake2b()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def get_data():
    cache_file = os.path.join(PATH_TO_CACHE, 'loaded_original_dataframe.pkl')

    # Check if the cache directory exists, if not create it
    if not os.path.exists(PATH_TO_CACHE):
        os.makedirs(PATH_TO_CACHE)

    # If the cache file exists, load and return the dataframe
    if os.path.exists(cache_file):
        #print("Loading dataframe from cache")
        with open(cache_file, 'rb') as file:
            df = pickle.load(file)
        return df

    dfs = []
    trial_count = 1
    hash_dict = {}
    unique_files = []

    for year in os.listdir(PATH_TO_DATA):
        # year looks like "07-21"
        if "Health_Status" not in year:
            for horse in os.listdir(os.path.join(PATH_TO_DATA, year)):
                # horse looks like "Hoof03_EB"

                # Determining if horse is laminitic
                isLaminitic = True
                if horse[-5:-3] in HEALTHY:
                    isLaminitic = False

                # Determine shoe type
                shoe = SHOE_DICT.get(horse[-2:])

                for trial in os.listdir(os.path.join(PATH_TO_DATA, year, horse)):
                    # trial looks like "Hoof03_EB_02.csv"

                    filepath = os.path.join(PATH_TO_DATA, year, horse, trial)
                    filehash = hash_file(filepath)

                    if filehash in hash_dict:
                        # print(f"Duplicate found and ignored: {filepath}")
                        continue
                    else:
                        hash_dict[filehash] = filepath
                        unique_files.append(filepath)
                        file = pd.read_csv(filepath)
                        file['Trial'] = trial_count
                        file['isLaminitic'] = isLaminitic
                        file['Shoe'] = shoe
                        trial_count += 1
                        dfs.append(file)

    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={'Marker 09.X': 'DW_x', 'Marker 09.Y': 'DW_y', 'Marker 09.Z': 'DW_z',
                            'Marker 12.X': 'SM_x', 'Marker 12.Y': 'SM_y', 'Marker 12.Z': 'SM_z',
                            'Marker 10.X': 'CB_x', 'Marker 10.Y': 'CB_y', 'Marker 10.Z': 'CB_z',
                            'Marker 11.X': 'P3_x', 'Marker 11.Y': 'P3_y', 'Marker 11.Z': 'P3_z'})

    # # P3 Medial coordinates will not be used in this project
    df.drop(columns=['Marker 05.X',	'Marker 05.Y',
            'Marker 05.Z'], axis=1, inplace=True)

    column_order = ['Trial', 'isLaminitic', 'Shoe', 'Time', 'DW_x', 'DW_y', 'DW_z',
                    'SM_x', 'SM_y', 'SM_z', 'CB_x', 'CB_y', 'CB_z', 'P3_x', 'P3_y', 'P3_z']
    df = df.reindex(columns=column_order)
    df = df.dropna()

    with open(cache_file, 'wb') as file:
        pickle.dump(df, file)
        print("Unprocessed dataframe cached at", cache_file)

    return df

# Preprocessing #

def interpolate(dataframe: pd.DataFrame):
    df = dataframe.copy()
    for t in df['Trial'].unique():
        trial = df[df['Trial'] == t]
        for coord in COORDS:
            values = trial[coord].values
            is_zero = (values == 0)
            zero_indices = np.flatnonzero(is_zero)
            non_zero_indices = np.flatnonzero(~is_zero)
            for i in zero_indices:
                values[i] = (values[non_zero_indices[non_zero_indices < i][-1]] +
                             values[non_zero_indices[non_zero_indices > i][0]]) / 2.0
            df.loc[df['Trial'] == t, coord] = values
    return df

def remove_bad_trials(dataframe: pd.DataFrame, bad_trials = [164, 165, 166, 167, 168]):
    df = dataframe.copy()
    # deleting bad p3 trials
    df = df[~df['Trial'].isin(bad_trials)]
    df['Trial'] = (df['Time'] == 0).cumsum()
    return df

def translate_data_to_zero(dataframe: pd.DataFrame):
    df = dataframe.copy()
    for t in df['Trial'].unique():
        trial = df['Trial'] == t
        df.loc[trial, COORDS] -= df.loc[trial &
                                        (df['Time'] == 0), COORDS].values
    return df


def truncate_trials_at_last_positive(df: pd.DataFrame, coords=['P3_y', 'P3_z']) -> pd.DataFrame:
    """
    Truncates each trial in the dataframe to the point just before the first negative value of the specified coordinates,
    considering only the last 50% of the data to avoid early negatives, matching the logic of v1.

    Parameters:
    - df: pandas DataFrame, the dataset containing multiple trials.
    - coords: List of strings, specifying the coordinates to check for negative values.

    Returns:
    - A truncated dataframe where each trial is cut off just before the first negative value within the last 50% of its data.
    """
    df_copy = df.copy()

    for trial in df_copy['Trial'].unique():
        trial_data = df_copy[df_copy['Trial'] == trial]
        # Adjust to consider the last 50% of data
        start_index_for_last_50 = trial_data.index[int(len(trial_data) * 0.5)]

        cutoff_indices = []

        for coord in coords:
            # Focusing on the last 50% of trial data
            last_50_trial_data = trial_data.loc[start_index_for_last_50:]

            # Identifying indices where the coordinate values are negative
            negative_crossings = last_50_trial_data[last_50_trial_data[coord] < 0].index

            if not negative_crossings.empty:
                # Determine the index just before the first negative value
                cutoff_index = negative_crossings[0] - \
                    1 if negative_crossings[0] > start_index_for_last_50 else None

                if cutoff_index is not None:
                    cutoff_indices.append(cutoff_index)

        # If cutoff indices were found, determine the earliest cutoff and retain data up to that point
        if cutoff_indices:
            earliest_cutoff_index = max(cutoff_indices)
            df_copy.drop(df_copy[(df_copy['Trial'] == trial) & (
                df_copy.index > earliest_cutoff_index)].index, inplace=True)

    return df_copy


def calculate_finite_difference(coord_series: pd.Series) -> pd.Series:
    """
    Calculate the finite difference of a pandas Series.

    Parameters:
    - coord_series: pandas Series, the series of coordinates from which to calculate the finite difference.

    Returns:
    - pandas Series representing the absolute finite difference.
    """
    return np.abs(np.diff(coord_series, prepend=np.nan))


def find_noise_spikes(df: pd.DataFrame, trial: int, coord_set: str, axis_label: str, threshold_value: float, visualize: bool = True) -> np.ndarray:
    """
    Find spikes in the data based on the finite difference exceeding a threshold.

    Parameters:
    - df: pandas DataFrame, the dataset containing the trials.
    - trial: int, the specific trial to analyze.
    - coord_set: str, the set of coordinates (e.g., 'P3').
    - axis_label: str, the axis label ('x', 'y', 'z').
    - threshold_value: float, the threshold for detecting spikes.
    - visualize: bool, flag to indicate whether to visualize the spikes.

    Returns:
    - An array of indices where spikes were detected.
    """
    trial_data = df[df['Trial'] == trial].copy()
    trial_data['Time'] = np.linspace(0, 100, len(trial_data))
    coord = f'{coord_set}_{axis_label}'
    trial_data[f'Finite_Difference_{coord}'] = calculate_finite_difference(
        trial_data[coord])
    spikes = np.where(
        trial_data[f'Finite_Difference_{coord}'] > threshold_value)[0]

    if visualize:
        plot_spikes(trial_data, coord, spikes, trial)

    return spikes


def plot_spikes(trial_data: pd.DataFrame, coord: str, spikes: np.ndarray, trial: int):
    """
    Visualize spikes on a plot.

    Parameters:
    - trial_data: pandas DataFrame, the dataset for the specific trial.
    - coord: str, the specific coordinate being analyzed.
    - spikes: np.ndarray, the indices of the detected spikes.
    - trial: int, the trial number.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(trial_data['Time'], trial_data[coord], label=coord)

    for i, spike in enumerate(spikes):
        if i == 0 or (spike - spikes[i - 1] > 1):
            # green dot for the previous point
            plt.plot(trial_data.iloc[spike - 1]['Time'],
                     trial_data.iloc[spike - 1][coord], 'go')
            plt.plot(trial_data.iloc[spike]['Time'],
                     trial_data.iloc[spike][coord], 'ro')  # red dot
            # green dot for the next point
            plt.plot(trial_data.iloc[spike + 1]['Time'],
                     trial_data.iloc[spike + 1][coord], 'go')

    plt.title(f'Trial {trial} - {coord} with Spikes')
    plt.xlabel('Time (%)')
    plt.ylabel(coord)
    plt.legend()
    plt.show()


def remove_spikes_and_average(df: pd.DataFrame, trial: int, coord_set: str, axis_label: str, spikes: np.ndarray):
    """
    Remove spikes from the data and replace them by averaging the neighboring points.

    Parameters:
    - df: pandas DataFrame, the dataset containing the trials.
    - trial: int, the specific trial to analyze.
    - coord_set: str, the set of coordinates (e.g., 'P3').
    - axis_label: str, the axis label ('x', 'y', 'z').
    - spikes: np.ndarray, the indices of the detected spikes.
    """
    coord = f'{coord_set}_{axis_label}'
    trial_indices = df[df['Trial'] == trial].index

    for spike in spikes:
        if 0 < spike < (len(trial_indices) - 1):
            prev_index = trial_indices[spike - 1]
            next_index = trial_indices[spike + 1]
            spike_index = trial_indices[spike]
            df.at[spike_index, coord] = (
                df.at[prev_index, coord] + df.at[next_index, coord]) / 2


def fix_noisy_trials(df: pd.DataFrame, trials: List[int], coord_set: str, axis_labels: List[str], thresholds: List[float]):
    """
    Process noisy trials by finding and removing noise spikes.

    Parameters:
    - df: pandas DataFrame, the dataset containing the trials.
    - trials: List[int], the list of noisy trials to process.
    - coord_set: str, the set of coordinates (e.g., 'P3').
    - axis_labels: List[str], the list of axis labels ('x', 'y', 'z') to process.
    - thresholds: List[float], the thresholds for detecting spikes for each axis.
    """
    for trial in trials:
        for axis_label, threshold in zip(axis_labels, thresholds):
            spikes = find_noise_spikes(
                df, trial, coord_set, axis_label, threshold, visualize=False)
            valid_spikes = [spike for spike in spikes if 0 <
                            spike < df[df['Trial'] == trial].shape[0] - 1]
            remove_spikes_and_average(
                df, trial, coord_set, axis_label, valid_spikes)


def suggest_cutoff(trial_df: pd.DataFrame, coord_sets: List[str], axis_labels: List[str], threshold_values: Optional[dict] = None, last_percent_of_trial: float = 0.25) -> Optional[int]:
    """
    Suggest a cutoff point in a trial data frame based on the threshold values for specified coordinates.

    Parameters:
    - trial_df: pandas DataFrame, the dataframe of a single trial.
    - coord_sets: List[str], the list of coordinate sets to be checked (e.g., ['P3']).
    - axis_labels: List[str], the list of axis labels to be checked (e.g., ['Y', 'Z']).
    - threshold_values: Optional[dict], a dictionary of axis labels to their threshold values.
    - last_percent_of_trial: float, the last percentage of the trial to consider for finding a cutoff.

    Returns:
    - The suggested cutoff index, or None if no cutoff is suggested.
    """
    stride_length = len(trial_df)
    relevant_data_start = int(stride_length * (1 - last_percent_of_trial))

    potential_cutoffs_all = []
    for coord_set in coord_sets:
        for axis_label in axis_labels:
            threshold_value = threshold_values.get(axis_label.upper())
            coord_column = f'{coord_set}_{axis_label.lower()}'
            finite_diff = calculate_finite_difference(trial_df[coord_column])

            potential_cutoffs = np.where(
                finite_diff[relevant_data_start:] > threshold_value)[0]
            potential_cutoffs += relevant_data_start

            if potential_cutoffs.size > 0:
                potential_cutoffs_all.append(potential_cutoffs[0])

    return min(potential_cutoffs_all) if potential_cutoffs_all else None


def trim_trials_at_cutoff(dataframe: pd.DataFrame, coord_set: str, axis_labels: List[str], threshold_values: List[float], last_percent_of_trial: float = 0.25) -> pd.DataFrame:
    """
    Trim trials in the dataframe to the suggested cutoff point based on the threshold values for specified coordinates,
    to remove unnecessary data at the end of each trial.

    Parameters:
    - dataframe: pandas DataFrame, the original dataframe containing multiple trials.
    - coord_set: str, the coordinate set to be checked (e.g., 'P3').
    - axis_labels: List[str], the list of axis labels to be checked (e.g., ['y', 'z']).
    - threshold_values: List[float], the threshold values corresponding to each axis label.
    - last_percent_of_trial: float, the percentage of the trial data from the end to consider for finding a cutoff.

    Returns:
    - A trimmed dataframe with each trial cut off at the suggested point.
    """
    # Convert the list of threshold values to a dictionary mapping axis labels to their threshold values
    threshold_dict = {axis.upper(): value for axis,
                      value in zip(axis_labels, threshold_values)}

    trimmed_dataframes = []
    for trial in dataframe['Trial'].unique():
        trial_df = dataframe[dataframe['Trial'] == trial].copy()
        cutoff = suggest_cutoff(
            trial_df, [coord_set], axis_labels, threshold_dict, last_percent_of_trial)

        if cutoff is not None:
            # Include the cutoff point itself in the trimmed data
            trimmed_df = trial_df.iloc[:cutoff - 1]
        else:
            trimmed_df = trial_df

        trimmed_dataframes.append(trimmed_df)

    final_trimmed_df = pd.concat(trimmed_dataframes, ignore_index=True)
    return final_trimmed_df


def normalize_data(dataframe: pd.DataFrame):
    """
    Normalize the coordinate data within each trial to a 0-1 range.

    Parameters:
    - dataframe: pandas DataFrame containing the data with a 'Trial' column and coordinate columns.

    Returns:
    - A tuple containing:
        - A dataframe with normalized coordinate data.
        - A dictionary with normalization parameters (min and max values) for each trial and coordinate.
    """
    df_normalized = dataframe.copy()
    normalization_params = {}

    # Group by 'Trial' and normalize within each group
    for coord in COORDS:
        # Calculate min and max for each trial and coordinate
        min_vals = df_normalized.groupby('Trial')[coord].transform('min')
        max_vals = df_normalized.groupby('Trial')[coord].transform('max')

        # Apply normalization
        df_normalized[coord] = (df_normalized[coord] -
                                min_vals) / (max_vals - min_vals)

        # Store normalization parameters
        for trial in df_normalized['Trial'].unique():
            if trial not in normalization_params:
                normalization_params[trial] = {}
            normalization_params[trial][f"{coord}_min"] = min_vals[df_normalized['Trial'] == trial].iloc[0]
            normalization_params[trial][f"{coord}_max"] = max_vals[df_normalized['Trial'] == trial].iloc[0]

    return df_normalized, normalization_params


def preprocess_data(dataframe: pd.DataFrame):
    cache_filename = 'preprocessed_dataframe.pkl'
    cache_file = os.path.join(PATH_TO_CACHE, cache_filename)

    # Check if the preprocessed cache file exists, load and return the dataframe if so
    if os.path.exists(cache_file):
        #print("Loading preprocessed dataframe from cache")
        with open(cache_file, 'rb') as file:
            df, normalization_params = pickle.load(file)
            return df, normalization_params

    # If not cached, proceed with preprocessing
    df = dataframe.copy()
    df = interpolate(df)
    df = remove_bad_trials(df)
    df = translate_data_to_zero(df)
    df = truncate_trials_at_last_positive(df)
    fix_noisy_trials(df, [129, 130, 131, 132], 'P3', ['y', 'z'], [0.25, 0.8])
    df = trim_trials_at_cutoff(df, 'P3', ['y'], [0.24])
    df, normalization_params = normalize_data(df)

    # Save the preprocessed dataframe to cache
    if not os.path.exists(PATH_TO_CACHE):
        os.makedirs(PATH_TO_CACHE)
    with open(cache_file, 'wb') as file:
        pickle.dump((df, normalization_params), file)
        print(f"Preprocessed dataframe cached at {cache_file}")

    return df, normalization_params
    

def plot_histogram(intervals=[0.2, 0.4, 0.6, 0.8], coord='P3_y'):
    df = get_data()
    df = interpolate(df)
    df = remove_bad_trials(df)
    df = translate_data_to_zero(df)
    df = truncate_trials_at_last_positive(df)
    fix_noisy_trials(df, [129, 130, 131, 132], 'P3', ['y', 'z'], [0.25, 0.8])
    df = trim_trials_at_cutoff(df, 'P3', ['y'], [0.24])
    for t in df['Trial'].unique():
        trial = df[df['Trial'] == t]
        df.loc[df['Trial'] == t, 'Time'] = np.linspace(
            0, 100, len(trial['Time']))

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
    

def plot_preprocessing_steps_to_pdf(plot_trial_function, pdfPages, trial=31, pdf_filename='preprocessing_steps.pdf'):
    df = get_data()

    preprocessing_steps = [
        ("Raw Data", lambda x: x),
        ("Interpolation", lambda x: interpolate(x)),
        # ("Removing Bad Trials", lambda x: remove_bad_trials(x)),
        ("Translation to Zero", lambda x: translate_data_to_zero(x)),
        ("Truncation at Last Positive", lambda x: truncate_trials_at_last_positive(x)),
        # ("Fix Noise", lambda x: fix_noisy_trials(x, [trial], 'P3', ['y', 'z'], [0.25, 0.8])),
        ("Trial Trimming", lambda x: trim_trials_at_cutoff(
            x, 'P3', ['y'], [0.24])),
        ("Normalization", lambda x: normalize_data(x)[0])
    ]
    # Create a figure with subplots for each preprocessing step
    fig, axes = plt.subplots(
        nrows=len(preprocessing_steps), ncols=2, figsize=(25, 30))

    processed_df = df[df['Trial'] == trial].copy()

    for i, (step_name, step_func) in enumerate(preprocessing_steps):
        # Apply the preprocessing step
        if step_name == 'Fix Noise':
            step_func(processed_df)
        else:
            processed_df = step_func(processed_df.copy())
        # Plot the trial for this preprocessing step
        for j, coord in enumerate(['P3_y', 'P3_z']):
            ax = axes[i, j]
            plot_trial_function(processed_df, trial, axs=ax, plot_directly=False,
                       labels=[coord], fontsize=15, legend=False)

            # Set individual titles for each subplot
            health = "Laminitic" if processed_df.iloc[0]['isLaminitic'] else "Healthy"
            shoe = processed_df.iloc[0]['Shoe']
            ax.set_title(
                f"Preprocessing: {step_name} - {coord}\nTrial {trial} - {health}, {shoe} shoe", fontsize=20)

    plt.tight_layout()

    pdf_path = f'../Media/{pdf_filename}'
    with pdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Preprocessing steps figure saved as {pdf_path}")
# Confidence Intervals #

def scale_data_back(data, norm_params, coord):
    """
    Scales normalized data back to original scale.
    
    Parameters:
    - data: Normalized data.
    - norm_params: Normalization parameters.
    - coord: Coordinate (e.g., 'P3_y' or 'P3_z').
    
    Returns:
    - Scaled data.
    """
    return data * (norm_params[f'{coord}_max'] - norm_params[f'{coord}_min']) + norm_params[f'{coord}_min']


def denormalize_data(df, normalization_params):
    """
    Denormalizes the data for each trial using the stored normalization parameters.
    
    Parameters:
    - df: The dataframe to be denormalized.
    - normalization_params: The normalization parameters stored during preprocessing.
    
    Returns:
    - A denormalized dataframe.
    """
    df_denormalized = df.copy()
    for trial in df['Trial'].unique():
        for coord in COORDS:
            if f'{coord}_max' in normalization_params[trial] and f'{coord}_min' in normalization_params[trial]:
                df_denormalized.loc[df_denormalized['Trial'] == trial, coord] = scale_data_back(
                    df_denormalized[df_denormalized['Trial'] == trial][coord],
                    normalization_params[trial], coord)
            else:
                print(
                    f"Missing normalization parameters for {coord} in trial {trial}.")
    return df_denormalized

def resample_array(original_indices, original_values, target_length):
    new_indices = np.linspace(0, 1, target_length)
    interpolator = interp1d(original_indices, original_values, kind='linear', fill_value='extrapolate')
    return interpolator(new_indices)

def resample_trials(df, columns_to_resample, n_samples):
    """
    Resamples each trial in the dataframe to have a uniform number of samples.
    
    Parameters:
    - df: The dataframe to resample.
    - n_samples: The number of samples to resample each trial to.
    
    Returns:
    - A resampled dataframe with uniform trial lengths.
    """
    resampled_trials = pd.DataFrame(
        columns=['Trial'] + columns_to_resample)

    trials = df['Trial'].unique()

    for trial in trials:
        trial_df = df[df['Trial'] == trial]
        trial_indices = np.linspace(0, 1, len(trial_df))
        new_indices = np.linspace(0, 1, n_samples)
        resampled_data = {'Trial': [trial] * n_samples}
        for column in columns_to_resample:
            interpolator = interp1d(
                trial_indices, trial_df[column], kind='linear', bounds_error=False, fill_value="extrapolate")
            resampled_data[column] = interpolator(new_indices)

        resampled_trial_df = pd.DataFrame(resampled_data)

        resampled_trials = pd.concat(
            [resampled_trials, resampled_trial_df], ignore_index=True)

    return resampled_trials


def calculate_confidence_intervals(df):
    """
    Calculates the confidence intervals for both P3_y and P3_z for each unique combination of 'isLaminitic' and 'Shoe'.
    It performs resampling internally to ensure uniform data points across trials.
    
    Parameters:
    - df: The preprocessed dataframe.
    
    Returns:
    - A dictionary with keys as tuples of (isLaminitic, Shoe) and values as dictionaries containing CI data for P3_y and P3_z.
    """
    results = {}
    combinations = df[['isLaminitic', 'Shoe']].drop_duplicates()

    for _, combo in combinations.iterrows():
        isLaminitic, shoe = combo['isLaminitic'], combo['Shoe']
        combo_df = df[(df['isLaminitic'] == isLaminitic)
                      & (df['Shoe'] == shoe)]
        combo_df = resample_trials(
            combo_df, ['P3_y', 'P3_z'], n_samples=combo_df.groupby('Trial').size().max())

        max_length = combo_df['Trial'].value_counts().max()

        ci_data = {'P3_y': [], 'P3_z': []}
        confidence_level = 0.95  # Define the confidence level here
        # Get the Z-score for the confidence level
        z_score = stats.norm.interval(confidence_level)[1]

        for column in ['P3_y', 'P3_z']:
            for count in range(max_length):
                point_values = combo_df.groupby('Trial').nth(count)
                if not point_values.empty:
                    mean = point_values[column].mean()
                    std = point_values[column].std(ddof=1)
                    n = len(point_values)
                    se = std / np.sqrt(n)

                    # Calculate CI using the Z-score from stats.norm.interval
                    ci_lower = mean - z_score * se
                    ci_upper = mean + z_score * se

                    ci_data[column].append({
                        'mean': mean,
                        'lower_ci': ci_lower,
                        'upper_ci': ci_upper,
                        'n': n
                    })
        results[(isLaminitic, shoe)] = ci_data
    return results


def save_confidence_intervals_to_json(results, filename="confidence_intervals.json"):
    # Convert tuple keys to strings
    results_str_keys = {str(key): value for key, value in results.items()}
    with open(f"../Data/Cache/{filename}", "w") as f:
        json.dump(results_str_keys, f, indent=4)


def load_confidence_intervals_from_json(filename="confidence_intervals.json"):
    try:
        with open(f"../Data/Cache/{filename}", "r") as f:
            results_str_keys = json.load(f)
        # Convert string keys back to tuples
        results_tuple_keys = {
            tuple(eval(key)): value for key, value in results_str_keys.items()}
        return results_tuple_keys
    except FileNotFoundError:
        # print("File not found. Ensure the confidence intervals have been generated and saved.")
        return None
    except json.JSONDecodeError:
        # print("Error decoding the JSON file.")
        return None


def get_confidence_intervals(filename='confidence_intervals.json'):
    # Attempt to load the confidence intervals from a JSON file
    results = load_confidence_intervals_from_json(filename)
    # If results are None, it means the file doesn't exist or an error occurred while reading the file
    if results is None:
        df, norm_params = preprocess_data(get_data())
        df = denormalize_data(df, norm_params)
        results = calculate_confidence_intervals(df)
        save_confidence_intervals_to_json(results, filename)
    return results
