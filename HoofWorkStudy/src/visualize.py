
from graphviz import Digraph
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from collections import defaultdict

plt.style.use('default')
plt.rcParams.update({'font.size': 14})

'''
Visualization
'''

    # return interval_values

def plot_derivative_with_threshold(df, trial, coord_sets, axis_label, threshold_value, suggest_cutoff_function, last_percent_of_trial=0.25):
    '''
    Visualizes the first derivative of specified coordinates for a given trial and highlights the suggested cutoff point.
    
    Parameters:
    - df: DataFrame containing trial data.
    - trial: The specific trial to visualize.
    - coord_sets: List of coordinate sets to visualize (e.g., ['DW', 'SM', 'CB', 'P3']).
    - axis_label: The axis label for which to calculate the derivative ('x', 'y', 'z').
    - threshold_value: The threshold value for suggesting a cutoff.
    - last_percent_of_trial: The last percentage of the trial to consider for finding a cutoff.

    Returns:
    - The index of the earliest suggested cutoff, or None if no cutoff is suggested.
    '''
    plt.figure(figsize=(10, 4))
    trial_df = df[df['Trial'] == trial].copy()
    stride_length = len(trial_df)

    # Adjusting for the new suggest_cutoff signature
    threshold_values = {axis_label.upper(): threshold_value}
    suggested_cutoff_index = suggest_cutoff_function(
        trial_df, coord_sets, [axis_label], threshold_values, last_percent_of_trial)

    plt.axvspan(100 - last_percent_of_trial * 100, 100,
                color='grey', alpha=0.2, label='Last part of trial')

    for coord_set in coord_sets:
        coord_column = f'{coord_set}_{axis_label.lower()}'
        trial_df[f'First_Derivative_{coord_set}'] = np.abs(
            np.diff(trial_df[coord_column], prepend=np.nan))

        plt.plot(np.linspace(0, 100, stride_length), trial_df[f'First_Derivative_{coord_set}'],
                 label=f'First Derivative {coord_set}')

    if suggested_cutoff_index is not None:
        cutoff_percentage = 100 * suggested_cutoff_index / (stride_length - 1)
        plt.scatter(cutoff_percentage, trial_df.iloc[suggested_cutoff_index][f'First_Derivative_{coord_sets[0]}'],
                    color='red', zorder=5, label='Earliest Suggested Cutoff')

    plt.title(
        f'First Derivative and Suggested Cutoff for {axis_label.upper()} Axis')
    plt.xlabel('Stride %')
    plt.ylabel('First Derivative')
    plt.legend()
    plt.show()

    return suggested_cutoff_index - 1




def plot_model(
        NeuralNetworkObject,
        input_features=['DW_y', 'DW_z', 'SM_y', 'SM_z', 'CB_y', 'CB_z'],
        output_features=['P3_y', 'P3_z'],
        hidden_neurons=32, 
        filename='neural_net_graph'):
    '''
    Visualizes the structure of the neural network
    '''
    model = NeuralNetworkObject(len(input_features), len(output_features), hidden_neurons)
    dot = Digraph()
    dot.node('X', label=f'Inputs = {input_features}', shape='box')
    layers = [str(layer) for layer in model.children()]
    for i, layer in enumerate(layers):
        dot.node(str(i), label=layer)
        if i == 0:
            dot.edge('X', str(i))
        else:
            dot.edge(str(i-1), str(i))
    dot.node('Output', label=f'Outputs = {output_features}', shape='box')
    dot.edge(str(len(layers)-1), 'Output')
    dot.render(f'../Media/{filename}', format='png', view=False)



def plot_trial(df, trial, axs=None, plot_directly=True, labels=('P3_y', 'P3_z'), title="auto", xlabel="Stride (%)", ylabel="Displacement (mm)", legend=True, fontsize=12):
    """
    Plots specified coordinates for a given trial, with options for direct plotting and customization.

    Parameters:
    - df: DataFrame containing the trial data.
    - trial: The trial number to plot.
    - axs: Optional tuple of matplotlib axes objects for the specified coordinates. If None, new axes are created.
    - plot_directly: If True, plots directly to a new figure. If False, uses provided axs.
    - labels: Tuple of labels for the plots, corresponding to dataframe columns.
    - title: Title for the plot. If 'auto', generates a default title based on trial data. If None, no title is set.
    - xlabel, ylabel: Labels for the x and y axes.
    - legend: Whether to display a legend.
    - fontsize: Font size for text elements.
    """
    t = df[df['Trial'] == trial].copy()
    t['Time'] = np.linspace(0, 100, len(t))

    if axs is None:
        fig, axs = plt.subplots(1, len(labels), figsize=(15, 5))
        created_axes = True
    else:
        created_axes = False
        if not isinstance(axs, (list, tuple)):
            axs = [axs]

    # Determine the title based on the 'auto' setting or the provided title
    determined_title = title
    if title == "auto":
        health = "Laminitic" if t.iloc[0]['isLaminitic'] else "Healthy"
        shoe = t.iloc[0]['Shoe']
        determined_title = f"P3_y/z Coordinates for trial {trial}\n{health}, {shoe} shoe"

    for ax, label in zip(axs, labels):
        ax.plot(t['Time'], t[label], label=label)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        if legend:
            ax.legend(fontsize=fontsize)

    # If this function created its own axes and is directed to plot immediately, then show the plot
    if created_axes and plot_directly:
        if determined_title is not None:
            plt.suptitle(determined_title, fontsize=fontsize + 2)
        plt.tight_layout()
        plt.show()


def plot_coords_to_pdf(dataframe: pd.DataFrame, coord_sets=['DW', 'SM', 'CB', 'P3'], colors=['red', 'green', 'blue', 'purple'], dimensions=['Y', 'Z'], suggest_cutoff_function=None, pdf_name='all_coords.pdf'):
    assert len(coord_sets) == len(
        colors), "Number of coord_sets and colors must be equal"

    df = dataframe.copy()
    unique_trials = df['Trial'].unique()

    with PdfPages(pdf_name) as pdf:
        for trial in unique_trials:
            fig, axes = plt.subplots(1, len(dimensions), figsize=(15, 5))
            trial_df = df[df['Trial'] == trial].reset_index(drop=True)
            is_laminitic = 'Laminitic' if trial_df['isLaminitic'].iloc[0] else 'Healthy'
            shoe = trial_df['Shoe'].iloc[0]
            title = f'Trial {trial}, {is_laminitic}, Shoe: {shoe}'

            cutoff_index = None
            if suggest_cutoff_function is not None:
                cutoff_index = suggest_cutoff_function(
                    trial_df, ['P3'], ['Y'], threshold_values={'Y': 0.24})
                if cutoff_index is not None and cutoff_index < len(trial_df):
                    cutoff_time_percentage = cutoff_index / \
                        (len(trial_df) - 1) * 100

            for i, axis_label in enumerate(dimensions):
                stride_percentage = np.linspace(0, 100, len(trial_df))
                for coord_set, color in zip(coord_sets, colors):
                    axes[i].plot(stride_percentage, trial_df[f'{coord_set}_{axis_label.lower()}'],
                                 label=f'{coord_set}_{axis_label}', color=color)
                    axes[i].set_xlabel('Stride %')
                    axes[i].set_ylabel(f'{axis_label}')
                    if suggest_cutoff_function is not None:
                        axes[i].axvspan(75, 100, color='grey', alpha=0.2)
                        if cutoff_index is not None:
                            cutoff_time_percentage = cutoff_index / \
                                (len(trial_df) - 1) * 100
                            for i, axis_label in enumerate(dimensions):
                                cutoff_y_value = trial_df.loc[cutoff_index,
                                                              f'{coord_set}_{axis_label.lower()}']
                                axes[i].plot(cutoff_time_percentage, cutoff_y_value,
                                             'ko', label='Suggested Cutoff', markersize=10)
                                axes[i].legend()

            fig.suptitle(title)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"All coordinates figure saved as {pdf_name}")




def plot_testing_set(y_test, test_args, trial_lengths, normalization_params, scaling_function):
    y_test_global = y_test.detach().numpy()

    for i, (trial, isLaminitic, shoe) in enumerate(test_args):
        start_index, end_index = trial_lengths[trial]

        y_test_trial = y_test_global[start_index:end_index]
        y = scaling_function(y_test_trial[:, 0], normalization_params[trial], 'P3_y')
        z = scaling_function(y_test_trial[:, 1], normalization_params[trial], 'P3_z')

        # Create a DataFrame that mimics the expected structure for plot_trial
        data_for_plotting = pd.DataFrame({
            'Trial': [trial] * len(y),
            'isLaminitic': [isLaminitic] * len(y),
            'Shoe': [shoe] * len(y),
            'Time': np.linspace(0, 100, len(y)),
            'P3_y': y,
            'P3_z': z
        })

        plot_trial(data_for_plotting, trial,
                   plot_directly=True, labels=['P3_y', 'P3_z'])

def plot_confidence_intervals_to_pdf(results):
    """
    Plots the confidence intervals for P3_y and P3_z and saves them to a PDF file.
    
    Parameters:
    - results: The confidence interval data as returned by get_confidence_intervals.
    """
    with PdfPages('../Media/confidence_interval_plots.pdf') as pdf:
        for (isLaminitic, shoe), data in results.items():
            for column in ['P3_y', 'P3_z']:
                ci_data = data[column]
                means = [d['mean'] for d in ci_data]
                lower_cis = [d['lower_ci'] for d in ci_data]
                upper_cis = [d['upper_ci'] for d in ci_data]

                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.linspace(0, 100, len(means))
                ax.fill_between(x, lower_cis, upper_cis, color='skyblue',
                                alpha=0.4, label='95% Confidence Interval')
                ax.plot(x, means, color='blue', label='Mean')
                ax.set_title(
                    f'{column} Confidence Interval & Mean - isLaminitic={isLaminitic}, Shoe={shoe}')
                ax.set_ylabel('Displacement (mm)')
                ax.set_xlabel('Stride (%)')
                ax.legend()

                plt.tight_layout()
                pdf.savefig()
                plt.close()


def plot_condition_trials_with_ci(dataframe, ci_results, resample_function):
    unique_conditions = dataframe[['isLaminitic', 'Shoe']].drop_duplicates()

    for index, condition in unique_conditions.iterrows():
        health_condition = 'Laminitic' if condition['isLaminitic'] else 'Healthy'
        shoe_condition = condition['Shoe']
        filename = f"{shoe_condition}_{health_condition}_with_CI.pdf"

        condition_df = dataframe[(dataframe['isLaminitic'] == condition['isLaminitic']) &
                                 (dataframe['Shoe'] == condition['Shoe'])]

        with PdfPages(f"../Media/{filename}") as pdf:
            for trial in condition_df['Trial'].unique():
                trial_df = condition_df[condition_df['Trial'] == trial]
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))

                for ax, coord in zip(axs, ['P3_y', 'P3_z']):
                    # Assume we have a function that can plot the actual curve
                    plot_actual_curve(ax, trial_df, coord)

                    # Resample CI data to match the length of trial_df for this coord
                    ci_data = ci_results.get(
                        (condition['isLaminitic'], condition['Shoe']), {}).get(coord, [])
                    if ci_data:
                        target_length = len(trial_df)
                        times = np.linspace(0, 100, target_length)
                        original_indices = np.linspace(0, 1, len(ci_data))
                        ci_lower_resampled = resample_function(
                            original_indices, [ci['lower_ci'] for ci in ci_data], target_length)
                        ci_upper_resampled = resample_function(
                            original_indices, [ci['upper_ci'] for ci in ci_data], target_length)

                        # Overlay CI
                        ax.fill_between(times, ci_lower_resampled,
                                        ci_upper_resampled, color='grey', alpha=0.5)

                fig.suptitle(
                    f"Trial {trial} - {health_condition}, {shoe_condition}")
                pdf.savefig(fig)
                plt.close(fig)


def plot_combined_trials_with_ci_and_std(dataframe, ci_results, resample_function):
    unique_conditions = dataframe[['isLaminitic', 'Shoe']].drop_duplicates()
    pdf_filename = "../Media/combined_conditions_with_CI_and_STD.pdf"

    with PdfPages(pdf_filename) as pdf:
        for _, condition in unique_conditions.iterrows():
            health_condition = 'Laminitic' if condition['isLaminitic'] else 'Healthy'
            shoe_condition = condition['Shoe']

            fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
            fig.suptitle(f"{health_condition}, {shoe_condition}")

            condition_df = dataframe[(dataframe['isLaminitic'] == condition['isLaminitic']) &
                                     (dataframe['Shoe'] == condition['Shoe'])]

            for ax, coord in zip(axs, ['P3_y', 'P3_z']):
                # Plot all trial curves in muted color without adding them to the legend
                for trial in condition_df['Trial'].unique():
                    trial_df = condition_df[condition_df['Trial'] == trial]
                    plot_actual_curve(ax, trial_df, coord,
                                      use_legend=False, alpha=0.2)

                # Plot CI and mean curve
                ci_data = ci_results.get(
                    (condition['isLaminitic'], condition['Shoe']), {}).get(coord, [])
                if ci_data:
                    target_length = max(len(trial_df)
                                        for trial in condition_df['Trial'].unique())
                    times = np.linspace(0, 100, target_length)
                    means = [ci['mean'] for ci in ci_data]
                    ci_lower = [ci['lower_ci'] for ci in ci_data]
                    ci_upper = [ci['upper_ci'] for ci in ci_data]
                    std_dev = [ci['std'] for ci in ci_data]

                    # Resample for plotting
                    resampled_means = resample_function(
                        np.linspace(0, 1, len(means)), means, target_length)
                    resampled_ci_lower = resample_function(np.linspace(
                        0, 1, len(ci_lower)), ci_lower, target_length)
                    resampled_ci_upper = resample_function(np.linspace(
                        0, 1, len(ci_upper)), ci_upper, target_length)
                    resampled_std = resample_function(np.linspace(
                        0, 1, len(std_dev)), std_dev, target_length)

                    resampled_std_lower = resampled_means - resampled_std
                    resampled_std_upper = resampled_means + resampled_std

                    # Plotting with emphasis on mean, CI, and STD
                    ax.plot(times, resampled_means, label='Mean',
                            color='black', linewidth=2, linestyle='--')
                    ax.fill_between(times, resampled_std_lower, resampled_std_upper,
                                    color='salmon', alpha=0.5, label='1 STD')
                    ax.fill_between(times, resampled_ci_lower, resampled_ci_upper,
                                    color='skyblue', alpha=1, label='95% CI')

            # Adjust the legend to only show unique descriptors
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axs[0].legend(by_label.values(),
                          by_label.keys(), loc='upper right')
            axs[1].legend(by_label.values(),
                          by_label.keys(), loc='upper right')

            pdf.savefig(fig)
            plt.close(fig)


def plot_actual_curve(ax, trial_df, coord, use_legend=True, **kwargs):
    """
    Placeholder for function to plot the actual data curve for a given coordinate.
    Replace this with the actual logic to plot the curve based on 'trial_df' and 'coord'.
    """
    # Example plotting logic
    label = coord if use_legend else None

    times = np.linspace(0, 100, len(trial_df))
    ax.plot(times, trial_df[coord], label=label, **kwargs)
    ax.set_xlabel("Stride (%)")
    ax.set_ylabel(coord)
    if use_legend:
        ax.legend()

    
def print_model_accuracy(model, X_test, y_test):
    predictions = model(X_test)
    y_test_global = y_test.detach().numpy()
    predictions_global = predictions.detach().numpy()
    print(f"Accuracy of Predictions for P3 Model")
    print("="*50)
    r2 = r2_score(y_test_global, predictions_global)
    mse = mean_squared_error(y_test_global, predictions_global)
    print(f"R² = {r2:.4f}")
    print(f"MSE = {mse:.4f}")
    print("="*50)


def plot_prediction_vs_actual(ax, times, actual, predicted, ci_lower, ci_upper, ylabel):
    """
    Plots the predicted vs actual values for a given coordinate (P3_y or P3_z) including confidence intervals.

    Parameters:
    - ax: Matplotlib axis object to plot on.
    - times: Time percentage for the stride.
    - actual: Actual values for the coordinate.
    - predicted: Predicted values for the coordinate.
    - ci_lower: Lower bound of the confidence interval.
    - ci_upper: Upper bound of the confidence interval.
    - ylabel: Label for the y-axis.
    """
    ax.fill_between(times, ci_lower, ci_upper,
                    color='lightblue', alpha=0.5, label='95% CI')
    ax.plot(times, actual, 'k-', label='Actual', linewidth=2)
    ax.plot(times, predicted, 'r--', label='Predicted', linewidth=2)
    ax.set_xlabel('Stride (%)')
    ax.set_ylabel(ylabel)
    ax.legend()


def get_yz(data, trial, normalization_params):
    Y, Z = [], []
    norm_params = normalization_params[trial]

    for y, z in data:
        Y.append(
            y * (norm_params['P3_y_max'] - norm_params['P3_y_min']) + norm_params['P3_y_min'])
        Z.append(
            z * (norm_params['P3_z_max'] - norm_params['P3_z_min']) + norm_params['P3_z_min'])

    return np.array(Y), np.array(Z)


def plot_model_results_to_pdf(model, X_test, y_test, test_args, trial_lengths, normalization_params, ci_function, resample_function, pdf_filename='model_results.pdf'):
    model.eval()

    print_model_accuracy(model, X_test, y_test)
    predictions = model(X_test).detach().numpy()
    true = y_test.detach().numpy()

    cis = ci_function()

    r2_scores_y = defaultdict(list)
    mse_scores_y = defaultdict(list)
    r2_scores_z = defaultdict(list)
    mse_scores_z = defaultdict(list)

    with PdfPages(f"../Media/{pdf_filename}") as pdf:
        for i, (trial, isLaminitic, shoe) in enumerate(test_args):

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            start, end = trial_lengths[trial]

            true_y_curve, true_z_curve = get_yz(
                true, trial, normalization_params)
            predicted_y_curve, predicted_z_curve = get_yz(
                predictions, trial, normalization_params)

            true_y_curve = true_y_curve[start:end]
            true_z_curve = true_z_curve[start:end]
            predicted_y_curve = predicted_y_curve[start:end]
            predicted_z_curve = predicted_z_curve[start:end]

            ci_y = cis[isLaminitic, shoe]['P3_y']
            ci_z = cis[isLaminitic, shoe]['P3_z']

            # Prepare resampling of CI data to match the length of true/predicted curves
            # Assuming len(true_y_curve) == len(true_z_curve)
            target_length = len(true_y_curve)

            # Resample CI data
            # Assuming ci_y and ci_z have the same length
            original_indices = np.linspace(0, 1, len(ci_y))
            ci_y_lower_resampled = resample_function(
                original_indices, [ci['lower_ci'] for ci in ci_y], target_length)
            ci_y_upper_resampled = resample_function(
                original_indices, [ci['upper_ci'] for ci in ci_y], target_length)
            ci_z_lower_resampled = resample_function(
                original_indices, [ci['lower_ci'] for ci in ci_z], target_length)
            ci_z_upper_resampled = resample_function(
                original_indices, [ci['upper_ci'] for ci in ci_z], target_length)

            time = np.linspace(0, 100, target_length)

            plot_prediction_vs_actual(axs[0], time, true_y_curve, predicted_y_curve,
                                      ci_y_lower_resampled, ci_y_upper_resampled, 'P3_y')
            plot_prediction_vs_actual(axs[1], time, true_z_curve, predicted_z_curve,
                                      ci_z_lower_resampled, ci_z_upper_resampled, 'P3_z')

            fig.suptitle(
                f'Trial {trial}: {shoe}, {"Laminitic" if isLaminitic else "Healthy"}')
            pdf.savefig(fig)
            plt.close(fig)

            # Calculate and collect R² and MSE scores for later box plotting
            category = f"{isLaminitic}_{shoe}"
            r2_scores_y[category].append(
                r2_score(true_y_curve, predicted_y_curve))
            mse_scores_y[category].append(
                mean_squared_error(true_y_curve, predicted_y_curve))
            r2_scores_z[category].append(
                r2_score(true_z_curve, predicted_z_curve))
            mse_scores_z[category].append(
                mean_squared_error(true_z_curve, predicted_z_curve))

    label_mapping = {
        'True_Standard': 'SD_H', 'True_Unshod': 'US_H', 'True_Heartbar': 'HB_H', 'True_Eggbar': 'EB_H',
        'False_Standard': 'SD_L', 'False_Unshod': 'US_L', 'False_Heartbar': 'HB_L', 'False_Eggbar': 'EB_L'
    }

    with PdfPages(f"../Media/{pdf_filename[:-4]}_boxplot.pdf") as pdf:

        fig, axs = plt.subplots(2, 2, figsize=(14, 12))

        for i, (score_dict, title) in enumerate(zip(
            [r2_scores_y, mse_scores_y, r2_scores_z, mse_scores_z],
                ['R² Scores for P3_y', 'MSE Scores for P3_y', 'R² Scores for P3_z', 'MSE Scores for P3_z'])):

            ax = axs[i // 2, i % 2]
            data = []
            labels = []

            # Preparing data and labels for boxplot
            for key, values in score_dict.items():
                # Convert 'True/False_ShoeType' back to 'SD_H' format using label_mapping
                mapped_label = label_mapping[key]
                if values:  # If there are data values for the category
                    data.append(values)  # Append the scores
                    labels.append(mapped_label)  # Append the label

            # Plotting boxplot with corresponding data and labels
            ax.boxplot(data, labels=labels, patch_artist=True,
                       whiskerprops={'linestyle': '-',
                                     'linewidth': 1.5, 'color': 'darkorange'},
                       boxprops={'facecolor': 'skyblue', 'color': 'darkblue'},
                       medianprops={'linestyle': '-',
                                    'linewidth': 1.5, 'color': 'darkred'},
                       flierprops={'marker': 'o', 'markerfacecolor': 'green', 'markersize': 5, 'linestyle': 'none'})
            ax.set_title(title)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
