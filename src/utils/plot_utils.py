import os
import matplotlib.pyplot as plt
import numpy as np

# Relative imports
from .logging_utils import setup_logger
from ..configs import main_config as config

# Set up logger for this module
logger = setup_logger(__name__, log_file=config.APPLICATION_LOG_FILE)

def plot_training_history(history_dict, model_name_prefix, metrics_to_plot=None):
    """
    Plots the training and validation history for specified metrics and saves the plot.

    Args:
        history_dict (dict): A dictionary containing training history (e.g., from model.fit).
                             Expected keys include 'loss', 'accuracy', 'val_loss', 'val_accuracy'.
        model_name_prefix (str): A prefix for the model name, used for saving the plot.
        metrics_to_plot (list, optional): A list of metric names to plot (e.g., ['loss', 'accuracy']).
                                          If None, it defaults to 'loss' and 'accuracy' if present.
    """
    if not history_dict:
        logger.warning(f"Training history data is empty for plotting ({model_name_prefix}). No plot generated.")
        return

    if metrics_to_plot is None:
        metrics_to_plot = []
        if 'loss' in history_dict:
            metrics_to_plot.append('loss')
        if 'accuracy' in history_dict:
            metrics_to_plot.append('accuracy')

    if not metrics_to_plot:
        logger.warning(f"No known metrics found in history data to plot ({model_name_prefix}). No plot generated.")
        return

    plots_save_dir = os.path.join(config.PLOTS_PATH, model_name_prefix)
    try:
        os.makedirs(plots_save_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create plot save directory ({plots_save_dir}): {e}")
        return # If directory cannot be created, plot cannot be saved.

    num_metrics_to_plot = len(metrics_to_plot)
    if num_metrics_to_plot == 0:
        logger.warning(f"No metrics specified for plotting ({model_name_prefix}).")
        return

    plt.figure(figsize=(7 * num_metrics_to_plot, 6)) # Adjust plot size

    for i, metric_name in enumerate(metrics_to_plot):
        plt.subplot(1, num_metrics_to_plot, i + 1)

        # Training metric
        if metric_name in history_dict:
            train_metric_values = history_dict[metric_name]
            epochs_train = np.arange(len(train_metric_values))
            valid_indices_train = np.isfinite(train_metric_values)
            if np.any(valid_indices_train):
                plt.plot(epochs_train[valid_indices_train],
                         np.array(train_metric_values)[valid_indices_train],
                         label=f'Training {metric_name.capitalize()}', marker='o', linestyle='-')
            else:
                logger.warning(f"Training metric '{metric_name}' contains only NaN/inf values.")

        # Validation metric
        val_metric_name = f'val_{metric_name}'
        if val_metric_name in history_dict:
            val_metric_values = history_dict[val_metric_name]
            epochs_val = np.arange(len(val_metric_values))
            valid_indices_val = np.isfinite(val_metric_values)
            if np.any(valid_indices_val):
                plt.plot(epochs_val[valid_indices_val],
                         np.array(val_metric_values)[valid_indices_val],
                         label=f'Validation {metric_name.capitalize()}', marker='x', linestyle='--')
            else:
                logger.warning(f"Validation metric '{val_metric_name}' contains only NaN/inf values.")

        plt.title(f'{model_name_prefix}\nTraining & Validation {metric_name.capitalize()}', fontsize=12)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel(metric_name.capitalize(), fontsize=10)
        plt.legend(fontsize=9)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)


    plot_filename = f"{model_name_prefix}_training_history.png"
    plot_full_path = os.path.join(plots_save_dir, plot_filename)

    try:
        plt.tight_layout(pad=2.0)
        plt.savefig(plot_full_path, dpi=150)
        logger.info(f"Training history plot successfully saved: {plot_full_path}")
    except Exception as e:
        logger.error(f"Error saving plot file ({plot_full_path}): {e}")
    finally:
        plt.close()