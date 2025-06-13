import os
import json
import pickle
import tensorflow as tf
import re
import shutil
from datetime import datetime

# Relative imports
from .logging_utils import setup_logger # From the same utils package
from ..configs import main_config as config # From the configs package one level up


logger = setup_logger(__name__, log_file=config.APPLICATION_LOG_FILE)

def create_project_directories():

    logger.info("Checking/creating necessary directories for the project...")
    dirs_to_create = [
        config.DATA_PATH, config.FER2013_DIR, config.RAVDESS_DATA_PATH,
        config.TRAINED_MODELS_PATH, config.LOGS_PATH, config.PLOTS_PATH,
        os.path.join(config.LOGS_PATH, "fer"), # Specific log subfolders for TensorBoard
        os.path.join(config.LOGS_PATH, "ser"),
    ]
    # Add the directory of the main application log file
    app_log_dir = os.path.dirname(config.APPLICATION_LOG_FILE)
    if app_log_dir: # If APPLICATION_LOG_FILE is a path
        dirs_to_create.append(app_log_dir)


    for dir_path in dirs_to_create:
        if not dir_path:
            logger.warning(f"Skipped an empty directory path: '{dir_path}'")
            continue
        try:
            os.makedirs(dir_path, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create directory: {dir_path} - Error: {e}")
    logger.info("Checking/creating necessary directories completed.")


def save_model_and_history(model, history_data, model_name_prefix):

    model_save_dir = os.path.join(config.TRAINED_MODELS_PATH, model_name_prefix)
    try:
        os.makedirs(model_save_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create model save directory ({model_save_dir}): {e}")
        return False # Save failed

    model_path_h5 = os.path.join(model_save_dir, f"{model_name_prefix}_model.h5")
    history_path_pkl = os.path.join(model_save_dir, f"{model_name_prefix}_history.pkl")
    architecture_path_json = os.path.join(model_save_dir, f"{model_name_prefix}_architecture.json")

    success_flags = {"model": False, "history": False, "architecture": False}

    # Save the model in H5 format
    try:
        model.save(model_path_h5)
        logger.info(f"Model successfully saved: {model_path_h5}")
        success_flags["model"] = True
    except Exception as e:
        logger.error(f"Error saving model as H5 ({model_path_h5}): {e}")

    # Save training history with pickle
    if history_data and isinstance(history_data, dict) and history_data:
        try:
            with open(history_path_pkl, 'wb') as f:
                pickle.dump(history_data, f)
            logger.info(f"Training history successfully saved: {history_path_pkl}")
            success_flags["history"] = True
        except Exception as e:
            logger.error(f"Error saving training history with pickle ({history_path_pkl}): {e}", exc_info=True)
    else:
        logger.warning(f"Invalid or empty training history dictionary (history_data). History not saved: {model_name_prefix}. Type: {type(history_data)}, Content (keys): {list(history_data.keys()) if isinstance(history_data, dict) else 'Not a Dict'}")

    # Save model architecture as JSON
    try:
        model_json = model.to_json(indent=4) # Indent for more readable JSON
        with open(architecture_path_json, "w", encoding="utf-8") as json_file:
            json_file.write(model_json)
        logger.info(f"Model architecture successfully saved: {architecture_path_json}")
        success_flags["architecture"] = True
    except Exception as e:
        logger.error(f"Error saving model architecture as JSON ({architecture_path_json}): {e}")

    return all(success_flags.values()) # Returns True if all are successful


def load_trained_model(model_identifier_prefix, custom_objects=None):

    if not model_identifier_prefix:
        logger.error("Model prefix (folder name) not specified for loading model.")
        return None

    standard_model_filename = f"{model_identifier_prefix}_model.h5"
    model_dir_path = os.path.join(config.TRAINED_MODELS_PATH, model_identifier_prefix)
    model_path_h5 = os.path.join(model_dir_path, standard_model_filename)

    checkpoint_model_filename = "best_model_checkpoint.h5"
    checkpoint_model_path_h5 = os.path.join(model_dir_path, checkpoint_model_filename)

    if not os.path.exists(model_path_h5):
        logger.warning(f"Standard model file ({standard_model_filename}) not found: {model_path_h5}. Trying alternative checkpoint file...")
        if os.path.exists(checkpoint_model_path_h5):
            logger.info(f"Alternative checkpoint file ({checkpoint_model_filename}) found: {checkpoint_model_path_h5}. This file will be used.")
            model_path_h5 = checkpoint_model_path_h5
        else:
            logger.error(f"Neither standard model file ({standard_model_filename}) nor alternative checkpoint file ({checkpoint_model_filename}) found: {model_dir_path}")
            return None

    logger.info(f"Loading model: {model_path_h5}")
    try:
        model = tf.keras.models.load_model(model_path_h5, custom_objects=custom_objects, compile=True)
        logger.info(f"Model successfully loaded: {model_path_h5}")
        return model
    except Exception as e:
        logger.error(f"Error loading model ({model_path_h5}): {e}", exc_info=True)
        return None


def load_training_history(model_identifier_prefix):

    if not model_identifier_prefix:
        logger.error("Model prefix (folder name) not specified for loading history.")
        return None

    history_path_pkl = os.path.join(config.TRAINED_MODELS_PATH, model_identifier_prefix, f"{model_identifier_prefix}_history.pkl")

    if not os.path.exists(history_path_pkl):
        logger.warning(f"Training history file not found: {history_path_pkl}")
        return None

    try:
        with open(history_path_pkl, 'rb') as f:
            history_dict = pickle.load(f)
        logger.info(f"Training history successfully loaded: {history_path_pkl}")
        return history_dict
    except Exception as e:
        logger.error(f"Error loading training history ({history_path_pkl}): {e}")
        return None


def is_specific_model_version(model_name_str):

    if not model_name_str or not isinstance(model_name_str, str):
        return False
    match = re.search(r"_(\d{8}-\d{6})$", model_name_str)
    return bool(match)


def get_latest_model_directory_for_base(base_model_name):

    trained_models_dir = config.TRAINED_MODELS_PATH
    if not os.path.isdir(trained_models_dir):
        logger.warning(f"Trained models directory not found: {trained_models_dir}")
        return None

    candidate_folders = []
    for f_name in os.listdir(trained_models_dir):
        if os.path.isdir(os.path.join(trained_models_dir, f_name)) and f_name.startswith(base_model_name):
            parts = f_name.split('_')
            if len(parts) > 1:
                date_time_stamp = parts[-1]
                if len(date_time_stamp) == 15 and date_time_stamp[8] == '-':
                    candidate_folders.append(f_name)

    if not candidate_folders:
        logger.warning(f"No model folder in the correct format found under TRAINED_MODELS_PATH for '{base_model_name}'.")
        return None

    candidate_folders.sort(key=lambda x: x.split('_')[-1], reverse=True)

    latest_model_dir_name = candidate_folders[0]
    logger.info(f"Latest model folder found ({base_model_name}): {latest_model_dir_name}")
    return latest_model_dir_name