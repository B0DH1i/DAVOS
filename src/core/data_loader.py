# src/core/data_loader.py
import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import collections # Added for emotion counting

# Relative imports
# utils is under src/utils
# configs is under src/configs
from ..utils.logging_utils import setup_logger
from ..configs import main_config as config

# try-except for audiomentations (optional dependency)
try:
    from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
    AUDIOMENTATIONS_AVAILABLE = True
except ImportError:
    # If audiomentations is not found, we will log a warning later,
    # once the logger is initialized.
    Compose = None # Placeholder to prevent errors
    AUDIOMENTATIONS_AVAILABLE = False

# Global variables for Whisper (to load the model and processor only once)
WHISPER_PROCESSOR = None
WHISPER_MODEL = None
TORCH_DEVICE = None # To determine the device (CPU/GPU)

# Setup logger for this module

logger = setup_logger(__name__, log_file=config.APPLICATION_LOG_FILE)

# Log if audiomentations failed to load (logger is now defined)
if not AUDIOMENTATIONS_AVAILABLE:
    logger.warning("audiomentations library not found. Data augmentation for SER will not be available.")

def _initialize_whisper():
    """Initializes the Whisper model and processor if they are not already loaded."""
    global WHISPER_PROCESSOR, WHISPER_MODEL, TORCH_DEVICE
    if WHISPER_PROCESSOR is None or WHISPER_MODEL is None:
        try:
            import torch
            from transformers import WhisperProcessor, WhisperModel

            TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"PyTorch device for Whisper: {TORCH_DEVICE}")

            logger.info(f"Loading Whisper processor: {config.WHISPER_MODEL_NAME}")
            WHISPER_PROCESSOR = WhisperProcessor.from_pretrained(config.WHISPER_MODEL_NAME)
            
            logger.info(f"Loading Whisper model: {config.WHISPER_MODEL_NAME}")
            WHISPER_MODEL = WhisperModel.from_pretrained(config.WHISPER_MODEL_NAME).to(TORCH_DEVICE)
            WHISPER_MODEL.eval() # Set the model to evaluation mode
            logger.info("Whisper model and processor loaded successfully.")

        except ImportError:
            logger.error("To use Whisper, 'transformers' and 'torch' libraries are required. Please install them.")
            raise
        except Exception as e:
            logger.error(f"Error loading Whisper model or processor: {e}")
            raise

def load_fer_data(prepared_data_path=None, img_size=None, num_classes=None):
    """
    Loads the prepared FERPlus dataset (PNG images and label.csv files).
    The data should be located under 'prepared_data_path' in 'FER2013Train', 
    'FER2013Valid', and 'FER2013Test' subfolders. Each subfolder must contain
    a 'label.csv' and the corresponding '.png' image files.

    Returns:
        tuple: ((X_train, y_train), (X_val, y_val), (X_test, y_test)) or None on error.
               X: Image data (NumPy array, [samples, height, width, 1])
               y: Labels (NumPy array, [samples, num_classes] - one-hot encoded)
    """
    if prepared_data_path is None:
        prepared_data_path = config.FERPLUS_PREPARED_DATA_PATH
    if img_size is None:
        img_size = config.FER_IMG_SIZE
    if num_classes is None:
        num_classes = config.NUM_MODEL_OUTPUT_CLASSES

    logger.info(f"Loading prepared FERPlus dataset from: {prepared_data_path}")
    if not os.path.exists(prepared_data_path):
        logger.error(f"FERPlus prepared data folder not found: {prepared_data_path}")
        return None

    # OpenCV will be required to load the images.
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV (cv2) library is required to load FERPlus images. Please install it.")
        return None

    data_splits = {"train": "FER2013Train", "validation": "FER2013Valid", "test": "FER2013Test"}
    X_data_map = {split: [] for split in data_splits.keys()}
    y_data_map = {split: [] for split in data_splits.keys()}
    
    emotion_columns = config.MODEL_OUTPUT_EMOTIONS

    for split_name, folder_name in data_splits.items():
        split_folder_path = os.path.join(prepared_data_path, folder_name)
        labels_csv_path = os.path.join(split_folder_path, "label.csv")

        if not os.path.exists(split_folder_path) or not os.path.exists(labels_csv_path):
            logger.warning(f"For FERPlus, '{split_name}' data split folder ({split_folder_path}) or label.csv not found. Skipping this split.")
            continue
        
        logger.info(f"Reading label.csv for '{split_name}' ({folder_name})")
        try:
            labels_df = pd.read_csv(labels_csv_path, header=None)
            if labels_df.shape[1] < 12:
                logger.error(f"File '{labels_csv_path}' has insufficient columns (expected at least 12, found {labels_df.shape[1]}). Skipping this split.")
                continue
        except Exception as e:
            logger.error(f"Error reading '{labels_csv_path}': {e}. Skipping this split.")
            continue

        for index, row in tqdm(labels_df.iterrows(), total=labels_df.shape[0], desc=f"Processing FERPlus '{split_name}'"):
            try:
                image_filename = row[0]
                image_path = os.path.join(split_folder_path, image_filename)

                if not os.path.exists(image_path):
                    logger.warning(f"Image file not found: {image_path}. Skipping row {index}.")
                    continue

                img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img_array is None:
                    logger.warning(f"Image could not be loaded (cv2.imread returned None): {image_path}. Skipping row {index}.")
                    continue
                
                resized_img = cv2.resize(img_array, img_size, interpolation=cv2.INTER_AREA)
                reshaped_img = np.expand_dims(resized_img, axis=-1)
                normalized_img = reshaped_img / 255.0

                all_ferplus_scores = row.iloc[2:12].values.astype(float)
                
                if len(all_ferplus_scores) != len(config.FERPLUS_LABEL_COLUMNS):
                    logger.warning(f"For row {index} ({image_filename}), expected {len(config.FERPLUS_LABEL_COLUMNS)} emotion scores but got {len(all_ferplus_scores)}. Skipping this row.")
                    continue

                ferplus_scores_series = pd.Series(all_ferplus_scores, index=config.FERPLUS_LABEL_COLUMNS)
                selected_emotion_scores = ferplus_scores_series[emotion_columns].values.astype(float)
                dominant_emotion_index = np.argmax(selected_emotion_scores)
                
                one_hot_label = np.zeros(num_classes)
                if 0 <= dominant_emotion_index < num_classes:
                    one_hot_label[dominant_emotion_index] = 1.0
                else:
                    logger.warning(f"Invalid dominant emotion index {dominant_emotion_index} (for {num_classes} classes). Image: {image_filename}. Skipping this row.")
                    continue
                
                X_data_map[split_name].append(normalized_img)
                y_data_map[split_name].append(one_hot_label)

            except Exception as e:
                logger.warning(f"Error processing row {index} ('{image_filename}' if defined, else 'Unknown'): {e}. Skipping this row.")

    X_train = np.array(X_data_map['train'])
    y_train = np.array(y_data_map['train'])
    X_val = np.array(X_data_map['validation'])
    y_val = np.array(y_data_map['validation'])
    X_test = np.array(X_data_map['test'])
    y_test = np.array(y_data_map['test'])

    logger.info("FERPlus Loading Complete:")
    if X_train.size > 0: logger.info(f"  Training set: Images {X_train.shape}, Labels {y_train.shape}")
    else: logger.warning("FERPlus training set is empty!")
    if X_val.size > 0: logger.info(f"  Validation set: Images {X_val.shape}, Labels {y_val.shape}")
    else: logger.warning("FERPlus validation set is empty!")
    if X_test.size > 0: logger.info(f"  Test set: Images {X_test.shape}, Labels {y_test.shape}")
    else: logger.warning("FERPlus test set is empty (this may be expected).")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def extract_audio_features_from_file(audio_path,
                                     sr_target=config.WHISPER_SAMPLING_RATE,
                                     apply_augmentation=False,
                                     augmenter=None,
                                     emotion_label_for_debug=None,
                                     actor_id_for_debug=None,
                                     data_path_for_debug=None):
    """
    Extracts Whisper embedding features from a single audio file.

    Args:
        audio_path (str): Path to the audio file.
        sr_target (int): Target sampling rate for Whisper.
        apply_augmentation (bool): Whether to apply audio augmentation.
        augmenter (callable, optional): Object to use for audio augmentation.
        ..._for_debug params are for debugging purposes.

    Returns:
        tuple (np.array, int) or (None, 0):
               - np.array: Whisper embedding vector of shape (config.WHISPER_EMBEDDING_DIM,).
               - int: Embedding dimension.
               On error, returns (None, 0).
    """
    original_feature_info = 0

    try:
        _initialize_whisper() # Initialize Whisper model if necessary
        if WHISPER_PROCESSOR is None or WHISPER_MODEL is None:
            logger.error("extract_audio_features_from_file: Whisper model/processor could not be loaded.")
            return None, original_feature_info

        y, sr_loaded = librosa.load(audio_path, sr=sr_target)

        if apply_augmentation and augmenter and AUDIOMENTATIONS_AVAILABLE:
            try:
                y = augmenter(samples=y, sample_rate=sr_loaded)
            except Exception as aug_e:
                logger.warning(f"Error during audio augmentation for '{audio_path}': {aug_e}. Using original data.")

        inputs = WHISPER_PROCESSOR(y, sampling_rate=sr_target, return_tensors="pt").to(TORCH_DEVICE)

        import torch
        with torch.no_grad():
            encoder_outputs = WHISPER_MODEL.encoder(inputs.input_features)
            last_hidden_state = encoder_outputs.last_hidden_state

        embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        if embedding.ndim == 0:
            logger.warning(f"Whisper embedding resulted in a scalar value (likely a very short audio). File: {audio_path}.")
            return None, 0

        if embedding.shape[0] != config.WHISPER_EMBEDDING_DIM:
            logger.warning(f"Whisper embedding dimension is unexpected! Expected: {config.WHISPER_EMBEDDING_DIM}, Got: {embedding.shape[0]}. File: {audio_path}")
            return None, 0

        original_feature_info = embedding.shape[0]
        return embedding, original_feature_info

    except Exception as e:
        logger.error(f"Error processing audio file ({audio_path}) in extract_audio_features_from_file: {e}", exc_info=True)
        return None, 0


def load_ravdess_data(data_path=None,
                      use_augmentation_on_train=config.USE_DATA_AUGMENTATION_SER,
                      target_sample_rate=config.WHISPER_SAMPLING_RATE,
                      target_actors=None,
                      specific_split_only=None):
    """
    Loads the RAVDESS dataset, extracts Whisper embedding features from audio files,
    and splits them into training, validation, and test sets.
    """
    if data_path is None:
        data_path = config.RAVDESS_DATA_PATH

    logger.info(f"Loading RAVDESS dataset from: {data_path}")
    if not os.path.exists(data_path) or not os.path.isdir(data_path):
        logger.error(f"RAVDESS data path not found or is not a directory: {data_path}")
        return None

    features_map = {'train': [], 'val': [], 'test': []}
    labels_map = {'train': [], 'val': [], 'test': []}
    emotion_counts_map = {
        'train': collections.Counter(), 
        'val': collections.Counter(), 
        'test': collections.Counter()
    }
    
    # Labels will be created according to the model's output format (e.g., 8 classes)
    model_output_emotion_to_idx = {emotion: i for i, emotion in enumerate(config.MODEL_OUTPUT_EMOTIONS)}
    num_model_output_classes = len(config.MODEL_OUTPUT_EMOTIONS)

    # Augmenter for audio data (to be applied only to the training set)
    ravdess_augmenter = None
    if use_augmentation_on_train and AUDIOMENTATIONS_AVAILABLE and Compose:
        ravdess_augmenter = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.85, max_rate=1.15, p=0.5, leave_length_unchanged=False),
            PitchShift(min_semitones=-3, max_semitones=3, p=0.5)
        ])
        logger.info("Audio augmentation enabled for RAVDESS training data (p=0.5).")

    actor_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and d.startswith("Actor_")]
    if not actor_folders:
        logger.error(f"No folders starting with 'Actor_' found in RAVDESS data path ({data_path}).")
        return None
    logger.info(f"Found {len(actor_folders)} actor folders. Processing...")

    for actor_folder in tqdm(actor_folders, desc="Processing RAVDESS Actors"):
        actor_path = os.path.join(data_path, actor_folder)
        wav_files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]

        for wav_file_name in list(wav_files):
            try:
                parts = wav_file_name.split('.')[0].split('-')
                if len(parts) < 7:
                    logger.warning(f"Filename ({wav_file_name}) is shorter than expected (7 parts). Skipping.")
                    continue
                
                emotion_code_from_file = parts[2]
                actor_id_from_file = int(parts[6])

                ravdess_emotion_text = config.RAVDESS_EMOTIONS_FROM_FILENAME.get(emotion_code_from_file)
                if not ravdess_emotion_text:
                    logger.warning(f"Filename ({wav_file_name}): Unknown RAVDESS emotion code '{emotion_code_from_file}'. Skipping.")
                    continue
                
                model_emotion_text = config.RAVDESS_TO_MODEL_OUTPUT_MAP.get(ravdess_emotion_text)
                if not model_emotion_text:
                    logger.warning(f"File ({wav_file_name}): No match for RAVDESS emotion '{ravdess_emotion_text}' in MODEL_OUTPUT_EMOTIONS map. Skipping.")
                    continue
                
                numeric_label = model_output_emotion_to_idx.get(model_emotion_text)
                if numeric_label is None:
                    logger.error(f"Critical: Numeric label for model output emotion '{model_emotion_text}' (from RAVDESS: {ravdess_emotion_text}) not found in config.MODEL_OUTPUT_EMOTIONS. Please check config. File: {wav_file_name}")
                    continue
                
                label_one_hot = np.zeros(num_model_output_classes)
                label_one_hot[numeric_label] = 1.0

                current_set_key = None
                apply_aug_for_this_file = False

                if actor_id_from_file in config.RAVDESS_TRAIN_ACTORS:
                    current_set_key = 'train'
                    apply_aug_for_this_file = use_augmentation_on_train
                elif actor_id_from_file in config.RAVDESS_VAL_ACTORS:
                    current_set_key = 'val'
                elif actor_id_from_file in config.RAVDESS_TEST_ACTORS:
                    current_set_key = 'test'
                else:
                    continue
                
                file_path_str = os.path.join(actor_path, wav_file_name)

                features, feature_info = extract_audio_features_from_file(
                    file_path_str,
                    sr_target=target_sample_rate,
                    apply_augmentation=apply_aug_for_this_file,
                    augmenter=ravdess_augmenter
                )
                
                if features is not None:
                    features_map[current_set_key].append(features)
                    labels_map[current_set_key].append(label_one_hot)
                    emotion_counts_map[current_set_key][model_emotion_text] += 1
                else:
                    logger.warning(f"Could not extract features for file {wav_file_name}. Skipping.")
            except Exception as e:
                logger.error(f"General error processing file {wav_file_name} in actor {actor_folder}: {e}", exc_info=True)
                continue 
    
    # Convert lists to NumPy arrays
    X_train_list = features_map['train']
    y_train_list = labels_map['train']
    X_val_list = features_map['val']
    y_val_list = labels_map['val']
    X_test_list = features_map['test']
    y_test_list = labels_map['test']

    X_train = np.array(X_train_list, dtype=np.float32) if X_train_list else np.empty((0, config.WHISPER_EMBEDDING_DIM))
    y_train = np.array(y_train_list, dtype=np.float32) if y_train_list else np.empty((0, len(config.MODEL_OUTPUT_EMOTIONS)))
    X_val = np.array(X_val_list, dtype=np.float32) if X_val_list else np.empty((0, config.WHISPER_EMBEDDING_DIM))
    y_val = np.array(y_val_list, dtype=np.float32) if y_val_list else np.empty((0, len(config.MODEL_OUTPUT_EMOTIONS)))
    X_test = np.array(X_test_list, dtype=np.float32) if X_test_list else np.empty((0, config.WHISPER_EMBEDDING_DIM))
    y_test = np.array(y_test_list, dtype=np.float32) if y_test_list else np.empty((0, len(config.MODEL_OUTPUT_EMOTIONS)))

    logger.info("RAVDESS Loading and Feature Extraction Complete:")
    logger.info(f"  Training set:   Features {X_train.shape}, Labels {y_train.shape}")
    logger.info(f"  Validation set: Features {X_val.shape}, Labels {y_val.shape}")
    logger.info(f"  Test set:       Features {X_test.shape}, Labels {y_test.shape}")

    logger.info("RAVDESS Emotion Distribution (according to MODEL_OUTPUT_EMOTIONS):")
    for set_name in ['train', 'val', 'test']:
        if emotion_counts_map[set_name]:
            logger.info(f"  {set_name.capitalize()} set: {dict(emotion_counts_map[set_name])}")
        else:
            logger.info(f"  - (No data for {set_name.capitalize()} set)")

    if X_train.size == 0: logger.warning("RAVDESS training set is empty!")
    if X_val.size == 0: logger.warning("RAVDESS validation set is empty!")
    if X_test.size == 0: logger.warning("RAVDESS test set is empty (this may be expected).")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)