import os
import logging

# --- Project Root Directory ---
CONFIG_FILE_PATH = os.path.abspath(__file__)
SRC_PATH = os.path.dirname(os.path.dirname(CONFIG_FILE_PATH))
PROJECT_ROOT = os.path.dirname(SRC_PATH)

# --- Core Data Paths ---
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
FER2013_DATA_PATH = os.path.join(DATA_PATH, "fer2013", "fer2013.csv") 
FER2013_DIR = os.path.join(DATA_PATH, "fer2013") 
FERPLUS_PREPARED_DATA_PATH = os.path.join(DATA_PATH, "ferplus_prepared")
RAVDESS_DATA_PATH = os.path.join(DATA_PATH, "ravdess_speech")
TRAINED_MODELS_PATH = os.path.join(PROJECT_ROOT, "trained_models")
LOGS_PATH = os.path.join(PROJECT_ROOT, "logs")
PLOTS_PATH = os.path.join(PROJECT_ROOT, "plots")

# --- MODEL INPUT SHAPES ---
INPUT_SHAPE_FER = (48, 48, 1)  # (height, width, channel) - Grayscale

# --- Face Detection Settings (DNN Model) ---
FACE_DETECTOR_DNN_PROTOTXT_PATH = os.path.join(TRAINED_MODELS_PATH, "face_detector", "deploy.prototxt.txt")
FACE_DETECTOR_DNN_MODEL_PATH = os.path.join(TRAINED_MODELS_PATH, "face_detector", "Res10_300x300_SSD_iter_140000.caffemodel")
FACE_DETECTOR_DNN_CONFIDENCE_THRESHOLD = 0.3

# --- Emotion Labels and Mappings ---
RAVDESS_EMOTIONS_FROM_FILENAME = {    
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",    
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprise"
}
TARGET_EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "surprise", "disgust", "calm", "unknown"] # Standard target emotion list
MODEL_OUTPUT_EMOTIONS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"] # Common output emotion list for FER and SER models

FER_TO_TARGET_MAP = {    
    "neutral": "neutral",    
    "happiness": "happy",    
    "surprise": "surprise",    
    "sadness": "sad",        
    "anger": "angry",        
    "disgust": "disgust",    
    "fear": "fear",    
    "contempt": "unknown"    
}
SER_OUTPUT_TO_TARGET_MAP = {    
    "neutral": "neutral",    
    "happiness": "happy",    
    "surprise": "surprise",    
    "sadness": "sad",    
    "anger": "angry",    
    "disgust": "disgust",    
    "fear": "fear",    
    "contempt": "unknown"
}
RAVDESS_TO_MODEL_OUTPUT_MAP = {    
    "neutral": "neutral",    
    "calm": "neutral",       
    "happy": "happiness",    
    "sad": "sadness",        
    "angry": "anger",        
    "fearful": "fear",       
    "disgust": "disgust",    
    "surprise": "surprise"   
}
RAVDESS_TO_TARGET_MAP = {    
    "neutral": "neutral",     
    "calm": "calm",          
    "happy": "happy",     
    "sad": "sad",    
    "angry": "angry",     
    "fearful": "fear",       
    "disgust": "disgust",     
    "surprise": "surprise"
}
NUM_MODEL_OUTPUT_CLASSES = len(MODEL_OUTPUT_EMOTIONS) 
NUM_TARGET_CLASSES = len(TARGET_EMOTIONS) 

# --- FER Data Loading and Preprocessing Settings ---
FERPLUS_LABEL_COLUMNS = MODEL_OUTPUT_EMOTIONS + ["unknown", "NF"]
FERPLUS_IMAGE_NAME_COLUMN = "Image name"
FER_IMG_SIZE = (48, 48)

# --- RAVDESS Audio Data Loading and Preprocessing Settings ---
RAVDESS_EXPECTED_ACTORS = 24
RAVDESS_TRAIN_ACTORS = list(range(1, 19))
RAVDESS_VAL_ACTORS = list(range(19, 22))
RAVDESS_TEST_ACTORS = list(range(22, 25))

# === AUDIO FEATURE SETTINGS ===
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION_SECONDS = 3 

# --- Model Types ---
FER_MODEL_TYPE_VGG16_TRANSFER = "vgg16_transfer"
SER_MODEL_TYPE_SIMPLE_DENSE_WHISPER = "simple_dense_whisper"

# --- Whisper Settings ---
SER_FEATURE_TYPE = "whisper" 
WHISPER_MODEL_NAME = "openai/whisper-base"
WHISPER_SAMPLING_RATE = 16000 
WHISPER_EMBEDDING_DIM = 512 

# === DATA LOADING AND PREPROCESSING ===
DATA_AUGMENTATION_FER_ENABLED = True 
DATA_AUGMENTATION_SER_ENABLED = True 

# === MODEL TRAINING SETTINGS (to be used by train_pipeline.py) ===
# --- FER Model Training Settings ---
DEFAULT_FER_MODEL_CHOICE = FER_MODEL_TYPE_VGG16_TRANSFER
FER_MODEL_NAME_PREFIX = "fer"
DEFAULT_EPOCHS_FER = 300
DEFAULT_BATCH_SIZE_FER = 128
DEFAULT_OPTIMIZER_FER = "sgd"
DEFAULT_LEARNING_RATE_FER = 0.01
WEIGHT_DECAY_FER = 0.0001 
PATIENCE_EARLY_STOPPING_FER = 20
PATIENCE_REDUCE_LR_FER = 5
FACTOR_REDUCE_LR_FER = 0.75
MONITOR_METRIC_FER = "val_accuracy"
SAVE_BEST_ONLY_FER = True
USE_DATA_AUGMENTATION_FER = DATA_AUGMENTATION_FER_ENABLED 

# --- SER Model Training Settings ---
DEFAULT_SER_MODEL_CHOICE = SER_MODEL_TYPE_SIMPLE_DENSE_WHISPER 
SER_MODEL_NAME_PREFIX = "ser"
DEFAULT_EPOCHS_SER = 100
DEFAULT_BATCH_SIZE_SER = 32
DEFAULT_OPTIMIZER_SER = "adam"
DEFAULT_LEARNING_RATE_SER = 0.0001
PATIENCE_EARLY_STOPPING_SER = 20
PATIENCE_REDUCE_LR_SER = 7
FACTOR_REDUCE_LR_SER = 0.5
MONITOR_METRIC_SER = "val_accuracy"
SAVE_BEST_ONLY_SER = True
USE_DATA_AUGMENTATION_SER = DATA_AUGMENTATION_SER_ENABLED 
SER_L2_REG_STRENGTH = 0.001

# --- Prediction and Live Analysis Settings ---
DEFAULT_FER_MODEL_LOAD_NAME = "fer_vgg16_transfer_20250602-221124"
DEFAULT_SER_MODEL_LOAD_NAME = "ser_model_simple_dense_whisper_whisper_20250602-230346"
ANALYSIS_INTERVAL_FER = 0.1  # seconds (camera frame processing frequency)
ANALYSIS_INTERVAL_SER = 0.5  # seconds (live audio analysis frequency)
ANALYSIS_INTERVAL_INTEGRATION = 0.2 # Emotion integration frequency
LIVE_AUDIO_SEGMENT_DURATION = 3 # seconds (audio segment length to capture for live analysis)
MAX_CONSECUTIVE_EMPTY_FRAMES_VIDEO_FILE = 20 # Limit for stopping after consecutive empty frames in video file

# === Pygame Display Settings (for OutputController) ===
DISPLAY_MAX_WIDTH = 800
DISPLAY_MAX_HEIGHT = 600
APP_WINDOW_TITLE = "DOVAS - Live Emotion Analysis System v3"
TEXT_COLOR = (255, 255, 0)  # Yellow (R, G, B)
BACKGROUND_COLOR = (30, 30, 30) # Dark Gray (R, G, B)
INFO_AREA_HEIGHT_RATIO = 0.25
FONT_SCALE = 0.5

# PyAudio settings
PYAUDIO_FORMAT = 8 # pyaudio.paInt16 (16-bit)
AUDIO_NUMPY_DTYPE = "int16" 
PYAUDIO_CHANNELS = 1
PYAUDIO_RATE = AUDIO_SAMPLE_RATE 
PYAUDIO_FRAMES_PER_BUFFER = 1024 
CAMERA_INDEX = 0

# --- Multi-Modal Integration Settings ---
INTEGRATION_STRATEGY = "weighted_average" 
INTEGRATION_WEIGHT_FACE = 0.6
INTEGRATION_WEIGHT_SPEECH = 0.4
TEMPORAL_SMOOTHING_ENABLED = True
TEMPORAL_SMOOTHING_TYPE = "moving_average" 
TEMPORAL_SMOOTHING_WINDOW_SIZE = 5 

# --- Lazanov Intervention Settings ---
LAZANOV_MUSIC_LIBRARY_PATH = os.path.join(DATA_PATH, "lazanov_music_library")
LAZANOV_MUSIC_METADATA_FILENAME = "music_library.json"
LAZANOV_INTERVENTION_ENABLED = True
LAZANOV_TRIGGER_EMOTIONS = {    
    "anger": {"target_state": "calm_focus", "intensity_threshold": 0.6},    
    "sadness": {"target_state": "gentle_uplift", "intensity_threshold": 0.5},    
    "fear": {"target_state": "secure_calm", "intensity_threshold": 0.6},    
    "stress": {"target_state": "deep_relaxation_alpha", "intensity_threshold": 0.7}
}
LAZANOV_TRIGGER_CONFIDENCE_THRESHOLD = 0.6
LAZANOV_INTERVENTION_COOLDOWN_SECONDS = 300
LAZANOV_DEFAULT_MUSIC_VOLUME = 0.7
LAZANOV_FADE_IN_OUT_DURATION_SECONDS = 5

# Brainwave Entrainment (Binaural Beats) Settings
LAZANOV_BRAINWAVE_ENTRAINMENT_ENABLED = True
LAZANOV_BINAURAL_BEATS_ENABLED = True
LAZANOV_BINAURAL_BEATS_SETTINGS = {    
    "deep_sleep_delta_2hz": {        
        "id": "deep_sleep_delta_2hz", "beat_hz": 2, "carrier_hz": 100,        
        "brain_wave_band": "Delta (0.5-4 Hz)",        
        "associated_states": ["Deep sleep", "Unconscious regeneration", "Very deep relaxation"],        
        "primary_emotion_target": "N/A (Sleep-focused)",        
        "intended_use_description": "To support transition into deep and restful sleep during insomnia or extreme fatigue.",        
        "default_duration_seconds": 900, "default_volume": 0.25    
    },    
    "meditative_relaxation_theta_6hz": {        
        "id": "meditative_relaxation_theta_6hz", "beat_hz": 6, "carrier_hz": 90,        
        "brain_wave_band": "Theta (4-8 Hz)",        
        "associated_states": ["Deep relaxation", "Meditation", "Introspection", "Light sleep", "Creativity"],        
        "primary_emotion_target": "Extreme stress, anxiety",        
        "intended_use_description": "To calm the mind during moments of intense stress or anxiety, provide deep meditative relaxation, and increase inner peace.",        
        "default_duration_seconds": 600, "default_volume": 0.3    
    },    
    "calm_focus_alpha_8hz": {        
        "id": "calm_focus_alpha_8hz", "beat_hz": 8, "carrier_hz": 100,        
        "brain_wave_band": "Alpha (8-12 Hz)",        
        "associated_states": ["Calm alertness", "Light relaxation", "Stress reduction", "Readiness for learning"],        
        "primary_emotion_target": "Mild stress, restlessness",        
        "intended_use_description": "To slow down the mind and enter a state of calm focus when feeling mildly stressed or restless.",        
        "default_duration_seconds": 480, "default_volume": 0.35    
    },    
    "alert_focus_alpha_10hz": {        
        "id": "alert_focus_alpha_10hz", "beat_hz": 10, "carrier_hz": 110,        
        "brain_wave_band": "Alpha (8-12 Hz)",        
        "associated_states": ["Relaxed focus", "Optimal learning", "Positive thinking", "Stress resilience"],        
        "primary_emotion_target": "Distraction, lack of motivation",        
        "intended_use_description": "To enhance performance during tasks requiring mental clarity and focused attention.",        
        "default_duration_seconds": 300, "default_volume": 0.4    
    },    
    "active_thinking_beta_15hz": {        
        "id": "active_thinking_beta_15hz", "beat_hz": 15, "carrier_hz": 120,        
        "brain_wave_band": "Beta (12-30 Hz)",        
        "associated_states": ["Active thinking", "Problem solving", "Alertness", "Concentration"],        
        "primary_emotion_target": "Mental fatigue (stimulant), low energy",        
        "intended_use_description": "To increase mental alertness and analytical thinking (short-term use).",        
        "default_duration_seconds": 240, "default_volume": 0.35    
    },    
    "energized_motivation_beta_18hz": {        
        "id": "energized_motivation_beta_18hz", "beat_hz": 18, "carrier_hz": 130,        
        "brain_wave_band": "Beta (12-30 Hz)",        
        "associated_states": ["Energetic alertness", "Motivation", "High-performance tasks"],        
        "primary_emotion_target": "Procrastination, low energy",        
        "intended_use_description": "To boost motivation and initiate energy-demanding tasks (short-term and careful use).",        
        "default_duration_seconds": 180, "default_volume": 0.3    
    },    
    "peak_cognition_gamma_40hz": {        
        "id": "peak_cognition_gamma_40hz", "beat_hz": 40, "carrier_hz": 140,        
        "brain_wave_band": "Gamma (30-100 Hz)",        
        "associated_states": ["High-level cognitive processing", "Intense focus", "''Aha!'' moments in problem-solving"],        
        "primary_emotion_target": "N/A (Cognitive peak performance focused)",        
        "intended_use_description": "For moments of very intense concentration or creative problem-solving (very short-term).",        
        "default_duration_seconds": 120, "default_volume": 0.25    
    }
}
# --- Logging Settings ---
LOGGING_LEVEL = logging.INFO 
APPLICATION_LOG_FILE = os.path.join(LOGS_PATH, "application.log")
