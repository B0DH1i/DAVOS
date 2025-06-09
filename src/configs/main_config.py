# duygu_verimlilik_projesi/src/configs/main_config.py
import os
import logging

# --- Proje Kök Dizini ---
CONFIG_FILE_PATH = os.path.abspath(__file__)
SRC_PATH = os.path.dirname(os.path.dirname(CONFIG_FILE_PATH))
PROJECT_ROOT = os.path.dirname(SRC_PATH)

# --- Temel Veri Yolları ---
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
FER2013_DATA_PATH = os.path.join(DATA_PATH, "fer2013", "fer2013.csv") # Eski veri seti, hala referans olabilir
FER2013_DIR = os.path.join(DATA_PATH, "fer2013") 
FERPLUS_PREPARED_DATA_PATH = os.path.join(DATA_PATH, "ferplus_prepared")
RAVDESS_DATA_PATH = os.path.join(DATA_PATH, "ravdess_speech")

TRAINED_MODELS_PATH = os.path.join(PROJECT_ROOT, "trained_models")
LOGS_PATH = os.path.join(PROJECT_ROOT, "logs")
PLOTS_PATH = os.path.join(PROJECT_ROOT, "plots")
# SOUNDS_PATH kaldırıldı, Lazanov kendi yolunu kullanıyor, kısa bildirimler kalktı.

# --- MODEL GİRİŞ ŞEKİLLERİ ---
# VGG16 gibi modeller 3 kanal bekler, bu eğitim pipeline'ında ele alınacak.
INPUT_SHAPE_FER = (48, 48, 1)  # (yükseklik, genişlik, kanal) - Gri tonlamalı
# INPUT_SHAPE_SER, SER_FEATURE_TYPE = "whisper" olduğunda (WHISPER_EMBEDDING_DIM,) olur.
# MFCC kullanılıyorsa (13, 313, 1) gibi bir değer alırdı. Şimdilik SER modeli girişini kendi belirler.

# --- Yüz Algılama Ayarları (DNN Modeli) ---
FACE_DETECTOR_DNN_PROTOTXT_PATH = os.path.join(TRAINED_MODELS_PATH, "face_detector", "deploy.prototxt.txt")
FACE_DETECTOR_DNN_MODEL_PATH = os.path.join(TRAINED_MODELS_PATH, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
FACE_DETECTOR_DNN_CONFIDENCE_THRESHOLD = 0.5

# --- Duygu Etiketleri ve Haritalamalar ---
RAVDESS_EMOTIONS_FROM_FILENAME = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprise"
}

# Standart hedef duygu listesi (9 duygu, integration_engine tarafından kullanılır)
TARGET_EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "surprise", "disgust", "calm", "unknown"]

# FER ve SER modellerinin ortak çıktı duygu listesi (8 duygu, contempt dahil, calm/unknown hariç)
# Bu, predictor_engine'in model tahminlerinde kullandığı sıralı listedir.
MODEL_OUTPUT_EMOTIONS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

# TARGET_EMOTIONS_ORDERED eski adlandırmaydı, MODEL_OUTPUT_EMOTIONS olarak yeniden adlandırıldı ve amacı netleştirildi.
# Eski TARGET_EMOTIONS_ORDERED = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

# FER modelinin çıktısını (MODEL_OUTPUT_EMOTIONS) TARGET_EMOTIONS'a eşleme
FER_TO_TARGET_MAP = {
    "neutral": "neutral",
    "happiness": "happy",    # İsim farkı
    "surprise": "surprise",
    "sadness": "sad",        # İsim farkı
    "anger": "angry",        # İsim farkı (aslında aynı ama tutarlılık için)
    "disgust": "disgust",
    "fear": "fear",
    "contempt": "unknown"    # Contempt TARGET_EMOTIONS'da yok, unknown'a eşle
}

# SER modelinin çıktısını (MODEL_OUTPUT_EMOTIONS) TARGET_EMOTIONS'a eşleme
# SER modelinin çıktısı da MODEL_OUTPUT_EMOTIONS listesine göre olduğundan, map aynı olacak.
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

# RAVDESS dosya adlarından gelen duyguları MODEL_OUTPUT_EMOTIONS'a eşleme (SER data_loader için)
RAVDESS_TO_MODEL_OUTPUT_MAP = {
    "neutral": "neutral",    # RAVDESS "neutral" -> MODEL_OUTPUT "neutral"
    "calm": "neutral",       # RAVDESS "calm"    -> MODEL_OUTPUT "neutral" (MODEL_OUTPUT'ta "calm" yok)
    "happy": "happiness",    # RAVDESS "happy"   -> MODEL_OUTPUT "happiness"
    "sad": "sadness",        # RAVDESS "sad"     -> MODEL_OUTPUT "sadness"
    "angry": "anger",        # RAVDESS "angry"   -> MODEL_OUTPUT "anger"
    "fearful": "fear",       # RAVDESS "fearful" -> MODEL_OUTPUT "fear"
    "disgust": "disgust",    # RAVDESS "disgust" -> MODEL_OUTPUT "disgust"
    "surprise": "surprise"   # RAVDESS "surprise"-> MODEL_OUTPUT "surprise"
}

# RAVDESS dosya adlarından gelen duyguları TARGET_EMOTIONS'a eşleme (data_loader için)
RAVDESS_TO_TARGET_MAP = {
    "neutral": "neutral", 
    "calm": "calm",          # calm doğrudan TARGET_EMOTIONS'da var
    "happy": "happy", 
    "sad": "sad",
    "angry": "angry", 
    "fearful": "fear",       # fearful -> fear
    "disgust": "disgust", 
    "surprise": "surprise"
}

NUM_MODEL_OUTPUT_CLASSES = len(MODEL_OUTPUT_EMOTIONS) # Eski NUM_FER_CLASSES yerine
NUM_TARGET_CLASSES = len(TARGET_EMOTIONS) # Bu doğruydu


# --- FER Veri Yükleme ve Ön İşleme Ayarları ---
# FER_EXPECTED_COLUMNS = ['emotion', 'pixels', 'Usage'] # Orijinal fer2013.csv için, artık kullanılmıyor
FERPLUS_LABEL_COLUMNS = MODEL_OUTPUT_EMOTIONS + ["unknown", "NF"]
FERPLUS_IMAGE_NAME_COLUMN = "Image name"
FER_IMG_SIZE = (48, 48)
# FER_DATA_SPLIT kaldırıldı, FERPlus klasör yapısı kullanılıyor.

# --- RAVDESS Ses Veri Yükleme ve Ön İşleme Ayarları ---
RAVDESS_EXPECTED_ACTORS = 24
RAVDESS_TRAIN_ACTORS = list(range(1, 19))
RAVDESS_VAL_ACTORS = list(range(19, 22))
RAVDESS_TEST_ACTORS = list(range(22, 25))

# === SES ÖZNİTELİK AYARLARI ===
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION_SECONDS = 3 # Canlı analiz ve eğitim için ses segmenti uzunluğu
# MFCC parametreleri (AUDIO_N_MFCC, AUDIO_MAX_FRAMES_MFCC, AUDIO_FFT_WINDOW_SIZE, AUDIO_HOP_LENGTH) kaldırıldı, SER_FEATURE_TYPE = "whisper".

# --- Model Tipleri ---
FER_MODEL_TYPE_VGG16_TRANSFER = "vgg16_transfer"
SER_MODEL_TYPE_SIMPLE_DENSE_WHISPER = "simple_dense_whisper"

# --- Whisper Ayarları ---
SER_FEATURE_TYPE = "whisper" # "mfcc" veya "whisper" olabilir
WHISPER_MODEL_NAME = "openai/whisper-base"
WHISPER_SAMPLING_RATE = 16000 # Whisper modelinin beklediği örnekleme oranı
WHISPER_EMBEDDING_DIM = 512 # Whisper 'base' modelinden elde edilecek embedding boyutu.

# === VERİ YÜKLEME VE ÖNİŞLEME ===
DATA_AUGMENTATION_FER_ENABLED = True # ImageDataGenerator ile FER için
DATA_AUGMENTATION_SER_ENABLED = True # audiomentations ile SER için (RAVDESS_DATA_AUGMENTATION_ENABLED yerine)

# === MODEL EĞİTİM AYARLARI (train_pipeline.py tarafından kullanılacak) ===

# --- FER Modeli Eğitim Ayarları ---
DEFAULT_FER_MODEL_CHOICE = FER_MODEL_TYPE_VGG16_TRANSFER
FER_MODEL_NAME_PREFIX = "fer"
DEFAULT_EPOCHS_FER = 300
DEFAULT_BATCH_SIZE_FER = 128
DEFAULT_OPTIMIZER_FER = "sgd"
DEFAULT_LEARNING_RATE_FER = 0.01
WEIGHT_DECAY_FER = 0.0001 # kernel_regularizer olarak uygulanacak
PATIENCE_EARLY_STOPPING_FER = 20
PATIENCE_REDUCE_LR_FER = 5
FACTOR_REDUCE_LR_FER = 0.75
MONITOR_METRIC_FER = "val_accuracy"
SAVE_BEST_ONLY_FER = True
USE_DATA_AUGMENTATION_FER = DATA_AUGMENTATION_FER_ENABLED # Yukarıdaki genel flag'e bağlandı

# --- SER Modeli Eğitim Ayarları ---
DEFAULT_SER_MODEL_CHOICE = SER_MODEL_TYPE_SIMPLE_DENSE_WHISPER # SER_MODEL_CHOICE yerine
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
USE_DATA_AUGMENTATION_SER = DATA_AUGMENTATION_SER_ENABLED # Yukarıdaki genel flag'e bağlandı
SER_L2_REG_STRENGTH = 0.001

# Eski FER_MODEL_CHOICE, FER_EPOCHS vb. ve SER_MODEL_DEFAULT_BASE_NAME, SER_EPOCHS vb. bloklar kaldırıldı.

# --- Tahmin ve Canlı Analiz Ayarları ---
DEFAULT_FER_MODEL_LOAD_NAME = "fer_vgg16_transfer_20250602-221124"
DEFAULT_SER_MODEL_LOAD_NAME = "ser_model_simple_dense_whisper_whisper_20250602-230346"

ANALYSIS_INTERVAL_FER = 0.1  # saniye (kamera frame işleme sıklığı)
ANALYSIS_INTERVAL_SER = 0.5  # saniye (canlı ses analizi sıklığı)
ANALYSIS_INTERVAL_INTEGRATION = 0.2 # Duygu entegrasyonu sıklığı

LIVE_AUDIO_SEGMENT_DURATION = 3 # saniye (canlı ses analizi için yakalanacak ses segmenti uzunluğu)
MAX_CONSECUTIVE_EMPTY_FRAMES_VIDEO_FILE = 20 # Video dosyasında ardışık boş kare sonrası durdurma sınırı (EKLENDİ)

# === Pygame Görüntüleme Ayarları (OutputController için) ===
DISPLAY_MAX_WIDTH = 800
DISPLAY_MAX_HEIGHT = 600
APP_WINDOW_TITLE = "DOVAS - Canlı Duygu Analiz Sistemi v3"
TEXT_COLOR = (255, 255, 0)  # Sarı (R, G, B)
BACKGROUND_COLOR = (30, 30, 30) # Koyu Gri (R, G, B)
INFO_AREA_HEIGHT_RATIO = 0.25
FONT_SCALE = 0.5

# PyAudio ayarları
PYAUDIO_FORMAT = 8 # pyaudio.paInt16 (16-bit)
AUDIO_NUMPY_DTYPE = "int16" # NUMPY_AUDIO_DTYPE kaldırıldı, bu tek kalsın.
PYAUDIO_CHANNELS = 1
PYAUDIO_RATE = AUDIO_SAMPLE_RATE # Mikrofon için de aynı örnekleme oranı
PYAUDIO_FRAMES_PER_BUFFER = 1024 # Chunk size
# PYAUDIO_AUDIO_SEGMENT_BUFFER_SIZE kaldırıldı, input_handler'da dinamik hesaplanıyor.
CAMERA_INDEX = 0

# --- Multi-Modal Entegrasyon Ayarları ---
INTEGRATION_STRATEGY = "weighted_average" # "weighted_average", "highest_confidence"
INTEGRATION_WEIGHT_FACE = 0.6
INTEGRATION_WEIGHT_SPEECH = 0.4

TEMPORAL_SMOOTHING_ENABLED = True
TEMPORAL_SMOOTHING_TYPE = "moving_average" # "moving_average", "ewma" (Exponential Weighted Moving Average)
TEMPORAL_SMOOTHING_WINDOW_SIZE = 5 # Son N tahmini kullan (moving_average için)
# TEMPORAL_SMOOTHING_ALPHA = 0.3 # EWMA için alfa değeri (gerekirse)

# --- Lazanov Müdahale Ayarları ---
# SHORT_NOTIFICATION_SOUNDS_MAP ve MIN_SHORT_NOTIFICATION_INTERVAL_SECONDS kaldırıldı.
LAZANOV_MUSIC_LIBRARY_PATH = os.path.join(DATA_PATH, "lazanov_music_library")
LAZANOV_MUSIC_METADATA_FILENAME = "music_library.json"
LAZANOV_INTERVENTION_ENABLED = True
LAZANOV_TRIGGER_EMOTIONS = {
    "anger": {"target_state": "calm_focus", "intensity_threshold": 0.6},
    "sadness": {"target_state": "gentle_uplift", "intensity_threshold": 0.5},
    "fear": {"target_state": "secure_calm", "intensity_threshold": 0.6},
    "stress": {"target_state": "deep_relaxation_alpha", "intensity_threshold": 0.7}
}
# LAZANOV_TRIGGER_DURATION_SECONDS kaldırıldı (şimdilik kullanılmıyor, anlık duyguya göre çalışıyor).
LAZANOV_TRIGGER_CONFIDENCE_THRESHOLD = 0.6
LAZANOV_INTERVENTION_COOLDOWN_SECONDS = 300
LAZANOV_DEFAULT_MUSIC_VOLUME = 0.7
LAZANOV_FADE_IN_OUT_DURATION_SECONDS = 5

# Beyin Dalgası Sürükleme (Binaural Ritimler) Ayarları
LAZANOV_BRAINWAVE_ENTRAINMENT_ENABLED = True
LAZANOV_BINAURAL_BEATS_ENABLED = True
LAZANOV_BINAURAL_BEATS_SETTINGS = {
    "deep_sleep_delta_2hz": {
        "id": "deep_sleep_delta_2hz", "beat_hz": 2, "carrier_hz": 100,
        "brain_wave_band": "Delta (0.5-4 Hz)",
        "associated_states": ["Derin uyku", "Bilinçdışı yenilenme", "Çok derin rahatlama"],
        "primary_emotion_target": "N/A (Uyku odaklı)",
        "intended_use_description": "Uykusuzluk veya aşırı yorgunluk durumlarında derin ve dinlendirici bir uykuya geçişi desteklemek için.",
        "default_duration_seconds": 900, "default_volume": 0.25
    },
    "meditative_relaxation_theta_6hz": {
        "id": "meditative_relaxation_theta_6hz", "beat_hz": 6, "carrier_hz": 90,
        "brain_wave_band": "Theta (4-8 Hz)",
        "associated_states": ["Derin rahatlama", "Meditasyon", "İçe dönüklük", "Hafif uyku", "Yaratıcılık"],
        "primary_emotion_target": "Aşırı stres, anksiyete",
        "intended_use_description": "Yoğun stres veya anksiyete anlarında zihni sakinleştirmek, derin bir meditatif rahatlama sağlamak ve içsel huzuru artırmak için.",
        "default_duration_seconds": 600, "default_volume": 0.3
    },
    "calm_focus_alpha_8hz": {
        "id": "calm_focus_alpha_8hz", "beat_hz": 8, "carrier_hz": 100,
        "brain_wave_band": "Alfa (8-12 Hz)",
        "associated_states": ["Sakin uyanıklık", "Hafif rahatlama", "Stres azaltma", "Öğrenmeye hazırlık"],
        "primary_emotion_target": "Hafif stres, huzursuzluk",
        "intended_use_description": "Hafif stresli veya huzursuz hissedildiğinde, zihni yavaşlatmak ve sakin bir odaklanma durumuna geçmek için.",
        "default_duration_seconds": 480, "default_volume": 0.35
    },
    "alert_focus_alpha_10hz": {
        "id": "alert_focus_alpha_10hz", "beat_hz": 10, "carrier_hz": 110,
        "brain_wave_band": "Alfa (8-12 Hz)",
        "associated_states": ["Rahatlamış odaklanma", "Optimal öğrenme", "Pozitif düşünme", "Stres direnci"],
        "primary_emotion_target": "Dikkat dağınıklığı, motivasyon eksikliği",
        "intended_use_description": "Zihinsel berraklık ve odaklanmış dikkat gerektiren görevler sırasında performansı artırmak.",
        "default_duration_seconds": 300, "default_volume": 0.4
    },
    "active_thinking_beta_15hz": {
        "id": "active_thinking_beta_15hz", "beat_hz": 15, "carrier_hz": 120,
        "brain_wave_band": "Beta (12-30 Hz)",
        "associated_states": ["Aktif düşünme", "Problem çözme", "Uyanıklık", "Konsantrasyon"],
        "primary_emotion_target": "Zihinsel yorgunluk (uyarıcı), düşük enerji",
        "intended_use_description": "Zihinsel uyanıklığı ve analitik düşünmeyi artırmak için (kısa süreli kullanım).",
        "default_duration_seconds": 240, "default_volume": 0.35
    },
    "energized_motivation_beta_18hz": {
        "id": "energized_motivation_beta_18hz", "beat_hz": 18, "carrier_hz": 130,
        "brain_wave_band": "Beta (12-30 Hz)",
        "associated_states": ["Enerjik uyanıklık", "Motivasyon", "Yüksek performanslı görevler"],
        "primary_emotion_target": "Prokrastinasyon, enerji düşüklüğü",
        "intended_use_description": "Motivasyonu artırmak ve enerji gerektiren işlere başlamak için (kısa süreli ve dikkatli kullanım).",
        "default_duration_seconds": 180, "default_volume": 0.3
    },
    "peak_cognition_gamma_40hz": {
        "id": "peak_cognition_gamma_40hz", "beat_hz": 40, "carrier_hz": 140,
        "brain_wave_band": "Gamma (30-100 Hz)",
        "associated_states": ["Yüksek seviye bilişsel işleme", "Yoğun odaklanma", "Problem çözmede 'aha!' anları"],
        "primary_emotion_target": "N/A (Bilişsel zirve performansı odaklı)",
        "intended_use_description": "Çok yoğun konsantrasyon veya yaratıcı problem çözme anları için (çok kısa süreli).",
        "default_duration_seconds": 120, "default_volume": 0.25
    }
}

# --- Logging Ayarları ---
LOGGING_LEVEL = logging.INFO # logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR
APPLICATION_LOG_FILE = os.path.join(LOGS_PATH, "application.log")

# RAVDESS_EMOTIONS listesi zaten RAVDESS_EMOTIONS_FROM_FILENAME.values() ile aynı, kaldırılabilir.
# Veya veri yükleyicide kullanılmak üzere bırakılabilir ama tutarlılık için kontrol edilmeli.
# Şimdilik bırakıyorum, data_loader.py'ye bakarken değerlendiririz.

# Yapılandırma dosyasının doğru yüklendiğini teyit etmek için olan print'ler kaldırıldı.
# --- Dosyanın sonundaki eski/mükerrer model tipleri, giriş şekilleri ve eğitim parametreleri blokları temizlendi. ---