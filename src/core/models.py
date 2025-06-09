import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense,
    BatchNormalization, Activation, Add, GlobalAveragePooling2D,
    LSTM, Bidirectional, TimeDistributed, Reshape,
    SeparableConv2D, DepthwiseConv2D, Permute, multiply
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16
import numpy as np # Sadece test için
import os # Added import

# Göreli importlar
from ..utils.logging_utils import setup_logger
from ..configs import main_config as config

# Logger'ı bu modül için kur
logger = setup_logger(__name__, log_file=config.APPLICATION_LOG_FILE)

# --- Yüz İfadesi Tanıma (FER) Modelleri ---

def build_fer_model_vgg16_transfer(input_shape, num_classes, model_name="FER_VGG16_Transfer_Model"):
    """
    VGG16 modelini temel alarak transfer öğrenme ile FER modeli oluşturur.
    Giriş şekli (height, width, 3) olmalıdır.
    """
    logger.info(f"VGG16 tabanlı FER transfer öğrenme modeli oluşturuluyor: {model_name}")
    logger.info(f"Giriş şekli: {input_shape}")
    logger.info(f"Sınıf sayısı: {num_classes}")

    # VGG16 modelini ImageNet ağırlıklarıyla yükle, en üstteki sınıflandırma katmanlarını dahil etme
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # VGG16'nın evrişimsel tabanındaki katmanları dondur
    base_model.trainable = False
    logger.info(f"VGG16 temel modelinin katmanları donduruldu (base_model.trainable = {base_model.trainable}).")

    # Yeni sınıflandırma katmanları ekle
    x = base_model.output
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    
    # FC Katman 1
    x = Dense(1024, activation='relu', kernel_regularizer=l2(config.WEIGHT_DECAY_FER), name="fc1_relu")(x)
    x = BatchNormalization(name="fc1_bn")(x)
    x = Dropout(0.5, name="fc1_dropout")(x)
    
    # FC Katman 2
    x = Dense(512, activation='relu', kernel_regularizer=l2(config.WEIGHT_DECAY_FER), name="fc2_relu")(x)
    x = BatchNormalization(name="fc2_bn")(x)
    x = Dropout(0.5, name="fc2_dropout")(x) # Dropout oranı 0.3'ten 0.5'e çıkarıldı
    
    # FC Katman 3 (Çıkış Katmanı)
    output_layer = Dense(num_classes, activation='softmax', name='output_emotion')(x)

    # Yeni modeli oluştur
    model = Model(inputs=base_model.input, outputs=output_layer, name=model_name)
    
    logger.info(f"{model_name} başarıyla oluşturuldu.")
    # model.summary(print_fn=logger.info) # Eğitim sırasında summary loglanacak
    return model

# --- Sesli Duygu Tanıma (SER) Modeli ---

def build_ser_model_crnn(input_shape=None, num_classes=None):
    """
    RAVDESS (MFCC özellikleri) için bir CRNN (Convolutional Recurrent Neural Network) modeli.
    Girdi şekli: (n_mfcc, time_frames)
    """
    if input_shape is None:
        input_shape = (config.AUDIO_N_MFCC, config.AUDIO_MAX_FRAMES_MFCC)
    if num_classes is None:
        num_classes = len(config.TARGET_EMOTIONS) # SER modeli TARGET_EMOTIONS'a göre eğitilir

    logger.info(f"SER CRNN modeli oluşturuluyor. Giriş şekli: {input_shape}, Sınıf sayısı: {num_classes}")

    mfcc_input = Input(shape=input_shape, name='ser_input_mfcc_crnn')
    
    # Kanal boyutu ekle (Conv2D için): (n_mfcc, time_frames) -> (n_mfcc, time_frames, 1)
    x = Reshape((input_shape[0], input_shape[1], 1), name='ser_crnn_reshape_to_4d')(mfcc_input)

    # Konvolüsyonel Bloklar
    # Blok 1
    x = Conv2D(64, (3, 3), padding='same', name='ser_crnn_conv1')(x)
    x = BatchNormalization(name='ser_crnn_bn1')(x)
    x = Activation('relu', name='ser_crnn_relu1')(x)
    x = MaxPooling2D((2, 2), name='ser_crnn_pool1')(x) # Zaman ve frekans boyutunu azaltır
    x = Dropout(0.4, name='ser_crnn_drop1')(x)

    # Blok 2
    x = Conv2D(128, (3, 3), padding='same', name='ser_crnn_conv2')(x)
    x = BatchNormalization(name='ser_crnn_bn2')(x)
    x = Activation('relu', name='ser_crnn_relu2')(x)
    x = MaxPooling2D((2, 2), name='ser_crnn_pool2')(x)
    x = Dropout(0.4, name='ser_crnn_drop2')(x)

    # Blok 3
    x = Conv2D(256, (3, 3), padding='same', name='ser_crnn_conv3')(x)
    x = BatchNormalization(name='ser_crnn_bn3')(x)
    x = Activation('relu', name='ser_crnn_relu3')(x)
    # Zaman adımını çok azaltmamak için MaxPooling'de zaman ekseninde stride=1 olabilir.
    # (pool_size=(2,2), strides=(2,1)) -> frekansı azalt, zamanı daha az azalt.
    # Veya sadece frekansta pool: MaxPooling2D(pool_size=(2,1))
    x = MaxPooling2D((2, 1), name='ser_crnn_pool3')(x) # (height, width) = (frekans, zaman)
    x = Dropout(0.4, name='ser_crnn_drop3')(x)
    
    # RNN katmanlarına hazırlık:
    # Son konvolüsyonel katmanın çıktısı: (batch, new_freq_bins, new_time_frames, channels)
    # LSTM'e vermek için: (batch, new_time_frames, new_freq_bins * channels)
    s = x.shape # (None, H_conv, W_conv, C_conv)
    if s[1] is not None and s[2] is not None and s[3] is not None: # Statik şekil varsa
        x = Reshape((s[2], s[1] * s[3]), name='ser_crnn_reshape_for_lstm')(x)
    else: # Dinamik şekil için Lambda katmanı (daha güvenli)
        def dynamic_reshape_for_lstm(tensor_in):
            input_shape_dyn = tf.shape(tensor_in) # (batch, H, W, C)
            # (batch, W, H * C) olmalı
            return tf.reshape(tensor_in, [input_shape_dyn[0], input_shape_dyn[2], input_shape_dyn[1] * input_shape_dyn[3]])
        x = tf.keras.layers.Lambda(dynamic_reshape_for_lstm, name='ser_crnn_lambda_reshape_lstm')(x)

    # Tekrarlayan Bloklar (LSTM)
    x = Bidirectional(LSTM(64, return_sequences=True, name='ser_crnn_bilstm1', kernel_regularizer=l2(config.SER_L2_REG_STRENGTH)), name='ser_crnn_bidir1')(x)
    x = Dropout(0.4, name='ser_crnn_drop_lstm1')(x)

    x = Bidirectional(LSTM(32, return_sequences=False, name='ser_crnn_bilstm2', kernel_regularizer=l2(config.SER_L2_REG_STRENGTH)), name='ser_crnn_bidir2')(x)
    x = Dropout(0.4, name='ser_crnn_drop_lstm2')(x)

    # Yoğun Katmanlar
    x = Dense(128, activation='relu', kernel_initializer='he_normal', name='ser_crnn_fc1', kernel_regularizer=l2(config.SER_L2_REG_STRENGTH))(x)
    x = Dropout(0.4, name='ser_crnn_drop_fc1')(x)

    output = Dense(num_classes, activation='softmax', name='ser_crnn_output')(x)

    model = Model(inputs=mfcc_input, outputs=output, name='SER_CRNN_Model')
    logger.info("SER CRNN modeli başarıyla oluşturuldu.")
    return model

def build_ser_model_simple_dense_for_whisper(input_shape_dim, num_classes=config.NUM_TARGET_CLASSES, model_name="SER_SimpleDense_Whisper_Model"):
    """
    Whisper embeddingleri için basit bir Yoğun (Dense) katmanlı SER modeli oluşturur.
    Giriş şekli (embedding_dim,) olmalıdır.
    """
    logger.info(f"Basit Yoğun SER modeli (Whisper için) oluşturuluyor: {model_name}")
    logger.info(f"Giriş boyutları (embedding_dim): {input_shape_dim}")
    logger.info(f"Sınıf sayısı: {num_classes}")

    whisper_embedding_input = Input(shape=(input_shape_dim,), name="whisper_input") # (embedding_dim,)

    # Model katmanları
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(whisper_embedding_input)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    output_layer = Dense(num_classes, activation='softmax', name='output_emotion')(x)

    model = Model(inputs=whisper_embedding_input, outputs=output_layer, name=model_name)
    
    logger.info(f"{model_name} başarıyla oluşturuldu.")
    # model.summary(print_fn=logger.info) # Eğitim sırasında summary loglanacak
    return model


def create_ser_model(model_choice=None, input_shape=None, num_classes=None):
    """
    Yapılandırmaya göre seçilen SER modelini oluşturur ve döndürür.
    """
    if model_choice is None:
        model_choice = config.DEFAULT_SER_MODEL_CHOICE
    if num_classes is None:
        num_classes = len(config.TARGET_EMOTIONS)

    logger.info(f"SER Modeli oluşturuluyor. Seçim: {model_choice}")

    # CRNN modeliyle ilgili blok kaldırıldı, artık sadece simple_dense_whisper desteklenecek.
    # if model_choice == "crnn":
    #     n_features_input = config.AUDIO_N_MFCC * config.SER_FEATURE_DIM_MULTIPLIER if getattr(config, 'SER_USE_DELTA_MFCC', False) and config.SER_FEATURE_TYPE == 'mfcc' else config.AUDIO_N_MFCC
    #     input_shape_crnn = (n_features_input, config.AUDIO_MAX_FRAMES_MFCC)
    #     logger.info(f"CRNN için input_shape: {input_shape_crnn}")
    #     return build_ser_model_crnn(input_shape=input_shape_crnn, num_classes=num_classes)
    
    # Doğrudan config sabitini kullanarak kontrol et
    if model_choice == config.SER_MODEL_TYPE_SIMPLE_DENSE_WHISPER:
        if input_shape is None: 
            input_shape_whisper = config.WHISPER_EMBEDDING_DIM
        elif isinstance(input_shape, (tuple, list)) and len(input_shape) == 1: 
            input_shape_whisper = input_shape[0]
        elif isinstance(input_shape, int):
            input_shape_whisper = input_shape
        else:
            logger.error(f"'{config.SER_MODEL_TYPE_SIMPLE_DENSE_WHISPER}' için beklenmedik input_shape formatı: {input_shape}. Beklenen (embedding_dim,) veya int.")
            raise ValueError(f"'{config.SER_MODEL_TYPE_SIMPLE_DENSE_WHISPER}' için geçersiz input_shape: {input_shape}")
        logger.info(f"Simple Dense (Whisper) için input_shape_dim: {input_shape_whisper}")
        return build_ser_model_simple_dense_for_whisper(input_shape_dim=input_shape_whisper, num_classes=num_classes)
    else:
        logger.error(f"Bilinmeyen veya desteklenmeyen SER model seçimi: {model_choice}. Sadece '{config.SER_MODEL_TYPE_SIMPLE_DENSE_WHISPER}' desteklenmektedir.")
        raise ValueError(f"Geçersiz SER model seçimi: {model_choice}")

# Model seçimi için fabrika deseni (dictionary)
MODEL_FACTORY = {
    # config.FER_MODEL_TYPE_MINI_XCEPTION: build_fer_model_mini_xception, # Bu satır silindi
    config.FER_MODEL_TYPE_VGG16_TRANSFER: build_fer_model_vgg16_transfer,
    # config.SER_MODEL_TYPE_CRNN: build_ser_model_crnn, # CRNN modeli ve ilgili satır kaldırıldı.
    config.SER_MODEL_TYPE_SIMPLE_DENSE_WHISPER: build_ser_model_simple_dense_for_whisper,
}

# Ana çalıştırma bloğu (modül testi için)
if __name__ == '__main__':
    # Bu modülü test etmek için proje kökünden `python -m src.core.models`
    logger.info("--- Model Tanımlama Modülü Test Ediliyor (models.py) ---")

    # Gerekli dizinleri oluştur (config.APPLICATION_LOG_FILE için logs/ dizini gerekebilir)
    # Normalde bu ana scriptlerde yapılır.
    log_dir_test = os.path.dirname(config.APPLICATION_LOG_FILE)
    if log_dir_test and not os.path.exists(log_dir_test):
        os.makedirs(log_dir_test, exist_ok=True)

    # FER Modeli Testleri
    # logger.info("\nFER Basit CNN Testi:") # Silinecek
    # fer_model_s = build_fer_model_simple_cnn() # Silinecek
    # fer_model_s.summary(print_fn=logger.info) # Silinecek
    # dummy_fer_input_np = np.random.rand(2, config.FER_IMG_SIZE[0], config.FER_IMG_SIZE[1], 1).astype(np.float32) # Silinecek
    # try: # Silinecek
    #     pred_s = fer_model_s.predict(dummy_fer_input_np) # Silinecek
    #     logger.info(f"  FER Basit CNN örnek tahmin şekli: {pred_s.shape}") # Silinecek
    # except Exception as e: # Silinecek
    #     logger.error(f"  FER Basit CNN tahmin hatası: {e}") # Silinecek

    # Mini-Xception test bloğu da buradan silinecek (logger.info("\nFER Mini-Xception Testi:") ile başlayan kısım)

    # Whisper için Basit Yoğun Model Testi
    logger.info("\nSER Basit Yoğun Model (Whisper için) Testi:")
    # SER_FEATURE_TYPE kontrolü artık gereksiz, çünkü create_ser_model sadece whisper destekliyor.
    # if config.SER_FEATURE_TYPE == 'whisper': 
    whisper_input_dim_test = config.WHISPER_EMBEDDING_DIM
    # create_ser_model artık config.SER_MODEL_CHOICE'u (simple_dense_whisper) kullanacak.
    # num_classes_target'ın tanımlanması gerekiyor, eğer daha önce tanımlanmadıysa.
    # Genellikle config.NUM_TARGET_CLASSES kullanılır. Test için geçici olarak tanımlayalım veya config'den alalım.
    num_classes_target = config.NUM_TARGET_CLASSES
    ser_model_w = create_ser_model(input_shape=whisper_input_dim_test, num_classes=num_classes_target) 
    if ser_model_w: # Model None değilse devam et
        ser_model_w.summary(print_fn=logger.info)
        dummy_ser_input_whisper_np = np.random.rand(2, whisper_input_dim_test).astype(np.float32)
        try:
            pred_w = ser_model_w.predict(dummy_ser_input_whisper_np)
            logger.info(f"  SER Basit Yoğun (Whisper) örnek tahmin şekli: {pred_w.shape}")
        except Exception as e:
            logger.error(f"  SER Basit Yoğun (Whisper) tahmin hatası: {e}")
    else:
        logger.error("SER Basit Yoğun (Whisper) test modeli oluşturulamadı.")


    logger.info("--- Model Tanımlama Testleri Tamamlandı ---")
