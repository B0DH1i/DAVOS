# src/core/data_loader.py
import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import collections # Duygu sayımı için eklendi

# Göreli importlar
# utils, src/utils altında
# configs, src/configs altında
from ..utils.logging_utils import setup_logger
from ..configs import main_config as config

# audiomentations için try-except (opsiyonel bağımlılık)
try:
    from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
    AUDIOMENTATIONS_AVAILABLE = True
except ImportError:
    # setup_logger burada henüz çağrılmadığı için print kullanabiliriz veya
    # bu importu __init__.py gibi bir yerde yapıp global bir flag set edebiliriz.
    # Şimdilik, eğer bu modül import edildiğinde logger yoksa geçici bir print yapalım.
    # print("UYARI: audiomentations kütüphanesi bulunamadı. SER için veri artırma yapılamayacak.") # Logger kullanılacak
    Compose = None # Hata vermemesi için placeholder
    AUDIOMENTATIONS_AVAILABLE = False

# Whisper için global değişkenler (modeli ve işlemciyi bir kere yüklemek için)
WHISPER_PROCESSOR = None
WHISPER_MODEL = None
TORCH_DEVICE = None # Cihazı (CPU/GPU) belirlemek için

# Logger'ı bu modül için kur
# config.APPLICATION_LOG_FILE config modülünde tanımlı
logger = setup_logger(__name__, log_file=config.APPLICATION_LOG_FILE)

# audiomentations yüklenemediğinde log bas (logger artık tanımlı)
if not AUDIOMENTATIONS_AVAILABLE:
    logger.warning("audiomentations kütüphanesi bulunamadı. SER için veri artırma yapılamayacak.")

def _initialize_whisper():
    global WHISPER_PROCESSOR, WHISPER_MODEL, TORCH_DEVICE
    if WHISPER_PROCESSOR is None or WHISPER_MODEL is None:
        try:
            import torch
            from transformers import WhisperProcessor, WhisperModel

            TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Whisper için PyTorch cihazı: {TORCH_DEVICE}")

            logger.info(f"Whisper işlemcisi yükleniyor: {config.WHISPER_MODEL_NAME}")
            WHISPER_PROCESSOR = WhisperProcessor.from_pretrained(config.WHISPER_MODEL_NAME)
            
            logger.info(f"Whisper modeli yükleniyor: {config.WHISPER_MODEL_NAME}")
            WHISPER_MODEL = WhisperModel.from_pretrained(config.WHISPER_MODEL_NAME).to(TORCH_DEVICE)
            WHISPER_MODEL.eval() # Modeli değerlendirme moduna al (dropout vs. etkilenmesin)
            logger.info("Whisper modeli ve işlemcisi başarıyla yüklendi.")

        except ImportError:
            logger.error("Whisper kullanmak için 'transformers' ve 'torch' kütüphaneleri gerekli. Lütfen kurun.")
            raise
        except Exception as e:
            logger.error(f"Whisper modeli veya işlemcisi yüklenirken hata: {e}")
            raise

# Logger'ı bu modül için kur
# config.APPLICATION_LOG_FILE config modülünde tanımlı
logger = setup_logger(__name__, log_file=config.APPLICATION_LOG_FILE)


def load_fer_data(prepared_data_path=None, img_size=None, num_classes=None):
    """
    Hazırlanmış FERPlus veri setini (PNG görüntüleri ve label.csv dosyaları) yükler.
    Veri, 'prepared_data_path' altında 'FER2013Train', 'FER2013Valid', 'FER2013Test'
    alt klasörlerinde bulunmalıdır. Her bir alt klasörde bir 'label.csv' ve 
    karşılık gelen '.png' görüntü dosyaları olmalıdır.

    Returns:
        tuple: ((X_train, y_train), (X_val, y_val), (X_test, y_test)) veya hata durumunda None.
               X: Görüntü verileri (NumPy array, [samples, height, width, 1])
               y: Etiketler (NumPy array, [samples, num_classes] - one-hot encoded)
    """
    if prepared_data_path is None:
        prepared_data_path = config.FERPLUS_PREPARED_DATA_PATH
    if img_size is None:
        img_size = config.FER_IMG_SIZE # (48, 48)
    if num_classes is None:
        num_classes = config.NUM_MODEL_OUTPUT_CLASSES # YENİ: NUM_FER_CLASSES yerine NUM_MODEL_OUTPUT_CLASSES kullanıldı

    logger.info(f"Hazırlanmış FERPlus veri seti yükleniyor: {prepared_data_path}")
    if not os.path.exists(prepared_data_path):
        logger.error(f"FERPlus hazırlanmış veri klasörü bulunamadı: {prepared_data_path}")
        return None

    # Görüntüleri yüklemek için OpenCV gerekli olacak.
    try:
        import cv2
    except ImportError:
        logger.error("FERPlus görüntülerini yüklemek için OpenCV (cv2) kütüphanesi gerekli. Lütfen kurun.")
        return None

    data_splits = {"train": "FER2013Train", "validation": "FER2013Valid", "test": "FER2013Test"}
    X_data_map = {split: [] for split in data_splits.keys()}
    y_data_map = {split: [] for split in data_splits.keys()}
    
    # config.FER_EMOTIONS ana duygu etiketlerini içerir (contempt dahil, unknown ve NF hariç)
    emotion_columns = config.MODEL_OUTPUT_EMOTIONS # YENİ: FER_EMOTIONS yerine MODEL_OUTPUT_EMOTIONS kullanıldı

    for split_name, folder_name in data_splits.items():
        split_folder_path = os.path.join(prepared_data_path, folder_name)
        labels_csv_path = os.path.join(split_folder_path, "label.csv")

        if not os.path.exists(split_folder_path) or not os.path.exists(labels_csv_path):
            logger.warning(f"FERPlus için '{split_name}' veri split klasörü ({split_folder_path}) veya label.csv bulunamadı. Bu split atlanıyor.")
            continue
        
        logger.info(f"'{split_name}' ({folder_name}) için label.csv okunuyor: {labels_csv_path}")
        try:
            # Başlık satırı olmadığını belirtiyoruz, sütunlar sayılarla indekslenecek (0, 1, 2...)
            labels_df = pd.read_csv(labels_csv_path, header=None)
            # Beklenen minimum sütun sayısını kontrol edelim (görüntü adı + koordinat + 10 duygu skoru = 12 sütun)
            if labels_df.shape[1] < 12:
                logger.error(f"'{labels_csv_path}' dosyasında yeterli sütun bulunmuyor (en az 12 bekleniyordu, {labels_df.shape[1]} bulundu). Bu split atlanıyor.")
                continue
        except Exception as e:
            logger.error(f"'{labels_csv_path}' okunurken hata: {e}. Bu split atlanıyor.")
            continue

        # Yorumlar kaldırılıyor
        logger.info(f"'{split_name}' için {len(labels_df)} etiket satırı okundu (başlıksız CSV).")

        for index, row in tqdm(labels_df.iterrows(), total=labels_df.shape[0], desc=f"FERPlus '{split_name}' İşleniyor"):
            try:
                image_filename = row[0] # Görüntü adı ilk sütunda (indeks 0)
                # row[1] koordinat bilgisi, şimdilik kullanılmıyor.
                image_path = os.path.join(split_folder_path, image_filename)

                if not os.path.exists(image_path):
                    logger.warning(f"Görüntü dosyası bulunamadı: {image_path}. Satır {index} atlanıyor.")
                    continue

                img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img_array is None:
                    logger.warning(f"Görüntü yüklenemedi (cv2.imread None döndü): {image_path}. Satır {index} atlanıyor.")
                    continue
                
                resized_img = cv2.resize(img_array, img_size, interpolation=cv2.INTER_AREA)
                reshaped_img = np.expand_dims(resized_img, axis=-1)
                normalized_img = reshaped_img / 255.0

                # 10 FERPlus duygu skorunu al (3. ila 12. sütunlar, yani indeks 2'den 11'e kadar)
                all_ferplus_scores = row.iloc[2:12].values.astype(float)
                
                if len(all_ferplus_scores) != len(config.FERPLUS_LABEL_COLUMNS):
                    logger.warning(f"Satır {index} ({image_filename}) için beklenen 10 duygu skoru ({len(config.FERPLUS_LABEL_COLUMNS)}) alınamadı, {len(all_ferplus_scores)} skor alındı. Bu satır atlanıyor.")
                    continue

                # Bu 10 skoru, config.FERPLUS_LABEL_COLUMNS sırasına göre bir Series'e dönüştürelim
                # Bu, config.FER_EMOTIONS (8 ana duygu) için doğru skorları seçmeyi kolaylaştırır.
                ferplus_scores_series = pd.Series(all_ferplus_scores, index=config.FERPLUS_LABEL_COLUMNS)
                
                # Sadece modelin eğitileceği 8 ana duygunun skorlarını al
                # emotion_columns = config.FER_EMOTIONS (bu zaten doğru)
                selected_emotion_scores = ferplus_scores_series[emotion_columns].values.astype(float)
                
                dominant_emotion_index = np.argmax(selected_emotion_scores)
                
                one_hot_label = np.zeros(num_classes) # num_classes = len(config.FER_EMOTIONS) = 8
                if 0 <= dominant_emotion_index < num_classes:
                    one_hot_label[dominant_emotion_index] = 1.0
                else:
                    logger.warning(f"Geçersiz dominant duygu indeksi {dominant_emotion_index} ({num_classes} sınıf için). Görüntü: {image_filename}. Bu satır atlanıyor.")
                    continue
                
                X_data_map[split_name].append(normalized_img)
                y_data_map[split_name].append(one_hot_label)

            except Exception as e:
                logger.warning(f"Satır {index} ({image_filename if 'image_filename' in locals() else 'Bilinmeyen'}) işlenirken hata: {e}. Bu satır atlanıyor.")

    X_train = np.array(X_data_map['train'])
    y_train = np.array(y_data_map['train'])
    X_val = np.array(X_data_map['validation'])
    y_val = np.array(y_data_map['validation'])
    X_test = np.array(X_data_map['test'])
    y_test = np.array(y_data_map['test'])

    logger.info("FERPlus Yükleme Tamamlandı:")
    if X_train.size > 0: logger.info(f"  Eğitim seti: Görüntüler {X_train.shape}, Etiketler {y_train.shape}")
    else: logger.warning("FERPlus eğitim seti boş!")
    if X_val.size > 0: logger.info(f"  Doğrulama seti: Görüntüler {X_val.shape}, Etiketler {y_val.shape}")
    else: logger.warning("FERPlus doğrulama seti boş!")
    if X_test.size > 0: logger.info(f"  Test seti: Görüntüler {X_test.shape}, Etiketler {y_test.shape}")
    else: logger.warning("FERPlus test seti boş (bu beklenen bir durum olabilir).")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def extract_audio_features_from_file(audio_path,
                                   sr_target=config.WHISPER_SAMPLING_RATE, # Whisper için hedef sr
                                   apply_augmentation=False,
                                   augmenter=None,
                                   # Debugging parameters, not used by core logic directly
                                   emotion_label_for_debug=None,
                                   actor_id_for_debug=None,
                                   data_path_for_debug=None):
    """
    Bir ses dosyasından Whisper embedding özelliklerini çıkarır.

    Args:
        audio_path (str): Ses dosyasının yolu.
        sr_target (int): Whisper için hedef örnekleme oranı.
        apply_augmentation (bool): Ses artırma uygulanıp uygulanmayacağı.
        augmenter (callable, opsiyonel): Ses artırma için kullanılacak nesne.
        emotion_label_for_debug, actor_id_for_debug, data_path_for_debug: Hata ayıklama için.

    Returns:
        tuple (np.array, int) or (None, 0):
            - np.array: Whisper embedding vektörü (config.WHISPER_EMBEDDING_DIM,) boyutunda.
            - int: Embedding boyutu.
            Hata durumunda (None, 0).
    """
    original_feature_info = 0 # Embedding boyutu

    try:
        _initialize_whisper() # Gerekliyse Whisper modelini yükle
        if WHISPER_PROCESSOR is None or WHISPER_MODEL is None: # Yükleme başarısız olduysa
            logger.error("extract_audio_features_from_file: Whisper modeli/işlemcisi yüklenemedi.")
            return None, original_feature_info

        y, sr_loaded = librosa.load(audio_path, sr=sr_target) # sr_target config.WHISPER_SAMPLING_RATE olmalı

        if apply_augmentation and augmenter and AUDIOMENTATIONS_AVAILABLE:
            try:
                y = augmenter(samples=y, sample_rate=sr_loaded)
            except Exception as aug_e:
                logger.warning(f"'{audio_path}' için Whisper öncesi ses artırma sırasında hata: {aug_e}. Orijinal veri kullanılıyor.")

        inputs = WHISPER_PROCESSOR(y, sampling_rate=sr_target, return_tensors="pt").to(TORCH_DEVICE)

        import torch # torch.no_grad() için
        with torch.no_grad():
            encoder_outputs = WHISPER_MODEL.encoder(inputs.input_features)
            last_hidden_state = encoder_outputs.last_hidden_state

        embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        if embedding.ndim == 0: # Eğer skaler bir değere sıkışırsa (çok kısa seslerde olabilir)
             logger.warning(f"Whisper embedding skaler bir değere dönüştü (muhtemelen çok kısa ses). Dosya: {audio_path}. Boyut: {embedding.shape}")
             # Uygun bir şekilde ele almak için None döndürebilir veya uygun boyutta bir sıfır vektörü döndürebiliriz.
             # Şimdilik None döndürmek daha güvenli.
             return None, 0

        if embedding.shape[0] != config.WHISPER_EMBEDDING_DIM:
            logger.warning(f"Whisper embedding boyutu beklenenden farklı! Beklenen: {config.WHISPER_EMBEDDING_DIM}, Gelen: {embedding.shape[0]}. Dosya: {audio_path}")
            return None, 0

        original_feature_info = embedding.shape[0]
        return embedding, original_feature_info

    except Exception as e:
        logger.error(f"Ses dosyası ({audio_path}) işlenirken hata (extract_audio_features_from_file): {e}", exc_info=True)
        return None, 0


def load_ravdess_data(data_path=None,
                      use_augmentation_on_train=config.USE_DATA_AUGMENTATION_SER, # config'den al
                      target_sample_rate=config.WHISPER_SAMPLING_RATE, # Whisper için
                      target_actors=None, # Belirli aktörleri yüklemek için (örn: sadece test)
                      specific_split_only=None): # 'train', 'validation', 'test' (target_actors ile kullanılır)
    """
    RAVDESS veri setini yükler, ses dosyalarından Whisper embedding özelliklerini çıkarır
    ve eğitim, doğrulama, test setlerine böler.
    """
    if data_path is None:
        data_path = config.RAVDESS_DATA_PATH

    logger.info(f"RAVDESS veri seti yükleniyor: {data_path}")
    if not os.path.exists(data_path) or not os.path.isdir(data_path):
        logger.error(f"RAVDESS veri yolu bulunamadı veya bir dizin değil: {data_path}")
        return None

    features_map = {'train': [], 'val': [], 'test': []}
    labels_map = {'train': [], 'val': [], 'test': []}
    emotion_counts_map = {
        'train': collections.Counter(), 
        'val': collections.Counter(), 
        'test': collections.Counter()
    } # Hedef duygu sayılarını tutmak için -> MODEL_OUTPUT_EMOTIONS'a göre olacak
    all_original_feature_info_list = [] # Tüm orijinal frame sayılarını tutmak için -> all_original_feature_info olarak genelleştirilebilir
    
    # YENİ: Modelin çıktı formatına göre etiketler oluşturulacak (8 sınıf)
    model_output_emotion_to_idx = {emotion: i for i, emotion in enumerate(config.MODEL_OUTPUT_EMOTIONS)}
    num_model_output_classes = len(config.MODEL_OUTPUT_EMOTIONS)

    # Ses artırma için augmenter (sadece eğitim setine uygulanacak)
    ravdess_augmenter = None
    if use_augmentation_on_train and AUDIOMENTATIONS_AVAILABLE and Compose:
        ravdess_augmenter = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.85, max_rate=1.15, p=0.5, leave_length_unchanged=False), # leave_length_unchanged=False önemli
            PitchShift(min_semitones=-3, max_semitones=3, p=0.5)
        ])
        logger.info("RAVDESS eğitim verisi için ses artırma etkinleştirildi (p=0.5).")

    actor_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and d.startswith("Actor_")]
    if not actor_folders:
        logger.error(f"RAVDESS veri yolunda ({data_path}) 'Actor_' ile başlayan klasör bulunamadı.")
        return None
    logger.info(f"{len(actor_folders)} aktör klasörü bulundu. İşleniyor...")

    for actor_folder in tqdm(actor_folders, desc="RAVDESS Aktörleri İşleniyor"):
        actor_path = os.path.join(data_path, actor_folder)
        
        files_in_actor_path = os.listdir(actor_path)
        # logger.debug(f"Aktör: {actor_folder}, os.listdir sonucu: {files_in_actor_path} (type: {type(files_in_actor_path)})") # Bu log artık gereksiz
        
        wav_files = [f for f in files_in_actor_path if f.endswith('.wav')]
        # logger.debug(f"Aktör: {actor_folder}, Filtrelenmiş wav_files: {wav_files}") # Bu log artık gereksiz
        # logger.debug(f"Aktör: {actor_folder}, len(wav_files): {len(wav_files)}") # Bu log artık gereksiz
        # if wav_files:
            # logger.debug(f"Aktör: {actor_folder}, type(wav_files[0]): {type(wav_files[0])}") # Bu log artık gereksiz

        for wav_file_name in list(wav_files):
            # logger.debug(f"  Processing file: {wav_file_name} in actor_folder: {actor_folder}") # Bu log da artık gereksiz olabilir
            try:
                parts = wav_file_name.split('.')[0].split('-')
                if len(parts) < 7:
                    logger.warning(f"Dosya adı ({wav_file_name}) beklenenden kısa (7 bölüm bekleniyor). Atlanıyor.")
                    continue
                
                emotion_code_from_file = parts[2]
                actor_id_from_file = int(parts[6])

                # DEBUG: Log the raw emotion code and actor ID
                # logger.info(f"DEBUG: Dosya: {wav_file_name}, Ham Duygu Kodu: {emotion_code_from_file}, Aktör ID: {actor_id_from_file}")

                ravdess_emotion_text = config.RAVDESS_EMOTIONS_FROM_FILENAME.get(emotion_code_from_file)
                if not ravdess_emotion_text:
                    logger.warning(f"Dosya adı ({wav_file_name}): Bilinmeyen RAVDESS duygu kodu '{emotion_code_from_file}'. Atlanıyor.")
                    continue
                
                # DEBUG: Log the parsed RAVDESS emotion
                # logger.info(f"DEBUG: Dosya: {wav_file_name}, Çözümlenmiş RAVDESS Duygusu: {ravdess_emotion_text}")

                model_emotion_text = config.RAVDESS_TO_MODEL_OUTPUT_MAP.get(ravdess_emotion_text)
                if not model_emotion_text:
                    logger.warning(f"Dosya ({wav_file_name}): RAVDESS duygusu '{ravdess_emotion_text}' için MODEL_OUTPUT_EMOTIONS'a eşleşme yok (RAVDESS_TO_MODEL_OUTPUT_MAP). Atlanıyor.")
                    continue
                
                # DEBUG: Log the mapped target emotion
                # logger.info(f"DEBUG: Dosya: {wav_file_name}, Hedef Duyguya Eşlendi: {target_emotion_text}")
                
                numeric_label = model_output_emotion_to_idx.get(model_emotion_text)
                if numeric_label is None:
                    logger.error(f"Kritik: Model çıktısı duygusu '{model_emotion_text}' (RAVDESS: {ravdess_emotion_text}) için config.MODEL_OUTPUT_EMOTIONS içinde sayısal etiket bulunamadı. Lütfen config'i kontrol edin. Dosya: {wav_file_name}")
                    continue
                
                label_one_hot = np.zeros(num_model_output_classes)
                label_one_hot[numeric_label] = 1.0

                # --- DOĞRU current_set_key ve apply_aug_for_this_file ATAMA BLOĞU ---
                current_set_key = None
                apply_aug_for_this_file = False # Varsayılan olarak artırma kapalı

                if actor_id_from_file in config.RAVDESS_TRAIN_ACTORS:
                    current_set_key = 'train'
                    apply_aug_for_this_file = use_augmentation_on_train # Sadece eğitim setine artırma uygula
                elif actor_id_from_file in config.RAVDESS_VAL_ACTORS:
                    current_set_key = 'val'
                elif actor_id_from_file in config.RAVDESS_TEST_ACTORS:
                    current_set_key = 'test'
                else:
                    logger.debug(f"Aktör {actor_id_from_file} (dosya: {wav_file_name}) herhangi bir sete atanmadı. Atlanıyor.")
                    continue
                # --- BİTİŞ: DOĞRU current_set_key ve apply_aug_for_this_file ATAMA BLOĞU ---
                
                file_path_str = os.path.join(actor_path, wav_file_name)

                features, feature_info = extract_audio_features_from_file(
                    file_path_str,
                    sr_target=target_sample_rate, # Whisper için
                    apply_augmentation=apply_aug_for_this_file,
                    augmenter=ravdess_augmenter,
                    # Debug
                    emotion_label_for_debug=model_emotion_text, # Debug için eklendi
                    actor_id_for_debug=actor_id_from_file,  # Debug için eklendi
                    data_path_for_debug=data_path           # Debug için eklendi
                )
                
                if features is not None:
                    features_map[current_set_key].append(features)
                    labels_map[current_set_key].append(label_one_hot)
                    emotion_counts_map[current_set_key][model_emotion_text] += 1
                    # Sadece sayısal olan (frame sayısı veya embedding boyutu) bilgiyi sakla
                    if isinstance(feature_info, (int, float)):
                         all_original_feature_info_list.append(feature_info)
                else:
                    logger.warning(f"Dosya {wav_file_name} için özellik çıkarılamadı. Atlanıyor.")
            except Exception as e:
                logger.error(f"Aktör {actor_folder}, dosya {wav_file_name} işlenirken genel hata: {e}", exc_info=True)
                # Hata durumunda bu dosyayı atla ve devam et
                continue 
    
    # Verileri NumPy array'lerine çevirme
    # dtype=object geçici olarak kullanılıp sonra vstack ile birleştiriliyordu, bu MFCC için sorun yaratabilir.
    # Doğrudan float32 ile ve doğru şekli koruyarak oluşturmalıyız.
    
    X_train_list = features_map['train']
    y_train_list = labels_map['train']
    X_val_list = features_map['val']
    y_val_list = labels_map['val']
    X_test_list = features_map['test']
    y_test_list = labels_map['test']

    # Özelliklerin (X) ve etiketlerin (y) boş olup olmadığını kontrol et ve NumPy array'e çevir.
    # extract_audio_features_from_file zaten float32 numpy array döndürmeli.
    X_train = np.array(X_train_list, dtype=np.float32) if X_train_list else np.empty((0, config.WHISPER_EMBEDDING_DIM))
    y_train = np.array(y_train_list, dtype=np.float32) if y_train_list else np.empty((0, len(config.MODEL_OUTPUT_EMOTIONS)))
    X_val = np.array(X_val_list, dtype=np.float32) if X_val_list else np.empty((0, config.WHISPER_EMBEDDING_DIM))
    y_val = np.array(y_val_list, dtype=np.float32) if y_val_list else np.empty((0, len(config.MODEL_OUTPUT_EMOTIONS)))
    X_test = np.array(X_test_list, dtype=np.float32) if X_test_list else np.empty((0, config.WHISPER_EMBEDDING_DIM))
    y_test = np.array(y_test_list, dtype=np.float32) if y_test_list else np.empty((0, len(config.MODEL_OUTPUT_EMOTIONS)))

    logger.info(f"RAVDESS Yükleme ve Özellik Çıkarma Tamamlandı:")
    logger.info(f"  Eğitim seti: Özellikler {X_train.shape}, Etiketler {y_train.shape}")
    logger.info(f"  Doğrulama seti: Özellikler {X_val.shape}, Etiketler {y_val.shape}")
    logger.info(f"  Test seti: Özellikler {X_test.shape}, Etiketler {y_test.shape}")

    logger.info("RAVDESS Duygu Dağılımları (MODEL_OUTPUT_EMOTIONS'a göre):")
    for set_name in ['train', 'val', 'test']:
        if emotion_counts_map[set_name]:
            logger.info(f"  {set_name.capitalize()} seti: {dict(emotion_counts_map[set_name])}")
        else:
            logger.info(f"    - ({set_name.capitalize()} seti için veri yok veya sayılamadı)")

    if all_original_feature_info_list:
        feature_info_array = np.array(all_original_feature_info_list)
        logger.info("Tüm veri seti için orijinal özellik bilgisi istatistikleri:")
        logger.info(f"  İşlenen dosya sayısı (özellik bilgisi olan): {len(feature_info_array)}")
        logger.info(f"  Min: {np.min(feature_info_array)}")
        logger.info(f"  Max: {np.max(feature_info_array)}")
        logger.info(f"  Ortalama: {np.mean(feature_info_array):.2f}")
        logger.info(f"  Medyan: {np.median(feature_info_array)}")
    else:
        logger.warning("Hiçbir ses dosyası için orijinal özellik bilgisi (frame sayısı/embedding boyutu) toplanamadı.")

    if X_train.size == 0: logger.warning("RAVDESS eğitim seti boş!")
    if X_val.size == 0: logger.warning("RAVDESS doğrulama seti boş!")
    # Test seti boş olabilir, bu bir uyarı olmalı ama kritik hata değil.
    if X_test.size == 0: logger.warning("RAVDESS test seti boş (bu beklenen bir durum olabilir).")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
