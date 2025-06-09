# src/utils/file_utils.py
import os
import json
import pickle
import tensorflow as tf # Standart import'a dön
import re # Regex için
import shutil
from datetime import datetime

# Göreli importlar
from .logging_utils import setup_logger # Aynı utils paketi içinden
from ..configs import main_config as config # Bir üstteki configs paketinden

# Logger'ı bu modül için kur
logger = setup_logger(__name__, log_file=config.APPLICATION_LOG_FILE)

def create_project_directories():
    """
    Proje için gerekli olan temel dizinleri (config'de tanımlanan) oluşturur.
    Bu fonksiyon, ana script'ler tarafından programın başında çağrılmalıdır.
    """
    logger.info("Proje için gerekli dizinler kontrol ediliyor/oluşturuluyor...")
    dirs_to_create = [
        config.DATA_PATH, config.FER2013_DIR, config.RAVDESS_DATA_PATH,
        config.TRAINED_MODELS_PATH, config.LOGS_PATH, config.PLOTS_PATH,
        os.path.join(config.LOGS_PATH, "fer"), # TensorBoard için spesifik log alt klasörleri
        os.path.join(config.LOGS_PATH, "ser"),
    ]
    # Ana uygulama log dosyasının dizinini de ekleyelim
    app_log_dir = os.path.dirname(config.APPLICATION_LOG_FILE)
    if app_log_dir: # Eğer APPLICATION_LOG_FILE bir yol ise
        dirs_to_create.append(app_log_dir)


    for dir_path in dirs_to_create:
        if not dir_path: # Eğer yol boşsa atla (örn: SOUNDS_PATH boş tanımlanmışsa)
            logger.warning(f"Boş bir dizin yolu atlandı: '{dir_path}'")
            continue
        try:
            os.makedirs(dir_path, exist_ok=True)
            # logger.debug(f"Dizin başarıyla kontrol edildi/oluşturuldu: {dir_path}")
        except OSError as e:
            logger.error(f"Dizin oluşturulamadı: {dir_path} - Hata: {e}")
    logger.info("Gerekli dizinlerin kontrolü/oluşturulması tamamlandı.")


def save_model_and_history(model, history_data, model_name_prefix):
    """
    Eğitilmiş modeli (.h5), eğitim geçmişini (.pkl) ve model yapısını (.json) kaydeder.
    Kaydedilen dosyalar TRAINED_MODELS_PATH/model_name_prefix/ altına yerleştirilir.

    Args:
        model (tf.keras.Model): Eğitilmiş Keras modeli.
        history_data (dict): Modelin eğitim geçmişi sözlüğü (model.fit().history).
        model_name_prefix (str): Model için benzersiz bir önek (genellikle model adı + zaman damgası).
    """
    model_save_dir = os.path.join(config.TRAINED_MODELS_PATH, model_name_prefix)
    try:
        os.makedirs(model_save_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Model kayıt dizini ({model_save_dir}) oluşturulamadı: {e}")
        return False # Kayıt başarısız

    model_path_h5 = os.path.join(model_save_dir, f"{model_name_prefix}_model.h5")
    history_path_pkl = os.path.join(model_save_dir, f"{model_name_prefix}_history.pkl")
    architecture_path_json = os.path.join(model_save_dir, f"{model_name_prefix}_architecture.json")

    success_flags = {"model": False, "history": False, "architecture": False}

    # Modeli H5 formatında kaydet
    try:
        model.save(model_path_h5)
        logger.info(f"Model başarıyla kaydedildi: {model_path_h5}")
        success_flags["model"] = True
    except Exception as e:
        logger.error(f"Model H5 olarak kaydedilirken hata oluştu ({model_path_h5}): {e}")

    # Eğitim geçmişini pickle ile kaydet
    if history_data and isinstance(history_data, dict) and history_data:
        try:
            with open(history_path_pkl, 'wb') as f:
                pickle.dump(history_data, f)
            logger.info(f"Eğitim geçmişi başarıyla kaydedildi: {history_path_pkl}")
            success_flags["history"] = True
        except Exception as e:
            logger.error(f"Eğitim geçmişi pickle ile kaydedilirken hata ({history_path_pkl}): {e}", exc_info=True)
    else:
        logger.warning(f"Geçersiz veya boş eğitim geçmişi sözlüğü (history_data). Geçmiş kaydedilmedi: {model_name_prefix}. Tip: {type(history_data)}, İçerik (anahtarlar): {list(history_data.keys()) if isinstance(history_data, dict) else 'Dict değil'}")

    # Model mimarisini JSON olarak kaydet
    try:
        model_json = model.to_json(indent=4) # Daha okunaklı JSON için indent
        with open(architecture_path_json, "w", encoding="utf-8") as json_file:
            json_file.write(model_json)
        logger.info(f"Model mimarisi başarıyla kaydedildi: {architecture_path_json}")
        success_flags["architecture"] = True
    except Exception as e:
        logger.error(f"Model mimarisi JSON olarak kaydedilirken hata ({architecture_path_json}): {e}")
        
    return all(success_flags.values()) # Hepsi başarılıysa True döner


def load_trained_model(model_identifier_prefix, custom_objects=None):
    """
    Kaydedilmiş bir Keras modelini yükler.
    model_identifier_prefix, TRAINED_MODELS_PATH altındaki klasör adıdır.

    Args:
        model_identifier_prefix (str): Modelin kaydedildiği klasörün adı
                                       (örn: "fer_mini_xception_YYYYMMDD-HHMMSS").
        custom_objects (dict, opsiyonel): Modelde özel katmanlar veya fonksiyonlar varsa.

    Returns:
        tf.keras.Model or None: Yüklenen model veya hata durumunda None.
    """
    if not model_identifier_prefix:
        logger.error("Model yüklemek için model öneki (klasör adı) belirtilmedi.")
        return None
        
    # Standart model adını oluştur
    standard_model_filename = f"{model_identifier_prefix}_model.h5"
    model_dir_path = os.path.join(config.TRAINED_MODELS_PATH, model_identifier_prefix)
    model_path_h5 = os.path.join(model_dir_path, standard_model_filename)
    
    # Alternatif checkpoint dosya adını oluştur
    checkpoint_model_filename = "best_model_checkpoint.h5"
    checkpoint_model_path_h5 = os.path.join(model_dir_path, checkpoint_model_filename)

    # Önce standart adı kontrol et
    if not os.path.exists(model_path_h5):
        logger.warning(f"Standart model dosyası ({standard_model_filename}) bulunamadı: {model_path_h5}. Alternatif checkpoint dosyası deneniyor...")
        # Standart dosya yoksa, checkpoint dosyasını kullanmayı dene
        if os.path.exists(checkpoint_model_path_h5):
            logger.info(f"Alternatif checkpoint dosyası ({checkpoint_model_filename}) bulundu: {checkpoint_model_path_h5}. Bu dosya kullanılacak.")
            model_path_h5 = checkpoint_model_path_h5 # Yüklenecek yolu güncelle
        else:
            logger.error(f"Ne standart model dosyası ({standard_model_filename}) ne de alternatif checkpoint dosyası ({checkpoint_model_filename}) bulunamadı: {model_dir_path}")
            return None
    
    # Eğer buraya gelindiyse, model_path_h5 ya orijinal standart yoldur ya da geçerli bir checkpoint yoludur.
    # Artık model_path_h5'in var olduğunu varsayabiliriz (önceki bloktan None dönmediyse)
    logger.info(f"Model yükleniyor: {model_path_h5}")
    try:
        model = tf.keras.models.load_model(model_path_h5, custom_objects=custom_objects, compile=True)
        # compile=True, modeli eğitimden geldiği gibi (optimizer state dahil) yükler.
        # Eğer sadece çıkarım yapılacaksa compile=False daha hızlı olabilir ve optimizer state'ini atlar.
        # Ancak, model.evaluate() gibi fonksiyonlar için compile=True gerekebilir.
        logger.info(f"Model başarıyla yüklendi: {model_path_h5}")
        return model
    except Exception as e:
        logger.error(f"Model yüklenirken hata ({model_path_h5}): {e}", exc_info=True)
        return None


def load_training_history(model_identifier_prefix):
    """
    Kaydedilmiş eğitim geçmişini (pickle dosyası) yükler.

    Args:
        model_identifier_prefix (str): Modelin kaydedildiği klasörün adı.

    Returns:
        dict or None: Yüklenen eğitim geçmişi sözlüğü veya hata durumunda None.
    """
    if not model_identifier_prefix:
        logger.error("Geçmiş yüklemek için model öneki (klasör adı) belirtilmedi.")
        return None

    history_path_pkl = os.path.join(config.TRAINED_MODELS_PATH, model_identifier_prefix, f"{model_identifier_prefix}_history.pkl")

    if not os.path.exists(history_path_pkl):
        logger.warning(f"Eğitim geçmişi dosyası bulunamadı: {history_path_pkl}")
        return None
        
    try:
        with open(history_path_pkl, 'rb') as f:
            history_dict = pickle.load(f)
        logger.info(f"Eğitim geçmişi başarıyla yüklendi: {history_path_pkl}")
        return history_dict
    except Exception as e:
        logger.error(f"Eğitim geçmişi yüklenirken hata ({history_path_pkl}): {e}")
        return None


def is_specific_model_version(model_name_str):
    """
    Verilen model adının belirli bir versiyonu (zaman damgası içeren)
    olup olmadığını kontrol eder.
    Örnek format: *_YYYYMMDD-HHMMSS
    """
    if not model_name_str or not isinstance(model_name_str, str):
        return False
    # Desen: Herhangi bir karakterle başlayıp (_) sonra (8 rakam)-(6 rakam) ile bitiyor mu?
    # Örnek: fer_model_mini_xception_20230101-120000
    match = re.search(r"_(\d{8}-\d{6})$", model_name_str)
    return bool(match)


def get_latest_model_directory_for_base(base_model_name):
    """
    TRAINED_MODELS_PATH içinde base_model_name ile başlayan
    en son tarihli klasörün tam adını (prefix) döndürür.
    Örn: base_model_name='fer_mini_xception' ise
         'fer_mini_xception_YYYYMMDD-HHMMSS' gibi bir şey döner.

    Args:
        base_model_name (str): Modelin temel adı (örn: 'fer_mini_xception').

    Returns:
        str or None: En son modelin klasör adı veya bulunamazsa None.
    """
    trained_models_dir = config.TRAINED_MODELS_PATH
    if not os.path.isdir(trained_models_dir):
        logger.warning(f"Eğitilmiş modeller dizini bulunamadı: {trained_models_dir}")
        return None

    candidate_folders = []
    for f_name in os.listdir(trained_models_dir):
        if os.path.isdir(os.path.join(trained_models_dir, f_name)) and f_name.startswith(base_model_name):
            # Klasör adının sonundaki tarih damgasını kontrol et
            # Örnek format: base_model_name_YYYYMMDD-HHMMSS
            parts = f_name.split('_') # _ ile ayır
            if len(parts) > 1:
                date_time_stamp = parts[-1] # Son parça tarih damgası olmalı
                if len(date_time_stamp) == 15 and date_time_stamp[8] == '-': # YYYYMMDD-HHMMSS formatı
                    candidate_folders.append(f_name)
    
    if not candidate_folders:
        logger.warning(f"'{base_model_name}' için TRAINED_MODELS_PATH altında uygun formatta model klasörü bulunamadı.")
        return None

    # Tarih damgasına göre tersten sırala (en son olan en başa gelsin)
    # Tarih damgası string formatında (YYYYMMDD-HHMMSS) olduğu için string karşılaştırması işe yarar.
    candidate_folders.sort(key=lambda x: x.split('_')[-1], reverse=True)
    
    latest_model_dir_name = candidate_folders[0]
    logger.info(f"En son model klasörü bulundu ({base_model_name}): {latest_model_dir_name}")
    return latest_model_dir_name
