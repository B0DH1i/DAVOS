# src/main_trainer.py
import argparse
import sys

# Proje kök dizinini Python path'ine eklemek gerekebilir,
# eğer src bir paket olarak tanınmıyorsa ve bu script src dışından çalıştırılıyorsa.
# Ancak, `python -m src.main_trainer` ile çalıştırılırsa genellikle gerek kalmaz.
# import os
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # src/main_trainer.py'nin olduğu dizin (src)
# PROJECT_ROOT_FROM_TRAINER = os.path.dirname(SCRIPT_DIR) # src'nin bir üstü (proje kökü)
# if PROJECT_ROOT_FROM_TRAINER not in sys.path:
#    sys.path.append(PROJECT_ROOT_FROM_TRAINER)

# Göreli importlar (eğer bu dosya src içinde bir modül gibi kabul ediliyorsa)
# veya mutlak importlar (eğer src PYTHONPATH'te ise)
# python -m src.main_trainer için göreli importlar çalışmalı.
from .utils.logging_utils import setup_logger
from .utils import file_utils # Dizin oluşturma için
from .configs import main_config as config # Varsayılan ayarlar için
from .training import train_pipeline # Eğitim fonksiyonlarını içerir

# Logger'ı bu ana script için kur
# config.APPLICATION_LOG_FILE log dosyası yolu olarak kullanılacak
logger = setup_logger("MainTrainer", log_file=config.APPLICATION_LOG_FILE)

def main():
    parser = argparse.ArgumentParser(
        description="Duygu Tanıma Modellerini Eğitme Script'i.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Varsayılan değerleri yardım mesajında göster
    )
    
    # Hangi modelin eğitileceği
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=['fer', 'ser'],
        help="Eğitilecek model tipi: 'fer' (Yüz İfadesi Tanıma) veya 'ser' (Sesli Duygu Tanıma)."
    )
    
    # FER modeline özel argümanlar
    parser.add_argument(
        "--fer_model_choice",
        type=str,
        default=config.DEFAULT_FER_MODEL_CHOICE,
        choices=[config.FER_MODEL_TYPE_VGG16_TRANSFER],
        help="Eğer model_type='fer' ise kullanılacak FER mimarisi."
    )
    
    # Genel eğitim parametreleri (config'deki değerleri ezer)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None, # None ise train_pipeline içinde config'den alınır
        help="Eğitim için epoch sayısı. Belirtilmezse config dosyasındaki değer kullanılır."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Eğitim için batch boyutu. Belirtilmezse config dosyasındaki değer kullanılır."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Optimizer için öğrenme oranı. Belirtilmezse config dosyasındaki değer kullanılır."
    )
    parser.add_argument(
        "--no_fer_augmentation",
        action="store_false", # Eğer flag kullanılırsa değeri False olur
        dest="fer_augmentation", # config.DATA_AUGMENTATION_FER_ENABLED'ı etkileyecek
        help="FER eğitimi için veri artırmayı devre dışı bırakır."
    )
    parser.set_defaults(fer_augmentation=config.DATA_AUGMENTATION_FER_ENABLED)

    parser.add_argument(
        "--optimizer_fer",
        type=str,
        default=config.DEFAULT_OPTIMIZER_FER,
        choices=['adam', 'rmsprop', 'sgd'], # train_pipeline'deki seçeneklerle uyumlu olmalı
        help="FER modeli için kullanılacak optimizer tipi. Belirtilmezse config dosyasındaki değer kullanılır."
    )

    parser.add_argument(
        "--no_ser_augmentation",
        action="store_false",
        dest="ser_augmentation",
        help="SER eğitimi için veri artırmayı devre dışı bırakır (eğer implemente edilmişse)."
    )
    parser.set_defaults(ser_augmentation=config.USE_DATA_AUGMENTATION_SER)

    args = parser.parse_args()

    logger.info("Ana Eğitim Script'i Başlatıldı.")
    logger.info(f"Komut Satırı Argümanları: {args}")

    # Gerekli proje dizinlerini oluştur/kontrol et
    try:
        file_utils.create_project_directories()
    except Exception as e:
        logger.error(f"Proje dizinleri oluşturulurken/kontrol edilirken hata: {e}")
        logger.error("Eğitim devam etmeden önce bu sorunu çözün.")
        return 1 # Hata koduyla çık

    # config'deki veri artırma flag'lerini argümanlara göre güncelle
    # Bu, config'in global durumunu değiştirir, dikkatli olunmalı.
    # Daha iyi bir yol, bu flag'leri doğrudan train_pipeline fonksiyonlarına parametre olarak geçmek olabilir.
    # Şimdilik, config'i güncelleyelim, train_pipeline zaten config'i okuyor.
    config.DATA_AUGMENTATION_FER_ENABLED = args.fer_augmentation
    config.DATA_AUGMENTATION_SER_ENABLED = args.ser_augmentation
    logger.info(f"FER Veri Artırma: {'Etkin' if config.DATA_AUGMENTATION_FER_ENABLED else 'Devre Dışı'}")
    logger.info(f"SER Veri Artırma: {'Etkin' if config.DATA_AUGMENTATION_SER_ENABLED else 'Devre Dışı'}")


    training_successful = False
    if args.model_type == 'fer':
        logger.info(f"FER modeli eğitimi başlatılıyor: {args.fer_model_choice}")
        training_successful = train_pipeline.train_fer_model(
            model_type=args.fer_model_choice,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            optimizer_type=args.optimizer_fer
        )
    elif args.model_type == 'ser':
        logger.info("SER (CRNN) modeli eğitimi başlatılıyor.")
        training_successful = train_pipeline.train_ser_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
            # SER için veri artırma zaten train_ser_model içinde config.DATA_AUGMENTATION_SER_ENABLED'a göre yönetiliyor.
        )
    else:
        # Bu durum argparse choices ile zaten engellenmiş olmalı, ama yine de kontrol
        logger.error(f"Geçersiz model_type: {args.model_type}")
        parser.print_help()
        return 1

    if training_successful:
        logger.info(f"{args.model_type.upper()} modeli eğitimi başarıyla tamamlandı (veya en azından hatasız bitti).")
        return 0 # Başarı kodu
    else:
        logger.error(f"{args.model_type.upper()} modeli eğitimi sırasında bir sorun oluştu veya başarısız oldu.")
        return 1 # Hata kodu

if __name__ == '__main__':
    # Script'i çalıştırmak için:
    # python -m src.main_trainer --model_type fer --fer_model_choice mini_xception --epochs 50
    # veya
    # python -m src.main_trainer --model_type ser --epochs 100
    
    # PYTHONPATH'ın proje kök dizinini içerdiğinden emin olun veya
    # proje kök dizinindeyken `python -m src.main_trainer ...` komutunu kullanın.
    
    return_code = main()
    if return_code == 0:
        print("\nEğitim süreci başarıyla tamamlandı.")
    else:
        print("\nEğitim sürecinde hatalar oluştu. Detaylar için logları kontrol edin.")
    sys.exit(return_code)
