import argparse
import os
import sys

# Göreli importlar
from .utils.logging_utils import setup_logger
from .utils import file_utils # Dizin oluşturma için
from .configs import main_config as config
from .core.predictor_engine import EmotionPredictorEngine # Ana tahmin motorumuz

# Logger'ı bu ana script için kur
logger = setup_logger("MainPredictor", log_file=config.APPLICATION_LOG_FILE)

def main():
    parser = argparse.ArgumentParser(
        description="Eğitilmiş modellerle tek bir dosyadan duygu tahmini yapma script'i.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=['image', 'audio'],
        help="Tahmin yapılacak dosyanın tipi: 'image' (görüntü) veya 'audio' (ses)."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Giriş görüntü veya ses dosyasının tam yolu."
    )
    parser.add_argument(
        "--fer_model_base",
        type=str,
        default=config.DEFAULT_FER_MODEL_BASE_NAME,
        help="Kullanılacak FER modelinin temel adı (örn: 'fer_mini_xception'). "
             "En son eğitilmiş olan otomatik olarak seçilir."
    )
    parser.add_argument(
        "--ser_model_base",
        type=str,
        default=config.DEFAULT_SER_MODEL_BASE_NAME,
        help="Kullanılacak SER modelinin temel adı (örn: 'ser_crnn'). "
             "En son eğitilmiş olan otomatik olarak seçilir."
    )

    args = parser.parse_args()

    logger.info("Tekil Dosya Tahmin Script'i Başlatıldı.")
    logger.info(f"Komut Satırı Argümanları: {args}")

    try:
        file_utils.create_project_directories()
    except Exception as e:
        logger.error(f"Proje dizinleri oluşturulurken/kontrol edilirken hata: {e}")
        return 1 

    try:
        predictor = EmotionPredictorEngine(
            fer_model_base_name=args.fer_model_base if args.type == 'image' else None, 
            ser_model_base_name=args.ser_model_base if args.type == 'audio' else None
        )
    except Exception as e:
        logger.error(f"EmotionPredictorEngine başlatılırken hata: {e}", exc_info=True)
        return 1

    if not os.path.exists(args.input_path):
        logger.error(f"Giriş dosyası bulunamadı: {args.input_path}")
        return 1

    predicted_label = None
    probabilities_dict = None
    success = False

    if args.type == 'image':
        if predictor.fer_model is None:
            logger.error(f"FER modeli ({args.fer_model_base}) yüklenemedi veya belirtilmedi. Görüntü tahmini yapılamıyor.")
            return 1
        logger.info(f"Görüntü dosyasından ({args.input_path}) duygu tahmini yapılıyor...")
        predicted_label, probabilities_dict = predictor.predict_from_image_file(args.input_path)
        if predicted_label and probabilities_dict is not None: # Check if prediction was successful
            success = True
            
    elif args.type == 'audio':
        if predictor.ser_model is None:
            logger.error(f"SER modeli ({args.ser_model_base}) yüklenemedi veya belirtilmedi. Ses tahmini yapılamıyor.")
            return 1
        logger.info(f"Ses dosyasından ({args.input_path}) duygu tahmini yapılıyor...")
        predicted_label, probabilities_dict = predictor.predict_from_audio_file(args.input_path)
        if predicted_label and probabilities_dict is not None: # Check if prediction was successful
            success = True
            
    else:
        logger.error(f"Geçersiz tahmin tipi: {args.type}")
        parser.print_help()
        return 1

    if success and predicted_label:
        print(f"\n--- TAHMİN SONUCU ({args.type.upper()}) ---")
        print(f"  Giriş Dosyası: {os.path.abspath(args.input_path)}")
        print(f"  Tahmin Edilen Duygu: {predicted_label}")
        if probabilities_dict:
            print("  Olasılıklar:")
            for emotion, prob in sorted(probabilities_dict.items(), key=lambda item: item[1], reverse=True):
                if prob > 0.001: 
                    print(f"    - {emotion:<10}: {prob:.4f}")
        logger.info(f"Tahmin başarıyla tamamlandı. Sonuç: {predicted_label}")
        return 0
    else:
        print(f"\n--- TAHMİN BAŞARISIZ ({args.type.upper()}) ---")
        print(f"  Giriş Dosyası: {os.path.abspath(args.input_path)}")
        # More specific error message based on predictor_engine's possible error returns
        if predicted_label in ["HataFER", "HataSER", "Onisleme Hatasi"]:
             logger.error(f"{args.type.upper()} tipi için duygu tahmini sırasında hata oluştu: {predicted_label}. Detaylar için logları kontrol edin.")
        else:
             logger.error(f"{args.type.upper()} tipi için duygu tahmini yapılamadı veya model geçerli bir sonuç döndürmedi. Detaylar için logları kontrol edin.")
        return 1

if __name__ == '__main__':
    # Script'i çalıştırmak için örnekler (proje kök dizininden):
    # python -m src.main_predictor --type image --input_path ./test_face.jpg
    # python -m src.main_predictor --type audio --input_path ./test_audio.wav
    # python -m src.main_predictor --type image --input_path ./test_face.jpg --fer_model_base fer_mini_xception
    # (Yukarıdaki fer_mini_xception, config.DEFAULT_FER_MODEL_BASE_NAME ile aynı olmalı veya geçerli bir alternatif olmalı)
    
    return_code = main()
    # No need to print additional messages here as main() already prints success/failure
    sys.exit(return_code)
