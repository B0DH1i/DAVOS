import gradio as gr
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd

# Proje içi importlar
from src.configs import main_config as config
from src.core.predictor_engine import EmotionPredictorEngine
from src.utils.logging_utils import setup_logger
from src.utils import file_utils

# Logger ve Gerekli Dizinlerin Kurulumu
logger = setup_logger("GuiApp")
try:
    file_utils.create_project_directories() # log gibi dizinlerin var olduğundan emin ol
    logger.info("Proje dizinleri başarıyla kontrol edildi/oluşturuldu.")
except Exception as e:
    logger.error(f"Proje dizinleri oluşturulurken hata: {e}", exc_info=True)


# Tahmin motorunu global olarak başlat (modellerin tekrar tekrar yüklenmesini önler)
predictor_engine = None
try:
    logger.info("EmotionPredictorEngine GUI için başlatılıyor...")
    predictor_engine = EmotionPredictorEngine(
        fer_model_name=config.DEFAULT_FER_MODEL_LOAD_NAME,
        ser_model_name=config.DEFAULT_SER_MODEL_LOAD_NAME
    )
    if predictor_engine.fer_model is None and predictor_engine.ser_model is None:
        logger.error("GUI için ne FER ne de SER modeli yüklenemedi. Lütfen modelleri kontrol edin.")
    elif predictor_engine.fer_model is None:
        logger.warning("GUI için FER modeli yüklenemedi. Görüntüden ve canlı kameradan duygu tahmini yapılamayacak.")
    elif predictor_engine.ser_model is None:
        logger.warning("GUI için SER modeli yüklenemedi. Sesten ve canlı mikrofondan duygu tahmini yapılamayacak.")
    else:
        logger.info("EmotionPredictorEngine GUI için başarıyla başlatıldı.")
except Exception as e:
    logger.error(f"EmotionPredictorEngine GUI için başlatılırken kritik hata: {e}", exc_info=True)

def predict_fer_from_image_upload(image_pil):
    """Yüklenen bir görüntüden Yüz İfadesi Tanıma (FER) yapar."""
    logger.info("predict_fer_from_image_upload çağrıldı.")
    if predictor_engine is None:
        logger.warning("predict_fer_from_image_upload: Tahmin motoru None.")
        return "Hata: Tahmin motoru başlatılamadı.", None, image_pil
    if predictor_engine.fer_model is None:
        logger.warning("predict_fer_from_image_upload: FER modeli None.")
        return "FER modeli yüklenemedi. Lütfen yapılandırmayı kontrol edin.", None, image_pil
    
    if image_pil is None:
        logger.warning("predict_fer_from_image_upload: Gelen image_pil None.")
        return "Görüntü yüklenmedi.", None, None

    try:
        logger.debug("predict_fer_from_image_upload: Kareyi BGR'ye çeviriyor.")
        frame_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        processed_bgr, detected_faces_list, dominant_emotion, raw_probs = predictor_engine.predict_from_image_frame(frame_bgr)
        logger.debug(f"predict_fer_from_image_upload: predict_from_image_frame sonucu: dominant_emotion='{dominant_emotion}', #faces={len(detected_faces_list) if detected_faces_list else 0}")
        
        if processed_bgr is not None:
            output_image_pil = Image.fromarray(cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB))
        else:
            logger.warning("predict_fer_from_image_upload: processed_bgr None geldi, orijinal kare kullanılıyor.")
            output_image_pil = image_pil
        
        if dominant_emotion and raw_probs is not None:
            df_probs = pd.DataFrame({
                "label": config.MODEL_OUTPUT_EMOTIONS,
                "value": [float(prob) for prob in raw_probs]
            })
            logger.info(f"predict_fer_from_image_upload: Sonuç döndürülüyor - Duygu: {dominant_emotion}")
            return f"Baskın Duygu: {dominant_emotion}", df_probs, output_image_pil
        elif detected_faces_list:
             logger.info("FER GUI (Upload): Yüz tespit edildi ancak duygu tam olarak belirlenemedi.")
             return "Duygu tahmin edilemedi (ancak yüz bulundu).", None, output_image_pil
        else:
            logger.info("FER GUI (Upload): Görüntüde yüz tespit edilemedi.")
            return "Görüntüde yüz tespit edilemedi.", None, output_image_pil
            
    except Exception as e:
        logger.error(f"FER GUI (Upload) tahmin hatası: {e}", exc_info=True)
        return f"Hata oluştu: {str(e)}", None, image_pil

def predict_ser_from_audio_upload(audio_file_obj):
    """Yüklenen bir ses dosyasından Konuşma Duygusu Tanıma (SER) yapar."""
    logger.info("predict_ser_from_audio_upload çağrıldı.")
    if predictor_engine is None:
        logger.warning("predict_ser_from_audio_upload: Tahmin motoru None.")
        return "Hata: Tahmin motoru başlatılamadı.", None
    if predictor_engine.ser_model is None:
        logger.warning("predict_ser_from_audio_upload: SER modeli None.")
        return "SER modeli yüklenemedi. Lütfen yapılandırmayı kontrol edin.", None
        
    if audio_file_obj is None:
        logger.warning("predict_ser_from_audio_upload: Gelen audio_file_obj None.")
        return "Ses dosyası yüklenmedi.", None

    try:
        audio_filepath = audio_file_obj.name 
        logger.info(f"SER için işlenecek ses dosyası: {audio_filepath}")
        dominant_emotion, emotion_probabilities = predictor_engine.predict_from_audio_file(audio_filepath)
        
        if dominant_emotion and emotion_probabilities:
            # emotion_probabilities zaten dict {etiket: olasılık} formatında geliyor
            df_probs = pd.DataFrame(list(emotion_probabilities.items()), columns=["label", "value"])
            df_probs["value"] = df_probs["value"].astype(float) # Değerlerin float olduğundan emin ol
            logger.info(f"predict_ser_from_audio_upload: Sonuç döndürülüyor - Duygu: {dominant_emotion}")
            return f"Baskın Duygu: {dominant_emotion}", df_probs
        else:
            # Tahmin başarısızsa veya duygu bulunamazsa
            logger.warning(f"predict_ser_from_audio_upload: Sesten duygu tahmin edilemedi veya sonuç None. Dönüş: {dominant_emotion}, {emotion_probabilities}")
            return "Sesten duygu tahmin edilemedi.", None # None, BarPlot'u temizler
            
    except Exception as e:
        logger.error(f"SER GUI (Upload) tahmin hatası: {e}", exc_info=True)
        return f"Hata: {str(e)}", None

def predict_fer_from_live_camera(camera_frame_pil):
    """Canlı kamera görüntüsünden Yüz İfadesi Tanıma (FER) yapar."""
    logger.info("predict_fer_from_live_camera çağrıldı.")
    if predictor_engine is None:
        logger.warning("predict_fer_from_live_camera: Tahmin motoru None.")
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return "Hata: Tahmin motoru başlatılamadı.", None, Image.fromarray(empty_frame)
    if predictor_engine.fer_model is None:
        logger.warning("predict_fer_from_live_camera: FER modeli None.")
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return "FER modeli yüklenemedi.", None, Image.fromarray(empty_frame)

    if camera_frame_pil is None:
        logger.warning("predict_fer_from_live_camera: Gelen camera_frame_pil None.")
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return "Kamera karesi alınamadı.", None, Image.fromarray(empty_frame)

    try:
        logger.debug("predict_fer_from_live_camera: Kareyi BGR'ye çeviriyor.")
        frame_bgr = cv2.cvtColor(np.array(camera_frame_pil), cv2.COLOR_RGB2BGR)
        processed_bgr, detected_faces_list, dominant_emotion, raw_probs = predictor_engine.predict_from_image_frame(frame_bgr)
        logger.debug(f"predict_fer_from_live_camera: predict_from_image_frame sonucu: dominant_emotion='{dominant_emotion}', #faces={len(detected_faces_list) if detected_faces_list else 0}")

        if processed_bgr is not None:
            output_image_pil = Image.fromarray(cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB))
        else: 
            logger.warning("predict_fer_from_live_camera: processed_bgr None geldi, orijinal kare kullanılıyor.")
            output_image_pil = camera_frame_pil
        
        if dominant_emotion and raw_probs is not None:
            df_probs = pd.DataFrame({
                "label": config.MODEL_OUTPUT_EMOTIONS,
                "value": [float(prob) for prob in raw_probs]
            })
            logger.info(f"predict_fer_from_live_camera: Sonuç döndürülüyor - Duygu: {dominant_emotion}")
            return f"Baskın Duygu: {dominant_emotion}", df_probs, output_image_pil
        elif detected_faces_list: 
             logger.info("predict_fer_from_live_camera: Sonuç döndürülüyor - Duygu tahmin edilemedi (yüz bulundu).")
             return "Duygu tahmin edilemedi (yüz bulundu).", None, output_image_pil
        else: 
            logger.info("predict_fer_from_live_camera: Sonuç döndürülüyor - Kamerada yüz tespit edilemedi.")
            return "Kamerada yüz tespit edilemedi.", None, output_image_pil
            
    except Exception as e:
        logger.error(f"FER GUI (Canlı Kamera) tahmin hatası: {e}", exc_info=True)
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        try: 
            return f"Hata: {str(e)}", None, camera_frame_pil if camera_frame_pil else Image.fromarray(empty_frame)
        except: 
            return f"Hata: {str(e)}", None, Image.fromarray(empty_frame)

def predict_ser_from_live_microphone(microphone_audio_input):
    """Canlı mikrofon girişinden Konuşma Duygusu Tanıma (SER) yapar."""
    logger.info("predict_ser_from_live_microphone çağrıldı.")
    if predictor_engine is None:
        logger.warning("predict_ser_from_live_microphone: Tahmin motoru None.")
        return "Hata: Tahmin motoru başlatılamadı.", None
    if predictor_engine.ser_model is None:
        logger.warning("predict_ser_from_live_microphone: SER modeli None.")
        return "SER modeli yüklenemedi.", None
    
    if microphone_audio_input is None:
        logger.warning("predict_ser_from_live_microphone: Mikrofon girdisi None.")
        # Streaming sırasında bu normal olabilir, sessizce None döndürerek grafiği temiz tutabiliriz.
        return "Mikrofon verisi bekleniyor...", None 

    try:
        sample_rate, audio_data_np = microphone_audio_input
        logger.debug(f"predict_ser_from_live_microphone: Ses alındı - Örnekleme Hızı: {sample_rate}, Veri Uzunluğu: {len(audio_data_np)}")

        if audio_data_np is None or len(audio_data_np) == 0:
            logger.debug("predict_ser_from_live_microphone: Boş ses verisi alındı.")
            return "Mikrofon verisi bekleniyor...", None # Veya son başarılı sonucu tutabilir

        # predictor_engine.predict_from_audio_segment zaten (etiket, olasılık_dict) döndürüyor.
        dominant_emotion, emotion_probabilities = predictor_engine.predict_from_audio_segment(audio_data_np, sample_rate)
        
        if dominant_emotion and emotion_probabilities:
            df_probs = pd.DataFrame(list(emotion_probabilities.items()), columns=["label", "value"])
            df_probs["value"] = df_probs["value"].astype(float)
            logger.info(f"predict_ser_from_live_microphone: Sonuç döndürülüyor - Duygu: {dominant_emotion}")
            return f"Baskın Duygu: {dominant_emotion}", df_probs
        elif dominant_emotion: # Olasılıklar None ise ama etiket varsa (örn: "Onisleme Hatasi")
            logger.warning(f"predict_ser_from_live_microphone: Duygu etiketi '{dominant_emotion}' alındı ama olasılıklar None.")
            return f"Durum: {dominant_emotion}", None
        else:
            logger.info("predict_ser_from_live_microphone: Sesten duygu tahmin edilemedi (canlı).")
            return "Duygu analiz ediliyor...", None # veya "Duygu Tespit Edilemedi"
            
    except Exception as e:
        logger.error(f"SER GUI (Canlı Mikrofon) tahmin hatası: {e}", exc_info=True)
        return f"Hata: {str(e)}", None


# Gradio Arayüzünü Oluşturma
with gr.Blocks(theme=gr.themes.Soft(), title="Duygu Verimlilik Projesi Test Arayüzü") as demo:
    gr.Markdown(
        """
        # Duygu Verimlilik Projesi - Gelişmiş Test Arayüzü
        Bu arayüzü kullanarak yüklediğiniz görüntü, ses dosyalarından, canlı kameradan ve canlı mikrofondan duygu analizlerini test edebilirsiniz.
        """
    )
    
    with gr.Tab("Yüz İfadesi Tanıma (FER) - Görüntüden"):
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Görüntü Yükle (.jpg, .png)")
                fer_button = gr.Button("Duyguyu Tahmin Et (Görüntü)")
            with gr.Column(scale=2):
                image_output_display = gr.Image(type="pil", label="İşlenmiş Görüntü")
        with gr.Row():
            fer_result_text = gr.Textbox(label="Baskın Duygu", interactive=False)
        with gr.Row():
            fer_probs_output = gr.BarPlot(label="Duygu Olasılıkları (FER)", x="label", y="value", 
                                          x_title="Duygu", y_title="Olasılık",
                                          vertical=False, min_width=300)
        
        fer_button.click(
            predict_fer_from_image_upload,
            inputs=[image_input],
            outputs=[fer_result_text, fer_probs_output, image_output_display]
        )

    with gr.Tab("Yüz İfadesi Tanıma (FER) - Canlı Kamera"):
        with gr.Row():
            with gr.Column(scale=1):
                live_camera_input = gr.Image(sources=["webcam"], type="pil", label="Canlı Kamera Görüntüsü", streaming=True)
            with gr.Column(scale=2):
                live_camera_output_display = gr.Image(type="pil", label="İşlenmiş Kamera Görüntüsü")
        with gr.Row():
            live_fer_result_text = gr.Textbox(label="Baskın Duygu (Canlı Kamera)", interactive=False)
        with gr.Row():
            live_fer_probs_output = gr.BarPlot(label="Duygu Olasılıkları (Canlı Kamera FER)", x="label", y="value",
                                               x_title="Duygu", y_title="Olasılık",
                                               vertical=False, min_width=300)
        
        live_camera_input.stream(
            predict_fer_from_live_camera,
            inputs=[live_camera_input],
            outputs=[live_fer_result_text, live_fer_probs_output, live_camera_output_display]
        )


    with gr.Tab("Konuşma Duygusu Tanıma (SER) - Ses Dosyasından"):
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.File(label="Ses Dosyası Yükle (.wav, .mp3)", file_types=[".wav", ".mp3"]) 
                ser_button = gr.Button("Duyguyu Tahmin Et (Ses Dosyası)")
            with gr.Column(scale=2):
                ser_result_text = gr.Textbox(label="Baskın Duygu (Ses Dosyası)", interactive=False)
                ser_probs_output = gr.BarPlot(label="Duygu Olasılıkları (Ses Dosyası SER)", x="label", y="value",
                                              x_title="Duygu", y_title="Olasılık",
                                              vertical=False, min_width=300)

        ser_button.click(
            predict_ser_from_audio_upload,
            inputs=[audio_input],
            outputs=[ser_result_text, ser_probs_output]
        )
        
    with gr.Tab("Konuşma Duygusu Tanıma (SER) - Canlı Mikrofon"):
        with gr.Row():
            with gr.Column(scale=1):
                # type="numpy" bize (sample_rate, data_array) olarak veri verir
                # streaming=True ile sürekli akış alınır
                microphone_input = gr.Audio(sources=["microphone"], type="numpy", label="Canlı Mikrofon Girişi", streaming=True, show_label=True)
            with gr.Column(scale=2):
                live_ser_result_text = gr.Textbox(label="Baskın Duygu (Canlı Mikrofon)", interactive=False)
                live_ser_probs_output = gr.BarPlot(label="Duygu Olasılıkları (Canlı Mikrofon SER)", x="label", y="value",
                                                   x_title="Duygu", y_title="Olasılık",
                                                   vertical=False, min_width=300)
        
        # streaming=True olduğu için 'stream' metodunu kullanıyoruz.
        # Her yeni ses parçası geldiğinde predict_ser_from_live_microphone fonksiyonu çağrılacak.
        microphone_input.stream(
            predict_ser_from_live_microphone,
            inputs=[microphone_input],
            outputs=[live_ser_result_text, live_ser_probs_output]
        )
    
    gr.Markdown(
        """
        ---
        **Kullanım Notları:**
        - **FER (Görüntü):** Bir görüntü (`.jpg`, `.png`) yükleyin ve tahmin butonuna basın.
        - **FER (Canlı Kamera):** Tarayıcınız kameraya erişim izni istedikten sonra canlı görüntüden duygu analizi başlar. (Fiziksel bir kameranızın bağlı ve çalışır durumda olması gerekir.)
        - **SER (Ses Dosyası):** `.wav` veya `.mp3` formatında bir ses dosyası yükleyin ve tahmin butonuna basın.
        - **SER (Canlı Mikrofon):** Tarayıcınız mikrofona erişim izni istedikten sonra konuşmaya başlayarak canlı duygu analizi yapabilirsiniz.
        - Modellerin ilk yüklenmesi biraz zaman alabilir. Hata alırsanız, terminal loglarını ve `logs/app.log` dosyasını kontrol edin.
        - Bu arayüz temel testler ve gösterimler için tasarlanmıştır.
        """
    )

# --- Ana Çalıştırma Bloğu ---
if __name__ == "__main__":
    logger.info("Gradio arayüzü başlatılıyor...")
    # inbrowser=True -> Arayüzü varsayılan tarayıcıda yeni bir sekmede açar (yerelde çalıştırırken kullanışlı).
    # share=False -> Hugging Face Spaces kendi paylaşım mekanizmasını yönetir, burada True yapmak gereksiz.
    # server_name="0.0.0.0" -> Uygulamanın bir Docker konteyneri içinde dışarıya açık olmasını sağlar.
    demo.launch(server_name="0.0.0.0")
    logger.info("Gradio GUI kapatıldı.") 