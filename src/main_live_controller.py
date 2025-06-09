# main_live_controller.py
import cv2
import time
import numpy as np
import queue
import threading
import argparse
import librosa # Added librosa
import os # Added os module

# Göreli importlar
from .configs import main_config as config # Changed from import *
from .utils.logging_utils import setup_logger
from .utils import file_utils # For create_project_directories

from .live.input_handler import InputController
from .live.output_handler import OutputController # Changed from OutputHandler
from .core.predictor_engine import EmotionPredictorEngine
from .core.integration_engine import integrate_emotion_probabilities, PROBABILITY_HISTORY # PROBABILITY_HISTORY'yi de alalım
from .core.lazanov_audio_manager import LazanovAudioManager # LAZANOV Entegrasyonu

# Logger
logger = setup_logger("MainLiveController", log_file=config.APPLICATION_LOG_FILE) # Updated logger name and file

# Bu, canlı analizde gecikmeyi önlemeye yardımcı olur.
FACE_PROCESSING_INPUT_QUEUE = queue.Queue(maxsize=1)
FACE_PROCESSING_OUTPUT_QUEUE = queue.Queue(maxsize=1)
# AUDIO_PROCESSING_INPUT_QUEUE = queue.Queue(maxsize=1) # REMOVED: Unused queue
AUDIO_PROCESSING_OUTPUT_QUEUE = queue.Queue(maxsize=1)

# Thread'lerin çalışıp çalışmadığını kontrol etmek için bir olay (event)
THREAD_RUNNING_EVENT = threading.Event()

def face_analysis_worker(predictor_engine):
    logger.info("Face analysis worker started (stub).")
    # Bu stub fonksiyon şimdilik bir şey yapmayacak.
    # Gerçek implementasyonda FACE_PROCESSING_INPUT_QUEUE'dan kare alıp işleyip
    # FACE_PROCESSING_OUTPUT_QUEUE'ya sonuç yazmalı.
    while THREAD_RUNNING_EVENT.is_set():
        time.sleep(0.1) # Döngünün çok hızlı dönmemesi için
    logger.info("Face analysis worker stopped (stub).")

def audio_analysis_worker(predictor_engine, input_controller, cli_args): # Added cli_args
    logger.info(f"Audio analysis worker started. Mode: {'File ('+cli_args.audio_file+')' if cli_args.audio_file else 'Live Microphone'}")
    
    try:
        if cli_args.audio_file:
            # Ses dosyasını işle
            if not predictor_engine.ser_model:
                logger.warning("SER model not loaded. Audio file processing skipped by worker.")
                return

            try:
                audio_file_path = os.path.abspath(cli_args.audio_file) # Convert to absolute path
                logger.info(f"Loading audio file: {audio_file_path}")
                # Dosya yolu PROJECT_ROOT ile birleştirilmemişse, buradan birleştirilebilir
                # Ancak genellikle argparse'dan gelen yol tam olur veya ana dizine göre olur
                audio_data, sample_rate = librosa.load(audio_file_path, sr=config.AUDIO_SAMPLE_RATE, mono=True) # Use absolute path
                logger.info(f"Audio file loaded successfully. Duration: {len(audio_data)/sample_rate:.2f}s, SR: {sample_rate}")

                if audio_data is not None and len(audio_data) > 0:
                    ser_label, ser_probs = predictor_engine.predict_from_audio_segment(audio_data, sample_rate)
                    
                    if ser_label:
                        logger.info(f"Audio file prediction -> Label: {ser_label}")
                        try:
                            AUDIO_PROCESSING_OUTPUT_QUEUE.put_nowait((ser_label, ser_probs))
                        except queue.Full:
                            logger.warning("Audio processing output queue is full. Skipping this result.")
                    else:
                        logger.warning("SER prediction from audio file returned no label.")
                else:
                    logger.warning(f"No audio data loaded from file: {cli_args.audio_file}")

            except Exception as e:
                logger.error(f"Error processing audio file {audio_file_path} in worker: {e}", exc_info=True)
            
            # Dosya işlendikten sonra thread'in sonlanmaması için ana event'i bekleyebilir
            # Ya da sadece bir kez işleyip thread'i sonlandırabilir.
            # Şimdilik, sadece bir kez işleyip beklesin.
            logger.info("Audio file processing complete. Worker will remain idle.")
            while THREAD_RUNNING_EVENT.is_set(): # Remain idle until main event is cleared
                time.sleep(0.5)

        else:
            # Canlı mikrofonu işle (mevcut taslak gibi ama gerçek işlemeli)
            if not predictor_engine.ser_model or not input_controller.is_audio_ready:
                logger.warning("SER model not loaded or microphone not ready. Live audio processing skipped by worker.")
                return

            last_ser_processing_time = 0
            while THREAD_RUNNING_EVENT.is_set():
                current_time = time.time()
                if current_time - last_ser_processing_time >= config.ANALYSIS_INTERVAL_SER:
                    # Kapanış sırasında oluşabilecek hataları önlemek için ek kontrol
                    if not THREAD_RUNNING_EVENT.is_set() or not input_controller.is_audio_ready:
                        logger.info("Audio worker: Ana thread durduruldu veya ses girişi hazır değil. Ses yakalama atlanıyor.")
                        break # Döngüden çık

                    logger.debug("Capturing live audio segment for SER...")
                    audio_segment = input_controller.capture_audio_segment(duration_seconds=config.LIVE_AUDIO_SEGMENT_DURATION)
                    
                    if audio_segment is not None and len(audio_segment) > 0:
                        # print(f"DEBUG: Audio segment captured, len: {len(audio_segment)}, dtype: {audio_segment.dtype}, min: {np.min(audio_segment)}, max: {np.max(audio_segment)}")
                        ser_label, ser_probs = predictor_engine.predict_from_audio_segment(audio_segment, config.AUDIO_SAMPLE_RATE) # Use config.AUDIO_SAMPLE_RATE
                        if ser_label:
                            logger.debug(f"Live SER Prediction -> Label: {ser_label}")
                            try:
                                AUDIO_PROCESSING_OUTPUT_QUEUE.put_nowait((ser_label, ser_probs))
                            except queue.Full:
                                logger.warning("Audio processing output queue is full. Skipping live SER result.")
                        else:
                            logger.debug("Live SER prediction returned no label.")
                    else:
                        logger.debug("Empty audio segment captured in worker.")
                    last_ser_processing_time = current_time
                
                # Döngünün çok hızlı dönmemesi için kısa bir uyku
                # time.sleep(config.ANALYSIS_INTERVAL_SER / 5) # Daha sık kontrol edebilir veya interval kadar uyuyabilir
                time.sleep(0.05) # Daha sık döngü ama işlem aralığına göre

    except Exception as e:
        logger.error(f"Critical error in audio_analysis_worker: {e}", exc_info=True)
    finally:
        logger.info("Audio analysis worker stopped.")

def main_live_loop():
    """Ana canlı analiz döngüsü."""
    logger.info("Canlı Duygu Analiz Sistemi (DOVAS) v3 Başlatılıyor...")
    
    parser = argparse.ArgumentParser(description="DOVAS Live Emotion Analysis System Controller")
    parser.add_argument("--video_file", type=str, default=None,
                        help="Path to a video file to use instead of the live camera feed.")
    parser.add_argument("--audio_file", type=str, default=None, 
                        help="Path to an audio file to process for SER (overrides live microphone).")
    parser.add_argument("--fer_model", type=str, default=None,
                        help=f"Base name or path for the FER model. Defaults to config value: {config.DEFAULT_FER_MODEL_LOAD_NAME}")
    parser.add_argument("--ser_model", type=str, default=None,
                        help=f"Base name or path for the SER model. Defaults to config value: {config.DEFAULT_SER_MODEL_LOAD_NAME}")
    parser.add_argument("--window_width", type=int, default=config.DISPLAY_MAX_WIDTH, help="Görüntü penceresi genişliği.")
    parser.add_argument("--window_height", type=int, default=config.DISPLAY_MAX_HEIGHT, help="Görüntü penceresi yüksekliği.")
    parser.add_argument("--fullscreen", action="store_true", help="Tam ekran modunda başlat.")
    parser.add_argument("--show_fer_probs", action="store_true", help="FER olasılıklarını göster.")
    parser.add_argument("--show_ser_probs", action="store_true", help="SER olasılıklarını göster.")
    parser.add_argument("--show_integrated_probs", action="store_true", help="Entegre olasılıkları göster.")
    parser.add_argument("--camera_id", type=int, default=config.CAMERA_INDEX, help="Kullanılacak kamera ID'si.")
    parser.add_argument("--disable_microphone", action="store_true", help="Canlı mikrofon girişini devre dışı bırak.")

    args = parser.parse_args()
    
    input_ctrl = None
    output_ctrl = None
    predictor_engine_instance = None 
    lazanov_manager = None
    audio_worker_thread = None
    exit_code = 0

    try:
        logger.info("Proje dizinleri kontrol ediliyor/oluşturuluyor...")
        file_utils.create_project_directories()
        
        logger.info("EmotionPredictorEngine başlatılıyor...")
        predictor_engine_instance = EmotionPredictorEngine(
            fer_model_name=args.fer_model if args.fer_model else config.DEFAULT_FER_MODEL_LOAD_NAME,
            ser_model_name=args.ser_model if args.ser_model else config.DEFAULT_SER_MODEL_LOAD_NAME
        )
        logger.info("EmotionPredictorEngine BAŞARIYLA oluşturuldu.")
        
        # LAZANOV Entegrasyonu: LazanovAudioManager örneğini oluştur
        if config.LAZANOV_INTERVENTION_ENABLED:
            try:
                lazanov_manager = LazanovAudioManager()
                logger.info("LazanovAudioManager başarıyla başlatıldı.")
            except Exception as e:
                logger.error(f"LazanovAudioManager başlatılırken hata oluştu: {e}. Lazanov müdahaleleri devre dışı bırakılacak.")
                lazanov_manager = None
        else:
            lazanov_manager = None # Ensure it's None if not enabled
            logger.info("Lazanov müdahaleleri config dosyasında devre dışı bırakılmış.")

        THREAD_RUNNING_EVENT.set()

        # Analiz thread'lerini başlat (eğer kullanılacaksa)
        # Face analysis worker (şu an için stub)
        face_worker_thread = None
        if predictor_engine_instance.fer_model: # Sadece FER modeli varsa yüz analiz thread'i mantıklı
            face_worker_thread = threading.Thread(target=face_analysis_worker, args=(predictor_engine_instance,))
            face_worker_thread.daemon = True # Ana thread sonlandığında bu da sonlansın
            face_worker_thread.start()
            logger.info("Face analysis worker thread started.")

        # Audio analysis worker
        audio_worker_thread = None
        # Ses dosyası veya mikrofon için input_ctrl'a ihtiyaç var, predictor_engine başlatıldıktan sonra input_ctrl hazırlanmalı
        
        # if predictor_engine.fer_model is None and predictor_engine.ser_model is None: # predictor_engine_instance kullan
        if predictor_engine_instance.fer_model is None and predictor_engine_instance.ser_model is None:
            logger.critical("Neither FER nor SER model could be loaded by the Predictor Engine. System cannot run.")
            return # Exit if no models are loaded
        # elif predictor_engine.fer_model is None: # predictor_engine_instance kullan
        elif predictor_engine_instance.fer_model is None:
            logger.warning("FER model not loaded. Face emotion analysis will be skipped.")
        # elif predictor_engine.ser_model is None: # predictor_engine_instance kullan
        elif predictor_engine_instance.ser_model is None:
            logger.warning("SER model not loaded. Speech emotion analysis will be skipped.")
        
        # Yüz tanıma modeli kontrolü (predictor_engine_instance içindeki attribute'lara göre)
        # if predictor_engine.face_cascade is None and predictor_engine.dnn_face_detector_net is None and predictor_engine.fer_model is not None: # predictor_engine_instance kullan
        if predictor_engine_instance.dnn_face_detector_net is not None:
            logger.debug(f"MAIN_LOOP_DNN_CHECK: DNN detector is loaded.")
        else:
            logger.warning(f"MAIN_LOOP_DNN_CHECK: DNN detector is NOT loaded.")


        # InputController'ı video/audio kaynaklarına göre başlat
        enable_video = predictor_engine_instance.fer_model is not None and (args.video_file is not None or args.camera_id is not None)
        enable_audio = predictor_engine_instance.ser_model is not None and (args.audio_file is not None or not args.disable_microphone)

        video_input_for_controller = None
        if args.video_file: # Eğer bir video dosyası argümanı varsa, onu kullan
            video_input_for_controller = args.video_file
        # Eğer video dosyası yoksa ve enable_video True ise (yani kamera kullanılacaksa),
        # InputController zaten video_file_path=None aldığında config.CAMERA_INDEX'i kullanır.
        # Dolayısıyla video_input_for_controller None kalabilir.

        input_ctrl = InputController(
            video_file_path=video_input_for_controller,
            enable_audio=enable_audio,
            camera_id=args.camera_id
        )
        logger.info(f"InputController başlatıldı. Video etkin: {input_ctrl.is_camera_ready}, Ses etkin: {input_ctrl.is_audio_ready}")

        # Ses analiz thread'ini şimdi (input_ctrl hazır olduktan sonra) başlat
        if predictor_engine_instance.ser_model and (input_ctrl.is_audio_ready or args.audio_file):
            audio_worker_thread = threading.Thread(target=audio_analysis_worker, args=(predictor_engine_instance, input_ctrl, args))
            audio_worker_thread.daemon = True
            audio_worker_thread.start()
            logger.info("Audio analysis worker thread started.")
        elif predictor_engine_instance.ser_model:
             logger.warning("SER model loaded, but no audio input (microphone or file) is available/enabled. Audio worker not started.")


        output_ctrl = OutputController(
            pygame_window_size=(args.window_width, args.window_height),
            enable_fullscreen=args.fullscreen,
            show_fer_probs=args.show_fer_probs,
            show_ser_probs=args.show_ser_probs,
            show_integrated_probs=args.show_integrated_probs,
            text_color=config.TEXT_COLOR,
            bg_color=config.BACKGROUND_COLOR,
            info_area_height_ratio=config.INFO_AREA_HEIGHT_RATIO,
            font_scale=config.FONT_SCALE
        )
        logger.info("Input and output modules ready.")
        
        last_fer_analysis_time = 0
        last_ser_analysis_time = 0
        last_integration_time = 0
        
        display_info = {
            "fps": "0.0",
            "face_detection_status": "Başlatılıyor...",
            "fer_raw_label": "N/A",
            "fer_probs_dict": None,
            "ser_raw_label": "N/A",
            "ser_probs_dict": None,
            "integrated_label": "N/A",
            "integrated_probs_vector": None,
            "face_box_to_draw": None,
            "lazanov_music_status": "Lazanov Devre Dışı" if not (lazanov_manager and config.LAZANOV_INTERVENTION_ENABLED) else "Pasif",
            "lazanov_binaural_status": "Lazanov Devre Dışı" if not (lazanov_manager and config.LAZANOV_INTERVENTION_ENABLED and config.LAZANOV_BINAURAL_BEATS_ENABLED) else "Pasif"
        }
        
        # LAZANOV Entegrasyonu: Duygu süresi takibi için değişkenler
        last_processed_dominant_emotion = None
        dominant_emotion_start_time = None
        
        empty_frame_count = 0 # Video dosyası için ardışık boş kare sayacı

        logger.info("Ana canlı analiz döngüsü başlatılıyor...")
        while THREAD_RUNNING_EVENT.is_set(): # THREAD_RUNNING_EVENT ile kontrol
            current_time = time.time()
            frame_bgr = input_ctrl.capture_video_frame()

            if frame_bgr is None:
                if input_ctrl.is_video_file: 
                    empty_frame_count += 1
                    if empty_frame_count > getattr(config, 'MAX_CONSECUTIVE_EMPTY_FRAMES_VIDEO_FILE', 10): # Config'den al veya varsayılan kullan
                        logger.info(f"Video dosyasının sonuna ulaşıldı veya ardışık {empty_frame_count} boş kare. Sistem durduruluyor.")
                        THREAD_RUNNING_EVENT.clear()
                        break
                else: 
                    logger.warning("Canlı kameradan kare okunamadı (frame_bgr is None).")
                
                # Görüntü olmasa bile olayları işle (örn. çıkış)
                if output_ctrl and not output_ctrl.handle_events():
                    logger.info("OutputController (boş kare sırasında) çıkış istedi. Sistem durduruluyor.")
                    THREAD_RUNNING_EVENT.clear()
                    break
                time.sleep(0.01) 
                continue
            else:
                empty_frame_count = 0 

            # --- Yüz İfadesi Tahmini (ANA THREAD) ---
            # PredictorEngine.predict_from_image_frame yüz tanıma ve FER tahminini yapar.
            # Dönüş değerleri: predicted_emotion_label, fer_probabilities_dict, face_box_coordinates
            # face_box_coordinates, çizim için OutputController'a gönderilecek.
            
            if predictor_engine_instance.fer_model and predictor_engine_instance.dnn_face_detector_net:
                if current_time - last_fer_analysis_time >= config.ANALYSIS_INTERVAL_FER:
                    predicted_emotion_face, fer_probs_dict, face_box = predictor_engine_instance.predict_from_image_frame(frame_bgr)

                    display_info["face_box_to_draw"] = face_box # Çizim için OutputController'a

                    if predicted_emotion_face and predicted_emotion_face not in predictor_engine_instance.ERROR_LABELS_FER:
                        display_info["face_detection_status"] = "Yüz Bulundu" if face_box else "Yüz Bulundu (Kutu Yok)"
                        display_info["fer_raw_label"] = predicted_emotion_face
                        display_info["fer_probs_dict"] = fer_probs_dict
                    elif predicted_emotion_face == predictor_engine_instance.FACE_NOT_FOUND_LABEL:
                        display_info["face_detection_status"] = "Yüz Bulunamadı"
                        display_info["fer_raw_label"] = "N/A"
                        display_info["fer_probs_dict"] = None
                    else: 
                        display_info["face_detection_status"] = predicted_emotion_face if predicted_emotion_face else "FER Hatası"
                        display_info["fer_raw_label"] = "N/A"
                        display_info["fer_probs_dict"] = None
                    last_fer_analysis_time = current_time
            elif not predictor_engine_instance.fer_model:
                display_info["face_detection_status"] = "FER Modeli Yok"
                display_info["fer_raw_label"] = "N/A"
                display_info["fer_probs_dict"] = None
            elif not predictor_engine_instance.dnn_face_detector_net:
                display_info["face_detection_status"] = "Yüz Detektörü Yok"
                display_info["fer_raw_label"] = "N/A"
                display_info["fer_probs_dict"] = None
            
            # --- Sesli Duygu Tahmini (Canlı veya Dosyadan) ---
            # audio_analysis_worker thread'i dosyadan işlemeyi ve canlı için AUDIO_PROCESSING_OUTPUT_QUEUE'yu doldurmayı hedefler.
            # Eğer worker kullanılmıyorsa veya öncelik ana thread'deyse, burada direkt analiz yapılabilir.
            # Mevcut kurguda audio_worker_thread SER için output queue'yu dolduruyor.

            if predictor_engine_instance.ser_model:
                try:
                    # Worker'dan gelen sonucu öncelikle almaya çalış
                    ser_label_from_q, ser_probs_dict_from_q = AUDIO_PROCESSING_OUTPUT_QUEUE.get_nowait()
                    if ser_label_from_q and ser_probs_dict_from_q and ser_label_from_q not in predictor_engine_instance.ERROR_LABELS_SER:
                        display_info["ser_raw_label"] = ser_label_from_q
                        display_info["ser_probs_dict"] = ser_probs_dict_from_q
                    elif ser_label_from_q: # Hata etiketi geldiyse
                        display_info["ser_raw_label"] = ser_label_from_q
                        display_info["ser_probs_dict"] = None
                    # AUDIO_PROCESSING_OUTPUT_QUEUE.task_done() # get_nowait için task_done gerekmez
                except queue.Empty:
                    # Kuyruk boşsa ve canlı mikrofon analizi için zamanı geldiyse, ana thread'de yap
                    if not args.audio_file and input_ctrl.is_audio_ready and \
                       (current_time - last_ser_analysis_time >= config.ANALYSIS_INTERVAL_SER):
                        audio_segment = input_ctrl.capture_audio_segment(duration_seconds=config.LIVE_AUDIO_SEGMENT_DURATION)
                        if audio_segment is not None and len(audio_segment) > 0:
                            ser_label, ser_probs = predictor_engine_instance.predict_from_audio_segment(audio_segment, config.AUDIO_SAMPLE_RATE)
                            if ser_label and ser_label not in predictor_engine_instance.ERROR_LABELS_SER:
                                display_info["ser_raw_label"] = ser_label
                                display_info["ser_probs_dict"] = ser_probs
                            else:
                                display_info["ser_raw_label"] = ser_label if ser_label else "SER Hatası"
                                display_info["ser_probs_dict"] = None
                        else:
                            # display_info["ser_raw_label"] = "Ses Alınamadı" # Mevcut değeri koru
                            pass 
                        last_ser_analysis_time = current_time
                except Exception as e:
                    logger.error(f"Audio output kuyruğundan okuma hatası: {e}")
                    display_info["ser_raw_label"] = "SER Kuyruk Hatası"
                    display_info["ser_probs_dict"] = None
            else:
                display_info["ser_raw_label"] = "SER Modeli Yok"
                display_info["ser_probs_dict"] = None

            # --- Duygu Entegrasyonu ---
            if current_time - last_integration_time >= config.ANALYSIS_INTERVAL_INTEGRATION:
                integrated_label, integrated_probs_vector = integrate_emotion_probabilities(
                    face_emotion_probs_dict=display_info.get("fer_probs_dict"),
                    speech_emotion_probs_dict=display_info.get("ser_probs_dict")
                )
                
                display_info["integrated_label"] = integrated_label
                display_info["integrated_probs_vector"] = integrated_probs_vector
                
                logger.info(f"Entegre Sonuç -> Etiket: {integrated_label} (FER: {display_info.get('fer_raw_label', 'N/A')}, SER: {display_info.get('ser_raw_label', 'N/A')})")
                last_integration_time = current_time
            
            # --- LAZANOV MÜDAHALE YÖNETİMİ ---
            if lazanov_manager and config.LAZANOV_INTERVENTION_ENABLED:
                current_integrated_label = display_info.get("integrated_label", "N/A")
                current_integrated_probs = display_info.get("integrated_probs_vector")
                target_emotions_list = getattr(config, 'TARGET_EMOTIONS_ORDERED', list(config.FER_EMOTIONS))
                
                confidence_for_intervention = 0.0
                if current_integrated_probs is not None and current_integrated_label in target_emotions_list:
                    try:
                        emotion_idx = target_emotions_list.index(current_integrated_label)
                        if emotion_idx < len(current_integrated_probs):
                            confidence_for_intervention = current_integrated_probs[emotion_idx]
                    except ValueError:
                        logger.warning(f"Entegre etiket '{current_integrated_label}' hedef duygu listesinde bulunamadı.")
                elif current_integrated_label not in ["unknown", "N/A"]:
                    confidence_for_intervention = config.LAZANOV_TRIGGER_CONFIDENCE_THRESHOLD

                lazanov_music_status, lazanov_binaural_status = lazanov_manager.manage_intervention(
                    current_integrated_label, 
                    confidence_for_intervention,
                    current_time 
                )
                display_info["lazanov_music_status"] = lazanov_music_status
                display_info["lazanov_binaural_status"] = lazanov_binaural_status
            # --- LAZANOV MÜDAHALE YÖNETİMİ BİTTİ ---

            # Görüntüleme ve olayları OutputController üzerinden yönet
            if output_ctrl:
                # display_frame'e ham frame_bgr ve display_info gönderilir.
                # OutputController içindeki display_frame metodu yüz kutusunu (varsa) ve diğer bilgileri çizer.
                output_ctrl.display_frame(frame_bgr, display_info)
                
                if not output_ctrl.handle_events():
                    logger.info("OutputController çıkış istedi. Sistem durduruluyor.")
                    THREAD_RUNNING_EVENT.clear()
                    break 
            else:
                logger.warning("OutputController başlatılamamış, görüntüleme ve olay yönetimi atlanıyor.")
                # OutputController yoksa, basit bir OpenCV gösterimi ve çıkış tuşu (fallback - idealde olmamalı)
                if frame_bgr is not None:
                    try:
                        cv2.imshow("Fallback Display - OutputCtrl Error", frame_bgr)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("Fallback display 'q' tuşu ile kapatıldı.")
                            THREAD_RUNNING_EVENT.clear()
                            break
                    except Exception as e:
                        logger.error(f"Fallback cv2.imshow hatası: {e}")
                        THREAD_RUNNING_EVENT.clear() # Görüntüleme tamamen çöktüyse dur.
                        break 
                else: # Frame de yoksa
                    if cv2.waitKey(1) & 0xFF == ord('q'): # Sadece q tuşunu dinle
                        THREAD_RUNNING_EVENT.clear()
                        break
                    time.sleep(0.01) # Boşuna CPU tüketme

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected (Ctrl+C). Stopping system.")
        THREAD_RUNNING_EVENT.clear() # Set the event to signal threads to stop
        logger.info("Thread running event cleared. Waiting for worker threads to join...")
        
        if face_worker_thread and face_worker_thread.is_alive():
            face_worker_thread.join(timeout=2.0) # Wait for 2 seconds
            if face_worker_thread.is_alive():
                logger.warning("Face analysis worker thread did not join in time.")
        if audio_worker_thread and audio_worker_thread.is_alive():
            audio_worker_thread.join(timeout=2.0)
            if audio_worker_thread.is_alive():
                logger.warning("Audio analysis worker thread did not join in time.")
        
        logger.info("Worker threads joined or timed out.")

    except Exception as e:
        logger.critical(f"Critical error in main live analysis loop: {e}", exc_info=True)
        THREAD_RUNNING_EVENT.clear() # Set the event to signal threads to stop
    finally:
        logger.info("Cleaning up resources...")
        if input_ctrl:
            # input_ctrl.release_resources() # release_resources input_handler'da yok, release var.
            input_ctrl.release()
        if output_ctrl:
            # output_ctrl.cleanup() # cleanup output_handler'da yok, quit var.
            output_ctrl.quit()
        
        # OpenCV pencerelerini kapat (eğer fallback kullanıldıysa veya başka bir yerde açıldıysa)
        # Genelde OutputController.quit() Pygame'i kapattığı için bu gerekmeyebilir,
        # ama bir fallback cv2.imshow çağrısı varsa diye eklenebilir.
        # OutputController tam kontrolü aldığı için buna gerek kalmamalı.
        # cv2.destroyAllWindows() 
        
        # LAZANOV Entegrasyonu: LazanovAudioManager'ı kapat
        if lazanov_manager:
            logger.info("LazanovAudioManager kapatılıyor...")
            lazanov_manager.shutdown()
            logger.info("LazanovAudioManager başarıyla kapatıldı.")

        # Clear the event again in case it was set by an error before finally
        THREAD_RUNNING_EVENT.clear()
        logger.info("System Shutdown Complete.")

if __name__ == "__main__":
    # Gerekli dizinlerin varlığını kontrol etme ve oluşturma file_utils'e taşındı.
    # Ancak ana script çalışmadan önce log dizini kesin olmalı.
    try:
        if not os.path.exists(config.LOGS_PATH):
            os.makedirs(config.LOGS_PATH, exist_ok=True)
        # Diğer önemli dizinler file_utils.create_project_directories() içinde hallediliyor.
    except Exception as e:
        # Bu çok erken bir hata olduğu için logger henüz tam hazır olmayabilir.
        print(f"[CRITICAL] Log dizini ({config.LOGS_PATH}) oluşturulamadı: {e}. Çıkılıyor.")
        # exit(1) # Logger olmadan çıkış yap

    main_live_loop()
