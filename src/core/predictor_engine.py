# src/core/predictor_engine.py
import os
import tensorflow as tf # tensorflow importu tf olarak düzeltildi
import numpy as np
import cv2    # Görüntü işleme için (yüz tespiti, ön işleme)
import librosa # Ses işleme için (MFCC)

# Whisper için gerekli olabilecek importlar (şimdilik taslak)
from transformers import WhisperProcessor, TFWhisperModel # TFWhisperModel olarak değiştirildi

# Göreli importlar
from ..utils.logging_utils import setup_logger
from ..utils import file_utils # Model yükleme için
from ..configs import main_config as config
# data_loader'dan extract_mfcc_features_from_file'ı değil, MFCC mantığını kullanacağız.

# Logger'ı bu modül için kur
logger = setup_logger(__name__, log_file=config.APPLICATION_LOG_FILE)

class EmotionPredictorEngine:
    """
    Eğitilmiş FER ve SER modellerini yükleyip yöneten ve 
    hem dosyalardan hem de canlı veriden duygu tahmini yapan motor sınıfı.
    """
    def __init__(self,
                 fer_model_name=config.DEFAULT_FER_MODEL_LOAD_NAME,
                 ser_model_name=config.DEFAULT_SER_MODEL_LOAD_NAME,
                 video_emotion_buffer_size=5,
                 audio_emotion_buffer_size=5
                 ):
        logger.info(f"EmotionPredictorEngine başlatılıyor...")
        logger.info(f"FER Model Adı: {fer_model_name}")
        logger.info(f"SER Model Adı: {ser_model_name}")

        self.face_detector_type = "dnn"
        logger.info(f"Yüz Tanıma Modeli Tipi: {self.face_detector_type}")
        self.video_emotion_buffer_size = video_emotion_buffer_size
        self.audio_emotion_buffer_size = audio_emotion_buffer_size

        self.video_emotion_buffer = []
        self.audio_emotion_buffer = []

        # Yüz tanıma modelini yükle
        self.face_cascade = None
        self.dnn_face_detector_net = None
        self._load_face_detector()
        logger.info(f"Post _load_face_detector: self.face_detector_type='{self.face_detector_type}', self.face_cascade is None: {self.face_cascade is None}, self.dnn_face_detector_net is None: {self.dnn_face_detector_net is None}") # Bu log KALIYOR

        self.fer_emotions = config.MODEL_OUTPUT_EMOTIONS # FER_EMOTIONS -> MODEL_OUTPUT_EMOTIONS

        # FER modelini yükle
        self.fer_model = None
        self.input_shape_fer = config.INPUT_SHAPE_FER # (48, 48, 1)
        actual_fer_model_load_name = fer_model_name
        if fer_model_name == "latest":
            actual_fer_model_load_name = file_utils.get_latest_model_directory_for_base(config.FER_MODEL_NAME_PREFIX)
            if not actual_fer_model_load_name:
                 logger.warning(f"'latest' için uygun FER modeli bulunamadı ({config.FER_MODEL_NAME_PREFIX} ile başlayan). Varsayılan model kullanılmayacak.")
                 actual_fer_model_load_name = None # Model yüklenmeyecek
        
        self.fer_model_folder = actual_fer_model_load_name # Ensure self.fer_model_folder is set

        if actual_fer_model_load_name:
            logger.info(f"FER modeli ({actual_fer_model_load_name}) yükleniyor...")
            custom_obj_fer = {"Swish": "swish"} if "xception" in actual_fer_model_load_name.lower() else None
            self.fer_model = file_utils.load_trained_model(actual_fer_model_load_name, custom_objects=custom_obj_fer)
            if self.fer_model:
                logger.info(f"FER modeli başarıyla yüklendi: {actual_fer_model_load_name}")
                try:
                    logger.info(f"Yüklenen FER modelinin beklenen giriş şekli: {self.fer_model.input_shape}")
                except Exception as e:
                    logger.warning(f"FER modelinin giriş şekli alınamadı: {e}")
            else:
                logger.error(f"FER modeli yüklenemedi: {actual_fer_model_load_name}")
        else:
            logger.warning("Yüklenecek bir FER modeli belirtilmedi veya bulunamadı.")

        # SER modelini ve ilgili bileşenleri yükle
        self.ser_model = None
        self.whisper_processor = None
        self.tf_whisper_model = None
        self.input_shape_ser_actual = None

        actual_ser_model_load_name = ser_model_name
        if ser_model_name == "latest":
            # Sadece Whisper varsayıldığı için SER_FEATURE_TYPE kontrolüne gerek yok.
            ser_prefix_to_search = config.SER_MODEL_NAME_PREFIX + "_whisper" # Doğrudan whisper ekle
            
            actual_ser_model_load_name = file_utils.get_latest_model_directory_for_base(ser_prefix_to_search)
            if not actual_ser_model_load_name:
                logger.warning(f"'latest' için uygun SER modeli bulunamadı ({ser_prefix_to_search} ile başlayan). Varsayılan model kullanılmayacak.")
                actual_ser_model_load_name = None
        
        self.ser_model_folder = actual_ser_model_load_name

        # SER_FEATURE_TYPE artık sadece "whisper" olacağı için bu kontrol basitleştirildi.
        try:
            logger.info(f"Whisper işlemcisi ve TFWhisperModel yükleniyor: {config.WHISPER_MODEL_NAME}")
            self.whisper_processor = WhisperProcessor.from_pretrained(config.WHISPER_MODEL_NAME)
            self.tf_whisper_model = TFWhisperModel.from_pretrained(config.WHISPER_MODEL_NAME)
            logger.info("Whisper işlemcisi ve TFWhisperModel başarıyla yüklendi.")
            if self.tf_whisper_model:
                self.input_shape_ser_actual = (self.tf_whisper_model.config.d_model,)
                logger.info(f"SER için Whisper embedding boyutu {self.input_shape_ser_actual} olarak ayarlandı ({config.WHISPER_MODEL_NAME} d_model: {self.tf_whisper_model.config.d_model}).")
            else:
                # Bu durum pek olası değil eğer from_pretrained hata vermezse, ama yedek olarak.
                self.input_shape_ser_actual = (config.WHISPER_EMBEDDING_DIM,)
                logger.warning(f"TFWhisperModel yüklenemedi, SER için Whisper embedding boyutu varsayılan {self.input_shape_ser_actual} ({config.WHISPER_EMBEDDING_DIM}) olarak ayarlandı.")
        except Exception as e:
            logger.error(f"Whisper modeli veya işlemcisi yüklenirken hata oluştu ({config.WHISPER_MODEL_NAME}): {e}", exc_info=True)
            self.input_shape_ser_actual = (config.WHISPER_EMBEDDING_DIM,) # Hata durumunda varsayılan boyut
            logger.warning(f"Hata nedeniyle, SER için Whisper embedding boyutu varsayılan {self.input_shape_ser_actual} ({config.WHISPER_EMBEDDING_DIM}) olarak ayarlandı.")
        
        # MFCC ile ilgili blok kaldırıldı.
        # elif config.SER_FEATURE_TYPE == "mfcc":
        #     self.input_shape_ser_actual = config.INPUT_SHAPE_SER # Bu config artık yok
        #     logger.info(f"SER için özellik tipi MFCC olarak ayarlandı. Giriş şekli: {self.input_shape_ser_actual}")
        # else:
        #     logger.error(f"Geçersiz SER_FEATURE_TYPE: {config.SER_FEATURE_TYPE}. 'whisper' veya 'mfcc' olmalı.")

        if actual_ser_model_load_name:
            logger.info(f"SER modeli ({actual_ser_model_load_name}) yükleniyor...")
            self.ser_model = file_utils.load_trained_model(actual_ser_model_load_name)
            if self.ser_model:
                logger.info(f"SER modeli başarıyla yüklendi: {actual_ser_model_load_name}")
                try:
                    logger.info(f"Yüklenen SER modelinin beklenen giriş şekli: {self.ser_model.input_shape}")
                    if config.SER_FEATURE_TYPE == "whisper" and self.input_shape_ser_actual:
                        model_input_dim = self.ser_model.input_shape[-1]
                        if model_input_dim != self.input_shape_ser_actual[0]:
                            logger.warning(f"SER modelinin beklenen giriş boyutu ({model_input_dim}) ile Whisper embedding boyutu ({self.input_shape_ser_actual[0]}) eşleşmiyor!")
                        else:
                            logger.info(f"SER modelinin giriş boyutu ({model_input_dim}) Whisper embedding boyutuyla ({self.input_shape_ser_actual[0]}) uyumlu.")
                except Exception as e:
                    logger.warning(f"SER modelinin giriş şekli alınamadı: {e}")
            else:
                logger.error(f"SER modeli yüklenemedi: {actual_ser_model_load_name}")
        else:
            logger.warning("Yüklenecek bir SER modeli belirtilmedi veya bulunamadı.")

        logger.info("EmotionPredictorEngine başarıyla başlatıldı.")
        # INIT_ID_CHECK ve INIT_CONFIG_PATH_CHECK logları __init__ sonundan kaldırıldı.

    def _load_face_detector(self):
        """Yüz dedektörünü (DNN) yükler."""
        logger.info(f"Yüz tanıma modeli ({self.face_detector_type}) yükleniyor...")
        if os.path.exists(config.FACE_DETECTOR_DNN_PROTOTXT_PATH) and \
           os.path.exists(config.FACE_DETECTOR_DNN_MODEL_PATH):
            try:
                self.dnn_face_detector_net = cv2.dnn.readNetFromCaffe(
                    config.FACE_DETECTOR_DNN_PROTOTXT_PATH,
                    config.FACE_DETECTOR_DNN_MODEL_PATH
                )
                if self.dnn_face_detector_net is None:
                    logger.error(f"DNN yüz tanıma modeli cv2.dnn.readNetFromCaffe ile yüklenemedi (None döndü). Prototxt: {config.FACE_DETECTOR_DNN_PROTOTXT_PATH}, Model: {config.FACE_DETECTOR_DNN_MODEL_PATH}")
                else:
                    logger.info(f"DNN yüz tanıma modeli başarıyla yüklendi: {config.FACE_DETECTOR_DNN_MODEL_PATH}")
            except Exception as e:
                logger.error(f"DNN yüz tanıma modeli yüklenirken hata oluştu: {e}. Prototxt: {config.FACE_DETECTOR_DNN_PROTOTXT_PATH}, Model: {config.FACE_DETECTOR_DNN_MODEL_PATH}", exc_info=True)
                self.dnn_face_detector_net = None
        else:
            logger.error(f"DNN yüz tanıma model dosyaları bulunamadı. Prototxt: {config.FACE_DETECTOR_DNN_PROTOTXT_PATH}, Model: {config.FACE_DETECTOR_DNN_MODEL_PATH}")
            self.dnn_face_detector_net = None
        
        # Haar Cascade ve diğer type kontrolleri buradan kaldırıldı

    def _preprocess_face_for_fer(self, face_roi, target_size=(48,48)):
        """Kırpılmış yüzü FER modeline uygun hale getirir."""
        if face_roi is None or face_roi.size == 0:
            logger.warning("_preprocess_face_for_fer: Gelen face_roi boş veya None.")
            return None

        try:
            # Yeniden boyutlandır (OpenCV enterpolasyon yöntemleri genellikle iyidir)
            # Gelen face_roi'nin gri tonlamalı (H, W) olduğunu varsayıyoruz.
            resized_face = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_AREA)

            is_vgg16 = self.fer_model_folder and "vgg16" in self.fer_model_folder.lower()

            if is_vgg16:
                # VGG16 RGB (3 kanal) bekler. Gelen resized_face (H,W) gri tonlamalı.
                # Önce BGR'ye çevir, sonra normalize et.
                bgr_face = cv2.cvtColor(resized_face, cv2.COLOR_GRAY2BGR) # (H, W) -> (H, W, 3)
                normalized_face = bgr_face.astype(np.float32) / 255.0
                # Model (1, H, W, 3) bekler
                processed_face_for_model = np.expand_dims(normalized_face, axis=0)
            else:
                # Diğer modeller (örn: MiniXception) gri tonlama (1 kanal) bekler.
                # resized_face zaten gri (H, W).
                normalized_face = resized_face.astype(np.float32) / 255.0 # Normalizasyonu float'a çevirdikten sonra yap
                expanded_face = np.expand_dims(normalized_face, axis=-1)  # (H, W) -> (H, W, 1)
                # Model (1, H, W, 1) bekler
                processed_face_for_model = np.expand_dims(expanded_face, axis=0)
            
            return processed_face_for_model
        except cv2.error as e: # OpenCV hatası (örn: geçersiz görüntü)
            logger.error(f"_preprocess_face_for_fer içinde OpenCV hatası: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"_preprocess_face_for_fer içinde genel hata: {e}", exc_info=True)
            return None

    def _extract_whisper_embedding_for_ser(self, audio_segment_np, sample_rate=config.WHISPER_SAMPLING_RATE):
        if self.whisper_processor is None or self.tf_whisper_model is None:
            logger.error("Whisper işlemcisi veya TF modeli yüklenmedi. Embedding çıkarılamıyor.")
            return None
        try:
            # Ses verisini float32'ye dönüştür ve normalize et (eğer zaten değilse)
            if audio_segment_np.dtype != np.float32:
                audio_float = audio_segment_np.astype(np.float32) / np.iinfo(audio_segment_np.dtype).max
            else:
                audio_float = audio_segment_np

            if sample_rate != config.WHISPER_SAMPLING_RATE:
                audio_float = librosa.resample(audio_float, orig_sr=sample_rate, target_sr=config.WHISPER_SAMPLING_RATE)
            
            inputs = self.whisper_processor(audio_float, sampling_rate=config.WHISPER_SAMPLING_RATE, return_tensors="tf")
            input_features = inputs.input_features

            # Decoder için gerekli olan başlangıç token ID'sini al
            # TFWhisperModel'in config'inden decoder_start_token_id alınır
            # Bazı modellerde bu config.decoder.bos_token_id veya benzeri olabilir.
            # Whisper için genelde config.decoder_start_token_id kullanılır.
            # Hugging Face WhisperProcessor, `get_decoder_prompt_ids` metodunu da sunar ama o daha çok text generation için.
            # En güvenlisi modelin config'inden doğrudan almak.
            decoder_start_token_id = self.tf_whisper_model.config.decoder_start_token_id
            if decoder_start_token_id is None: # Nadir ama bazı modellerde farklı olabilir
                # Alternatif olarak processor'dan almayı deneyebiliriz veya varsayılan bir ID kullanabiliriz.
                # Örn: processor.tokenizer.bos_token_id. Ancak bu TF modeli için doğrudan config daha iyi.
                pass # Eğer None ise, aşağıdaki TF.constant hata verecek, bu istenen bir durum olabilir debugging için.

            decoder_input_ids = tf.constant([[decoder_start_token_id]]) # Shape: (batch_size, 1)

            model_outputs = self.tf_whisper_model(
                input_features,
                decoder_input_ids=decoder_input_ids, # Decoder girdisini sağla
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True
            )

            if not hasattr(model_outputs, 'encoder_last_hidden_state') or model_outputs.encoder_last_hidden_state is None:
                logger.error("Encoder çıktıları (encoder_last_hidden_state) TFWhisperModel çıktısında bulunamadı.")
                logger.error(f"Model çıktı tipi: {type(model_outputs)}, Mevcut anahtarlar/özellikler: {dir(model_outputs)}")
                return None
            
            encoder_last_hidden_state = model_outputs.encoder_last_hidden_state
            # Shape: (batch, seq_len_encoder, hidden_dim) -> (Örn: 1, 1500, 512) whisper-base için d_model=512
            
            # Zaman boyutu (axis=1) üzerinden ortalama alarak (mean pooling) SER modelinin beklediği şekle getir.
            # Bu, (batch, seq_len_encoder, hidden_dim) -> (batch, hidden_dim) dönüşümünü sağlar.
            pooled_embedding = tf.reduce_mean(encoder_last_hidden_state, axis=1)
            # Shape: (batch, hidden_dim) -> (Örn: 1, 512)

            # SER modelinin beklediği format numpy array.
            embedding_np = pooled_embedding.numpy() 
            
            # Boyut kontrolü (Bu log daha önce uyarı veriyordu, şimdi pooling sonrası kontrol edelim)
            if embedding_np.shape[1] != self.input_shape_ser_actual[0]: # input_shape_ser_actual (config.WHISPER_EMBEDDING_DIM,) idi
                logger.warning(f"Havuzlanmış Whisper embedding boyutu ({embedding_np.shape[1]}) ile beklenen ({self.input_shape_ser_actual[0]}) eşleşmiyor!")
            else:
                logger.debug(f"Havuzlanmış Whisper embedding başarıyla çıkarıldı ve boyutlar eşleşiyor: {embedding_np.shape}")

            return embedding_np
        except Exception as e:
            logger.error(f"Whisper embedding çıkarılırken hata oluştu: {e}", exc_info=True)
            return None

    def predict_from_image_frame(self, bgr_frame):
        """Verilen bir BGR görüntü çerçevesinden yüz ve duygu tahmini yapar."""
        if bgr_frame is None or bgr_frame.size == 0:
            logger.warning("predict_from_image_frame: Gelen bgr_frame boş veya None.")
            return None, [], None, None

        detected_faces_coords = []
        fer_predictions = []
        processed_frame = bgr_frame.copy()
        gray_frame_for_haar = None # Artık kullanılmayacak ama kalsa da zararı yok

        # --- Yüz Algılama (Sadece DNN) ---
        if self.dnn_face_detector_net is None:
            # logger.warning("DNN Yüz dedektörü yüklenmedi, atlanıyor.") # Bu log gereksiz, _load_face_detector zaten logluyor.
            return processed_frame, [], None, None # Yüz dedektörü yoksa işlem yapma

        (h, w) = processed_frame.shape[:2]
        # Giriş blobunu oluştur (300x300'e yeniden boyutlandır ve normalize et)
        blob = cv2.dnn.blobFromImage(cv2.resize(processed_frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.dnn_face_detector_net.setInput(blob)
        detections = self.dnn_face_detector_net.forward()

        # Algılanan yüzler üzerinde döngü
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > config.FACE_DETECTOR_DNN_CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Kutu koordinatlarının geçerli olduğundan emin ol
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w - 1, endX)
                endY = min(h - 1, endY)

                if startX >= endX or startY >= endY:
                    logger.debug(f"predict_from_image_frame (DNN): Geçersiz yüz kutusu koordinatları atlanıyor: ({startX},{startY})-({endX},{endY})")
                    continue
                
                detected_faces_coords.append((startX, startY, endX - startX, endY - startY)) # (x, y, w, h)
                face_roi_gray = cv2.cvtColor(bgr_frame[startY:endY, startX:endX], cv2.COLOR_BGR2GRAY)

                # --- FER Tahmini ---
                if self.fer_model is not None and face_roi_gray.size > 0:
                    preprocessed_face = self._preprocess_face_for_fer(face_roi_gray)
                    if preprocessed_face is not None:
                        try:
                            prediction = self.fer_model.predict(preprocessed_face)
                            fer_predictions.append(prediction[0]) # İlk (ve tek) batch elemanını al
                            # logger.debug(f"FER tahmini (olasılıklar): {prediction[0]}") # Bu çok fazla log üretir
                        except Exception as e:
                            logger.error(f"FER model tahmini sırasında hata: {e}", exc_info=True)
                            fer_predictions.append(np.zeros(len(self.fer_emotions))) # Hata durumunda boş tahmin
                    else:
                        fer_predictions.append(np.zeros(len(self.fer_emotions))) # Ön işleme hatası
                else:
                    # logger.debug("FER modeli yüklenmedi veya yüz ROI boş, FER tahmini atlanıyor.") # Bu log gereksiz
                    fer_predictions.append(np.zeros(len(self.fer_emotions))) # Model yoksa veya ROI boşsa

                # Yüzü çerçevele (DNN için de aynı)
                cv2.rectangle(processed_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        
        # Haar Cascade ile ilgili blok tamamen kaldırıldı.

        final_emotion_label = None
        final_emotion_probabilities_for_gui = None # GUI'ye gönderilecek tek bir olasılık seti

        if fer_predictions: # fer_predictions yüz başına olasılık vektörleri listesi
            # Şimdilik, tespit edilen son yüzün duygusunu ve olasılıklarını alalım
            # Daha gelişmiş bir mantık (örn: en büyük yüz, en yüksek güvenli tahmin vb.) eklenebilir.
            last_face_prediction_vector = fer_predictions[-1] # Son yüzün olasılık vektörü
            if last_face_prediction_vector is not None and last_face_prediction_vector.size > 0:
                final_emotion_label = self.fer_emotions[np.argmax(last_face_prediction_vector)]
                final_emotion_probabilities_for_gui = last_face_prediction_vector
            else: # Eğer son tahmin boşsa veya sorunluysa
                final_emotion_label = "Bilinmiyor" # veya None
                final_emotion_probabilities_for_gui = np.zeros(len(self.fer_emotions))
        else: # Hiç yüz tahmini yoksa
            final_emotion_label = None # veya "Yüz Yok"
            final_emotion_probabilities_for_gui = None # veya np.zeros(...)

        # Fonksiyon şimdi 4 değer döndürecek: işlenmiş kare, yüz koordinatları listesi, tek bir baskın duygu etiketi, ve o etikete ait olasılık vektörü
        return processed_frame, detected_faces_coords, final_emotion_label, final_emotion_probabilities_for_gui

    def predict_from_audio_segment(self, audio_segment_np, sample_rate=config.WHISPER_SAMPLING_RATE):
        """
        Verilen bir NumPy ses segmentinden duyguyu tahmin eder.
        Bu fonksiyon canlı analiz için kullanılır.

        Args:
            audio_segment_np (np.array): Mikrofondan alınan ses segmenti.
            sample_rate (int): Ses segmentinin örnekleme hızı.

        Returns:
            tuple: (predicted_emotion_label, emotion_probabilities_dict)
                   Hata durumunda (None, None) döner.
        """
        if self.ser_model is None:
            # logger.debug("SER modeli yüklü değil (canlı ses).") # Yorum kaldırıldı
            return None, None

        # SER_FEATURE_TYPE kontrolü kaldırıldı, sadece Whisper varsayılıyor.
        if not hasattr(self, 'whisper_processor') or not hasattr(self, 'tf_whisper_model') or \
           self.whisper_processor is None or self.tf_whisper_model is None:
            logger.error("Whisper işlemcisi/TF modeli başlatılmamış. SER tahmini atlanıyor.")
            return "Whisper Yok", None # Hata mesajı olarak döndürülebilir veya None, None
        
        processed_audio_input = self._extract_whisper_embedding_for_ser(audio_segment_np, sample_rate)
        
        if processed_audio_input is None:
            logger.warning(f"Ses ön işleme (Whisper embedding) sonucu boş. SER Tahmini atlanıyor.")
            return "Onisleme Hatasi", None # Hata mesajı olarak döndürülebilir veya None, None

        try:
            if processed_audio_input.ndim == 1:
                processed_audio_input = np.expand_dims(processed_audio_input, axis=0)
            
            expected_input_dim_model = self.ser_model.input_shape[-1]
            if processed_audio_input.shape[-1] != expected_input_dim_model:
                logger.error(f"SER modeli için işlenmiş ses girdisinin son boyutu ({processed_audio_input.shape[-1]}) modelin beklediği ({expected_input_dim_model}) ile eşleşmiyor! Girdi Şekli: {processed_audio_input.shape}")
                return "Boyut HataSER", None # Hata mesajı olarak döndürülebilir veya None, None

            predictions = self.ser_model.predict(processed_audio_input, verbose=0)
            probabilities_vector = predictions[0]
            
            predicted_index = np.argmax(probabilities_vector)
            predicted_label = config.MODEL_OUTPUT_EMOTIONS[predicted_index] # TARGET_EMOTIONS_ORDERED -> MODEL_OUTPUT_EMOTIONS
            
            probs_dict = {config.MODEL_OUTPUT_EMOTIONS[i]: float(probabilities_vector[i]) for i in range(len(probabilities_vector))} # TARGET_EMOTIONS_ORDERED -> MODEL_OUTPUT_EMOTIONS
            
            return predicted_label, probs_dict
        except Exception as e:
            logger.error(f"Canlı sesten duygu tahmini sırasında hata: {e}", exc_info=True) # exc_info=True eklendi
            return "HataSER", None # Hata mesajı olarak döndürülebilir veya None, None


    def predict_from_image_file(self, image_path):
        """
        Verilen bir görüntü dosyasından yüz ifadesi duygusunu tahmin eder.
        (Eski predictor.py'deki fonksiyonun sınıf içindeki hali)
        """
        if self.fer_model is None:
            logger.error("FER modeli yüklenmemiş. Dosyadan görüntü tahmini yapılamıyor.")
            return None, None
        if self.face_detector_type == "dnn" and self.dnn_face_detector_net is None:
            logger.error("Yüz dedektörü (DNN) yüklenmedi. Dosyadan görüntü tahmini yapılamıyor.")
            return None, None
            
        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                logger.error(f"Görüntü dosyası yüklenemedi: {image_path}")
                return None, None
            
            # predict_from_image_frame fonksiyonunu yeniden kullanabiliriz
            label, probs_dict, _ = self.predict_from_image_frame(img_bgr) # face_box'ı burada kullanmıyoruz
            
            if label:
                 logger.info(f"Görüntü dosyası ({os.path.basename(image_path)}) -> Tahmin: {label}")
            return label, probs_dict
        except Exception as e:
            logger.error(f"Görüntü dosyasından duygu tahmini sırasında hata: {e}", exc_info=True)
            return None, None


    def predict_from_audio_file(self, audio_path):
        """
        Verilen bir ses dosyasından duyguyu tahmin eder.
        (Eski predictor.py'deki fonksiyonun sınıf içindeki hali)
        """
        if self.ser_model is None:
            logger.error("SER modeli yüklenmemiş. Dosyadan ses tahmini yapılamıyor.")
            return None, None
        try:
            # Ses dosyasını Whisper modelinin beklediği örnekleme oranıyla yükle
            audio_data, sr = librosa.load(audio_path, sr=config.WHISPER_SAMPLING_RATE)
            if audio_data is None:
                logger.error(f"Ses dosyası yüklenemedi: {audio_path}")
                return None, None

            # predict_from_audio_segment zaten doğru örnekleme hızını bekliyor.
            label, probs_dict = self.predict_from_audio_segment(audio_data, sr) # sr burada WHISPER_SAMPLING_RATE olacak

            if label and label not in ["Whisper Yok", "Onisleme Hatasi", "Boyut HataSER", "HataSER"]:
                logger.info(f"Ses dosyası ({os.path.basename(audio_path)}) -> Tahmin: {label}")
            elif label:
                 logger.warning(f"Ses dosyası ({os.path.basename(audio_path)}) için SER tahmini başarısız oldu: {label}")
            return label, probs_dict
        except Exception as e:
            logger.error(f"Ses dosyasından duygu tahmini sırasında hata: {e}", exc_info=True)
            return None, None
