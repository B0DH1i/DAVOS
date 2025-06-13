
import os
import tensorflow as tf
import numpy as np
import cv2
import librosa


from transformers import WhisperProcessor, TFWhisperModel

# Relative imports
from ..utils.logging_utils import setup_logger
from ..utils import file_utils  # For model loading
from ..configs import main_config as config



logger = setup_logger(__name__, log_file=config.APPLICATION_LOG_FILE)

class EmotionPredictorEngine:

    def __init__(self,
                 fer_model_name=config.DEFAULT_FER_MODEL_LOAD_NAME,
                 ser_model_name=config.DEFAULT_SER_MODEL_LOAD_NAME,
                 video_emotion_buffer_size=5,
                 audio_emotion_buffer_size=5
                 ):
        logger.info(f"Initializing EmotionPredictorEngine...")
        logger.info(f"FER Model Name: {fer_model_name}")
        logger.info(f"SER Model Name: {ser_model_name}")

        self.face_detector_type = "dnn"
        logger.info(f"Face Detector Type: {self.face_detector_type}")
        self.video_emotion_buffer_size = video_emotion_buffer_size
        self.audio_emotion_buffer_size = audio_emotion_buffer_size

        self.video_emotion_buffer = []
        self.audio_emotion_buffer = []

        # Load face detector
        self.face_cascade = None
        self.dnn_face_detector_net = None
        self._load_face_detector()
        logger.info(f"Post _load_face_detector: self.face_detector_type='{self.face_detector_type}', self.face_cascade is None: {self.face_cascade is None}, self.dnn_face_detector_net is None: {self.dnn_face_detector_net is None}") # This log REMAINS

        self.fer_emotions = config.MODEL_OUTPUT_EMOTIONS

        # Load FER model
        self.fer_model = None
        self.input_shape_fer = config.INPUT_SHAPE_FER  # (48, 48, 1)
        actual_fer_model_load_name = fer_model_name
        if fer_model_name == "latest":
            actual_fer_model_load_name = file_utils.get_latest_model_directory_for_base(config.FER_MODEL_NAME_PREFIX)
            if not actual_fer_model_load_name:
                logger.warning(f"No suitable FER model found for 'latest' (starting with {config.FER_MODEL_NAME_PREFIX}). Default model will not be used.")
                actual_fer_model_load_name = None  # No model will be loaded

        self.fer_model_folder = actual_fer_model_load_name  # Ensure self.fer_model_folder is set

        if actual_fer_model_load_name:
            logger.info(f"Loading FER model: {actual_fer_model_load_name}...")
            custom_obj_fer = {"Swish": "swish"} if "xception" in actual_fer_model_load_name.lower() else None
            self.fer_model = file_utils.load_trained_model(actual_fer_model_load_name, custom_objects=custom_obj_fer)
            if self.fer_model:
                logger.info(f"FER model successfully loaded: {actual_fer_model_load_name}")
                try:
                    logger.info(f"Loaded FER model's expected input shape: {self.fer_model.input_shape}")
                except Exception as e:
                    logger.warning(f"Could not get FER model input shape: {e}")
            else:
                logger.error(f"Failed to load FER model: {actual_fer_model_load_name}")
        else:
            logger.warning("No FER model specified or found for loading.")


        self.ser_model = None
        self.whisper_processor = None
        self.tf_whisper_model = None
        self.input_shape_ser_actual = None

        actual_ser_model_load_name = ser_model_name
        if ser_model_name == "latest":

            ser_prefix_to_search = config.SER_MODEL_NAME_PREFIX + "_whisper"

            actual_ser_model_load_name = file_utils.get_latest_model_directory_for_base(ser_prefix_to_search)
            if not actual_ser_model_load_name:
                logger.warning(f"No suitable SER model found for 'latest' (starting with {ser_prefix_to_search}). Default model will not be used.")
                actual_ser_model_load_name = None

        self.ser_model_folder = actual_ser_model_load_name

        try:
            logger.info(f"Loading Whisper processor and TFWhisperModel: {config.WHISPER_MODEL_NAME}")
            self.whisper_processor = WhisperProcessor.from_pretrained(config.WHISPER_MODEL_NAME)
            self.tf_whisper_model = TFWhisperModel.from_pretrained(config.WHISPER_MODEL_NAME)
            logger.info("Whisper processor and TFWhisperModel loaded successfully.")
            if self.tf_whisper_model:
                self.input_shape_ser_actual = (self.tf_whisper_model.config.d_model,)
                logger.info(f"Whisper embedding size set to {self.input_shape_ser_actual} for SER ({config.WHISPER_MODEL_NAME} d_model: {self.tf_whisper_model.config.d_model}).")
            else:
                self.input_shape_ser_actual = (config.WHISPER_EMBEDDING_DIM,)
                logger.warning(f"TFWhisperModel could not be loaded, Whisper embedding size for SER defaulted to {self.input_shape_ser_actual} ({config.WHISPER_EMBEDDING_DIM}).")
        except Exception as e:
            logger.error(f"Error loading Whisper model or processor ({config.WHISPER_MODEL_NAME}): {e}", exc_info=True)
            self.input_shape_ser_actual = (config.WHISPER_EMBEDDING_DIM,)  # Default size in case of error
            logger.warning(f"Due to error, Whisper embedding size for SER defaulted to {self.input_shape_ser_actual} ({config.WHISPER_EMBEDDING_DIM}).")

        if actual_ser_model_load_name:
            logger.info(f"Loading SER model: {actual_ser_model_load_name}...")
            self.ser_model = file_utils.load_trained_model(actual_ser_model_load_name)
            if self.ser_model:
                logger.info(f"SER model successfully loaded: {actual_ser_model_load_name}")
                try:
                    logger.info(f"Loaded SER model's expected input shape: {self.ser_model.input_shape}")
                    if config.SER_FEATURE_TYPE == "whisper" and self.input_shape_ser_actual:
                        model_input_dim = self.ser_model.input_shape[-1]
                        if model_input_dim != self.input_shape_ser_actual[0]:
                            logger.warning(f"SER model's expected input dimension ({model_input_dim}) does not match Whisper embedding dimension ({self.input_shape_ser_actual[0]})!")
                        else:
                            logger.info(f"SER model input dimension ({model_input_dim}) is compatible with Whisper embedding dimension ({self.input_shape_ser_actual[0]}).")
                except Exception as e:
                    logger.warning(f"Could not get SER model input shape: {e}")
            else:
                logger.error(f"Failed to load SER model: {actual_ser_model_load_name}")
        else:
            logger.warning("No SER model specified or found for loading.")

        logger.info("EmotionPredictorEngine successfully initialized.")

    def _load_face_detector(self):
        """Loads the face detector (DNN)."""
        logger.info(f"Loading face detection model ({self.face_detector_type})...")
        if os.path.exists(config.FACE_DETECTOR_DNN_PROTOTXT_PATH) and \
           os.path.exists(config.FACE_DETECTOR_DNN_MODEL_PATH):
            try:
                self.dnn_face_detector_net = cv2.dnn.readNetFromCaffe(
                    config.FACE_DETECTOR_DNN_PROTOTXT_PATH,
                    config.FACE_DETECTOR_DNN_MODEL_PATH
                )
                if self.dnn_face_detector_net is None:
                    logger.error(f"DNN face detection model could not be loaded with cv2.dnn.readNetFromCaffe (returned None). Prototxt: {config.FACE_DETECTOR_DNN_PROTOTXT_PATH}, Model: {config.FACE_DETECTOR_DNN_MODEL_PATH}")
                else:
                    logger.info(f"DNN face detection model successfully loaded: {config.FACE_DETECTOR_DNN_MODEL_PATH}")
            except Exception as e:
                logger.error(f"Error loading DNN face detection model: {e}. Prototxt: {config.FACE_DETECTOR_DNN_PROTOTXT_PATH}, Model: {config.FACE_DETECTOR_DNN_MODEL_PATH}", exc_info=True)
                self.dnn_face_detector_net = None
        else:
            logger.error(f"DNN face detection model files not found. Prototxt: {config.FACE_DETECTOR_DNN_PROTOTXT_PATH}, Model: {config.FACE_DETECTOR_DNN_MODEL_PATH}")
            self.dnn_face_detector_net = None


    def _preprocess_face_for_fer(self, face_roi, target_size=(48,48)):
        """Preprocesses the cropped face for the FER model."""
        if face_roi is None or face_roi.size == 0:
            logger.warning("_preprocess_face_for_fer: Input face_roi is empty or None.")
            return None

        try:
            resized_face = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_AREA)

            is_vgg16 = self.fer_model_folder and "vgg16" in self.fer_model_folder.lower()

            if is_vgg16:
                # VGG16 expects RGB (3 channels). Input resized_face is grayscale (H,W).
                bgr_face = cv2.cvtColor(resized_face, cv2.COLOR_GRAY2BGR)  # (H, W) -> (H, W, 3)
                normalized_face = bgr_face.astype(np.float32) / 255.0
                # Model expects (1, H, W, 3)
                processed_face_for_model = np.expand_dims(normalized_face, axis=0)
            else:
                # resized_face is already grayscale (H, W).
                normalized_face = resized_face.astype(np.float32) / 255.0
                expanded_face = np.expand_dims(normalized_face, axis=-1)  # (H, W) -> (H, W, 1)
                # Model expects (1, H, W, 1)
                processed_face_for_model = np.expand_dims(expanded_face, axis=0)

            return processed_face_for_model
        except cv2.error as e:
            logger.error(f"OpenCV error in _preprocess_face_for_fer: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"General error in _preprocess_face_for_fer: {e}", exc_info=True)
            return None

    def _extract_whisper_embedding_for_ser(self, audio_segment_np, sample_rate=config.WHISPER_SAMPLING_RATE):
        """Extracts Whisper embeddings from an audio segment for SER."""
        if self.whisper_processor is None or self.tf_whisper_model is None:
            logger.error("Whisper processor or TF model not loaded. Cannot extract embedding.")
            return None
        try:
            if audio_segment_np.dtype != np.float32:
                audio_float = audio_segment_np.astype(np.float32) / np.iinfo(audio_segment_np.dtype).max
            else:
                audio_float = audio_segment_np

            if sample_rate != config.WHISPER_SAMPLING_RATE:
                audio_float = librosa.resample(audio_float, orig_sr=sample_rate, target_sr=config.WHISPER_SAMPLING_RATE)

            inputs = self.whisper_processor(audio_float, sampling_rate=config.WHISPER_SAMPLING_RATE, return_tensors="tf")
            input_features = inputs.input_features


            decoder_start_token_id = self.tf_whisper_model.config.decoder_start_token_id
            if decoder_start_token_id is None:
                pass

            decoder_input_ids = tf.constant([[decoder_start_token_id]])  # Shape: (batch_size, 1)

            model_outputs = self.tf_whisper_model(
                input_features,
                decoder_input_ids=decoder_input_ids,  # Provide decoder input
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True
            )

            if not hasattr(model_outputs, 'encoder_last_hidden_state') or model_outputs.encoder_last_hidden_state is None:
                logger.error("Encoder outputs (encoder_last_hidden_state) not found in TFWhisperModel output.")
                logger.error(f"Model output type: {type(model_outputs)}, Available keys/attributes: {dir(model_outputs)}")
                return None

            encoder_last_hidden_state = model_outputs.encoder_last_hidden_state

            pooled_embedding = tf.reduce_mean(encoder_last_hidden_state, axis=1)
            # Shape: (batch, hidden_dim) -> (e.g., 1, 512)

            # The SER model expects a numpy array.
            embedding_np = pooled_embedding.numpy()

            if embedding_np.shape[1] != self.input_shape_ser_actual[0]:  # input_shape_ser_actual (config.WHISPER_EMBEDDING_DIM,)
                logger.warning(f"Pooled Whisper embedding dimension ({embedding_np.shape[1]}) does not match expected ({self.input_shape_ser_actual[0]})!")
            else:
                logger.debug(f"Pooled Whisper embedding successfully extracted and dimensions match: {embedding_np.shape}")

            return embedding_np
        except Exception as e:
            logger.error(f"Error extracting Whisper embedding: {e}", exc_info=True)
            return None

    def predict_from_image_frame(self, bgr_frame):
        """Performs face and emotion prediction from a given BGR image frame."""
        if bgr_frame is None or bgr_frame.size == 0:
            logger.warning("predict_from_image_frame: Input bgr_frame is empty or None.")
            return None, [], None, None

        detected_faces_coords = []
        fer_predictions = []
        processed_frame = bgr_frame.copy()
        gray_frame_for_haar = None

        # --- Face Detection (DNN only) ---
        if self.dnn_face_detector_net is None:
            return processed_frame, [], None, None

        (h, w) = processed_frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(processed_frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.dnn_face_detector_net.setInput(blob)
        detections = self.dnn_face_detector_net.forward()

        # Loop over the detected faces
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > config.FACE_DETECTOR_DNN_CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure box coordinates are valid
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w - 1, endX)
                endY = min(h - 1, endY)

                if startX >= endX or startY >= endY:
                    logger.debug(f"predict_from_image_frame (DNN): Skipping invalid face box coordinates: ({startX},{startY})-({endX},{endY})")
                    continue

                detected_faces_coords.append((startX, startY, endX - startX, endY - startY))  # (x, y, w, h)
                face_roi_gray = cv2.cvtColor(bgr_frame[startY:endY, startX:endX], cv2.COLOR_BGR2GRAY)

                # --- FER Prediction ---
                if self.fer_model is not None and face_roi_gray.size > 0:
                    preprocessed_face = self._preprocess_face_for_fer(face_roi_gray)
                    if preprocessed_face is not None:
                        try:
                            prediction = self.fer_model.predict(preprocessed_face)
                            fer_predictions.append(prediction[0])  # Get the first (and only) batch element
                        except Exception as e:
                            logger.error(f"Error during FER model prediction: {e}", exc_info=True)
                            fer_predictions.append(np.zeros(len(self.fer_emotions)))  # Empty prediction on error
                    else:
                        fer_predictions.append(np.zeros(len(self.fer_emotions)))  # Preprocessing error
                else:
                    fer_predictions.append(np.zeros(len(self.fer_emotions)))  # If no model or empty ROI

                # Draw rectangle around face
                cv2.rectangle(processed_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        final_emotion_label = None
        final_emotion_probabilities_for_gui = None  # A single set of probabilities to send to GUI

        if fer_predictions:  # fer_predictions is a list of probability vectors per face
            last_face_prediction_vector = fer_predictions[-1]  # Probability vector of the last face
            if last_face_prediction_vector is not None and last_face_prediction_vector.size > 0:
                final_emotion_label = self.fer_emotions[np.argmax(last_face_prediction_vector)]
                final_emotion_probabilities_for_gui = last_face_prediction_vector
            else:  # If the last prediction is empty or problematic
                final_emotion_label = "Unknown"
                final_emotion_probabilities_for_gui = np.zeros(len(self.fer_emotions))
        else:  # If no face predictions
            final_emotion_label = None
            final_emotion_probabilities_for_gui = None

        return processed_frame, detected_faces_coords, final_emotion_label, final_emotion_probabilities_for_gui

    def predict_from_audio_segment(self, audio_segment_np, sample_rate=config.WHISPER_SAMPLING_RATE):
        """Predicts emotion from a given audio segment."""
        if self.ser_model is None:
            return None, None

        if not hasattr(self, 'whisper_processor') or not hasattr(self, 'tf_whisper_model') or \
           self.whisper_processor is None or self.tf_whisper_model is None:
            logger.error("Whisper processor/TF model not initialized. Skipping SER prediction.")
            return "No Whisper", None

        processed_audio_input = self._extract_whisper_embedding_for_ser(audio_segment_np, sample_rate)

        if processed_audio_input is None:
            logger.warning(f"Audio preprocessing (Whisper embedding) resulted in empty output. Skipping SER prediction.")
            return "Preprocessing Error", None

        try:
            if processed_audio_input.ndim == 1:
                processed_audio_input = np.expand_dims(processed_audio_input, axis=0)

            expected_input_dim_model = self.ser_model.input_shape[-1]
            if processed_audio_input.shape[-1] != expected_input_dim_model:
                logger.error(f"Last dimension of processed audio input for SER model ({processed_audio_input.shape[-1]}) does not match model's expected ({expected_input_dim_model})! Input Shape: {processed_audio_input.shape}")
                return "Dimension ErrorSER", None

            predictions = self.ser_model.predict(processed_audio_input, verbose=0)
            probabilities_vector = predictions[0]

            predicted_index = np.argmax(probabilities_vector)
            predicted_label = config.MODEL_OUTPUT_EMOTIONS[predicted_index]

            probs_dict = {config.MODEL_OUTPUT_EMOTIONS[i]: float(probabilities_vector[i]) for i in range(len(probabilities_vector))}

            return predicted_label, probs_dict
        except Exception as e:
            logger.error(f"Error during emotion prediction from live audio: {e}", exc_info=True)
            return "ErrorSER", None


    def predict_from_image_file(self, image_path):
        """
        Predicts facial emotion from a given image file.
        """
        if self.fer_model is None:
            logger.error("FER model not loaded. Cannot predict from image file.")
            return None, None
        if self.face_detector_type == "dnn" and self.dnn_face_detector_net is None:
            logger.error("Face detector (DNN) not loaded. Cannot predict from image file.")
            return None, None

        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                logger.error(f"Could not load image file: {image_path}")
                return None, None

            # Unpack only the relevant values
            _, _, label, probs_dict = self.predict_from_image_frame(img_bgr)

            if label:
                logger.info(f"Image file ({os.path.basename(image_path)}) -> Prediction: {label}")
            return label, probs_dict
        except Exception as e:
            logger.error(f"Error during emotion prediction from image file: {e}", exc_info=True)
            return None, None


    def predict_from_audio_file(self, audio_path):
        """
        Predicts emotion from a given audio file.
        """
        if self.ser_model is None:
            logger.error("SER model not loaded. Cannot predict from audio file.")
            return None, None
        try:
            audio_data, sr = librosa.load(audio_path, sr=config.WHISPER_SAMPLING_RATE)
            if audio_data is None:
                logger.error(f"Could not load audio file: {audio_path}")
                return None, None

            label, probs_dict = self.predict_from_audio_segment(audio_data, sr)

            if label and label not in ["No Whisper", "Preprocessing Error", "Dimension ErrorSER", "ErrorSER"]:
                logger.info(f"Audio file ({os.path.basename(audio_path)}) -> Prediction: {label}")
            elif label:
                logger.warning(f"SER prediction failed for audio file ({os.path.basename(audio_path)}): {label}")
            return label, probs_dict
        except Exception as e:
            logger.error(f"Error during emotion prediction from audio file: {e}", exc_info=True)
            return None, None
