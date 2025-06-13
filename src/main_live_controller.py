import cv2
import time
import numpy as np
import queue
import threading
import argparse
import librosa
import os
import sys 

from .configs import main_config as config
from .utils.logging_utils import setup_logger
from .utils import file_utils

from .live.input_handler import InputController
from .live.output_handler import OutputController
from .core.predictor_engine import EmotionPredictorEngine
from .core.integration_engine import integrate_emotion_probabilities, PROBABILITY_HISTORY
from .core.lazanov_audio_manager import LazanovAudioManager

# Logger
logger = setup_logger("MainLiveController", log_file=config.APPLICATION_LOG_FILE)

FACE_PROCESSING_INPUT_QUEUE = queue.Queue(maxsize=1)
FACE_PROCESSING_OUTPUT_QUEUE = queue.Queue(maxsize=1)
AUDIO_PROCESSING_OUTPUT_QUEUE = queue.Queue(maxsize=1)

THREAD_RUNNING_EVENT = threading.Event()

def face_analysis_worker(predictor_engine):
    logger.info("Face analysis worker started (stub).")

    while THREAD_RUNNING_EVENT.is_set():
        time.sleep(0.1)
    logger.info("Face analysis worker stopped (stub).")

def audio_analysis_worker(predictor_engine, input_controller, cli_args): 
    logger.info(f"Audio analysis worker started. Mode: {'File ('+cli_args.audio_file+')' if cli_args.audio_file else 'Live Microphone'}")

    try:
        if cli_args.audio_file:
            # Process audio file
            if not predictor_engine.ser_model:
                logger.warning("SER model not loaded. Audio file processing skipped by worker.")
                return

            try:
                audio_file_path = os.path.abspath(cli_args.audio_file) # Convert to absolute path
                logger.info(f"Loading audio file: {audio_file_path}")

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


            logger.info("Audio file processing complete. Worker will remain idle.")
            while THREAD_RUNNING_EVENT.is_set(): # Remain idle until main event is cleared
                time.sleep(0.5)

        else:
            # Process live microphone (similar to current stub but with actual processing)
            if not predictor_engine.ser_model or not input_controller.is_audio_ready:
                logger.warning("SER model not loaded or microphone not ready. Live audio processing skipped by worker.")
                return

            last_ser_processing_time = 0
            while THREAD_RUNNING_EVENT.is_set():
                current_time = time.time()
                if current_time - last_ser_processing_time >= config.ANALYSIS_INTERVAL_SER:
                    # Additional check to prevent errors during shutdown
                    if not THREAD_RUNNING_EVENT.is_set() or not input_controller.is_audio_ready:
                        logger.info("Audio worker: Main thread stopped or audio input not ready. Skipping audio capture.")
                        break # Exit loop

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


                time.sleep(0.05)
    except Exception as e:
        logger.error(f"Critical error in audio_analysis_worker: {e}", exc_info=True)
    finally:
        logger.info("Audio analysis worker stopped.")

def main_live_loop():
    """Main live analysis loop."""
    logger.info("Starting Live Emotion Analysis System (DOVAS) v3...")

    parser = argparse.ArgumentParser(description="DOVAS Live Emotion Analysis System Controller")
    parser.add_argument("--video_file", type=str, default=None,
                        help="Path to a video file to use instead of the live camera feed.")
    parser.add_argument("--audio_file", type=str, default=None,
                        help="Path to an audio file to process for SER (overrides live microphone).")
    parser.add_argument("--fer_model", type=str, default=None,
                        help=f"Base name or path for the FER model. Defaults to config value: {config.DEFAULT_FER_MODEL_LOAD_NAME}")
    parser.add_argument("--ser_model", type=str, default=None,
                        help=f"Base name or path for the SER model. Defaults to config value: {config.DEFAULT_SER_MODEL_LOAD_NAME}")
    parser.add_argument("--window_width", type=int, default=config.DISPLAY_MAX_WIDTH, help="Display window width.")
    parser.add_argument("--window_height", type=int, default=config.DISPLAY_MAX_HEIGHT, help="Display window height.")
    parser.add_argument("--fullscreen", action="store_true", help="Start in fullscreen mode.")
    parser.add_argument("--show_fer_probs", action="store_true", help="Show FER probabilities.")
    parser.add_argument("--show_ser_probs", action="store_true", help="Show SER probabilities.")
    parser.add_argument("--show_integrated_probs", action="store_true", help="Show integrated probabilities.")
    parser.add_argument("--camera_id", type=int, default=config.CAMERA_INDEX, help="Camera ID to use.")
    parser.add_argument("--disable_microphone", action="store_true", help="Disable live microphone input.")

    args = parser.parse_args()

    input_ctrl = None
    output_ctrl = None
    predictor_engine_instance = None
    lazanov_manager = None
    audio_worker_thread = None
    exit_code = 0

    try:
        logger.info("Checking/creating project directories...")
        file_utils.create_project_directories()

        logger.info("Initializing EmotionPredictorEngine...")
        predictor_engine_instance = EmotionPredictorEngine(
            fer_model_name=args.fer_model if args.fer_model else config.DEFAULT_FER_MODEL_LOAD_NAME,
            ser_model_name=args.ser_model if args.ser_model else config.DEFAULT_SER_MODEL_LOAD_NAME
        )
        logger.info("EmotionPredictorEngine successfully created.")

        if config.LAZANOV_INTERVENTION_ENABLED:
            try:
                lazanov_manager = LazanovAudioManager()
                logger.info("LazanovAudioManager successfully initialized.")
            except Exception as e:
                logger.error(f"Error initializing LazanovAudioManager: {e}. Lazanov interventions will be disabled.")
                lazanov_manager = None
        else:
            lazanov_manager = None # Ensure it's None if not enabled
            logger.info("Lazanov interventions disabled in config file.")

        THREAD_RUNNING_EVENT.set()

        # Face analysis worker (currently a stub)
        face_worker_thread = None
        if predictor_engine_instance.fer_model:
            face_worker_thread = threading.Thread(target=face_analysis_worker, args=(predictor_engine_instance,))
            face_worker_thread.daemon = True
            face_worker_thread.start()
            logger.info("Face analysis worker thread started.")

        # Audio analysis worker
        audio_worker_thread = None

        if predictor_engine_instance.fer_model is None and predictor_engine_instance.ser_model is None:
            logger.critical("Neither FER nor SER model could be loaded by the Predictor Engine. System cannot run.")
            return # Exit if no models are loaded
        elif predictor_engine_instance.fer_model is None:
            logger.warning("FER model not loaded. Face emotion analysis will be skipped.")
        elif predictor_engine_instance.ser_model is None:
            logger.warning("SER model not loaded. Speech emotion analysis will be skipped.")

        if predictor_engine_instance.dnn_face_detector_net is not None:
            logger.debug(f"MAIN_LOOP_DNN_CHECK: DNN detector is loaded.")
        else:
            logger.warning(f"MAIN_LOOP_DNN_CHECK: DNN detector is NOT loaded.")


        enable_video = predictor_engine_instance.fer_model is not None and (args.video_file is not None or args.camera_id is not None)
        enable_audio = predictor_engine_instance.ser_model is not None and (args.audio_file is not None or not args.disable_microphone)

        video_input_for_controller = None
        if args.video_file:
            video_input_for_controller = args.video_file


        input_ctrl = InputController(
            video_file_path=video_input_for_controller,
            enable_audio=enable_audio,
            camera_id=args.camera_id
        )
        logger.info(f"InputController initialized. Video enabled: {input_ctrl.is_camera_ready}, Audio enabled: {input_ctrl.is_audio_ready}")

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
            "face_detection_status": "Initializing...",
            "fer_raw_label": "N/A",
            "fer_probs_dict": None,
            "ser_raw_label": "N/A",
            "ser_probs_dict": None,
            "integrated_label": "N/A",
            "integrated_probs_vector": None,
            "face_box_to_draw": None,
            "lazanov_music_status": "Lazanov Disabled" if not (lazanov_manager and config.LAZANOV_INTERVENTION_ENABLED) else "Passive",
            "lazanov_binaural_status": "Lazanov Disabled" if not (lazanov_manager and config.LAZANOV_INTERVENTION_ENABLED and config.LAZANOV_BINAURAL_BEATS_ENABLED) else "Passive"
        }

        # LAZANOV Integration: Variables for tracking emotion duration
        last_processed_dominant_emotion = None
        dominant_emotion_start_time = None

        empty_frame_count = 0 # Consecutive empty frame counter for video file

        logger.info("Starting main live analysis loop...")
        while THREAD_RUNNING_EVENT.is_set(): # Control with THREAD_RUNNING_EVENT
            current_time = time.time()
            frame_bgr = input_ctrl.capture_video_frame()

            if frame_bgr is None:
                if input_ctrl.is_video_file:
                    empty_frame_count += 1
                    if empty_frame_count > getattr(config, 'MAX_CONSECUTIVE_EMPTY_FRAMES_VIDEO_FILE', 10): # Get from config or use default
                        logger.info(f"End of video file reached or {empty_frame_count} consecutive empty frames. Stopping system.")
                        THREAD_RUNNING_EVENT.clear()
                        break
                else:
                    logger.warning("Could not read frame from live camera (frame_bgr is None).")


                if output_ctrl and not output_ctrl.handle_events():
                    logger.info("OutputController requested exit (during empty frame). Stopping system.")
                    THREAD_RUNNING_EVENT.clear()
                    break
                time.sleep(0.01)
                continue
            else:
                empty_frame_count = 0


            if predictor_engine_instance.fer_model and predictor_engine_instance.dnn_face_detector_net:
                if current_time - last_fer_analysis_time >= config.ANALYSIS_INTERVAL_FER:
                    predicted_emotion_face, fer_probs_dict, face_box = predictor_engine_instance.predict_from_image_frame(frame_bgr)

                    display_info["face_box_to_draw"] = face_box # To OutputController for drawing

                    if predicted_emotion_face and predicted_emotion_face not in predictor_engine_instance.ERROR_LABELS_FER:
                        display_info["face_detection_status"] = "Face Found" if face_box else "Face Found (No Box)"
                        display_info["fer_raw_label"] = predicted_emotion_face
                        display_info["fer_probs_dict"] = fer_probs_dict
                    elif predicted_emotion_face == predictor_engine_instance.FACE_NOT_FOUND_LABEL:
                        display_info["face_detection_status"] = "Face Not Found"
                        display_info["fer_raw_label"] = "N/A"
                        display_info["fer_probs_dict"] = None
                    else:
                        display_info["face_detection_status"] = predicted_emotion_face if predicted_emotion_face else "FER Error"
                        display_info["fer_raw_label"] = "N/A"
                        display_info["fer_probs_dict"] = None
                    last_fer_analysis_time = current_time
            elif not predictor_engine_instance.fer_model:
                display_info["face_detection_status"] = "FER Model Not Found"
                display_info["fer_raw_label"] = "N/A"
                display_info["fer_probs_dict"] = None
            elif not predictor_engine_instance.dnn_face_detector_net:
                display_info["face_detection_status"] = "Face Detector Not Found"
                display_info["fer_raw_label"] = "N/A"
                display_info["fer_probs_dict"] = None


            if predictor_engine_instance.ser_model:
                try:
                    ser_label_from_q, ser_probs_dict_from_q = AUDIO_PROCESSING_OUTPUT_QUEUE.get_nowait()
                    if ser_label_from_q and ser_probs_dict_from_q and ser_label_from_q not in predictor_engine_instance.ERROR_LABELS_SER:
                        display_info["ser_raw_label"] = ser_label_from_q
                        display_info["ser_probs_dict"] = ser_probs_dict_from_q
                    elif ser_label_from_q: # If an error label came
                        display_info["ser_raw_label"] = ser_label_from_q
                        display_info["ser_probs_dict"] = None
                except queue.Empty:
                    # If queue is empty and it's time for live microphone analysis, do it in the main thread
                    if not args.audio_file and input_ctrl.is_audio_ready and \
                       (current_time - last_ser_analysis_time >= config.ANALYSIS_INTERVAL_SER):
                        audio_segment = input_ctrl.capture_audio_segment(duration_seconds=config.LIVE_AUDIO_SEGMENT_DURATION)
                        if audio_segment is not None and len(audio_segment) > 0:
                            ser_label, ser_probs = predictor_engine_instance.predict_from_audio_segment(audio_segment, config.AUDIO_SAMPLE_RATE)
                            if ser_label and ser_label not in predictor_engine_instance.ERROR_LABELS_SER:
                                display_info["ser_raw_label"] = ser_label
                                display_info["ser_probs_dict"] = ser_probs
                            else:
                                display_info["ser_raw_label"] = ser_label if ser_label else "SER Error"
                                display_info["ser_probs_dict"] = None
                        else:
                            pass
                        last_ser_analysis_time = current_time
                except Exception as e:
                    logger.error(f"Error reading from audio output queue: {e}")
                    display_info["ser_raw_label"] = "SER Queue Error"
                    display_info["ser_probs_dict"] = None
            else:
                display_info["ser_raw_label"] = "SER Model Not Found"
                display_info["ser_probs_dict"] = None

            # --- Emotion Integration ---
            if current_time - last_integration_time >= config.ANALYSIS_INTERVAL_INTEGRATION:
                integrated_label, integrated_probs_vector = integrate_emotion_probabilities(
                    face_emotion_probs_dict=display_info.get("fer_probs_dict"),
                    speech_emotion_probs_dict=display_info.get("ser_probs_dict")
                )

                display_info["integrated_label"] = integrated_label
                display_info["integrated_probs_vector"] = integrated_probs_vector

                logger.info(f"Integrated Result -> Label: {integrated_label} (FER: {display_info.get('fer_raw_label', 'N/A')}, SER: {display_info.get('ser_raw_label', 'N/A')})")
                last_integration_time = current_time

            # --- LAZANOV INTERVENTION MANAGEMENT ---
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
                        logger.warning(f"Integrated label '{current_integrated_label}' not found in target emotion list.")
                elif current_integrated_label not in ["unknown", "N/A"]:
                    confidence_for_intervention = config.LAZANOV_TRIGGER_CONFIDENCE_THRESHOLD

                lazanov_music_status, lazanov_binaural_status = lazanov_manager.manage_intervention(
                    current_integrated_label,
                    confidence_for_intervention,
                    current_time
                )
                display_info["lazanov_music_status"] = lazanov_music_status
                display_info["lazanov_binaural_status"] = lazanov_binaural_status

            if output_ctrl:
                output_ctrl.display_frame(frame_bgr, display_info)

                if not output_ctrl.handle_events():
                    logger.info("OutputController requested exit. Stopping system.")
                    THREAD_RUNNING_EVENT.clear()
                    break
            else:
                logger.warning("OutputController not initialized, skipping display and event management.")
                if frame_bgr is not None:
                    try:
                        cv2.imshow("Fallback Display - OutputCtrl Error", frame_bgr)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("Fallback display closed with 'q' key.")
                            THREAD_RUNNING_EVENT.clear()
                            break
                    except Exception as e:
                        logger.error(f"Fallback cv2.imshow error: {e}")
                        THREAD_RUNNING_EVENT.clear()
                        break
                else:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        THREAD_RUNNING_EVENT.clear()
                        break
                    time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected (Ctrl+C). Stopping system.")
        THREAD_RUNNING_EVENT.clear() # Set the event to signal threads to stop
        logger.info("Thread running event cleared. Waiting for worker threads to join...")

        if face_worker_thread and face_worker_thread.is_alive():
            face_worker_thread.join(timeout=2.0)
            if face_worker_thread.is_alive():
                logger.warning("Face analysis worker thread did not join in time.")
        if audio_worker_thread and audio_worker_thread.is_alive():
            audio_worker_thread.join(timeout=2.0)
            if audio_worker_thread.is_alive():
                logger.warning("Audio analysis worker thread did not join in time.")

        logger.info("Worker threads joined or timed out.")

    except Exception as e:
        logger.critical(f"Critical error in main live analysis loop: {e}", exc_info=True)
        THREAD_RUNNING_EVENT.clear()
    finally:
        logger.info("Cleaning up resources...")
        if input_ctrl:
            input_ctrl.release()
        if output_ctrl:
            output_ctrl.quit()


        if lazanov_manager:
            logger.info("Shutting down LazanovAudioManager...")
            lazanov_manager.shutdown()
            logger.info("LazanovAudioManager successfully shut down.")

        # Clear the event again in case it was set by an error before finally
        THREAD_RUNNING_EVENT.clear()
        logger.info("System Shutdown Complete.")

if __name__ == "__main__":

    try:
        if not os.path.exists(config.LOGS_PATH):
            os.makedirs(config.LOGS_PATH, exist_ok=True)
    except Exception as e:
        print(f"[CRITICAL] Could not create log directory ({config.LOGS_PATH}): {e}. Exiting.")
        sys.exit(1) 

    main_live_loop()