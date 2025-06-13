import gradio as gr
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd
from src.configs import main_config as config
from src.core.predictor_engine import EmotionPredictorEngine
from src.utils.logging_utils import setup_logger
from src.utils import file_utils

logger = setup_logger("GuiApp")
try:
    file_utils.create_project_directories() 
    logger.info("Project directories successfully checked/created.")
except Exception as e:
    logger.error(f"Error while creating project directories: {e}", exc_info=True)


predictor_engine = None
try:
    logger.info("Initializing EmotionPredictorEngine for GUI...")
    predictor_engine = EmotionPredictorEngine(
        fer_model_name=config.DEFAULT_FER_MODEL_LOAD_NAME,
        ser_model_name=config.DEFAULT_SER_MODEL_LOAD_NAME
    )
    if predictor_engine.fer_model is None and predictor_engine.ser_model is None:
        logger.error("Neither FER nor SER model could be loaded for the GUI. Please check the models.")
    elif predictor_engine.fer_model is None:
        logger.warning("FER model could not be loaded for the GUI. Emotion prediction from images and live camera will not be available.")
    elif predictor_engine.ser_model is None:
        logger.warning("SER model could not be loaded for the GUI. Emotion prediction from audio and live microphone will not be available.")
    else:
        logger.info("EmotionPredictorEngine successfully initialized for GUI.")
except Exception as e:
    logger.error(f"Critical error initializing EmotionPredictorEngine for GUI: {e}", exc_info=True)

def predict_fer_from_image_upload(image_pil):
    """
    Performs Facial Expression Recognition (FER) from an uploaded image.
    """
    logger.info("predict_fer_from_image_upload called.")
    if predictor_engine is None:
        logger.warning("predict_fer_from_image_upload: Predictor engine is None.")
        return "Error: Predictor engine not initialized.", None, image_pil
    if predictor_engine.fer_model is None:
        logger.warning("predict_fer_from_image_upload: FER model is None.")
        return "FER model could not be loaded. Please check configuration.", None, image_pil
    
    if image_pil is None:
        logger.warning("predict_fer_from_image_upload: Incoming image_pil is None.")
        return "No image uploaded.", None, None

    try:
        logger.debug("predict_fer_from_image_upload: Converting frame to BGR.")
        frame_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        processed_bgr, detected_faces_list, dominant_emotion, raw_probs = predictor_engine.predict_from_image_frame(frame_bgr)
        logger.debug(f"predict_fer_from_image_upload: predict_from_image_frame result: dominant_emotion='{dominant_emotion}', #faces={len(detected_faces_list) if detected_faces_list else 0}")
        
        if processed_bgr is not None:
            output_image_pil = Image.fromarray(cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB))
        else:
            logger.warning("predict_fer_from_image_upload: processed_bgr came as None, using original frame.")
            output_image_pil = image_pil
        
        if dominant_emotion and raw_probs is not None:
            df_probs = pd.DataFrame({
                "label": config.MODEL_OUTPUT_EMOTIONS,
                "value": [float(prob) for prob in raw_probs]
            })
            logger.info(f"predict_fer_from_image_upload: Returning result - Emotion: {dominant_emotion}")
            return f"Dominant Emotion: {dominant_emotion}", df_probs, output_image_pil
        elif detected_faces_list:
             logger.info("FER GUI (Upload): Face detected but emotion could not be fully determined.")
             return "Emotion could not be predicted (but face found).", None, output_image_pil
        else:
            logger.info("FER GUI (Upload): No face detected in the image.")
            return "No face detected in the image.", None, output_image_pil
            
    except Exception as e:
        logger.error(f"FER GUI (Upload) prediction error: {e}", exc_info=True)
        return f"Error occurred: {str(e)}", None, image_pil

def predict_ser_from_audio_upload(audio_file_obj):
    """
    Performs Speech Emotion Recognition (SER) from an uploaded audio file.
    """
    logger.info("predict_ser_from_audio_upload called.")
    if predictor_engine is None:
        logger.warning("predict_ser_from_audio_upload: Predictor engine is None.")
        return "Error: Predictor engine not initialized.", None
    if predictor_engine.ser_model is None:
        logger.warning("predict_ser_from_audio_upload: SER model is None.")
        return "SER model could not be loaded. Please check configuration.", None
        
    if audio_file_obj is None:
        logger.warning("predict_ser_from_audio_upload: Incoming audio_file_obj is None.")
        return "No audio file uploaded.", None

    try:
        audio_filepath = audio_file_obj.name 
        logger.info(f"Audio file to be processed for SER: {audio_filepath}")
        dominant_emotion, emotion_probabilities = predictor_engine.predict_from_audio_file(audio_filepath)
        
        if dominant_emotion and emotion_probabilities:
            df_probs = pd.DataFrame(list(emotion_probabilities.items()), columns=["label", "value"])
            df_probs["value"] = df_probs["value"].astype(float) # Ensure values are float
            logger.info(f"predict_ser_from_audio_upload: Returning result - Emotion: {dominant_emotion}")
            return f"Dominant Emotion: {dominant_emotion}", df_probs
        else:
            logger.warning(f"predict_ser_from_audio_upload: Emotion could not be predicted from audio or result is None. Return: {dominant_emotion}, {emotion_probabilities}")
            return "Emotion could not be predicted from audio.", None # None clears the BarPlot
            
    except Exception as e:
        logger.error(f"SER GUI (Upload) prediction error: {e}", exc_info=True)
        return f"Error: {str(e)}", None

def predict_fer_from_live_camera(camera_frame_pil):
    """
    Performs Facial Expression Recognition (FER) from a live camera feed.
    """
    logger.info("predict_fer_from_live_camera called.")
    if predictor_engine is None:
        logger.warning("predict_fer_from_live_camera: Predictor engine is None.")
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return "Error: Predictor engine not initialized.", None, Image.fromarray(empty_frame)
    if predictor_engine.fer_model is None:
        logger.warning("predict_fer_from_live_camera: FER model is None.")
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return "FER model could not be loaded.", None, Image.fromarray(empty_frame)

    if camera_frame_pil is None:
        logger.warning("predict_fer_from_live_camera: Incoming camera_frame_pil is None.")
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return "Camera frame could not be captured.", None, Image.fromarray(empty_frame)

    try:
        logger.debug("predict_fer_from_live_camera: Converting frame to BGR.")
        frame_bgr = cv2.cvtColor(np.array(camera_frame_pil), cv2.COLOR_RGB2BGR)
        processed_bgr, detected_faces_list, dominant_emotion, raw_probs = predictor_engine.predict_from_image_frame(frame_bgr)
        logger.debug(f"predict_fer_from_live_camera: predict_from_image_frame result: dominant_emotion='{dominant_emotion}', #faces={len(detected_faces_list) if detected_faces_list else 0}")

        if processed_bgr is not None:
            output_image_pil = Image.fromarray(cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB))
        else: 
            logger.warning("predict_fer_from_live_camera: processed_bgr came as None, using original frame.")
            output_image_pil = camera_frame_pil
        
        if dominant_emotion and raw_probs is not None:
            df_probs = pd.DataFrame({
                "label": config.MODEL_OUTPUT_EMOTIONS,
                "value": [float(prob) for prob in raw_probs]
            })
            logger.info(f"predict_fer_from_live_camera: Returning result - Emotion: {dominant_emotion}")
            return f"Dominant Emotion: {dominant_emotion}", df_probs, output_image_pil
        elif detected_faces_list: 
             logger.info("predict_fer_from_live_camera: Returning result - Emotion could not be predicted (face found).")
             return "Emotion could not be predicted (face found).", None, output_image_pil
        else: 
            logger.info("predict_fer_from_live_camera: Returning result - No face detected in camera.")
            return "No face detected in camera.", None, output_image_pil
            
    except Exception as e:
        logger.error(f"FER GUI (Live Camera) prediction error: {e}", exc_info=True)
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        try: 
            return f"Error: {str(e)}", None, camera_frame_pil if camera_frame_pil else Image.fromarray(empty_frame)
        except: 
            return f"Error: {str(e)}", None, Image.fromarray(empty_frame)

def predict_ser_from_live_microphone(microphone_audio_input):
    """
    Performs Speech Emotion Recognition (SER) from live microphone input.
    """
    logger.info("predict_ser_from_live_microphone called.")
    if predictor_engine is None:
        logger.warning("predict_ser_from_live_microphone: Predictor engine is None.")
        return "Error: Predictor engine not initialized.", None
    if predictor_engine.ser_model is None:
        logger.warning("predict_ser_from_live_microphone: SER model is None.")
        return "SER model could not be loaded.", None
    
    if microphone_audio_input is None:
        logger.warning("predict_ser_from_live_microphone: Microphone input is None.")
        return "Waiting for microphone data...", None 

    try:
        sample_rate, audio_data_np = microphone_audio_input
        logger.debug(f"predict_ser_from_live_microphone: Audio received - Sample Rate: {sample_rate}, Data Length: {len(audio_data_np)}")

        if audio_data_np is None or len(audio_data_np) == 0:
            logger.debug("predict_ser_from_live_microphone: Empty audio data received.")
            return "Waiting for microphone data...", None # Or hold last successful result

        dominant_emotion, emotion_probabilities = predictor_engine.predict_from_audio_segment(audio_data_np, sample_rate)
        
        if dominant_emotion and emotion_probabilities:
            df_probs = pd.DataFrame(list(emotion_probabilities.items()), columns=["label", "value"])
            df_probs["value"] = df_probs["value"].astype(float)
            logger.info(f"predict_ser_from_live_microphone: Returning result - Emotion: {dominant_emotion}")
            return f"Dominant Emotion: {dominant_emotion}", df_probs
        elif dominant_emotion: # If there's an emotion label but probabilities are None (e.g., "Preprocessing Error")
            logger.warning(f"predict_ser_from_live_microphone: Emotion label '{dominant_emotion}' received but probabilities are None.")
            return f"Status: {dominant_emotion}", None
        else:
            logger.info("predict_ser_from_live_microphone: Emotion could not be predicted from audio (live).")
            return "Analyzing emotion...", None 
            
    except Exception as e:
        logger.error(f"SER GUI (Live Microphone) prediction error: {e}", exc_info=True)
        return f"Error: {str(e)}", None


# Creating the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Emotion Productivity Project Test Interface") as demo:
    gr.Markdown(
        """
        # Emotion Productivity Project - Advanced Test Interface
        Use this interface to test emotion analysis from uploaded images, audio files, live camera, and live microphone.
        """
    )
    
    with gr.Tab("Facial Expression Recognition (FER) - From Image"):
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Image (.jpg, .png)")
                fer_button = gr.Button("Predict Emotion (Image)")
            with gr.Column(scale=2):
                image_output_display = gr.Image(type="pil", label="Processed Image")
        with gr.Row():
            fer_result_text = gr.Textbox(label="Dominant Emotion", interactive=False)
        with gr.Row():
            fer_probs_output = gr.BarPlot(label="Emotion Probabilities (FER)", x="label", y="value", 
                                          x_title="Emotion", y_title="Probability",
                                          vertical=False, min_width=300)
        
        fer_button.click(
            predict_fer_from_image_upload,
            inputs=[image_input],
            outputs=[fer_result_text, fer_probs_output, image_output_display]
        )

    with gr.Tab("Facial Expression Recognition (FER) - Live Camera"):
        with gr.Row():
            with gr.Column(scale=1):
                live_camera_input = gr.Image(sources=["webcam"], type="pil", label="Live Camera Feed", streaming=True)
            with gr.Column(scale=2):
                live_camera_output_display = gr.Image(type="pil", label="Processed Camera Feed")
        with gr.Row():
            live_fer_result_text = gr.Textbox(label="Dominant Emotion (Live Camera)", interactive=False)
        with gr.Row():
            live_fer_probs_output = gr.BarPlot(label="Emotion Probabilities (Live Camera FER)", x="label", y="value",
                                               x_title="Emotion", y_title="Probability",
                                               vertical=False, min_width=300)
        
        live_camera_input.stream(
            predict_fer_from_live_camera,
            inputs=[live_camera_input],
            outputs=[live_fer_result_text, live_fer_probs_output, live_camera_output_display]
        )


    with gr.Tab("Speech Emotion Recognition (SER) - From Audio File"):
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.File(label="Upload Audio File (.wav, .mp3)", file_types=[".wav", ".mp3"]) 
                ser_button = gr.Button("Predict Emotion (Audio File)")
            with gr.Column(scale=2):
                ser_result_text = gr.Textbox(label="Dominant Emotion (Audio File)", interactive=False)
                ser_probs_output = gr.BarPlot(label="Emotion Probabilities (Audio File SER)", x="label", y="value",
                                              x_title="Emotion", y_title="Probability",
                                              vertical=False, min_width=300)

        ser_button.click(
            predict_ser_from_audio_upload,
            inputs=[audio_input],
            outputs=[ser_result_text, ser_probs_output]
        )
        
    with gr.Tab("Speech Emotion Recognition (SER) - Live Microphone"):
        with gr.Row():
            with gr.Column(scale=1):
                # type="numpy" gives us data as (sample_rate, data_array)
                # streaming=True enables continuous streaming
                microphone_input = gr.Audio(sources=["microphone"], type="numpy", label="Live Microphone Input", streaming=True, show_label=True)
            with gr.Column(scale=2):
                live_ser_result_text = gr.Textbox(label="Dominant Emotion (Live Microphone)", interactive=False)
                live_ser_probs_output = gr.BarPlot(label="Emotion Probabilities (Live Microphone SER)", x="label", y="value",
                                                   x_title="Emotion", y_title="Probability",
                                                   vertical=False, min_width=300)
        
        microphone_input.stream(
            predict_ser_from_live_microphone,
            inputs=[microphone_input],
            outputs=[live_ser_result_text, live_ser_probs_output]
        )
    
    gr.Markdown(
        """
        ---
        **Usage Notes:**
        - **FER (Image):** Upload an image (`.jpg`, `.png`) and click the predict button.
        - **FER (Live Camera):** After your browser grants camera access, emotion analysis from the live feed will start. (A physical camera must be connected and operational.)
        - **SER (Audio File):** Upload a `.wav` or `.mp3` audio file and click the predict button.
        - **SER (Live Microphone):** After your browser grants microphone access, you can start speaking to perform live emotion analysis.
        - Initial model loading might take some time. If you encounter errors, check the terminal logs and `logs/app.log` file.
        - This interface is designed for basic testing and demonstrations.
        """
    )

# --- Main Execution Block ---
if __name__ == "__main__":
    logger.info("Launching Gradio interface...")
    demo.launch(server_name="0.0.0.0")
    logger.info("Gradio GUI closed.")