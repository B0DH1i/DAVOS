import argparse
import os
import sys

from .utils.logging_utils import setup_logger
from .utils import file_utils
from .configs import main_config as config
from .core.predictor_engine import EmotionPredictorEngine

logger = setup_logger("MainPredictor", log_file=config.APPLICATION_LOG_FILE)

def main():
    """
    Main function to perform emotion prediction from a single image or audio file
    using trained models.
    """
    parser = argparse.ArgumentParser(
        description="Script to perform emotion prediction from a single file using trained models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=['image', 'audio'],
        help="Type of the file to predict: 'image' or 'audio'."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Full path to the input image or audio file."
    )
    parser.add_argument(
        "--fer_model_base",
        type=str,
        default=config.DEFAULT_FER_MODEL_BASE_NAME,
        help="Base name for the FER model to use (e.g., 'fer_mini_xception'). "
             "The latest trained one will be automatically selected."
    )
    parser.add_argument(
        "--ser_model_base",
        type=str,
        default=config.DEFAULT_SER_MODEL_BASE_NAME,
        help="Base name for the SER model to use (e.g., 'ser_crnn'). "
             "The latest trained one will be automatically selected."
    )

    args = parser.parse_args()

    logger.info("Single File Prediction Script Started.")
    logger.info(f"Command Line Arguments: {args}")

    try:
        file_utils.create_project_directories()
    except Exception as e:
        logger.error(f"Error while creating/checking project directories: {e}")
        return 1

    try:
        predictor = EmotionPredictorEngine(
            fer_model_base_name=args.fer_model_base if args.type == 'image' else None,
            ser_model_base_name=args.ser_model_base if args.type == 'audio' else None
        )
    except Exception as e:
        logger.error(f"Error initializing EmotionPredictorEngine: {e}", exc_info=True)
        return 1

    if not os.path.exists(args.input_path):
        logger.error(f"Input file not found: {args.input_path}")
        return 1

    predicted_label = None
    probabilities_dict = None
    success = False

    if args.type == 'image':
        if predictor.fer_model is None:
            logger.error(f"FER model ({args.fer_model_base}) could not be loaded or was not specified. Image prediction cannot be performed.")
            return 1
        logger.info(f"Performing emotion prediction from image file ({args.input_path})...")
        predicted_label, probabilities_dict = predictor.predict_from_image_file(args.input_path)
        if predicted_label and probabilities_dict is not None:
            success = True

    elif args.type == 'audio':
        if predictor.ser_model is None:
            logger.error(f"SER model ({args.ser_model_base}) could not be loaded or was not specified. Audio prediction cannot be performed.")
            return 1
        logger.info(f"Performing emotion prediction from audio file ({args.input_path})...")
        predicted_label, probabilities_dict = predictor.predict_from_audio_file(args.input_path)
        if predicted_label and probabilities_dict is not None:
            success = True

    else:
        logger.error(f"Invalid prediction type: {args.type}")
        parser.print_help()
        return 1

    if success and predicted_label:
        print(f"\n--- PREDICTION RESULT ({args.type.upper()}) ---")
        print(f"    Input File: {os.path.abspath(args.input_path)}")
        print(f"    Predicted Emotion: {predicted_label}")
        if probabilities_dict:
            print("    Probabilities:")
            for emotion, prob in sorted(probabilities_dict.items(), key=lambda item: item[1], reverse=True):
                if prob > 0.001:
                    print(f"      - {emotion:<10}: {prob:.4f}")
        logger.info(f"Prediction completed successfully. Result: {predicted_label}")
        return 0
    else:
        print(f"\n--- PREDICTION FAILED ({args.type.upper()}) ---")
        print(f"    Input File: {os.path.abspath(args.input_path)}")
        if predicted_label in ["ErrorFER", "ErrorSER", "Preprocessing Error"]:
            logger.error(f"An error occurred during emotion prediction for {args.type.upper()} type: {predicted_label}. Check logs for details.")
        else:
            logger.error(f"Emotion prediction could not be performed for {args.type.upper()} type or model returned no valid result. Check logs for details.")
        return 1

if __name__ == '__main__':
    return_code = main()
    sys.exit(return_code)