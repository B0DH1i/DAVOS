import argparse
import sys
from .utils.logging_utils import setup_logger
from .utils import file_utils 
from .configs import main_config as config 
from .training import train_pipeline 


logger = setup_logger("MainTrainer", log_file=config.APPLICATION_LOG_FILE)

def main():
    parser = argparse.ArgumentParser(
        description="Script for Training Emotion Recognition Models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )

    # Which model to train
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=['fer', 'ser'],
        help="Type of model to train: 'fer' (Facial Emotion Recognition) or 'ser' (Speech Emotion Recognition)."
    )

    # Arguments specific to FER model
    parser.add_argument(
        "--fer_model_choice",
        type=str,
        default=config.DEFAULT_FER_MODEL_CHOICE,
        choices=[config.FER_MODEL_TYPE_VGG16_TRANSFER],
        help="FER architecture to use if model_type='fer'."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None, # If None, it's taken from config in train_pipeline
        help="Number of epochs for training. If not specified, the value from the config file is used."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for training. If not specified, the value from the config file is used."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate for the optimizer. If not specified, the value from the config file is used."
    )
    parser.add_argument(
        "--no_fer_augmentation",
        action="store_false",
        dest="fer_augmentation",
        help="Disables data augmentation for FER training."
    )
    parser.set_defaults(fer_augmentation=config.DATA_AUGMENTATION_FER_ENABLED)

    parser.add_argument(
        "--optimizer_fer",
        type=str,
        default=config.DEFAULT_OPTIMIZER_FER,
        choices=['adam', 'rmsprop', 'sgd'], # Must be compatible with options in train_pipeline
        help="Type of optimizer to use for the FER model. If not specified, the value from the config file is used."
    )

    parser.add_argument(
        "--no_ser_augmentation",
        action="store_false",
        dest="ser_augmentation",
        help="Disables data augmentation for SER training (if implemented)."
    )
    parser.set_defaults(ser_augmentation=config.USE_DATA_AUGMENTATION_SER)

    args = parser.parse_args()

    logger.info("Main Training Script Started.")
    logger.info(f"Command Line Arguments: {args}")

    try:
        file_utils.create_project_directories()
    except Exception as e:
        logger.error(f"Error creating/checking project directories: {e}")
        logger.error("Resolve this issue before continuing training.")
        return 1 # Exit with error code


    config.DATA_AUGMENTATION_FER_ENABLED = args.fer_augmentation
    config.DATA_AUGMENTATION_SER_ENABLED = args.ser_augmentation
    logger.info(f"FER Data Augmentation: {'Enabled' if config.DATA_AUGMENTATION_FER_ENABLED else 'Disabled'}")
    logger.info(f"SER Data Augmentation: {'Enabled' if config.DATA_AUGMENTATION_SER_ENABLED else 'Disabled'}")


    training_successful = False
    if args.model_type == 'fer':
        logger.info(f"Starting FER model training: {args.fer_model_choice}")
        training_successful = train_pipeline.train_fer_model(
            model_type=args.fer_model_choice,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            optimizer_type=args.optimizer_fer
        )
    elif args.model_type == 'ser':
        logger.info("Starting SER (CRNN) model training.")
        training_successful = train_pipeline.train_ser_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    else:
        logger.error(f"Invalid model_type: {args.model_type}")
        parser.print_help()
        return 1

    if training_successful:
        logger.info(f"{args.model_type.upper()} model training completed successfully (or at least finished without explicit errors).")
        return 0 # Success code
    else:
        logger.error(f"A problem occurred or {args.model_type.upper()} model training failed.")
        return 1 # Error code

if __name__ == '__main__':

    return_code = main()
    if return_code == 0:
        print("\nTraining process completed successfully.")
    else:
        print("\nErrors occurred during the training process. Check logs for details.")
    sys.exit(return_code)