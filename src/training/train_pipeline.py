import os
import datetime
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.utils import class_weight

# Relative imports
from ..utils.logging_utils import setup_logger
from ..utils import file_utils # Model saving, directory creation
from ..utils import plot_utils # Plotting history
from ..configs import main_config as config
from ..core import data_loader # Data loader
from ..core import models      # Model definitions

# Preprocessing function for VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input

# Set up logger for this module
logger = setup_logger(__name__, log_file=config.APPLICATION_LOG_FILE)

def train_fer_model(model_type=config.FER_MODEL_TYPE_VGG16_TRANSFER,
                    epochs=None,  # Set default to None, get from config below
                    batch_size=None, # Set default to None, get from config below
                    optimizer_type=config.DEFAULT_OPTIMIZER_FER,
                    learning_rate=None, # Set default to None, get from config below
                    patience_early_stopping=config.PATIENCE_EARLY_STOPPING_FER,
                    patience_reduce_lr=config.PATIENCE_REDUCE_LR_FER,
                    monitor_metric=config.MONITOR_METRIC_FER,
                    save_best_only=config.SAVE_BEST_ONLY_FER,
                    data_augmentation=config.USE_DATA_AUGMENTATION_FER,
                    base_save_name_prefix=config.FER_MODEL_NAME_PREFIX):
    """
    Trains the Facial Expression Recognition (FER) model.
    """
    try:

        # If parameters are None, get default values from config
        if epochs is None:
            epochs = config.DEFAULT_EPOCHS_FER
        if batch_size is None:
            batch_size = config.DEFAULT_BATCH_SIZE_FER
        if learning_rate is None:
            learning_rate = config.DEFAULT_LEARNING_RATE_FER

        logger.info(f"--- Starting FER Model Training (Model Type: {model_type}) ---")
        logger.info(f"Parameters: Epochs={epochs}, Batch={batch_size}, LR={learning_rate}, Optimizer={optimizer_type}")

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.load_fer_data()
        if X_train is None or X_train.size == 0:
            logger.error("FER training data could not be loaded or is empty. Training stopped.")
            return None, None

        input_shape = config.INPUT_SHAPE_FER
        num_classes = config.NUM_MODEL_OUTPUT_CLASSES

        if model_type == config.FER_MODEL_TYPE_VGG16_TRANSFER:
            logger.info("Performing data preprocessing for VGG16 transfer learning model...")
            input_shape = (config.INPUT_SHAPE_FER[0], config.INPUT_SHAPE_FER[1], 3) # (48, 48, 3)

            logger.info(f"X_train shape (before): {X_train.shape}, data range (min-max): {np.min(X_train)}-{np.max(X_train)}")

            X_train_rgb = (X_train * 255.0).astype(np.uint8)
            X_val_rgb = (X_val * 255.0).astype(np.uint8)
            X_test_rgb = (X_test * 255.0).astype(np.uint8)

            X_train_rgb = np.repeat(X_train_rgb, 3, axis=-1)
            X_val_rgb = np.repeat(X_val_rgb, 3, axis=-1)
            X_test_rgb = np.repeat(X_test_rgb, 3, axis=-1)
            logger.info(f"X_train shape (after replicating to 3 channels): {X_train_rgb.shape}")

            X_train = vgg16_preprocess_input(X_train_rgb)
            X_val = vgg16_preprocess_input(X_val_rgb)
            X_test = vgg16_preprocess_input(X_test_rgb)
            logger.info(f"VGG16 preprocessing completed. X_train shape (after): {X_train.shape}, data range (min-max): {np.min(X_train)}-{np.max(X_train)}")

        # Get MODEL_FACTORY from src.core.models
        if model_type not in models.MODEL_FACTORY:
            logger.error(f"Invalid FER model type: {model_type}. Not found in model factory. Available: {list(models.MODEL_FACTORY.keys())}")
            return None, None

        # Get the model creation function from the factory
        model_builder_func = models.MODEL_FACTORY[model_type]
        model = model_builder_func(input_shape=input_shape, num_classes=num_classes)

        model_save_prefix = f"{base_save_name_prefix}_{model_type}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # 3. Compile the Model
        if optimizer_type.lower() == "adam":
            optimizer = Adam(learning_rate=learning_rate)
            logger.info(f"Using Adam optimizer. Learning Rate: {learning_rate}")
        elif optimizer_type.lower() == "rmsprop":
            optimizer = RMSprop(learning_rate=learning_rate)
            logger.info(f"Using RMSprop optimizer. Learning Rate: {learning_rate}")
        elif optimizer_type.lower() == "sgd": # Newly added
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
            logger.info(f"Using SGD optimizer (with Nesterov momentum). Learning Rate: {learning_rate}, Momentum: 0.9")
        else:
            logger.warning(f"Unknown optimizer type: {optimizer_type}. Defaulting to SGD (Nesterov).") # Updated default to SGD
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        logger.info(f"FER Model ({model_type}) Compiled. Summary:")
        model.summary(print_fn=logger.info)

        # 4. Callbacks
        checkpoint_save_dir = os.path.join(config.TRAINED_MODELS_PATH, model_save_prefix)
        os.makedirs(checkpoint_save_dir, exist_ok=True)

        checkpoint_filepath = os.path.join(checkpoint_save_dir, "best_model_checkpoint.h5")
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor=monitor_metric,
            save_best_only=save_best_only,
            save_weights_only=False,
            verbose=1
        )
        early_stopping = EarlyStopping(monitor=config.MONITOR_METRIC_FER, patience=patience_early_stopping, verbose=1, restore_best_weights=True) # monitor updated
        reduce_lr = ReduceLROnPlateau(monitor=config.MONITOR_METRIC_FER, factor=config.FACTOR_REDUCE_LR_FER, patience=patience_reduce_lr, min_lr=1e-6, verbose=1) # monitor and factor updated

        tensorboard_log_dir = os.path.join(config.LOGS_PATH, "fer", model_save_prefix)
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
        callbacks = [model_checkpoint, early_stopping, reduce_lr, tensorboard_callback]

        # 5. Data Augmentation
        history = None
        if data_augmentation:
            logger.info("Using data augmentation for FER.")

            datagen_common = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

            if model_type == config.FER_MODEL_TYPE_VGG16_TRANSFER:

                logger.info("Data augmentation for VGG16 (without rescaling, pre-processed data).")
                train_generator = datagen_common.flow(X_train, y_train, batch_size=batch_size)
            else:

                logger.info("Data augmentation for standard FER model (without rescaling, data in [0,1] range).")
                train_generator = datagen_common.flow(X_train, y_train, batch_size=batch_size)


            calculated_steps_per_epoch = max(1, len(X_train) // batch_size)
            logger.info(f"FER Data Augmentation - X_train length: {len(X_train)}, batch_size: {batch_size}, calculated_steps_per_epoch: {calculated_steps_per_epoch}")

            history = model.fit(
                train_generator,
                steps_per_epoch=calculated_steps_per_epoch, # Prevent error if len(X_train) is 0
                epochs=epochs,
                validation_data=(X_val, y_val), # X_val, y_val should be provided as a tuple
                callbacks=callbacks,
                verbose=1
            )
        else:
            logger.info("Not using data augmentation for FER.")

            logger.info(f"FER No Data Augmentation - X_train length: {len(X_train)}, batch_size: {batch_size}")

            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val), # X_val, y_val should be provided as a tuple
                callbacks=callbacks,
                verbose=1
            )

        logger.info(f"FER Model ({model_type}) Training Completed.")


        if history:
            file_utils.save_model_and_history(model, history.history, model_save_prefix)
            # 7. Plot Training History (using plot_utils)
            logger.info(f"Creating training plot (FER): {model_save_prefix}")
            plot_utils.plot_training_history(history.history, model_save_prefix)
        else:
            logger.error("Training history could not be created. Model and plot cannot be saved.")
            return None, None

        # 8. Evaluation on Test Set
        if X_test.size > 0 and y_test.size > 0:
            logger.info("Evaluating FER on test set...")
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            logger.info(f"FER Test Set Results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
        else:
            logger.warning("FER test data is empty or could not be loaded. Skipping evaluation.")

        logger.info(f"--- FER Model Training Finished (Save prefix: {model_save_prefix}) ---")
        return model, history
    except Exception as e:
        logger.error(f"An error occurred during FER model training: {e}", exc_info=True)
        return None, None


def train_ser_model(epochs=None, batch_size=None, learning_rate=None,
                    optimizer_type=config.DEFAULT_OPTIMIZER_SER, # Optimizer type added
                    patience_early_stopping=None,
                    patience_reduce_lr=None,
                    monitor_metric=None,
                    save_best_only=config.SAVE_BEST_ONLY_SER, # save_best_only added
                    data_augmentation=config.USE_DATA_AUGMENTATION_SER, # USE_DATA_AUGMENTATION_SER used
                    base_save_name_prefix=config.SER_MODEL_NAME_PREFIX): # base_save_name_prefix added
    """Trains the Speech Emotion Recognition (SER) model."""
    # If parameters are None, get default values from config
    epochs = epochs if epochs is not None else config.DEFAULT_EPOCHS_SER
    batch_size = batch_size if batch_size is not None else config.DEFAULT_BATCH_SIZE_SER
    learning_rate = learning_rate if learning_rate is not None else config.DEFAULT_LEARNING_RATE_SER
    patience_early_stopping = patience_early_stopping if patience_early_stopping is not None else config.PATIENCE_EARLY_STOPPING_SER
    patience_reduce_lr = patience_reduce_lr if patience_reduce_lr is not None else config.PATIENCE_REDUCE_LR_SER
    monitor_metric = monitor_metric if monitor_metric is not None else config.MONITOR_METRIC_SER

    model_choice = config.DEFAULT_SER_MODEL_CHOICE # model_choice_from_config -> model_choice, SER_MODEL_CHOICE -> DEFAULT_SER_MODEL_CHOICE
    feature_type = config.SER_FEATURE_TYPE # feature_type_from_config -> feature_type

    logger.info(f"--- Starting SER Model Training (Model: {model_choice}, Feature: {feature_type}) ---")
    logger.info(f"Parameters: Epochs={epochs}, Batch={batch_size}, LR={learning_rate}, Optimizer={optimizer_type}")
    logger.info(f"Callback Settings: EarlyStopPatience={patience_early_stopping}, ReduceLRPatience={patience_reduce_lr}, Monitor={monitor_metric}")

    logger.info("Loading RAVDESS data...")
    ravdess_data = data_loader.load_ravdess_data(use_augmentation_on_train=data_augmentation)
    if ravdess_data is None:
        logger.error("RAVDESS data loading failed. Stopping SER training.")
        return None, None

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = ravdess_data

    if X_train is None or X_train.size == 0:
        logger.error("SER training data could not be loaded or is empty. Training stopped.")
        return None, None


    input_shape_ser = X_train.shape[1:]
    num_classes = config.NUM_MODEL_OUTPUT_CLASSES # NUM_TARGET_CLASSES -> NUM_MODEL_OUTPUT_CLASSES
    logger.info(f"Input shape for SER: {input_shape_ser}, Number of classes: {num_classes}")

    if y_train is not None and y_train.ndim > 1 and y_train.shape[0] > 0:
        y_train_indices = np.argmax(y_train, axis=1)
        class_weights_array = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_indices),
            y=y_train_indices
        )
        class_weights = {i : class_weights_array[i] for i in range(len(class_weights_array))}
        logger.info(f"Class weights calculated for SER: {class_weights}")
    else:
        logger.warning("SER y_train data is not in the correct format or is empty. Class weights could not be calculated.")
        class_weights = None

    if model_choice not in models.MODEL_FACTORY:
        logger.error(f"Invalid SER model type: {model_choice}. Not found in model factory. Available: {list(models.MODEL_FACTORY.keys())}")
        return None, None

    model_builder_func = models.MODEL_FACTORY[model_choice]
    if isinstance(input_shape_ser, tuple) and len(input_shape_ser) == 1:
        current_input_shape_dim = input_shape_ser[0]
    elif isinstance(input_shape_ser, int):
        current_input_shape_dim = input_shape_ser
    else:
        logger.error(f"Unexpected input_shape_ser format for SER model: {input_shape_ser}. Expected (embedding_dim,) tuple.")
        return False

    model = model_builder_func(input_shape_dim=current_input_shape_dim, num_classes=num_classes)

    model_save_prefix = f"{base_save_name_prefix}_{model_choice}_{feature_type}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # 3. Compile the Model
    # Optimizer selection (Adam, RMSprop, SGD)
    if optimizer_type.lower() == "adam":
        optimizer = Adam(learning_rate=learning_rate)
        logger.info(f"Using Adam optimizer. Learning Rate: {learning_rate}")
    elif optimizer_type.lower() == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
        logger.info(f"Using RMSprop optimizer. Learning Rate: {learning_rate}")
    elif optimizer_type.lower() == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        logger.info(f"Using SGD optimizer (with Nesterov momentum). Learning Rate: {learning_rate}")
    else:
        logger.warning(f"Unknown optimizer type: {optimizer_type}. Defaulting to Adam.")
        optimizer = Adam(learning_rate=learning_rate) # Updated default to Adam

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    logger.info(f"SER Model ({model_choice}) Compiled. Summary:")
    model.summary(print_fn=logger.info)

    # 4. Callbacks
    checkpoint_save_dir = os.path.join(config.TRAINED_MODELS_PATH, model_save_prefix)
    os.makedirs(checkpoint_save_dir, exist_ok=True)
    checkpoint_filepath = os.path.join(checkpoint_save_dir, "best_model_checkpoint.h5")

    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor=monitor_metric,
        save_best_only=save_best_only,
        save_weights_only=False,
        verbose=1
    )
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=patience_early_stopping, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor=monitor_metric, factor=config.FACTOR_REDUCE_LR_SER, patience=patience_reduce_lr, min_lr=1e-6, verbose=1)

    tensorboard_log_dir = os.path.join(config.LOGS_PATH, "ser", model_save_prefix)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
    callbacks = [model_checkpoint, early_stopping, reduce_lr, tensorboard_callback]

    # 5. Train the Model
    logger.info(f"Starting SER Model Training - X_train length: {len(X_train)}, batch_size: {batch_size}")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights, # Class weights added
        verbose=1
    )
    logger.info(f"SER Model Training Completed.")

    if history and hasattr(history, 'history') and isinstance(history.history, dict) and history.history:
        logger.info(f"Saving training history. Available metrics: {list(history.history.keys())}")
        save_success = file_utils.save_model_and_history(model, history.history, model_save_prefix)
        if save_success:
            logger.info(f"Model and training history successfully saved as '{model_save_prefix}'.")
        else:
            logger.warning(f"Model or training history could not be fully saved for '{model_save_prefix}'. See logs for details.")

        logger.info(f"Creating training plot: {model_save_prefix}")
        plot_utils.plot_training_history(history.history, model_save_prefix)
    else:
        logger.error("Training history (history.history) was not properly created, is empty, or model.fit returned None for SER. History and plot cannot be saved.")

    # 8. Evaluation on Test Set
    if X_test.size > 0 and y_test.size > 0:
        logger.info("Evaluating SER on test set...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"SER Test Set Results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
    else:
        logger.warning("SER test data is empty or could not be loaded. Skipping evaluation.")

    logger.info(f"--- SER Model Training Finished (Save prefix: {model_save_prefix}) ---")
    return True # Success


def evaluate_specific_ser_model_on_test_set(model_path: str):

    logger.info(f"--- Evaluating Specified SER Model on Test Set ---")
    logger.info(f"Model path: {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return

    # 1. Load the Model
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model successfully loaded: {model_path}")
        model.summary(print_fn=logger.info)
    except Exception as e:
        logger.error(f"Error loading model ({model_path}): {e}", exc_info=True)
        return

    # 2. Load Test Data (without augmentation)
    logger.info("Loading RAVDESS test data (without augmentation)...")
    ravdess_data = data_loader.load_ravdess_data(use_augmentation_on_train=False) # Augmentation is not used during testing
    if ravdess_data is None:
        logger.error("RAVDESS test data could not be loaded. Stopping evaluation.")
        return
    (_, _), (_, _), (X_test, y_test) = ravdess_data # Get only the test set

    if X_test.size == 0 or y_test.size == 0:
        logger.error("RAVDESS test data is empty. Stopping evaluation.")
        return

    # Reshape test data to 4D format required for CRNN
    if X_test.ndim == 3:
        X_test = np.expand_dims(X_test, axis=-1)
        logger.info(f"X_test reshaped for evaluation: {X_test.shape}")

    if X_test.ndim != 4 or (X_test.ndim == 4 and X_test.shape[3] != 1):
        logger.error(f"Test data (X_test) is not in the expected 4D format. Current shape: {X_test.shape}.")
        return


    if not hasattr(model, 'optimizer') or model.optimizer is None:
        logger.warning("The loaded model appears not to be compiled. Compiling with default Adam optimizer and categorical_crossentropy.")
        optimizer = Adam(learning_rate=config.SER_LEARNING_RATE) # Can be taken from config
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        logger.info(f"Model already compiled. Optimizer: {type(model.optimizer).__name__}, Loss: {model.loss}")


    # 4. Evaluation on Test Set
    logger.info("Evaluating specified SER model on test set...")
    try:
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1) # verbose=1 to see progress
        logger.info(f"--- Evaluation Results ({os.path.basename(model_path)}) ---")
        logger.info(f"   Test Loss      : {test_loss:.4f}")
        logger.info(f"   Test Accuracy  : {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    except Exception as e:
        logger.error(f"An error occurred while evaluating the model: {e}", exc_info=True)

    logger.info(f"--- SER Model Evaluation Completed ({model_path}) ---")


if __name__ == '__main__':
    logger.warning("This script (train_pipeline.py) is intended to be called via "
                     "main_trainer.py, not directly.")
    logger.info("Nevertheless, starting a FER training for testing purposes (with default settings)..")

    try:
        file_utils.create_project_directories()
    except Exception as e:
        logger.error(f"Could not create project directories for testing: {e}")
        logger.error("Please try running `python -m src.utils.file_utils` first to test.")