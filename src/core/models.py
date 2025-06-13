import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense,
    BatchNormalization, Activation, Add, GlobalAveragePooling2D,
    LSTM, Bidirectional, TimeDistributed, Reshape,
    SeparableConv2D, DepthwiseConv2D, Permute, multiply
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16
import numpy as np # Sadece test için
import os # Added import

# Göreli importlar
from ..utils.logging_utils import setup_logger
from ..configs import main_config as config

# Setup logger for this module
logger = setup_logger(__name__, log_file=config.APPLICATION_LOG_FILE)

# --- Facial Emotion Recognition (FER) Models ---

def build_fer_model_vgg16_transfer(input_shape, num_classes, model_name="FER_VGG16_Transfer_Model"):
    """
    Creates a FER model using transfer learning based on the VGG16 model.
    Input shape must be (height, width, 3).
    """
    logger.info(f"Creating VGG16-based FER transfer learning model: {model_name}")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Number of classes: {num_classes}")

    # Load the VGG16 model with ImageNet weights, excluding the top classification layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers in the convolutional base of VGG16
    base_model.trainable = False
    logger.info(f"Layers of the VGG16 base model have been frozen (base_model.trainable = {base_model.trainable}).")

    # Add new classification layers
    x = base_model.output
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    
    # FC Layer 1
    x = Dense(1024, activation='relu', kernel_regularizer=l2(config.WEIGHT_DECAY_FER), name="fc1_relu")(x)
    x = BatchNormalization(name="fc1_bn")(x)
    x = Dropout(0.5, name="fc1_dropout")(x)
    
    # FC Layer 2
    x = Dense(512, activation='relu', kernel_regularizer=l2(config.WEIGHT_DECAY_FER), name="fc2_relu")(x)
    x = BatchNormalization(name="fc2_bn")(x)
    x = Dropout(0.5, name="fc2_dropout")(x) # Dropout rate increased from 0.3 to 0.5
    
    # FC Layer 3 (Output Layer)
    output_layer = Dense(num_classes, activation='softmax', name='output_emotion')(x)

    # Create the new model
    model = Model(inputs=base_model.input, outputs=output_layer, name=model_name)
    
    logger.info(f"{model_name} created successfully.")
    # model.summary() will be logged during training
    return model

# --- Speech Emotion Recognition (SER) Model ---

def build_ser_model_crnn(input_shape=None, num_classes=None):
    """
    A CRNN (Convolutional Recurrent Neural Network) model for RAVDESS (MFCC features).
    Input shape: (n_mfcc, time_frames)
    """
    if input_shape is None:
        input_shape = (config.AUDIO_N_MFCC, config.AUDIO_MAX_FRAMES_MFCC)
    if num_classes is None:
        num_classes = len(config.TARGET_EMOTIONS) # SER model is trained based on TARGET_EMOTIONS

    logger.info(f"Creating SER CRNN model. Input shape: {input_shape}, Number of classes: {num_classes}")

    mfcc_input = Input(shape=input_shape, name='ser_input_mfcc_crnn')
    
    # Add channel dimension (for Conv2D): (n_mfcc, time_frames) -> (n_mfcc, time_frames, 1)
    x = Reshape((input_shape[0], input_shape[1], 1), name='ser_crnn_reshape_to_4d')(mfcc_input)

    # Convolutional Blocks
    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='ser_crnn_conv1')(x)
    x = BatchNormalization(name='ser_crnn_bn1')(x)
    x = Activation('relu', name='ser_crnn_relu1')(x)
    x = MaxPooling2D((2, 2), name='ser_crnn_pool1')(x) # Reduces time and frequency dimension
    x = Dropout(0.4, name='ser_crnn_drop1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='ser_crnn_conv2')(x)
    x = BatchNormalization(name='ser_crnn_bn2')(x)
    x = Activation('relu', name='ser_crnn_relu2')(x)
    x = MaxPooling2D((2, 2), name='ser_crnn_pool2')(x)
    x = Dropout(0.4, name='ser_crnn_drop2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='ser_crnn_conv3')(x)
    x = BatchNormalization(name='ser_crnn_bn3')(x)
    x = Activation('relu', name='ser_crnn_relu3')(x)
    
    x = MaxPooling2D((2, 1), name='ser_crnn_pool3')(x) # (height, width) = (frequency, time)
    x = Dropout(0.4, name='ser_crnn_drop3')(x)
    
    
    s = x.shape # (None, H_conv, W_conv, C_conv)
    if s[1] is not None and s[2] is not None and s[3] is not None: # If static shape exists
        x = Reshape((s[2], s[1] * s[3]), name='ser_crnn_reshape_for_lstm')(x)
    else: 
        def dynamic_reshape_for_lstm(tensor_in):
            input_shape_dyn = tf.shape(tensor_in) # (batch, H, W, C)
            # Should be (batch, W, H * C)
            return tf.reshape(tensor_in, [input_shape_dyn[0], input_shape_dyn[2], input_shape_dyn[1] * input_shape_dyn[3]])
        x = tf.keras.layers.Lambda(dynamic_reshape_for_lstm, name='ser_crnn_lambda_reshape_lstm')(x)

    # Recurrent Blocks (LSTM)
    x = Bidirectional(LSTM(64, return_sequences=True, name='ser_crnn_bilstm1', kernel_regularizer=l2(config.SER_L2_REG_STRENGTH)), name='ser_crnn_bidir1')(x)
    x = Dropout(0.4, name='ser_crnn_drop_lstm1')(x)

    x = Bidirectional(LSTM(32, return_sequences=False, name='ser_crnn_bilstm2', kernel_regularizer=l2(config.SER_L2_REG_STRENGTH)), name='ser_crnn_bidir2')(x)
    x = Dropout(0.4, name='ser_crnn_drop_lstm2')(x)

    # Dense Layers
    x = Dense(128, activation='relu', kernel_initializer='he_normal', name='ser_crnn_fc1', kernel_regularizer=l2(config.SER_L2_REG_STRENGTH))(x)
    x = Dropout(0.4, name='ser_crnn_drop_fc1')(x)

    output = Dense(num_classes, activation='softmax', name='ser_crnn_output')(x)

    model = Model(inputs=mfcc_input, outputs=output, name='SER_CRNN_Model')
    logger.info("SER CRNN model created successfully.")
    return model

def build_ser_model_simple_dense_for_whisper(input_shape_dim, num_classes=config.NUM_TARGET_CLASSES, model_name="SER_SimpleDense_Whisper_Model"):
    """
    Creates a simple Dense layered SER model for Whisper embeddings.
    Input shape must be (embedding_dim,).
    """
    logger.info(f"Creating Simple Dense SER model (for Whisper): {model_name}")
    logger.info(f"Input dimensions (embedding_dim): {input_shape_dim}")
    logger.info(f"Number of classes: {num_classes}")

    whisper_embedding_input = Input(shape=(input_shape_dim,), name="whisper_input") # (embedding_dim,)

    # Model layers
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(whisper_embedding_input)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    output_layer = Dense(num_classes, activation='softmax', name='output_emotion')(x)

    model = Model(inputs=whisper_embedding_input, outputs=output_layer, name=model_name)
    
    logger.info(f"{model_name} created successfully.")
    
    return model


def create_ser_model(model_choice=None, input_shape=None, num_classes=None):
    """
    Creates and returns the selected SER model based on the configuration.
    """
    if model_choice is None:
        model_choice = config.DEFAULT_SER_MODEL_CHOICE
    if num_classes is None:
        num_classes = len(config.TARGET_EMOTIONS)

    logger.info(f"Creating SER Model. Choice: {model_choice}")

    
    
    # Check directly using the config constant
    if model_choice == config.SER_MODEL_TYPE_SIMPLE_DENSE_WHISPER:
        if input_shape is None: 
            input_shape_whisper = config.WHISPER_EMBEDDING_DIM
        elif isinstance(input_shape, (tuple, list)) and len(input_shape) == 1: 
            input_shape_whisper = input_shape[0]
        elif isinstance(input_shape, int):
            input_shape_whisper = input_shape
        else:
            logger.error(f"Unexpected input_shape format for '{config.SER_MODEL_TYPE_SIMPLE_DENSE_WHISPER}': {input_shape}. Expected (embedding_dim,) or int.")
            raise ValueError(f"Invalid input_shape for '{config.SER_MODEL_TYPE_SIMPLE_DENSE_WHISPER}': {input_shape}")
        logger.info(f"input_shape_dim for Simple Dense (Whisper): {input_shape_whisper}")
        return build_ser_model_simple_dense_for_whisper(input_shape_dim=input_shape_whisper, num_classes=num_classes)
    else:
        logger.error(f"Unknown or unsupported SER model choice: {model_choice}. Only '{config.SER_MODEL_TYPE_SIMPLE_DENSE_WHISPER}' is supported.")
        raise ValueError(f"Invalid SER model choice: {model_choice}")

# Factory pattern (dictionary) for model selection
MODEL_FACTORY = {
   
    config.FER_MODEL_TYPE_VGG16_TRANSFER: build_fer_model_vgg16_transfer,
    
    config.SER_MODEL_TYPE_SIMPLE_DENSE_WHISPER: build_ser_model_simple_dense_for_whisper,
}


if __name__ == '__main__':
   
    logger.info("--- Testing Model Definition Module (models.py) ---")

    
    log_dir_test = os.path.dirname(config.APPLICATION_LOG_FILE)
    if log_dir_test and not os.path.exists(log_dir_test):
        os.makedirs(log_dir_test, exist_ok=True)

   

    # Simple Dense Model Test for Whisper
    logger.info("\nSER Simple Dense Model (for Whisper) Test:")
    
    whisper_input_dim_test = config.WHISPER_EMBEDDING_DIM
    
    num_classes_target = config.NUM_TARGET_CLASSES
    ser_model_w = create_ser_model(input_shape=whisper_input_dim_test, num_classes=num_classes_target) 
    if ser_model_w: # Proceed if model is not None
        ser_model_w.summary(print_fn=logger.info)
        dummy_ser_input_whisper_np = np.random.rand(2, whisper_input_dim_test).astype(np.float32)
        try:
            pred_w = ser_model_w.predict(dummy_ser_input_whisper_np)
            logger.info(f"  SER Simple Dense (Whisper) sample prediction shape: {pred_w.shape}")
        except Exception as e:
            logger.error(f"  SER Simple Dense (Whisper) prediction error: {e}")
    else:
        logger.error("SER Simple Dense (Whisper) test model could not be created.")


    logger.info("--- Model Definition Tests Completed ---")
