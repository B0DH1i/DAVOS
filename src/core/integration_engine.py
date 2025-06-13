import numpy as np
import collections

# Relative imports
from ..utils.logging_utils import setup_logger
from ..configs import main_config as config

# Setup logger for this module
logger = setup_logger(__name__, log_file=config.APPLICATION_LOG_FILE)


PROBABILITY_HISTORY = collections.deque(maxlen=config.TEMPORAL_SMOOTHING_WINDOW_SIZE)


def _map_source_probs_to_target_vector(source_probs_dict, 
                                       source_emotion_list_config_key, # e.g., "FER_EMOTIONS"
                                       target_emotion_list_config_key, # e.g., "TARGET_EMOTIONS"
                                       source_to_target_map_config_key): # e.g., "FER_TO_TARGET_MAP"
   
    if source_probs_dict is None:
        return None

    # Get the relevant lists and map from config
    try:
        source_emotion_list = getattr(config, source_emotion_list_config_key)
        target_emotion_list = getattr(config, target_emotion_list_config_key)
        source_to_target_map = getattr(config, source_to_target_map_config_key)
    except AttributeError as e:
        logger.error(f"Expected key not found in config: {e}. Cannot perform probability mapping.")
        return None

    num_target_classes = len(target_emotion_list)
    target_probs_vector = np.zeros(num_target_classes)
    target_emotion_to_idx = {emotion: i for i, emotion in enumerate(target_emotion_list)}

    for source_emotion_name, source_prob in source_probs_dict.items():
        # Convert the source emotion to the target emotion
        target_emotion_name = source_to_target_map.get(source_emotion_name)
        
        if target_emotion_name and target_emotion_name in target_emotion_to_idx:
            target_idx = target_emotion_to_idx[target_emotion_name]
            target_probs_vector[target_idx] += source_prob
        
            
    return target_probs_vector


def integrate_emotion_probabilities(face_emotion_probs_dict=None, speech_emotion_probs_dict=None,
                                    strategy=None, weight_fer=None, weight_ser=None,
                                    apply_temporal_smoothing=None, temporal_smoothing_strategy=None):
    """
    Combines emotion probabilities from face and voice to determine a final emotional state.
    Can also apply temporal smoothing.
    Gets default values from config if parameters are None.
    """
   
    # Default values for parameters from config
    strategy = strategy if strategy is not None else config.INTEGRATION_STRATEGY
    weight_fer = weight_fer if weight_fer is not None else config.INTEGRATION_WEIGHT_FACE
    weight_ser = weight_ser if weight_ser is not None else config.INTEGRATION_WEIGHT_SPEECH
    apply_temporal_smoothing = apply_temporal_smoothing if apply_temporal_smoothing is not None else config.TEMPORAL_SMOOTHING_ENABLED
    temporal_smoothing_strategy = temporal_smoothing_strategy if temporal_smoothing_strategy is not None else config.TEMPORAL_SMOOTHING_TYPE

    target_emotion_list = config.TARGET_EMOTIONS
    num_target_classes = len(target_emotion_list)

    target_probs_fer_vector = _map_source_probs_to_target_vector(
        face_emotion_probs_dict,
        "FER_EMOTIONS",
        "TARGET_EMOTIONS",
        "FER_TO_TARGET_MAP"
    )
    
    target_probs_ser_vector = _map_source_probs_to_target_vector(
        speech_emotion_probs_dict,
        "TARGET_EMOTIONS_ORDERED", # Source label list (should be in config)
        "TARGET_EMOTIONS",         # Target label list (should be in config)
        "SER_OUTPUT_TO_TARGET_MAP" # This map TARGET_EMOTIONS_ORDERED -> TARGET_EMOTIONS (self-mapping) (should be in config)
    )

    # Case 1: Both modalities are None
    if target_probs_fer_vector is None and target_probs_ser_vector is None:
        if "unknown" in target_emotion_list:
            unknown_idx = target_emotion_list.index("unknown")
            final_probs_vector = np.zeros(num_target_classes, dtype=np.float32)
            final_probs_vector[unknown_idx] = 1.0
            # logger.debug("No valid probabilities from any modality. Returning 'unknown'.")
            return target_emotion_list[unknown_idx], final_probs_vector
        else:
            logger.error("'unknown' not in TARGET_EMOTIONS and no input probabilities. Returning first target emotion or 'Error'.")
            # Return the first emotion or "Error" if target_emotion_list is empty
            return target_emotion_list[0] if target_emotion_list else "Error", np.zeros(num_target_classes, dtype=np.float32)

    # Case 2: Only one modality has data
    elif target_probs_fer_vector is not None and target_probs_ser_vector is None:
        current_combined_probs_vector = target_probs_fer_vector
        # logger.debug("Only FER probabilities available.")
    elif target_probs_fer_vector is None and target_probs_ser_vector is not None:
        current_combined_probs_vector = target_probs_ser_vector
        # logger.debug("Only SER probabilities available.")
    # Case 3: Both modalities have data - apply strategy
    else:
        # logger.debug(f"Integrating with strategy: {strategy}")
        if strategy == "weighted_average":
            current_combined_probs_vector = (weight_fer * target_probs_fer_vector) + \
                                          (weight_ser * target_probs_ser_vector)
        elif strategy == "highest_confidence":
            max_conf_fer = np.max(target_probs_fer_vector)
            max_conf_ser = np.max(target_probs_ser_vector)
            if max_conf_fer >= max_conf_ser:
                current_combined_probs_vector = target_probs_fer_vector
            else:
                current_combined_probs_vector = target_probs_ser_vector
        else:
            logger.error(f"Unknown integration strategy: {strategy}. Defaulting to FER if available, else SER, else unknown.")
            # Fallback: prefer FER, then SER, then unknown
            if target_probs_fer_vector is not None:
                current_combined_probs_vector = target_probs_fer_vector
            elif target_probs_ser_vector is not None:
                current_combined_probs_vector = target_probs_ser_vector
            else: # Should not be reached if the first None-None check is correct
                if "unknown" in target_emotion_list:
                    unknown_idx = target_emotion_list.index("unknown")
                    current_combined_probs_vector = np.zeros(num_target_classes, dtype=np.float32)
                    current_combined_probs_vector[unknown_idx] = 1.0
                else:
                    current_combined_probs_vector = np.zeros(num_target_classes, dtype=np.float32)
                    current_combined_probs_vector[0] = 1.0 # Default to first emotion

    # Normalize combined probabilities
    sum_probs = np.sum(current_combined_probs_vector)
    if sum_probs > 1e-6:
        current_combined_probs_vector = current_combined_probs_vector / sum_probs
    else:
        # logger.warning("Sum of combined probabilities is near zero. Defaulting to 'unknown' or first emotion.")
        current_combined_probs_vector = np.zeros(num_target_classes, dtype=np.float32)
        if "unknown" in target_emotion_list:
            current_combined_probs_vector[target_emotion_list.index("unknown")] = 1.0
        else:
            current_combined_probs_vector[0] = 1.0

    # Apply temporal smoothing
    final_probs_vector = current_combined_probs_vector
    if apply_temporal_smoothing:
        PROBABILITY_HISTORY.append(current_combined_probs_vector) # Add current unsmoothed to history
        if len(PROBABILITY_HISTORY) > 0 : # Check if history is not empty
            if temporal_smoothing_strategy == "moving_average":
                try:
                    history_array = np.array(list(PROBABILITY_HISTORY))
                    if history_array.ndim == 2 and history_array.shape[0] > 0 : # Ensure not empty and 2D
                        final_probs_vector = np.mean(history_array, axis=0)
                    elif history_array.ndim == 1: # Single item in history
                         final_probs_vector = history_array
                    # Normalize again after smoothing
                    sum_smoothed_probs = np.sum(final_probs_vector)
                    if sum_smoothed_probs > 1e-6:
                        final_probs_vector = final_probs_vector / sum_smoothed_probs
                    else:
                        # Default to unknown or first emotion if sum is still zero
                        final_probs_vector = np.zeros(num_target_classes, dtype=np.float32)
                        if "unknown" in target_emotion_list:
                            final_probs_vector[target_emotion_list.index("unknown")] = 1.0
                        else:
                            final_probs_vector[0] = 1.0
                except Exception as e:
                    logger.warning(f"Error during moving average temporal smoothing: {e}. Using unsmoothed probabilities.")
                    # final_probs_vector remains current_combined_probs_vector
            else:
                logger.warning(f"Unknown temporal smoothing strategy: {temporal_smoothing_strategy}. Using unsmoothed probabilities.")
        # else: if PROBABILITY_HISTORY is empty, no smoothing can be applied yet.

    # Determine final label
    predicted_label_index = np.argmax(final_probs_vector)
    predicted_label = target_emotion_list[predicted_label_index]

    # logger.debug(f"Final integrated emotion: {predicted_label}, Probs: {final_probs_vector}")
    return predicted_label, final_probs_vector
