import pygame
import os
import json
import time
import random
import numpy as np
from ..configs import main_config as config
from ..utils.logging_utils import setup_logger

_lazanov_logger_name = "LazanovAudioManager"
_lazanov_log_file_path = os.path.join(config.LOGS_PATH, f"{_lazanov_logger_name}.log")
logger = setup_logger(
    _lazanov_logger_name,
    level=config.LOGGING_LEVEL,
    log_file=_lazanov_log_file_path
)

class LazanovAudioManager:
    def __init__(self):
        self.music_library = []
        self.current_music_entry = None
        self.is_music_playing = False
        self.music_channel = None # For pygame.mixer.Channel object
        self.active_intervention_type = None # Stores which type of intervention is active

        self.is_binaural_playing = False
        self.binaural_sound = None # For the stereo Sound object
        self.binaural_channel = None # A separate channel for binaural beats

        logger.info("Initializing LazanovAudioManager...")

        try:
            pygame.mixer.init(frequency=config.AUDIO_SAMPLE_RATE, channels=2, buffer=2048)
            logger.info(f"Pygame mixer initialized successfully at {config.AUDIO_SAMPLE_RATE} Hz, stereo.")
            # Allocate separate channels for interventions
            self.music_channel = pygame.mixer.Channel(0) # Channel 0 for music
            if config.LAZANOV_BRAINWAVE_ENTRAINMENT_ENABLED and config.LAZANOV_BINAURAL_BEATS_ENABLED:
                if pygame.mixer.get_num_channels() > 1:
                    self.binaural_channel = pygame.mixer.Channel(1) # Channel 1 for binaural
                    logger.info("A separate mixer channel (1) has been allocated for binaural beats.")
                else:
                    logger.warning("Not enough mixer channels for binaural beats. Binaural will be disabled.")
                    config.LAZANOV_BINAURAL_BEATS_ENABLED = False
        except Exception as e:
            logger.error(f"Failed to initialize Pygame mixer: {e}")
            raise RuntimeError(f"Failed to initialize Pygame mixer: {e}")

        self.load_music_library()

        if not self.music_library:
            logger.warning("Music library is empty or could not be loaded.")

    def load_music_library(self):
        metadata_path = os.path.join(config.LAZANOV_MUSIC_LIBRARY_PATH, config.LAZANOV_MUSIC_METADATA_FILENAME)
        try:
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.music_library = json.load(f)
                logger.info(f"{len(self.music_library)} music tracks loaded from {metadata_path}.")
                for track in self.music_library:
                    track['filepath'] = os.path.join(config.LAZANOV_MUSIC_LIBRARY_PATH, track['filename'])
                    if not os.path.exists(track['filepath']):
                        logger.warning(f"Music file not found: {track['filepath']}. Removing this track from the library.")
                        self.music_library.remove(track)
            else:
                logger.warning(f"Music metadata file not found: {metadata_path}")
                self.music_library = []
        except Exception as e:
            logger.error(f"Error loading music library: {e}")
            self.music_library = []

    def _generate_stereo_sine_wave(self, freq_left, freq_right, duration_sec, sample_rate, amplitude=0.5):
        """Generates a stereo sine wave (numpy array) at specified frequencies."""
        num_samples = int(sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        
        wave_left = amplitude * np.sin(2 * np.pi * freq_left * t)
        wave_right = amplitude * np.sin(2 * np.pi * freq_right * t)
        
        # Convert to 16-bit integer format
        wave_left_int = (wave_left * 32767).astype(np.int16)
        wave_right_int = (wave_right * 32767).astype(np.int16)
        
        # Create stereo interleaved array (LRLRLR...)
        stereo_wave = np.empty((num_samples * 2,), dtype=np.int16)
        stereo_wave[0::2] = wave_left_int
        stereo_wave[1::2] = wave_right_int
        return stereo_wave

    def select_music_for_intervention(self, target_state):
        if not self.music_library:
            logger.warning("Cannot select music, library is empty.")
            return None

        suitable_tracks = [track for track in self.music_library if target_state in track.get("target_states", [])]
        
        if not suitable_tracks:
            logger.warning(f"No suitable music found for target state '{target_state}'.")
            suitable_tracks = [track for track in self.music_library if "calm" in track.get("primary_mood_tags", [])]
            if not suitable_tracks:
                logger.warning("No suitable music found for fallback either. Selecting a random track.")
                suitable_tracks = self.music_library 
        
        if suitable_tracks:
            selected_track = random.choice(suitable_tracks)
            logger.info(f"Music selected for '{target_state}': {selected_track['title']}")
            return selected_track
        return None

    def play_music(self, music_entry, volume=None, on_finish_callback=None):
        if not music_entry or not os.path.exists(music_entry['filepath']):
            logger.error(f"Music entry to be played is invalid or file does not exist: {music_entry}")
            return

        self.stop_music() # Stop any currently playing music first

        try:
            sound = pygame.mixer.Sound(music_entry['filepath'])
            self.current_music_entry = music_entry
            effective_volume = volume if volume is not None else config.LAZANOV_DEFAULT_MUSIC_VOLUME
            
            self.music_channel.set_volume(0) # Initial volume 0 for fade-in
            self.music_channel.play(sound, loops=-1 if music_entry.get("loop", False) else 0)
            self.is_music_playing = True
            logger.info(f"Playing music: {music_entry['title']} (Volume: {effective_volume})")

           
            fade_steps = 50 
            for i in range(fade_steps + 1):
                self.music_channel.set_volume(effective_volume * (i / fade_steps))
                time.sleep(config.LAZANOV_FADE_IN_OUT_DURATION_SECONDS / fade_steps)
            
            if on_finish_callback:
                
                pass 

        except Exception as e:
            logger.error(f"Error while playing music: {e} (File: {music_entry['filepath']})")
            self.is_music_playing = False

    def stop_music(self, fade_out_duration_seconds=None):
        if self.is_music_playing and self.music_channel.get_busy():
            logger.info(f"Stopping music: {self.current_music_entry['title'] if self.current_music_entry else 'Unknown'}")
            duration = fade_out_duration_seconds if fade_out_duration_seconds is not None else config.LAZANOV_FADE_IN_OUT_DURATION_SECONDS
            initial_volume = self.music_channel.get_volume()
            fade_steps = 50

            try:
                for i in range(fade_steps + 1):
                    self.music_channel.set_volume(initial_volume * (1 - (i / fade_steps)))
                    time.sleep(duration / fade_steps)
                self.music_channel.stop()
            except Exception as e:
                 logger.warning(f"Error during music stop (fade-out): {e}. Stopping directly.")
                 self.music_channel.stop()
            
        self.is_music_playing = False
        self.current_music_entry = None

    def start_binaural_beats(self, target_beat_hz, duration_sec=600, carrier_hz=None, volume=None):
        if not config.LAZANOV_BRAINWAVE_ENTRAINMENT_ENABLED or not config.LAZANOV_BINAURAL_BEATS_ENABLED or not self.binaural_channel:
            logger.debug("Binaural beats are disabled or no channel is available.")
            return

        self.stop_binaural_beats() # Stop any current beats first

        carrier_hz = carrier_hz if carrier_hz is not None else config.LAZANOV_BINAURAL_BEATS_CARRIER_FREQ_HZ
        effective_volume = volume if volume is not None else config.LAZANOV_BINAURAL_BEATS_VOLUME

        freq_left = carrier_hz - (target_beat_hz / 2.0)
        freq_right = carrier_hz + (target_beat_hz / 2.0)

        try:
            stereo_wave_data = self._generate_stereo_sine_wave(freq_left, freq_right, duration_sec, config.AUDIO_SAMPLE_RATE, amplitude=0.8)
            self.binaural_sound = pygame.sndarray.make_sound(stereo_wave_data)
            
            self.binaural_channel.set_volume(effective_volume) 
            self.binaural_channel.play(self.binaural_sound, loops=-1) # loops=-1 for continuous play
            self.is_binaural_playing = True
            logger.info(f"Binaural beats started with target {target_beat_hz} Hz (Left: {freq_left:.2f} Hz, Right: {freq_right:.2f} Hz, Volume: {effective_volume}).")
        except Exception as e:
            logger.error(f"Error starting binaural beats: {e}")
            self.is_binaural_playing = False

    def stop_binaural_beats(self):
        if self.is_binaural_playing and self.binaural_channel and self.binaural_channel.get_busy():
            self.binaural_channel.stop()
            logger.info("Binaural beats stopped.")
        if self.binaural_sound:
             self.binaural_sound = None # Clear reference
        self.is_binaural_playing = False

    def manage_intervention(self, primary_emotion, emotion_confidence):
        if not config.LAZANOV_INTERVENTION_ENABLED:
            # logger.debug("Lazanov interventions are globally disabled.")
            return

        intervention_details = config.LAZANOV_TRIGGER_EMOTIONS.get(primary_emotion)
        
        if intervention_details and emotion_confidence >= config.LAZANOV_TRIGGER_CONFIDENCE_THRESHOLD:
            target_state = intervention_details["target_state"]
            logger.info(f"Evaluating intervention status: Emotion='{primary_emotion}', Confidence={emotion_confidence:.2f}, Target State='{target_state}'")
            
            if self.active_intervention_type == target_state and (self.is_music_playing or self.is_binaural_playing):
                # logger.debug(f"An intervention is already active for '{target_state}'.")
                return

           

            selected_music = self.select_music_for_intervention(target_state)
            if selected_music:
                self.play_music(selected_music)
                self.active_intervention_type = target_state # Set intervention type for music
            # Binaural beats update active_intervention_type only if music is not playing or triggered separately

            if config.LAZANOV_BRAINWAVE_ENTRAINMENT_ENABLED and config.LAZANOV_BINAURAL_BEATS_ENABLED:
                binaural_setting = config.LAZANOV_BINAURAL_BEATS_SETTINGS.get(target_state)
                
                if binaural_setting:
                    beat_hz = binaural_setting.get("beat_hz")
                    carrier_hz = binaural_setting.get("carrier_hz")
                    duration = binaural_setting.get("default_duration_seconds", 600) # Default duration
                    volume = binaural_setting.get("default_volume", 0.3) # Default volume level
                    
                    if beat_hz is not None: # Proceed if beat_hz is defined
                        logger.info(f"Binaural beat settings found for '{target_state}': Beat={beat_hz}Hz, Carrier={carrier_hz}Hz, Duration={duration}s, Volume={volume}")
                        self.start_binaural_beats(target_beat_hz=beat_hz, 
                                                  duration_sec=duration, 
                                                  carrier_hz=carrier_hz, 
                                                  volume=volume)
                        if not self.is_music_playing: # If music is not playing, set the intervention type for binaural
                            self.active_intervention_type = target_state
                    else:
                        logger.warning(f"'beat_hz' not found in binaural settings for '{target_state}'. Not starting binaural beats.")
                        self.stop_binaural_beats() # If no appropriate setting, stop any that are playing
                else:
                    logger.info(f"No specific binaural beat setting found for '{target_state}'. Stopping any current binaural beats.")
                    self.stop_binaural_beats() # If no suitable binaural for the target state, stop any that are playing.
        
        elif self.is_music_playing or self.is_binaural_playing: # If no trigger emotion or confidence is low, and something is playing, stop it
            logger.info("Triggering emotion state no longer present or confidence is below threshold. Stopping active interventions.")
            self.stop_music()
            self.stop_binaural_beats()
            self.active_intervention_type = None

    def shutdown(self):
        logger.info("Shutting down LazanovAudioManager...")
        self.stop_music()
        self.stop_binaural_beats()
        pygame.mixer.quit()
        logger.info("LazanovAudioManager shut down successfully.")