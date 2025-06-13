import cv2
import pyaudio
import numpy as np
import time
import os

# Relative imports
from ..utils.logging_utils import setup_logger
from ..configs import main_config as config

# Setup logger for this module
logger = setup_logger(__name__, log_file=config.APPLICATION_LOG_FILE)

class InputController:
    """
    Manages camera and microphone inputs.
    """
    def __init__(self, video_file_path=None, enable_audio=True, camera_id=None):
        self.cap = None
        self.p_audio = None
        self.audio_stream = None
        self.is_camera_ready = False
        self.is_audio_ready = False
        self.video_file_path = video_file_path
        self.audio_enabled = enable_audio
        self.is_video_file = False
        self.camera_id_to_use = camera_id if camera_id is not None else config.CAMERA_INDEX

        self._initialize_camera()
        if self.audio_enabled:
            self._initialize_audio()
        else:
            logger.info("Audio input (microphone) is disabled for this session.")

    def _initialize_camera(self):
        """Initializes the camera or video file."""
        if self.video_file_path and os.path.exists(self.video_file_path):
            logger.info(f"Initializing video file: {self.video_file_path}")
            try:
                self.cap = cv2.VideoCapture(self.video_file_path)
                if not self.cap.isOpened():
                    logger.error(f"Could not open video file ({self.video_file_path}).")
                    self.is_camera_ready = False
                    self.is_video_file = False
                else:
                    width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    logger.info(f"Video file opened successfully. Resolution: {int(width)}x{int(height)}, FPS: {fps:.2f}, Frame Count: {frame_count}")
                    self.is_camera_ready = True
                    self.is_video_file = True
            except Exception as e:
                logger.error(f"General error initializing video file ({self.video_file_path}): {e}", exc_info=False)
                self.is_camera_ready = False
                self.is_video_file = False
        else:
            if self.video_file_path:
                logger.warning(f"Specified video file ({self.video_file_path}) not found. Default camera will be used.")

            self.is_video_file = False

            logger.info(f"Initializing camera (index: {self.camera_id_to_use})...")
            try:
                self.cap = cv2.VideoCapture(self.camera_id_to_use)
                if not self.cap.isOpened():
                    logger.error(f"Could not open camera (index: {self.camera_id_to_use}).")
                    self.is_camera_ready = False
                else:
                    width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    logger.info(f"Camera successfully initialized. Resolution: {int(width)}x{int(height)}, FPS: {fps:.2f}")
                    self.is_camera_ready = True
            except Exception as e:
                logger.error(f"General error initializing camera: {e}", exc_info=False)
                self.is_camera_ready = False

    def _initialize_audio(self):
        """Initializes the microphone and PyAudio."""
        if not self.audio_enabled:
            self.is_audio_ready = False
            return

        logger.info("Initializing microphone (PyAudio)...")
        try:
            self.p_audio = pyaudio.PyAudio()
            self.audio_stream = self.p_audio.open(
                format=config.PYAUDIO_FORMAT,
                channels=config.PYAUDIO_CHANNELS,
                rate=config.PYAUDIO_RATE,
                input=True,
                frames_per_buffer=config.PYAUDIO_FRAMES_PER_BUFFER,
                stream_callback=None
            )
            self.is_audio_ready = True
            logger.info(f"Microphone successfully initialized. Settings: Format={config.PYAUDIO_FORMAT}, Channels={config.PYAUDIO_CHANNELS}, Rate={config.PYAUDIO_RATE}, Buffer={config.PYAUDIO_FRAMES_PER_BUFFER}")
        except AttributeError as ae:
            logger.error(f"Config file error (e.g., missing PYAUDIO_FORMAT) during PyAudio initialization: {ae}")
            self.p_audio = None
            self.audio_stream = None
        except Exception as e:
            logger.error(f"Error initializing PyAudio or microphone stream: {e}", exc_info=True)
            self.is_audio_ready = False
            if self.p_audio:
                try:
                    self.p_audio.terminate()
                except Exception as term_e:
                    logger.error(f"Additional error terminating PyAudio: {term_e}")
                self.p_audio = None

    def capture_video_frame(self):
        """Captures a single video frame from the camera."""
        if not self.is_camera_ready or not self.cap or not self.cap.isOpened():
            return None

        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Could not read frame from camera (ret=False). Check camera connection.")
                return None
            return frame
        except Exception as e:
            logger.error(f"Error capturing video frame: {e}", exc_info=False)
            return None

    def capture_audio_segment(self, duration_seconds=None):
        
        if not self.audio_enabled:
            return None

        if duration_seconds is None:
            duration_seconds = config.LIVE_AUDIO_SEGMENT_DURATION

        if not self.is_audio_ready or not self.audio_stream or not self.audio_stream.is_active():
            return None

        frames = []
        num_chunks_to_read = int(config.PYAUDIO_RATE / config.PYAUDIO_FRAMES_PER_BUFFER * duration_seconds)
        if num_chunks_to_read == 0:
            num_chunks_to_read = 1
        if num_chunks_to_read <= 0:
            logger.warning(f"Invalid number of chunks to read ({num_chunks_to_read}) for duration: {duration_seconds}s. Audio not captured.")
            return None

        try:
            for _ in range(num_chunks_to_read):
                data = self.audio_stream.read(config.PYAUDIO_FRAMES_PER_BUFFER, exception_on_overflow=False)
                frames.append(data)
        except IOError as e:
            errnum = None
            if hasattr(e, 'errno'):
                errnum = e.errno
            elif hasattr(e, 'args') and len(e.args) > 0 and isinstance(e.args[0], int):
                errnum = e.args[0]

            if errnum == pyaudio.paInputOverflowed or "Input overflowed" in str(e):
                logger.warning("Microphone input overflowed! Some audio data might be lost.")
            elif errnum == -9988 or "Stream is not active" in str(e).lower():
                logger.warning(f"Audio stream is not active (likely closing): {e}")
                return None
            elif errnum == -9983 or "Stream is stopped" in str(e).lower():
                logger.warning(f"Audio stream is stopped (likely closing): {e}")
                return None
            else:
                logger.error(f"IOError/OSError while reading audio data: {e} (Errno: {errnum})", exc_info=False)
                return None
        except Exception as e:
            logger.error(f"General and unexpected error while capturing audio segment: {e}", exc_info=True)
            return None

        if not frames:
            logger.debug("No audio frames captured.")
            return None

        audio_segment_np = np.frombuffer(b''.join(frames), dtype=config.AUDIO_NUMPY_DTYPE)
        return audio_segment_np

    def release(self):
        """Releases camera and microphone resources."""
        logger.info("Releasing input resources (camera and microphone)...")
        if self.cap and self.is_camera_ready:
            try:
                if self.cap.isOpened(): self.cap.release()
                logger.info("Camera released.")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
        self.is_camera_ready = False

        if self.audio_enabled:
            if self.audio_stream and self.is_audio_ready:
                try:
                    if self.audio_stream.is_active(): self.audio_stream.stop_stream()
                    self.audio_stream.close()
                    logger.info("Microphone stream stopped and closed.")
                except Exception as e:
                    logger.error(f"Error stopping/closing microphone stream: {e}")
            self.is_audio_ready = False

            if self.p_audio:
                try:
                    self.p_audio.terminate()
                    logger.info("PyAudio terminated.")
                except Exception as e:
                    logger.error(f"Error terminating PyAudio: {e}")
                self.p_audio = None