import cv2
import pyaudio
import numpy as np
import time # Hata durumunda kısa bekleme için
import os # For __main__ test directory creation fallback

# Göreli importlar
from ..utils.logging_utils import setup_logger
from ..configs import main_config as config

# Logger'ı bu modül için kur
logger = setup_logger(__name__, log_file=config.APPLICATION_LOG_FILE)

class InputController: # Changed class name from InputHandler to InputController
    """
    Kamera ve mikrofon girdilerini yöneten sınıf.
    """
    def __init__(self, video_file_path=None, enable_audio=True, camera_id=None):
        self.cap = None
        self.p_audio = None
        self.audio_stream = None
        self.is_camera_ready = False
        self.is_audio_ready = False
        self.video_file_path = video_file_path
        self.audio_enabled = enable_audio # Yeni özellik
        self.is_video_file = False # YENİ ÖZELLİK: Video dosyası mı işleniyor?
        self.camera_id_to_use = camera_id if camera_id is not None else config.CAMERA_INDEX # Kullanılacak kamera ID'si

        self._initialize_camera()
        if self.audio_enabled:
            self._initialize_audio()
        else:
            logger.info("Ses girişi (mikrofon) bu oturum için devre dışı bırakıldı.")

    def _initialize_camera(self):
        """Kamerayı veya video dosyasını başlatır."""
        if self.video_file_path and os.path.exists(self.video_file_path):
            logger.info(f"Video dosyası başlatılıyor: {self.video_file_path}")
            try:
                self.cap = cv2.VideoCapture(self.video_file_path)
                if not self.cap.isOpened():
                    logger.error(f"Video dosyası ({self.video_file_path}) açılamadı.")
                    self.is_camera_ready = False
                    self.is_video_file = False # Dosya açılamadı
                else:
                    width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    logger.info(f"Video dosyası başarıyla açıldı. Çözünürlük: {int(width)}x{int(height)}, FPS: {fps:.2f}, Kare Sayısı: {frame_count}")
                    self.is_camera_ready = True
                    self.is_video_file = True # Video dosyası başarıyla açıldı
            except Exception as e:
                logger.error(f"Video dosyası ({self.video_file_path}) başlatılırken genel hata: {e}", exc_info=False)
                self.is_camera_ready = False
                self.is_video_file = False # Hata durumunda false
        else:
            if self.video_file_path:
                logger.warning(f"Belirtilen video dosyası ({self.video_file_path}) bulunamadı. Varsayılan kamera kullanılacak.")
            
            self.is_video_file = False # Canlı kamera kullanılıyor, dosya değil
            # Kullanılacak kamera ID'sini logla
            logger.info(f"Kamera başlatılıyor (indeks: {self.camera_id_to_use})...")
            try:
                self.cap = cv2.VideoCapture(self.camera_id_to_use)
                if not self.cap.isOpened():
                    logger.error(f"Kamera (indeks: {self.camera_id_to_use}) açılamadı.")
                    self.is_camera_ready = False
                else:
                    width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    logger.info(f"Kamera başarıyla başlatıldı. Çözünürlük: {int(width)}x{int(height)}, FPS: {fps:.2f}")
                    self.is_camera_ready = True
            except Exception as e:
                logger.error(f"Kamera başlatılırken genel hata: {e}", exc_info=False)
                self.is_camera_ready = False

    def _initialize_audio(self):
        """Mikrofonu ve PyAudio'yu başlatır."""
        if not self.audio_enabled: # Eğer ses devre dışıysa, başlatma
            self.is_audio_ready = False
            return
            
        logger.info("Mikrofon (PyAudio) başlatılıyor...")
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
            logger.info(f"Mikrofon başarıyla başlatıldı. Ayarlar: Format={config.PYAUDIO_FORMAT}, Kanallar={config.PYAUDIO_CHANNELS}, Rate={config.PYAUDIO_RATE}, Buffer={config.PYAUDIO_FRAMES_PER_BUFFER}")
        except AttributeError as ae:
            logger.error(f"PyAudio başlatılırken config dosyası hatası (örn: PYAUDIO_FORMAT eksik): {ae}")
            self.p_audio = None
            self.audio_stream = None
        except Exception as e:
            logger.error(f"PyAudio veya mikrofon akışı başlatılırken hata: {e}", exc_info=True) # exc_info=True for more details
            self.is_audio_ready = False
            if self.p_audio:
                try:
                    self.p_audio.terminate()
                except Exception as term_e:
                    logger.error(f"PyAudio sonlandırılırken ek hata: {term_e}")
                self.p_audio = None

    def capture_video_frame(self):
        """Kameradan tek bir video karesi yakalar."""
        if not self.is_camera_ready or not self.cap or not self.cap.isOpened(): # Added isOpened() check
            return None
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Kameradan kare okunamadı (ret=False). Kamera bağlantısını kontrol edin.")
                return None
            return frame
        except Exception as e:
            logger.error(f"Video karesi yakalanırken hata: {e}", exc_info=False)
            return None

    def capture_audio_segment(self, duration_seconds=None):
        """
        Belirtilen süre boyunca mikrofondan ses segmenti yakalar.
        Eğer duration_seconds None ise config'den LIVE_AUDIO_SEGMENT_DURATION kullanılır.
        """
        if not self.audio_enabled: # Eğer ses devre dışıysa, yakalama
            return None
            
        if duration_seconds is None:
            duration_seconds = config.LIVE_AUDIO_SEGMENT_DURATION

        if not self.is_audio_ready or not self.audio_stream or not self.audio_stream.is_active(): # Added is_active check
            return None

        frames = []
        num_chunks_to_read = int(config.PYAUDIO_RATE / config.PYAUDIO_FRAMES_PER_BUFFER * duration_seconds)
        if num_chunks_to_read == 0:
            num_chunks_to_read = 1
        if num_chunks_to_read <= 0:
            logger.warning(f"Okunacak chunk sayısı geçersiz ({num_chunks_to_read}) süre: {duration_seconds}s. Ses yakalanamadı.")
            return None
            
        try:
            for _ in range(num_chunks_to_read):
                data = self.audio_stream.read(config.PYAUDIO_FRAMES_PER_BUFFER, exception_on_overflow=False)
                frames.append(data)
        except IOError as e: # Catches general I/O errors including some from PyAudio
            # PyAudio uses specific error codes; paInputOverflowed is -9981
            # Stream is not active is -9988, Stream is stopped is -9983
            errnum = None
            if hasattr(e, 'errno'): # For standard OSError
                errnum = e.errno
            elif hasattr(e, 'args') and len(e.args) > 0 and isinstance(e.args[0], int): # For some PyAudio errors
                errnum = e.args[0]
            
            if errnum == pyaudio.paInputOverflowed or "Input overflowed" in str(e):
                 logger.warning("Mikrofon girişi taştı (Overflow)! Bazı ses verileri kaybolmuş olabilir.")
            elif errnum == -9988 or "Stream is not active" in str(e).lower(): # pyaudio.paStreamIsNotActive (may not be directly available)
                 logger.warning(f"Ses akışı aktif değil (muhtemelen kapanıyor): {e}")
                 return None
            elif errnum == -9983 or "Stream is stopped" in str(e).lower(): # pyaudio.paStreamIsStopped (may not be directly available)
                 logger.warning(f"Ses akışı durdurulmuş (muhtemelen kapanıyor): {e}")
                 return None
            else:
                logger.error(f"Ses verisi okunurken IOError/OSError: {e} (Errno: {errnum})", exc_info=False) # Reduce log noise for common issues
                return None 
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Ses segmenti yakalanırken genel ve beklenmedik hata: {e}", exc_info=True)
            return None
        
        if not frames:
            logger.debug("Hiç ses frame'i yakalanamadı.")
            return None
            
        audio_segment_np = np.frombuffer(b''.join(frames), dtype=config.AUDIO_NUMPY_DTYPE) # Use from config
        return audio_segment_np

    def release(self):
        """Kamera ve mikrofon kaynaklarını serbest bırakır."""
        logger.info("Giriş kaynakları (kamera ve mikrofon) serbest bırakılıyor...")
        if self.cap and self.is_camera_ready:
            try:
                if self.cap.isOpened(): self.cap.release()
                logger.info("Kamera serbest bırakıldı.")
            except Exception as e:
                logger.error(f"Kamera serbest bırakılırken hata: {e}")
        self.is_camera_ready = False # Set to false even if release fails
        
        if self.audio_enabled: # Sadece ses aktifse ses kaynaklarını serbest bırak
            if self.audio_stream and self.is_audio_ready:
                try:
                    if self.audio_stream.is_active(): self.audio_stream.stop_stream()
                    self.audio_stream.close()
                    logger.info("Mikrofon akışı durduruldu ve kapatıldı.")
                except Exception as e:
                    logger.error(f"Mikrofon akışı durdurulurken/kapatılırken hata: {e}")
            self.is_audio_ready = False # Set to false even if stream ops fail

            if self.p_audio:
                try:
                    self.p_audio.terminate()
                    logger.info("PyAudio sonlandırıldı.")
                except Exception as e:
                    logger.error(f"PyAudio sonlandırılırken hata: {e}")
            self.p_audio = None
        
