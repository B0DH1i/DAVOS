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
        self.music_channel = None # pygame.mixer.Channel objesi için
        self.active_intervention_type = None # Hangi tür müdahalenin aktif olduğunu tutar

        self.is_binaural_playing = False
        self.binaural_sound = None # Stereo Sound objesi için
        self.binaural_channel = None # Binaural ritimler için ayrı bir channel

        logger.info("LazanovAudioManager başlatılıyor...")

        try:
            pygame.mixer.init(frequency=config.AUDIO_SAMPLE_RATE, channels=2, buffer=2048)
            logger.info(f"Pygame mixer {config.AUDIO_SAMPLE_RATE} Hz, stereo olarak başarıyla başlatıldı.")
            # Müdahaleler için ayrı kanallar ayır
            self.music_channel = pygame.mixer.Channel(0) # Müzik için kanal 0
            if config.LAZANOV_BRAINWAVE_ENTRAINMENT_ENABLED and config.LAZANOV_BINAURAL_BEATS_ENABLED:
                if pygame.mixer.get_num_channels() > 1:
                    self.binaural_channel = pygame.mixer.Channel(1) # Binaural için kanal 1
                    logger.info("Binaural ritimler için ayrı bir mixer kanalı (1) ayrıldı.")
                else:
                    logger.warning("Binaural ritimler için yeterli mixer kanalı yok. Binaural devre dışı bırakılacak.")
                    config.LAZANOV_BINAURAL_BEATS_ENABLED = False
        except Exception as e:
            logger.error(f"Pygame mixer başlatılamadı: {e}")
            raise RuntimeError(f"Pygame mixer başlatılamadı: {e}")

        self.load_music_library()

        if not self.music_library:
            logger.warning("Müzik kütüphanesi boş veya yüklenemedi.")

    def load_music_library(self):
        metadata_path = os.path.join(config.LAZANOV_MUSIC_LIBRARY_PATH, config.LAZANOV_MUSIC_METADATA_FILENAME)
        try:
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.music_library = json.load(f)
                logger.info(f"{len(self.music_library)} adet müzik parçası {metadata_path} dosyasından yüklendi.")
                for track in self.music_library:
                    track['filepath'] = os.path.join(config.LAZANOV_MUSIC_LIBRARY_PATH, track['filename'])
                    if not os.path.exists(track['filepath']):
                        logger.warning(f"Müzik dosyası bulunamadı: {track['filepath']}. Bu parça kütüphaneden kaldırılıyor.")
                        self.music_library.remove(track)
            else:
                logger.warning(f"Müzik metadata dosyası bulunamadı: {metadata_path}")
                self.music_library = []
        except Exception as e:
            logger.error(f"Müzik kütüphanesi yüklenirken hata oluştu: {e}")
            self.music_library = []

    def _generate_stereo_sine_wave(self, freq_left, freq_right, duration_sec, sample_rate, amplitude=0.5):
        """Belirli frekanslarda stereo sinüs dalgası (numpy array) üretir."""
        num_samples = int(sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        
        wave_left = amplitude * np.sin(2 * np.pi * freq_left * t)
        wave_right = amplitude * np.sin(2 * np.pi * freq_right * t)
        
        # 16-bit integer formatına dönüştür
        wave_left_int = (wave_left * 32767).astype(np.int16)
        wave_right_int = (wave_right * 32767).astype(np.int16)
        
        # Stereo interleaved array oluştur (LRLRLR...)
        stereo_wave = np.empty((num_samples * 2,), dtype=np.int16)
        stereo_wave[0::2] = wave_left_int
        stereo_wave[1::2] = wave_right_int
        return stereo_wave

    def select_music_for_intervention(self, target_state):
        if not self.music_library:
            logger.warning("Müzik seçilemiyor, kütüphane boş.")
            return None

        suitable_tracks = [track for track in self.music_library if target_state in track.get("target_states", [])]
        
        if not suitable_tracks:
            logger.warning(f"'{target_state}' hedef durumu için uygun müzik bulunamadı.")
            # Fallback: Genel "calm" veya "relaxing" etiketli bir müzik dene
            suitable_tracks = [track for track in self.music_library if "calm" in track.get("primary_mood_tags", [])]
            if not suitable_tracks:
                logger.warning("Fallback için de uygun müzik bulunamadı. Rastgele bir müzik seçiliyor.")
                suitable_tracks = self.music_library # En kötü ihtimalle rastgele birini seç
        
        if suitable_tracks:
            selected_track = random.choice(suitable_tracks)
            logger.info(f"'{target_state}' için müzik seçildi: {selected_track['title']}")
            return selected_track
        return None

    def play_music(self, music_entry, volume=None, on_finish_callback=None):
        if not music_entry or not os.path.exists(music_entry['filepath']):
            logger.error(f"Çalınacak müzik girdisi geçersiz veya dosya yok: {music_entry}")
            return

        self.stop_music() # Önce varsa çalanı durdur

        try:
            sound = pygame.mixer.Sound(music_entry['filepath'])
            self.current_music_entry = music_entry
            effective_volume = volume if volume is not None else config.LAZANOV_DEFAULT_MUSIC_VOLUME
            
            self.music_channel.set_volume(0) # Fade-in için başlangıç sesi 0
            self.music_channel.play(sound, loops=-1 if music_entry.get("loop", False) else 0)
            self.is_music_playing = True
            logger.info(f"Müzik çalınıyor: {music_entry['title']} (Ses: {effective_volume})")

            # Fade-in
            fade_steps = 50 # Daha pürüzsüz geçiş için adım sayısı
            for i in range(fade_steps + 1):
                self.music_channel.set_volume(effective_volume * (i / fade_steps))
                time.sleep(config.LAZANOV_FADE_IN_OUT_DURATION_SECONDS / fade_steps)
            
            if on_finish_callback:
                # Pygame channel'ın bitişini doğrudan izlemek için event kullanılabilir veya süre ile takip edilebilir.
                # Şimdilik basitlik adına, eğer loop etmiyorsa, süresi kadar sonra callback çağrılır gibi varsayalım.
                # Daha robust bir çözüm için pygame eventleri (pygame.USEREVENT + MUSIC_END_EVENT) kullanılmalı.
                pass # İleriye dönük not

        except Exception as e:
            logger.error(f"Müzik çalınırken hata: {e} (Dosya: {music_entry['filepath']})")
            self.is_music_playing = False

    def stop_music(self, fade_out_duration_seconds=None):
        if self.is_music_playing and self.music_channel.get_busy():
            logger.info(f"Müzik durduruluyor: {self.current_music_entry['title'] if self.current_music_entry else 'Bilinmeyen'}")
            duration = fade_out_duration_seconds if fade_out_duration_seconds is not None else config.LAZANOV_FADE_IN_OUT_DURATION_SECONDS
            initial_volume = self.music_channel.get_volume()
            fade_steps = 50

            try:
                for i in range(fade_steps + 1):
                    self.music_channel.set_volume(initial_volume * (1 - (i / fade_steps)))
                    time.sleep(duration / fade_steps)
                self.music_channel.stop()
            except Exception as e:
                 logger.warning(f"Müzik durdurulurken (fade-out) hata: {e}. Direkt durduruluyor.")
                 self.music_channel.stop()
            
        self.is_music_playing = False
        self.current_music_entry = None

    def start_binaural_beats(self, target_beat_hz, duration_sec=600, carrier_hz=None, volume=None):
        if not config.LAZANOV_BRAINWAVE_ENTRAINMENT_ENABLED or not config.LAZANOV_BINAURAL_BEATS_ENABLED or not self.binaural_channel:
            logger.debug("Binaural ritimler devre dışı veya kanal yok.")
            return

        self.stop_binaural_beats() # Varsa önce durdur

        carrier_hz = carrier_hz if carrier_hz is not None else config.LAZANOV_BINAURAL_BEATS_CARRIER_FREQ_HZ
        effective_volume = volume if volume is not None else config.LAZANOV_BINAURAL_BEATS_VOLUME

        freq_left = carrier_hz - (target_beat_hz / 2.0)
        freq_right = carrier_hz + (target_beat_hz / 2.0)

        try:
            stereo_wave_data = self._generate_stereo_sine_wave(freq_left, freq_right, duration_sec, config.AUDIO_SAMPLE_RATE, amplitude=0.8)
            self.binaural_sound = pygame.sndarray.make_sound(stereo_wave_data)
            
            self.binaural_channel.set_volume(effective_volume) 
            self.binaural_channel.play(self.binaural_sound, loops=-1) # Sürekli çalması için loops=-1
            self.is_binaural_playing = True
            logger.info(f"{target_beat_hz} Hz hedefli binaural ritimler başlatıldı (Sol: {freq_left:.2f} Hz, Sağ: {freq_right:.2f} Hz, Ses: {effective_volume}).")
        except Exception as e:
            logger.error(f"Binaural ritimler başlatılırken hata: {e}")
            self.is_binaural_playing = False

    def stop_binaural_beats(self):
        if self.is_binaural_playing and self.binaural_channel and self.binaural_channel.get_busy():
            self.binaural_channel.stop()
            logger.info("Binaural ritimler durduruldu.")
        if self.binaural_sound:
             self.binaural_sound = None # Referansı temizle
        self.is_binaural_playing = False

    def manage_intervention(self, primary_emotion, emotion_confidence):
        if not config.LAZANOV_INTERVENTION_ENABLED:
            # logger.debug("Lazanov müdahaleleri genel olarak devre dışı.")
            return

        intervention_details = config.LAZANOV_TRIGGER_EMOTIONS.get(primary_emotion)
        
        if intervention_details and emotion_confidence >= config.LAZANOV_TRIGGER_CONFIDENCE_THRESHOLD:
            target_state = intervention_details["target_state"]
            logger.info(f"Müdahale durumu değerlendiriliyor: Duygu='{primary_emotion}', Güven={emotion_confidence:.2f}, Hedef Durum='{target_state}'")
            
            if self.active_intervention_type == target_state and (self.is_music_playing or self.is_binaural_playing):
                # logger.debug(f"'{target_state}' için zaten bir müdahale aktif.")
                return

            # Cooldown check (henüz implemente edilmedi, basit bir zamanlayıcı eklenebilir)
            # if time.time() - self.last_intervention_time < config.LAZANOV_INTERVENTION_COOLDOWN_SECONDS:
            #     logger.debug("Müdahale cooldown süresinde.")
            #     return

            selected_music = self.select_music_for_intervention(target_state)
            if selected_music:
                self.play_music(selected_music)
                self.active_intervention_type = target_state # Müzik için müdahale tipini ayarla
            # Binaural ritimler sadece müzik yoksa veya ayrıca tetikleniyorsa active_intervention_type'ı günceller

            if config.LAZANOV_BRAINWAVE_ENTRAINMENT_ENABLED and config.LAZANOV_BINAURAL_BEATS_ENABLED:
                # target_state'in LAZANOV_BINAURAL_BEATS_SETTINGS içinde bir ID olduğunu varsayıyoruz.
                binaural_setting = config.LAZANOV_BINAURAL_BEATS_SETTINGS.get(target_state)
                
                if binaural_setting:
                    beat_hz = binaural_setting.get("beat_hz")
                    carrier_hz = binaural_setting.get("carrier_hz")
                    duration = binaural_setting.get("default_duration_seconds", 600) # Varsayılan süre
                    volume = binaural_setting.get("default_volume", 0.3) # Varsayılan ses seviyesi
                    
                    if beat_hz is not None: # beat_hz tanımlıysa devam et
                        logger.info(f"'{target_state}' için binaural ritim ayarları bulundu: Beat={beat_hz}Hz, Carrier={carrier_hz}Hz, Süre={duration}s, Ses={volume}")
                        self.start_binaural_beats(target_beat_hz=beat_hz, 
                                                  duration_sec=duration, 
                                                  carrier_hz=carrier_hz, 
                                                  volume=volume)
                        if not self.is_music_playing: # Eğer müzik çalmıyorsa, binaural müdahale tipini ayarla
                            self.active_intervention_type = target_state
                    else:
                        logger.warning(f"'{target_state}' için binaural ayarlarında 'beat_hz' bulunamadı. Binaural ritimler başlatılmıyor.")
                        self.stop_binaural_beats() # Uygun ayar yoksa, varsa çalanı durdur
                else:
                    logger.info(f"'{target_state}' için özel bir binaural ritim ayarı bulunamadı. Mevcut binaural ritimler (varsa) durduruluyor.")
                    self.stop_binaural_beats() # Hedef duruma uygun binaural yoksa, çalanı durdur.
        
        elif self.is_music_playing or self.is_binaural_playing: # Tetikleyici duygu yoksa veya güven düşükse ve bir şey çalıyorsa durdur
            logger.info("Müdahale için tetikleyici duygu durumu kalmadı veya güven eşiğin altında. Aktif müdahaleler durduruluyor.")
            self.stop_music()
            self.stop_binaural_beats()
            self.active_intervention_type = None

    def shutdown(self):
        logger.info("LazanovAudioManager kapatılıyor...")
        self.stop_music()
        self.stop_binaural_beats()
        pygame.mixer.quit()
        logger.info("LazanovAudioManager başarıyla kapatıldı.")

# Test bloğu silinecek 