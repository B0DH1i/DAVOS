import os
import time
import pygame
import numpy as np
from pathlib import Path
import cv2
from ..configs import main_config as config
from src.utils.logging_utils import setup_logger

logger = setup_logger("OutputHandler", log_file=config.APPLICATION_LOG_FILE)

class OutputController:
    """
    Handles Pygame display for video and emotion information.
    """

    def __init__(self,
                 pygame_window_size=(config.DISPLAY_MAX_WIDTH, config.DISPLAY_MAX_HEIGHT),
                 enable_fullscreen=False,
                 show_fer_probs=False,
                 show_ser_probs=False,
                 show_integrated_probs=False,
                 text_color=config.TEXT_COLOR if hasattr(config, 'TEXT_COLOR') else (255, 255, 0),
                 bg_color=config.BACKGROUND_COLOR if hasattr(config, 'BACKGROUND_COLOR') else (0, 0, 0),
                 info_area_height_ratio=config.INFO_AREA_HEIGHT_RATIO if hasattr(config, 'INFO_AREA_HEIGHT_RATIO') else 0.25,
                 font_scale=config.FONT_SCALE if hasattr(config, 'FONT_SCALE') else 0.5):


        self.window_size = pygame_window_size
        self.enable_fullscreen = enable_fullscreen
        self.show_fer_probs = show_fer_probs
        self.show_ser_probs = show_ser_probs
        self.show_integrated_probs = show_integrated_probs
        self.text_color = text_color
        self.bg_color = bg_color
        self.info_area_height_ratio = info_area_height_ratio
        self.font_scale_cv = font_scale

        self.screen = None
        self.pygame_font = None
        self.base_font_size = 20
        self.line_height = 22

        self._initialize_display()

        logger.info(f"Output controller initialized. Display initialized: {self.screen is not None}")

    def _initialize_display(self):
        """Initialize pygame display and font."""
        logger.info("Initializing Pygame display...")
        try:
            if not pygame.get_init():
                pygame.init()

            if self.enable_fullscreen:
                self.screen = pygame.display.set_mode(self.window_size, pygame.FULLSCREEN | pygame.DOUBLEBUF)
            else:
                self.screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE | pygame.DOUBLEBUF)
            pygame.display.set_caption(config.APP_WINDOW_TITLE)

            effective_font_scale = self.font_scale_cv if self.font_scale_cv > 0 else 0.5
            self.base_font_size = int(self.window_size[1] * 0.035 / effective_font_scale)

            self.base_font_size = max(10, min(self.base_font_size, 40))
            self.line_height = int(self.base_font_size * 1.25)

            try:
                self.pygame_font = pygame.font.SysFont("Arial", self.base_font_size)
            except Exception:
                self.pygame_font = pygame.font.Font(None, self.base_font_size)

            logger.info(f"Pygame display successfully initialized. Size: {self.window_size}, Font size: {self.base_font_size}")
        except pygame.error as e:
            logger.error(f"Error initializing Pygame display: {e}")
            self.screen = None
        except Exception as e:
            logger.error(f"A general error occurred while initializing Pygame display: {e}")
            self.screen = None

    def display_frame(self, frame_bgr, display_info):

        if not self.screen or not self.pygame_font:
            return

        self.screen.fill(self.bg_color)

        if frame_bgr is not None:
            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_pygame_surface = pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")

                screen_w, screen_h = self.screen.get_size()
                img_w, img_h = frame_pygame_surface.get_size()

                target_w, target_h = screen_w, screen_h

                scale_w = target_w / img_w
                scale_h = target_h / img_h
                scale = min(scale_w, scale_h)

                scaled_w, scaled_h = int(img_w * scale), int(img_h * scale)

                if scaled_w > 0 and scaled_h > 0:
                    scaled_surface = pygame.transform.smoothscale(frame_pygame_surface, (scaled_w, scaled_h))
                    blit_x = (screen_w - scaled_w) // 2
                    blit_y = (screen_h - scaled_h) // 2
                    self.screen.blit(scaled_surface, (blit_x, blit_y))

            except Exception as e:
                logger.error(f"Error converting or blitting video frame to Pygame: {e}", exc_info=False)

        y_offset = 10

        info_lines_content = {
            "FPS (Approx)": f"{display_info.get('fps', 'N/A'):.1f}" if isinstance(display_info.get('fps'), float) else display_info.get('fps', 'N/A'),
            "Face Status": display_info.get("face_detection_status", "N/A"),
            "FER Raw": display_info.get("fer_raw_label", "N/A"),
            "SER Raw": display_info.get("ser_raw_label", "N/A"),
            "Integrated": display_info.get("integrated_label", "N/A"),
            "Lazanov Music": display_info.get("lazanov_music_status", "N/A"),
            "Binaural Beats": display_info.get("lazanov_binaural_status", "N/A")
        }

        for key, value in info_lines_content.items():
            try:
                text_surface = self.pygame_font.render(f"{key}: {str(value)}", True, self.text_color)
                self.screen.blit(text_surface, (10, y_offset))
                y_offset += self.line_height
            except Exception as e:
                logger.error(f"Error drawing text ({key}): {e}")
                y_offset += self.line_height

        prob_y_start = y_offset + self.line_height // 2

        def render_probabilities_list(title, probs_dict_or_list, y_current, target_labels=None):
            if not probs_dict_or_list: return y_current

            text_surface = self.pygame_font.render(title, True, self.text_color)
            self.screen.blit(text_surface, (15, y_current))
            y_current += self.line_height

            items_to_render = []
            if isinstance(probs_dict_or_list, dict):
                sorted_probs = sorted(probs_dict_or_list.items(), key=lambda item: item[1], reverse=True)
                items_to_render = sorted_probs
            elif isinstance(probs_dict_or_list, (list, np.ndarray)) and target_labels:
                if len(probs_dict_or_list) == len(target_labels):
                    temp_dict = {target_labels[i]: probs_dict_or_list[i] for i in range(len(target_labels))}
                    items_to_render = sorted(temp_dict.items(), key=lambda item: item[1], reverse=True)
                else:
                    logger.warning(f"Probability list and label list lengths do not match for '{title}'.")

            for emotion, prob_val in items_to_render:
                prob_val_float = float(prob_val)
                if prob_val_float > 0.01:
                    prob_text = f"   {str(emotion)[:10]}: {prob_val_float:.2f}"
                    text_surface = self.pygame_font.render(prob_text, True, self.text_color)
                    self.screen.blit(text_surface, (20, y_current))
                    y_current += int(self.line_height * 0.9)
                    if y_current > self.window_size[1] - self.line_height: break
            return y_current + self.line_height // 2

        if self.show_fer_probs:
            prob_y_start = render_probabilities_list("FER Probs:", display_info.get("fer_probs_dict"), prob_y_start)
        if self.show_ser_probs:
            prob_y_start = render_probabilities_list("SER Probs:", display_info.get("ser_probs_dict"), prob_y_start)
        if self.show_integrated_probs:
            target_emotions_for_display = config.TARGET_EMOTIONS
            if target_emotions_for_display:
                prob_y_start = render_probabilities_list("Integrated Probs:",
                                                         display_info.get("integrated_probs_vector"),
                                                         prob_y_start,
                                                         target_labels=target_emotions_for_display)
            else:
                logger.warning("config.TARGET_EMOTIONS not found for integrated probs display.")

        pygame.display.flip()

    def handle_events(self):

        if not self.screen: return True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logger.info("Pygame window close request received.")
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    logger.info("User pressed 'q' or 'ESC'. Exiting.")
                    return False
                if event.key == pygame.K_f:
                    self.enable_fullscreen = not self.enable_fullscreen
                    if self.enable_fullscreen:
                        self.screen = pygame.display.set_mode(self.window_size, pygame.FULLSCREEN | pygame.DOUBLEBUF)
                    else:
                        self.screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE | pygame.DOUBLEBUF)
                    logger.info(f"Fullscreen mode {'enabled' if self.enable_fullscreen else 'disabled'}.")

            if event.type == pygame.VIDEORESIZE:
                self.window_size = event.size
                if not self.enable_fullscreen:
                    self.screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE | pygame.DOUBLEBUF)

                effective_font_scale = self.font_scale_cv if self.font_scale_cv > 0 else 0.5
                self.base_font_size = int(self.window_size[1] * 0.035 / effective_font_scale)
                self.base_font_size = max(10, min(self.base_font_size, 40))
                self.line_height = int(self.base_font_size * 1.25)
                try:
                    self.pygame_font = pygame.font.SysFont("Arial", self.base_font_size)
                except Exception:
                    self.pygame_font = pygame.font.Font(None, self.base_font_size)
                logger.info(f"Pygame window resized: {self.window_size}, New font size: {self.base_font_size}")

        return True

    def quit(self):
        logger.info("Releasing Pygame display and resources...")
        try:
            if pygame.display.get_init():
                pygame.display.quit()
                logger.info("Pygame display closed.")

            if pygame.get_init():
                pygame.quit()
                logger.info("General pygame.quit() called.")

        except Exception as e:
            logger.error(f"Error while quitting Pygame: {e}", exc_info=True)