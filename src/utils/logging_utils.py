import logging
import sys
import os
from ..configs import main_config as config

def setup_logger(logger_name, level=None, log_file=None, add_console_handler=True, add_file_handler=True):
    """
    Proje genelinde kullanılacak bir logger kurar ve döndürür.
    Eğer logger zaten kurulmuşsa (handler'ları varsa), mevcut olanı döndürür.

    Args:
        logger_name (str): Logger için isim (genellikle __name__ kullanılır).
        level (int, opsiyonel): Logger seviyesi (örn: logging.INFO, logging.DEBUG).
                                Eğer None ise config dosyasından alınır.
        log_file (str, opsiyonel): Logların yazılacağı dosya yolu.
                                 Eğer None ise config dosyasından alınır.
        add_console_handler (bool): Konsola log basılıp basılmayacağı.
        add_file_handler (bool): Dosyaya log yazılıp yazılmayacağı.

    Returns:
        logging.Logger: Kurulmuş logger nesnesi.
    """
    if level is None:
        level = config.LOGGING_LEVEL

    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if add_console_handler:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if add_file_handler:
        if log_file is None:
            log_file = config.APPLICATION_LOG_FILE
        
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except OSError as e:
                print(f"KRİTİK HATA: Log dizini ({log_dir}) oluşturulamadı: {e}. Dosyaya loglama yapılamayacak.")
        
        if log_dir and os.path.exists(log_dir):
            fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        else:
            print(f"UYARI: Log dosyası için dizin ({log_dir}) mevcut değil. Dosyaya loglama yapılamayacak.")

    return logger
