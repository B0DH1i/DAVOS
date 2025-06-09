# src/utils/plot_utils.py
import os
import matplotlib.pyplot as plt
import numpy as np # Bazen history'de NaN değerler olabilir, bunları handle etmek için

# Göreli importlar
from .logging_utils import setup_logger # Aynı utils paketi içinden
from ..configs import main_config as config # Bir üstteki configs paketinden

# Logger'ı bu modül için kur
logger = setup_logger(__name__, log_file=config.APPLICATION_LOG_FILE)

def plot_training_history(history_dict, model_name_prefix, metrics_to_plot=None):
    """
    Eğitim ve doğrulama metriklerini (kayıp ve doğruluk gibi) çizdirir ve kaydeder.

    Args:
        history_dict (dict): Eğitim geçmişini içeren sözlük
                             (örn: model.fit().history'den gelen).
        model_name_prefix (str): Grafik dosyalarının adlandırılması ve kaydedileceği
                                 alt klasör için kullanılacak önek.
        metrics_to_plot (list of str, opsiyonel): Çizdirilecek metriklerin listesi.
                                                Eğer None ise, 'loss' ve 'accuracy'
                                                (eğer varsa) çizdirilir.
    """
    if not history_dict:
        logger.warning(f"Çizdirilecek eğitim geçmişi verisi boş ({model_name_prefix}). Grafik oluşturulmuyor.")
        return

    if metrics_to_plot is None:
        metrics_to_plot = []
        if 'loss' in history_dict:
            metrics_to_plot.append('loss')
        if 'accuracy' in history_dict:
            metrics_to_plot.append('accuracy')
        # Eğer başka yaygın metrikler varsa eklenebilir: 'mae', 'mse' vb.
        # if 'mae' in history_dict: metrics_to_plot.append('mae')
        
    if not metrics_to_plot:
        logger.warning(f"Geçmiş verisinde çizdirilecek bilinen metrik bulunamadı ({model_name_prefix}). Grafik oluşturulmuyor.")
        return

    # Grafiklerin kaydedileceği dizin (PLOTS_PATH/model_name_prefix/)
    plots_save_dir = os.path.join(config.PLOTS_PATH, model_name_prefix)
    try:
        os.makedirs(plots_save_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Grafik kayıt dizini ({plots_save_dir}) oluşturulamadı: {e}")
        return # Dizin oluşturulamazsa grafik kaydedilemez.

    num_metrics_to_plot = len(metrics_to_plot)
    if num_metrics_to_plot == 0:
        logger.warning(f"Çizdirilecek metrik bulunamadı ({model_name_prefix}).")
        return

    plt.figure(figsize=(7 * num_metrics_to_plot, 6)) # Grafik boyutunu ayarla

    for i, metric_name in enumerate(metrics_to_plot):
        plt.subplot(1, num_metrics_to_plot, i + 1)
        
        # Eğitim metriği
        if metric_name in history_dict:
            train_metric_values = history_dict[metric_name]
            # NaN veya sonsuz değerleri filtrele (matplotlib hata verebilir)
            epochs_train = np.arange(len(train_metric_values))
            valid_indices_train = np.isfinite(train_metric_values)
            if np.any(valid_indices_train):
                 plt.plot(epochs_train[valid_indices_train], 
                          np.array(train_metric_values)[valid_indices_train], 
                          label=f'Training {metric_name.capitalize()}', marker='o', linestyle='-')
            else:
                logger.warning(f"Eğitim metriği '{metric_name}' sadece NaN/inf değerler içeriyor.")

        # Doğrulama metriği
        val_metric_name = f'val_{metric_name}'
        if val_metric_name in history_dict:
            val_metric_values = history_dict[val_metric_name]
            epochs_val = np.arange(len(val_metric_values))
            valid_indices_val = np.isfinite(val_metric_values)
            if np.any(valid_indices_val):
                plt.plot(epochs_val[valid_indices_val], 
                         np.array(val_metric_values)[valid_indices_val], 
                         label=f'Validation {metric_name.capitalize()}', marker='x', linestyle='--')
            else:
                logger.warning(f"Doğrulama metriği '{val_metric_name}' sadece NaN/inf değerler içeriyor.")

        plt.title(f'{model_name_prefix}\nTraining & Validation {metric_name.capitalize()}', fontsize=12)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel(metric_name.capitalize(), fontsize=10)
        plt.legend(fontsize=9)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)

    # Tüm metrikleri tek bir dosyaya kaydet
    plot_filename = f"{model_name_prefix}_training_history.png"
    plot_full_path = os.path.join(plots_save_dir, plot_filename)

    try:
        plt.tight_layout(pad=2.0) # Alt grafikler arası boşluk
        plt.savefig(plot_full_path, dpi=150) # Çözünürlüğü artır
        logger.info(f"Eğitim geçmişi grafiği başarıyla kaydedildi: {plot_full_path}")
    except Exception as e:
        logger.error(f"Grafik dosyası kaydedilirken hata ({plot_full_path}): {e}")
    finally:
        plt.close() # Figürü kapatarak hafızayı serbest bırak
