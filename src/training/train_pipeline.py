# src/training/train_pipeline.py
import os
import datetime
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # TensorFlow Python loglarını azalt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.utils import class_weight # Sınıf ağırlıklandırma için eklendi

# Göreli importlar
from ..utils.logging_utils import setup_logger
from ..utils import file_utils # Model kaydetme, dizin oluşturma
from ..utils import plot_utils # Geçmiş çizdirme
from ..configs import main_config as config
from ..core import data_loader # Veri yükleyici
from ..core import models      # Model tanımları

# VGG16 için ön işleme fonksiyonu
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input

# Logger'ı bu modül için kur
logger = setup_logger(__name__, log_file=config.APPLICATION_LOG_FILE)

def train_fer_model(model_type=config.FER_MODEL_TYPE_VGG16_TRANSFER, 
                    epochs=None,  # Varsayılanı None yap, aşağıda config'den al
                    batch_size=None, # Varsayılanı None yap, aşağıda config'den al
                    optimizer_type=config.DEFAULT_OPTIMIZER_FER,
                    learning_rate=None, # Varsayılanı None yap, aşağıda config'den al
                    patience_early_stopping=config.PATIENCE_EARLY_STOPPING_FER,
                    patience_reduce_lr=config.PATIENCE_REDUCE_LR_FER,
                    monitor_metric=config.MONITOR_METRIC_FER,
                    save_best_only=config.SAVE_BEST_ONLY_FER,
                    data_augmentation=config.USE_DATA_AUGMENTATION_FER,
                    base_save_name_prefix=config.FER_MODEL_NAME_PREFIX):
    """
    Yüz İfadesi Tanıma (FER) modelini eğitir.
    """
    try:
        # Cihaz yerleşimini logla (GPU kullanımını kontrol etmek için)
        # tf.debugging.set_log_device_placement(True) # Kaldırıldı, çok fazla log üretiyor

        # Parametreler None ise config'den varsayılan değerleri al
        if epochs is None:
            epochs = config.DEFAULT_EPOCHS_FER
        if batch_size is None:
            batch_size = config.DEFAULT_BATCH_SIZE_FER
        if learning_rate is None:
            learning_rate = config.DEFAULT_LEARNING_RATE_FER
        # optimizer_type, patience_early_stopping vb. zaten config'den varsayılan alıyor.

        logger.info(f"--- FER Model Eğitimi Başlatılıyor (Model Tipi: {model_type}) ---")
        logger.info(f"Parametreler: Epochs={epochs}, Batch={batch_size}, LR={learning_rate}, Optimizer={optimizer_type}")
        # 1. Veriyi Yükle
        # (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data_main(dataset_name='fer2013', 
        #                                                                     model_type_for_input_shape='fer')
        # data_loader.load_fer2013_data() kullanalım, çünkü load_data_main SER için de kullanılıyor ve karmaşıklık yaratabilir
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.load_fer_data() # load_fer2013_data -> load_fer_data olarak güncellendi

        if X_train is None or X_train.size == 0: # X_train.size == 0 kontrolü eklendi
            logger.error("FER eğitim verisi yüklenemedi veya boş. Eğitim durduruluyor.")
            return None, None

        input_shape = config.INPUT_SHAPE_FER # (48, 48, 1) MiniXception için varsayılan
        num_classes = config.NUM_MODEL_OUTPUT_CLASSES # config.FER_EMOTIONS yerine NUM_MODEL_OUTPUT_CLASSES kullanıldı

        if model_type == config.FER_MODEL_TYPE_VGG16_TRANSFER:
            logger.info("VGG16 transfer öğrenme modeli için veri ön işleme yapılıyor...")
            input_shape = (config.INPUT_SHAPE_FER[0], config.INPUT_SHAPE_FER[1], 3) # (48, 48, 3)
            
            logger.info(f"X_train şekli (önce): {X_train.shape}, veri aralığı (min-max): {np.min(X_train)}-{np.max(X_train)}")

            X_train_rgb = (X_train * 255.0).astype(np.uint8)
            X_val_rgb = (X_val * 255.0).astype(np.uint8)
            X_test_rgb = (X_test * 255.0).astype(np.uint8)

            X_train_rgb = np.repeat(X_train_rgb, 3, axis=-1)
            X_val_rgb = np.repeat(X_val_rgb, 3, axis=-1)
            X_test_rgb = np.repeat(X_test_rgb, 3, axis=-1)
            logger.info(f"X_train şekli (3 kanala çoğaltıldıktan sonra): {X_train_rgb.shape}")

            X_train = vgg16_preprocess_input(X_train_rgb)
            X_val = vgg16_preprocess_input(X_val_rgb)
            X_test = vgg16_preprocess_input(X_test_rgb)
            logger.info(f"VGG16 ön işlemesi tamamlandı. X_train şekli (sonra): {X_train.shape}, veri aralığı (min-max): {np.min(X_train)}-{np.max(X_train)}")

        # 2. Modeli Oluştur
        # MODEL_FACTORY'yi src.core.models'dan alalım
        if model_type not in models.MODEL_FACTORY:
            logger.error(f"Geçersiz FER model tipi: {model_type}. Model fabrikasında bulunamadı. Kullanılabilir: {list(models.MODEL_FACTORY.keys())}")
            return None, None
        
        # Model oluşturma fonksiyonunu fabrikadan al
        model_builder_func = models.MODEL_FACTORY[model_type]
        model = model_builder_func(input_shape=input_shape, num_classes=num_classes)
        
        model_save_prefix = f"{base_save_name_prefix}_{model_type}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # 3. Modeli Derleme
        if optimizer_type.lower() == "adam":
            optimizer = Adam(learning_rate=learning_rate)
            logger.info(f"Adam optimizer kullanılıyor. Öğrenme Oranı: {learning_rate}")
        elif optimizer_type.lower() == "rmsprop":
            optimizer = RMSprop(learning_rate=learning_rate)
            logger.info(f"RMSprop optimizer kullanılıyor. Öğrenme Oranı: {learning_rate}")
        elif optimizer_type.lower() == "sgd": # Yeni eklendi
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
            logger.info(f"SGD optimizer (Nesterov momentum ile) kullanılıyor. Öğrenme Oranı: {learning_rate}, Momentum: 0.9")
        else:
            logger.warning(f"Bilinmeyen optimizer tipi: {optimizer_type}. Varsayılan olarak SGD (Nesterov) kullanılacak.") # Varsayılan SGD'ye güncellendi
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
            
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        logger.info(f"FER Modeli ({model_type}) Derlendi. Özet:")
        model.summary(print_fn=logger.info)

        # 4. Callback'ler
        # ModelCheckpoint: En iyi modeli kaydetmek için (file_utils'daki save_model bunu yapmıyor)
        # Kayıt yolu: TRAINED_MODELS_PATH / model_save_prefix / best_model_checkpoint.h5
        checkpoint_save_dir = os.path.join(config.TRAINED_MODELS_PATH, model_save_prefix)
        os.makedirs(checkpoint_save_dir, exist_ok=True) # file_utils.save_model_and_history de yapar ama burada da garanti
        
        checkpoint_filepath = os.path.join(checkpoint_save_dir, "best_model_checkpoint.h5")
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor=monitor_metric, # İzlenecek metrik (config.MONITOR_METRIC_FER)
            save_best_only=save_best_only,    # Sadece en iyi modeli kaydet
            save_weights_only=False, # Tüm modeli kaydet (mimari + ağırlıklar + optimizer state)
            verbose=1
        )
        early_stopping = EarlyStopping(monitor=config.MONITOR_METRIC_FER, patience=patience_early_stopping, verbose=1, restore_best_weights=True) # monitor güncellendi
        reduce_lr = ReduceLROnPlateau(monitor=config.MONITOR_METRIC_FER, factor=config.FACTOR_REDUCE_LR_FER, patience=patience_reduce_lr, min_lr=1e-6, verbose=1) # monitor ve factor güncellendi
        
        # TensorBoard logları (LOGS_PATH / fer / model_save_prefix)
        tensorboard_log_dir = os.path.join(config.LOGS_PATH, "fer", model_save_prefix)
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
        callbacks = [model_checkpoint, early_stopping, reduce_lr, tensorboard_callback]

        # 5. Veri Artırma (Data Augmentation)
        history = None
        if data_augmentation:
            logger.info("FER için veri artırma kullanılıyor.")
            
            # Genel ImageDataGenerator (rescale olmadan, çünkü veri zaten [0,1] veya VGG preprocessed)
            datagen_common = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

            if model_type == config.FER_MODEL_TYPE_VGG16_TRANSFER:
                # VGG16 için X_train zaten vgg16_preprocess_input ile işlendi.
                # Bu işlenmiş veri ImageDataGenerator'a verilecek.
                logger.info("VGG16 için veri artırma (rescale olmadan, önceden işlenmiş veri).")
                train_generator = datagen_common.flow(X_train, y_train, batch_size=batch_size)
            else:
                # Diğer modeller için X_train'in data_loader tarafından [0,1] aralığına getirildiği varsayılır.
                # Bu yüzden rescale burada da olmamalı.
                logger.info("Standart FER modeli için veri artırma (rescale olmadan, [0,1] aralığında veri).")
                train_generator = datagen_common.flow(X_train, y_train, batch_size=batch_size)
            
            # Eğer X_val boş değilse validation_data olarak kullan, yoksa kullanma
            # validation verisine artırma UYGULANMAZ. Sadece eğitim verisine.
            # Ancak VGG16 ise onun da vgg16_preprocess_input'tan geçmesi lazım. Bu zaten yukarıda yapıldı.
            
            calculated_steps_per_epoch = max(1, len(X_train) // batch_size)
            logger.info(f"FER Data Augmentation - X_train length: {len(X_train)}, batch_size: {batch_size}, calculated_steps_per_epoch: {calculated_steps_per_epoch}")

            history = model.fit(
                train_generator,
                steps_per_epoch=calculated_steps_per_epoch, # len(X_train) 0 ise hata vermesin
                epochs=epochs,
                validation_data=(X_val, y_val), # X_val, y_val tuple olarak verilmeli
                callbacks=callbacks,
                verbose=1
            )
        else:
            logger.info("FER için veri artırma kullanılmıyor.")
            # validation_data_feed = (X_val, y_val) if X_val.size > 0 else None # Eski hali
            
            # Veri artırma olmadığında steps_per_epoch Keras tarafından otomatik ayarlanır,
            # ancak yine de ne beklendiğini loglayabiliriz.
            # model.fit NumPy arrayleri ile çağrıldığında steps_per_epoch'a gerek duymaz.
            logger.info(f"FER No Data Augmentation - X_train length: {len(X_train)}, batch_size: {batch_size}")

            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val), # X_val, y_val tuple olarak verilmeli
                callbacks=callbacks,
                verbose=1
            )
        
        logger.info(f"FER Model ({model_type}) Eğitimi Tamamlandı.")

        # 6. Modeli ve Geçmişi Kaydet (file_utils kullanarak)
        # ModelCheckpoint zaten en iyi modeli checkpoint_filepath'e kaydetti.
        # Eğer restore_best_weights=True ise, 'model' nesnesi zaten en iyi ağırlıklara sahip olur.
        # Bu yüzden, son (veya en iyi restore edilmiş) modeli ve tüm geçmişi file_utils ile tekrar kaydediyoruz.
        # Bu, model_save_prefix altında standart bir yapı oluşturur.
        if history:
            file_utils.save_model_and_history(model, history.history, model_save_prefix)
            # 7. Eğitim Geçmişini Çizdir (plot_utils kullanarak)
            logger.info(f"Eğitim grafiği oluşturuluyor (FER): {model_save_prefix}")
            plot_utils.plot_training_history(history.history, model_save_prefix)
        else:
            logger.error("Eğitim geçmişi oluşturulamadı. Model ve grafik kaydedilemiyor.")
            return None, None

        # 8. Test Seti Üzerinde Değerlendirme
        if X_test.size > 0 and y_test.size > 0:
            logger.info("FER Test seti üzerinde değerlendirme yapılıyor...")
            # En iyi modeli yükleyip onunla değerlendirme yapmak daha doğru olabilir,
            # eğer restore_best_weights=True değilse veya emin olmak istiyorsak.
            # Şu an model zaten en iyi ağırlıkları içeriyor olmalı (EarlyStopping sayesinde).
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            logger.info(f"FER Test Seti Sonuçları - Kayıp: {test_loss:.4f}, Doğruluk: {test_accuracy:.4f}")
        else:
            logger.warning("FER test verisi boş veya yüklenemedi. Değerlendirme atlanıyor.")
        
        logger.info(f"--- FER Model Eğitimi Bitti (Kayıt öneki: {model_save_prefix}) ---")
        return model, history
    except Exception as e:
        logger.error(f"FER model eğitimi sırasında hata oluştu: {e}", exc_info=True)
        return None, None


def train_ser_model(epochs=None, batch_size=None, learning_rate=None,
                    optimizer_type=config.DEFAULT_OPTIMIZER_SER, # Optimizer type eklendi
                    patience_early_stopping=None,
                    patience_reduce_lr=None,
                    monitor_metric=None,
                    save_best_only=config.SAVE_BEST_ONLY_SER, # save_best_only eklendi
                    data_augmentation=config.USE_DATA_AUGMENTATION_SER, # USE_DATA_AUGMENTATION_SER kullanıldı
                    base_save_name_prefix=config.SER_MODEL_NAME_PREFIX): # base_save_name_prefix eklendi
    """Sesli Duygu Tanıma (SER) modelini eğitir."""
    # Parametreler None ise config'den varsayılan değerleri al
    epochs = epochs if epochs is not None else config.DEFAULT_EPOCHS_SER
    batch_size = batch_size if batch_size is not None else config.DEFAULT_BATCH_SIZE_SER
    learning_rate = learning_rate if learning_rate is not None else config.DEFAULT_LEARNING_RATE_SER
    patience_early_stopping = patience_early_stopping if patience_early_stopping is not None else config.PATIENCE_EARLY_STOPPING_SER
    patience_reduce_lr = patience_reduce_lr if patience_reduce_lr is not None else config.PATIENCE_REDUCE_LR_SER
    monitor_metric = monitor_metric if monitor_metric is not None else config.MONITOR_METRIC_SER
    
    model_choice = config.DEFAULT_SER_MODEL_CHOICE # model_choice_from_config -> model_choice, SER_MODEL_CHOICE -> DEFAULT_SER_MODEL_CHOICE
    feature_type = config.SER_FEATURE_TYPE # feature_type_from_config -> feature_type

    logger.info(f"--- SER Model Eğitimi Başlatılıyor (Model: {model_choice}, Öznitelik: {feature_type}) ---")
    logger.info(f"Parametreler: Epochs={epochs}, Batch={batch_size}, LR={learning_rate}, Optimizer={optimizer_type}")
    logger.info(f"Callback Ayarları: EarlyStopPatience={patience_early_stopping}, ReduceLRPatience={patience_reduce_lr}, Monitor={monitor_metric}")

    # 1. Veri Yükleme
    logger.info("RAVDESS verisi yükleniyor...")
    # data_augmentation parametresi load_ravdess_data'ya eklendi
    ravdess_data = data_loader.load_ravdess_data(use_augmentation_on_train=data_augmentation)
    if ravdess_data is None:
        logger.error("RAVDESS veri yüklemesi başarısız oldu. SER eğitimi durduruluyor.")
        return None, None
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = ravdess_data

    if X_train is None or X_train.size == 0:
        logger.error("SER eğitim verisi yüklenemedi veya boş. Eğitim durduruluyor.")
        return None, None

    # Giriş şeklini ve sınıf sayısını belirle
    # MFCC durumunda X_train.shape[1:] (örn: (13, 313, 1)) olurdu.
    # Whisper durumunda X_train.shape[1] (örn: (512,)) olur, expand_dims ile (512,1) veya (1, 512) yapılabilir modele göre.
    # Modelin kendisi (None, embedding_dim) bekler, bu yüzden X_train.shape[1:] [(embedding_dim,)] olur.
    input_shape_ser = X_train.shape[1:] 
    num_classes = config.NUM_MODEL_OUTPUT_CLASSES # NUM_TARGET_CLASSES -> NUM_MODEL_OUTPUT_CLASSES
    logger.info(f"SER için giriş şekli: {input_shape_ser}, Sınıf sayısı: {num_classes}")

    # Sınıf ağırlıklarını hesapla (dengesiz veri seti için)
    if y_train is not None and y_train.ndim > 1 and y_train.shape[0] > 0:
        y_train_indices = np.argmax(y_train, axis=1)
        class_weights_array = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_indices),
            y=y_train_indices
        )
        class_weights = {i : class_weights_array[i] for i in range(len(class_weights_array))}
        logger.info(f"SER için sınıf ağırlıkları hesaplandı: {class_weights}")
    else:
        logger.warning("SER için y_train verisi uygun formatta değil veya boş. Sınıf ağırlıkları hesaplanamadı.")
        class_weights = None

    # 2. Modeli Oluştur
    if model_choice not in models.MODEL_FACTORY:
        logger.error(f"Geçersiz SER model tipi: {model_choice}. Model fabrikasında bulunamadı. Kullanılabilir: {list(models.MODEL_FACTORY.keys())}")
        return None, None
        
    model_builder_func = models.MODEL_FACTORY[model_choice]
    # input_shape_ser'in doğru olduğundan emin ol. Whisper için (embedding_dim,) olmalı.
    # Eğer gelen X_train.shape (batch_size, embedding_dim) ise X_train.shape[1:] (embedding_dim,) olur.
    if isinstance(input_shape_ser, tuple) and len(input_shape_ser) == 1:
        current_input_shape_dim = input_shape_ser[0]
    elif isinstance(input_shape_ser, int): # Doğrudan int gelirse (beklenmiyor ama olabilir)
        current_input_shape_dim = input_shape_ser
    else:
        logger.error(f"SER modeli için beklenmedik input_shape_ser formatı: {input_shape_ser}. Beklenen (embedding_dim,) tuple'ı.")
        return False # veya None, None

    model = model_builder_func(input_shape_dim=current_input_shape_dim, num_classes=num_classes)

    model_save_prefix = f"{base_save_name_prefix}_{model_choice}_{feature_type}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # 3. Modeli Derleme
    # Optimizer seçimi (Adam, RMSprop, SGD)
    if optimizer_type.lower() == "adam":
        optimizer = Adam(learning_rate=learning_rate)
        logger.info(f"Adam optimizer kullanılıyor. Öğrenme Oranı: {learning_rate}")
    elif optimizer_type.lower() == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
        logger.info(f"RMSprop optimizer kullanılıyor. Öğrenme Oranı: {learning_rate}")
    elif optimizer_type.lower() == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        logger.info(f"SGD optimizer (Nesterov momentum ile) kullanılıyor. Öğrenme Oranı: {learning_rate}")
    else:
        logger.warning(f"Bilinmeyen optimizer tipi: {optimizer_type}. Varsayılan olarak Adam kullanılacak.")
        optimizer = Adam(learning_rate=learning_rate) # Varsayılan Adam'a güncellendi

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    logger.info(f"SER Modeli ({model_choice}) Derlendi. Özet:")
    model.summary(print_fn=logger.info)

    # 4. Callback'ler
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

    # 5. Modeli Eğit
    # SER için veri artırma data_loader içinde halledildiği için burada ImageDataGenerator yok.
    logger.info(f"SER Model Eğitimi Başlıyor - X_train length: {len(X_train)}, batch_size: {batch_size}")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights, # Sınıf ağırlıkları eklendi
        verbose=1
    )
    logger.info(f"SER Model Eğitimi Tamamlandı.")

    # 6. Modeli ve Geçmişi Kaydet (file_utils kullanarak)
    # ModelCheckpoint zaten en iyi modeli checkpoint_filepath'e kaydetti.
    # Eğer restore_best_weights=True ise, 'model' nesnesi zaten en iyi ağırlıklara sahip olur.
    if history and hasattr(history, 'history') and isinstance(history.history, dict) and history.history:
        # history.history'nin boş bir sözlük olmadığını da kontrol edelim.
        logger.info(f"Eğitim geçmişi kaydediliyor. Mevcut metrikler: {list(history.history.keys())}")
        save_success = file_utils.save_model_and_history(model, history.history, model_save_prefix)
        if save_success:
            logger.info(f"Model ve eğitim geçmişi başarıyla '{model_save_prefix}' olarak kaydedildi.")
        else:
            logger.warning(f"Model veya eğitim geçmişi '{model_save_prefix}' için tam olarak kaydedilemedi. Detaylar için loglara bakın.")
        
        # 7. Eğitim Geçmişini Çizdir (plot_utils kullanarak)
        # plot_utils.plot_training_history history.history (sözlük) bekler ve kendi loglamasını yapar.
        logger.info(f"Eğitim grafiği oluşturuluyor: {model_save_prefix}")
        plot_utils.plot_training_history(history.history, model_save_prefix)
    else:
        logger.error("SER için eğitim geçmişi (history.history) düzgün oluşturulamadı, boş veya model.fit None döndü. Geçmiş ve grafik kaydedilemiyor.")
        # return False # Eğer bu kritikse eğitimi başarısız sayabiliriz.

    # 8. Test Seti Üzerinde Değerlendirme
    if X_test.size > 0 and y_test.size > 0:
        logger.info("SER Test seti üzerinde değerlendirme yapılıyor...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"SER Test Seti Sonuçları - Kayıp: {test_loss:.4f}, Doğruluk: {test_accuracy:.4f}")
    else:
        logger.warning("SER test verisi boş veya yüklenemedi. Değerlendirme atlanıyor.")
        
    logger.info(f"--- SER Model Eğitimi Bitti (Kayıt öneki: {model_save_prefix}) ---")
    return True # Başarılı


def evaluate_specific_ser_model_on_test_set(model_path: str):
    """
    Belirtilen yoldaki bir SER modelini yükler ve RAVDESS test seti üzerinde değerlendirir.
    """
    logger.info(f"--- Belirtilen SER Modelini Test Setinde Değerlendirme ---")
    logger.info(f"Model yolu: {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"Model dosyası bulunamadı: {model_path}")
        return

    # 1. Modeli Yükle
    try:
        # model = tf.keras.models.load_model(model_path) # Bu bazen özel nesnelerle sorun çıkarabilir
        # Daha güvenli yükleme için custom_objects gerekebilir, ama build_ser_model_crnn basit katmanlar kullanıyor.
        # Şimdilik doğrudan load_model deneyelim. Eğer hata verirse, file_utils.load_trained_model kullanılabilir.
        # Ancak file_utils.load_trained_model bir dizin öneki bekliyor, .h5 değil.
        # Bu yüzden en iyisi modeli tekrar build edip ağırlıkları yüklemek veya custom_objects sağlamak.
        # Basitlik için, eğer load_model hata verirse, eğitilmiş modelin yapısını build edip sadece ağırlıkları yükleyebiliriz.
        # VEYA, modelin tam yolu verildiği için, tf.keras.models.load_model yeterli olmalı.
        
        # En güvenli yöntem, modeli yeniden oluşturup ağırlıkları yüklemek olabilir,
        # özellikle custom Lambda katmanı varsa. Ancak ModelCheckpoint tüm modeli (.h5) kaydettiği için
        # tf.keras.models.load_model genellikle çalışır.
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model başarıyla yüklendi: {model_path}")
        model.summary(print_fn=logger.info)
    except Exception as e:
        logger.error(f"Model yüklenirken hata oluştu ({model_path}): {e}", exc_info=True)
        # Alternatif: Modeli yeniden build et ve ağırlıkları yükle
        # try:
        #     logger.info("load_model başarısız, modeli build edip ağırlıklar yüklenecek...")
        #     # Bu kısım için input_shape ve num_classes gibi bilgilere ihtiyaç var.
        #     # Bu bilgiler model_path\'in olduğu klasördeki bir config\'den veya doğrudan fonksiyona parametre olarak gelebilir.
        #     # Şimdilik bu kısmı atlıyoruz, load_model\'ın çalışacağını varsayıyoruz.
        #     # dummy_input_shape = (config.AUDIO_N_MFCC, config.AUDIO_MAX_FRAMES_MFCC) 
        #     # dummy_num_classes = len(config.TARGET_EMOTIONS)
        #     # built_model = models.build_ser_model_crnn(input_shape=dummy_input_shape, num_classes=dummy_num_classes)
        #     # built_model.load_weights(model_path) # .h5 dosyası tüm modeli içeriyorsa bu yanlış olur.
        #     # model = built_model
        #     # logger.info("Model build edilip ağırlıklar yüklendi.")
        # except Exception as e_build:
        #     logger.error(f"Modeli build edip ağırlıkları yüklerken de hata: {e_build}", exc_info=True)
        #     return
        return

    # 2. Test Verisini Yükle (Augmentasyon olmadan)
    logger.info("RAVDESS test verisi yükleniyor (augmentasyonsuz)...")
    ravdess_data = data_loader.load_ravdess_data(use_augmentation_on_train=False) # Augmentasyon testte kullanılmaz
    if ravdess_data is None:
        logger.error("RAVDESS test verisi yüklenemedi. Değerlendirme durduruluyor.")
        return
    (_, _), (_, _), (X_test, y_test) = ravdess_data # Sadece test setini al

    if X_test.size == 0 or y_test.size == 0:
        logger.error("RAVDESS test verisi boş. Değerlendirme durduruluyor.")
        return

    # Test verisini CRNN için gereken 4D formata getir
    if X_test.ndim == 3:
        X_test = np.expand_dims(X_test, axis=-1)
        logger.info(f"X_test değerlendirme için yeniden şekillendirildi: {X_test.shape}")

    if X_test.ndim != 4 or (X_test.ndim == 4 and X_test.shape[3] != 1):
        logger.error(f"Test verisi (X_test) beklenen 4D formatta değil. Güncel şekil: {X_test.shape}.")
        return
        
    # 3. Modeli Derle (Eğer yüklenen model derlenmemişse veya farklı bir optimizer/loss ile test etmek istiyorsak)
    # load_model zaten derlenmiş modeli yükler (optimizer state dahil).
    # Eğer sadece ağırlıklar yüklenseydi (load_weights), o zaman compile gerekirdi.
    # Emin olmak için, modelin optimizer\'ını kontrol edebiliriz.
    if not hasattr(model, 'optimizer') or model.optimizer is None:
        logger.warning("Yüklenen model derlenmemiş görünüyor. Varsayılan Adam optimizer ve categorical_crossentropy ile derleniyor.")
        # Öğrenme oranı burada çok önemli değil, sadece değerlendirme yapılacak.
        optimizer = Adam(learning_rate=config.SER_LEARNING_RATE) # Config'den alınabilir
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        logger.info(f"Model zaten derlenmiş. Optimizer: {type(model.optimizer).__name__}, Loss: {model.loss}")


    # 4. Test Seti Üzerinde Değerlendirme
    logger.info("Belirtilen SER modeli test seti üzerinde değerlendiriliyor...")
    try:
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1) # verbose=1 ile ilerlemeyi gör
        logger.info(f"--- Değerlendirme Sonuçları ({os.path.basename(model_path)}) ---")
        logger.info(f"  Test Kayıp     : {test_loss:.4f}")
        logger.info(f"  Test Doğruluk  : {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    except Exception as e:
        logger.error(f"Model değerlendirilirken hata oluştu: {e}", exc_info=True)

    logger.info(f"--- SER Model Değerlendirmesi Tamamlandı ({model_path}) ---")

# Bu dosya doğrudan çalıştırılmayacak, main_trainer.py tarafından çağrılacak.
# Ama test için if __name__ bloğu eklenebilir.
if __name__ == '__main__':
    logger.warning("Bu script (train_pipeline.py) doğrudan çalıştırılmak yerine "
                   "main_trainer.py üzerinden çağrılmak üzere tasarlanmıştır.")
    logger.info("Yine de, test amaçlı bir FER eğitimi başlatılıyor (varsayılan ayarlarla)..")
    
    # Önce config'den dizin oluşturma fonksiyonunu çağıralım
    # Normalde bu ana scriptlerde yapılır.
    try:
        file_utils.create_project_directories() # file_utils'dan çağırıyoruz
    except Exception as e: # Geniş tutalım, import hatası da olabilir
        logger.error(f"Test için proje dizinleri oluşturulamadı: {e}")
        logger.error("Lütfen önce `python -m src.utils.file_utils` çalıştırarak test edin.")


    # FER modelini varsayılan ayarlarla test et
    # Gerçek veri setlerinizin data/ altında olduğundan emin olun.
    # Bu test uzun sürebilir.
    # train_fer_model(epochs=2, batch_size=8) # Çok kısa bir test için
    train_fer_model() # epochs=1 parametresini kaldırarak orijinal haline getir
    
    # SER modelini varsayılan ayarlarla test et
    # train_ser_model(epochs=2, batch_size=4)

    # logger.info("train_pipeline.py __main__ bloğu tamamlandı.")

    # Belirli bir modeli test etmek için aşağıdaki satırları etkinleştirin:
    # specific_model_to_test = os.path.join(
    #     config.TRAINED_MODELS_PATH,
    #     "ser_model_crnn_20250602-022617", # Bir önceki eğitimdeki klasör adı
    #     "best_model_checkpoint.h5" # ModelCheckpoint'in kaydettiği dosya
    # )
    # logger.info(f"__main__ bloğunda belirtilen modeli test etme: {specific_model_to_test}")
    # evaluate_specific_ser_model_on_test_set(specific_model_to_test)
