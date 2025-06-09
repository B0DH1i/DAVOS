---
title: DAVOS - Duygu Analizi ve Verimlilik Otomasyon Sistemi
emoji: 🧠🎧
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.31.0
app_file: duygu_verimlilik_projesi/gui_main.py
pinned: false
---

# DAVOS - Duygu Analizi ve Verimlilik Otomasyon Sistemi

Bu proje, canlı kamera ve mikrofon girdilerini kullanarak yüz ve ses ifadelerinden duygu analizi yapar. Tespit edilen duygu durumuna göre proaktif olarak rahatlatıcı veya motive edici müzik/binaural ritimler çalarak kullanıcının verimliliğini ve ruh halini iyileştirmeyi hedefler.

## 🚀 Temel Özellikler

*   **Çoklu Model Entegrasyonu:**
    *   Yüz Tespiti (DNN Modeli)
    *   Yüz İfadesinden Duygu Tanıma (FER - VGG16)
    *   Sesten Duygu Tanıma (SER - Whisper Tabanlı)
*   **Proaktif Müdahale:** Tespit edilen duyguya (örn: stresli, yorgun) göre `Lazanov Müzik Kütüphanesi`'nden uygun sesleri otomatik olarak çalar.
*   **İki Farklı Çalışma Modu:**
    1.  `gui_main.py`: İnteraktif bir Gradio arayüzü ile anlık test ve analiz.
    2.  `main_live_controller.py`: Arka planda sürekli çalışan tam otomatik analiz ve müdahale sistemi.

## 🛠️ Kurulum ve Çalıştırma (Hugging Face)

Bu Space, GitHub deposundan otomatik olarak oluşturulmuştur. Gerekli tüm Python kütüphaneleri `requirements.txt` ve sistem bağımlılıkları (`ffmpeg`) `packages.txt` dosyaları aracılığıyla yüklenir.

Uygulamanın çalışması için herhangi bir ek adım gerekmemektedir. Gradio arayüzünün yüklenmesini beklemeniz yeterlidir. 