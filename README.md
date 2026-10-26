# 🛡️ DotGuard - Standalone Threat & NSFW Analysis Tool

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-4D4D4D)](https://github.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**DotGuard**, Twitch, Kick, Discord, Telegram ve YouTube gibi platformlarda metin filtrelerini (Automod, Nightbot, OpenAI Moderation vb.) atlatmak için kullanılan **Unicode Braille sanatı (`⣿⣟...`), ASCII çizimleri ve doğrudan resim/fotoğraf dosyalarını** analiz eden bağımsız bir güvenlik ve moderasyon aracıdır.

Herhangi bir canlı yayın botuna **bağlı olmadan**, kendi başına terminalden metin dosyaları (`.txt`), doğrudan fotoğraflar (`.jpg`, `.png`) veya kopyalanıp yapıştırılan chat mesajlarıyla anında çalışır.

---

## ⚡ Problem ve Çözüm

1. **Metin Filtrelerinin (NLP) Kör Noktası:** NLP botları kelime bazlı okur. Karakterler birleşip yasaklı bir silah, gamalı haç veya hakaret çizdiğinde standart botlar bunu sadece anlamsız semboller dizisi olarak görür.
2. **Görsel Modellerin Kör Noktası:** Klasik görsel yapay zeka modelleri (YOLO, ResNet) dosya (`.jpg`, `.png`) bekler; chatteki ham metin karakterlerini göremez.
3. **DotGuard Çözümü:** 
   - Metin karakterlerini matematiksel **ikili piksel matrisine (Rasterization)** dönüştürür.
   - **Morfolojik görüntü işleme (Dilation & Closing)** ile noktaları birleştirip katı silüetler oluşturur.
   - 4 katlı rotasyonel simetri, açıdan bağımsız uzama oranları (`MinAreaRect`), steganografi ve NSFW ten segmentasyonu ile tehdit analizini milisaniyeler içinde tamamlar.

---

## 🚀 Hızlı Başlangıç

### 1. Kurulum
```bash
git clone https://github.com/kullanici_adiniz/dotguard.git
cd dotguard
pip install -r requirements.txt
```

*(Opsiyonel)* Sistemi küresel bir terminal komutu olarak kurmak için:
```bash
pip install -e .
# Artık doğrudan 'dotguard' komutunu her yerden çağırabilirsiniz!
```

---

## 💻 Kullanım Yöntemleri

### Yöntem 1: İnteraktif Menü (En Kolayı)
Hiçbir parametre yazmadan doğrudan çalıştırabilirsiniz:
```bash
python cli.py
```
Açılan menüden:
- `[1]` Çok satırlı chat mesajını doğrudan terminale yapıştırabilirsiniz.
- `[2]` Bir `.txt` dosyasının yolunu girebilir veya dosyayı terminale sürükleyip bırakabilirsiniz.
- `[3]` Bir resim/fotoğraf dosyasını (`.jpg`, `.png`) tarayabilirsiniz.
- `[4]` Hazır demo örneklerini çalıştırabilirsiniz.

---

### Yöntem 2: Tek Komutla Otomatik Tarama (Scan)
Dosya uzantısına (`.txt` veya `.jpg`/`.png`) göre otomatik analiz yapar:

```bash
# Metin / Braille dosyasını tara:
python cli.py scan samples/swastika_braille.txt

# Bıçak / Silah silüeti içeren metni tara:
python cli.py scan samples/blade_silhouette.txt

# Doğrudan fotoğraf tara ve görsel rapor kaydet:
python cli.py scan samples/test_blade.jpg --visual sonuc.png

# Çıktıyı botlar için JSON formatında al:
python cli.py scan samples/test_nsfw.jpg --json
```

---

### Yöntem 3: Ham Mesaj Dizesini Tarama
```bash
python cli.py scan-text "⣿⣿⣿⣿⠀⠀⠀⠀⠀⣿⣿⣿⣿..." --visual rapor.png
```

---

### Yöntem 4: Linux / PowerShell Boru Hattı (Pipe / Stdin)
```bash
# Linux / macOS:
cat suspicious.txt | python cli.py scan -

# Windows PowerShell:
$OutputEncoding = [System.Text.Encoding]::UTF8; Get-Content samples/swastika_braille.txt -Raw | python cli.py scan -
```

---

## 📊 Örnek Çıktı Raporu

```text
================================================================
 [*] THREAT ANALYSIS REPORT
================================================================
 Input Type       : BRAILLE
 Resolution       : 274 x 258 px
 Risk Assessment  : CRITICAL
 Threat Score     : 97.5 / 100
 Contours Scanned : 1
----------------------------------------------------------------
 [!] DETECTED VIOLATIONS & THREATS (1):
   1. [CRITICAL] Swastika / Rotational Cross Pattern
      Confidence: 75.0% | Location: (0, 0, 274, 258)
================================================================
```

---

## 📂 Proje Yapısı

```text
dotguard/
├── cli.py                  # İnteraktif ve argümanlı CLI arayüzü
├── requirements.txt        # Gerekli bağımlılıklar (OpenCV, NumPy, Pillow)
├── pyproject.toml          # Standart Python paketleme yapılandırması
├── setup.py                # Pip kurulum dosyası
├── LICENSE                 # MIT Lisansı
├── dotguard/               # Çekirdek Python paketi
│   ├── __init__.py
│   ├── analyzer.py         # Braille & ASCII analiz orkestratörü
│   ├── image_analyzer.py   # Doğrudan resim/fotoğraf analiz motoru
│   ├── rasterizer.py       # Metinden piksele dönüştürücü (Bit-matrix mapping)
│   ├── preprocessor.py     # Morfoloji, kontur analizi ve MinAreaRect
│   ├── visualizer.py       # 3 panelli görsel rapor oluşturucu (.png)
│   └── detectors/
│       ├── geometry.py     # Rotasyonel simetri, gamalı haç, bıçak/kesici alet
│       ├── nsfw_rules.py   # Anatomik lob ve dışbükeylik analizi
│       ├── stego.py        # Görünmez Zero-Width ve bitstream steganografi
│       ├── image_nsfw.py   # Çift renk uzayı (YCrCb + HSV) ten segmentasyonu
│       ├── image_weapon.py # OpenCV DNN YOLO + geometrik namlu/bıçak analizi
│       └── image_text_threat.py # Türkçe & İngilizce tehdit sözlüğü ve OCR
├── samples/                # Test için hazır örnek dosyalar (.txt ve .jpg)
└── tests/
    ├── test_engine.py      # Otomatik birim testleri (Unittest)
    └── test_samples.py     # Sentetik tehdit modelleri kütüphanesi
```

---

## 🧪 Birim Testleri Çalıştırma

Tüm testleri saniyeler içinde çalıştırmak için:
```bash
python -m unittest discover tests
```

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır. Açık kaynaklı ve ticari projelerde özgürce kullanılabilir.
