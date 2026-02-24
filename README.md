# Analisis Performa YOLO Untuk Emotions Detections Pada Wajah Manusia

Dashboard Streamlit untuk visualisasi dan analisis performa berbagai model YOLO (YOLOv8–YOLOv12).

---

## Instalasi

1. Clone repository ini:

```bash
git clone <repository_url>
cd <repository_folder>
```

2. Install dependency:

```bash
pip install -r requirements.txt
```

---

## Menjalankan Project

Jalankan Streamlit:

```bash
streamlit run yolo.py
```

Aplikasi akan terbuka di browser. Gunakan sidebar untuk navigasi halaman:

- 📊 **Tabel**: Menampilkan data performa model.
- 📈 **Chart**: Visualisasi akurasi, latency, training time, dsb.
- 🧪 **Demo**: Halaman eksperimen atau demo.

---

## Catatan

- Pastikan semua file CSV berada di folder `/data`.
- Sidebar dapat menampilkan logo (opsional) dari folder `/assets`.
- Semua chart sudah mendukung filter model dan sorting untuk analisis lebih fleksibel.

## TODO
- Tambahkan Halaman Untuk Dataset AffectNet
- Enhance UI/UX
