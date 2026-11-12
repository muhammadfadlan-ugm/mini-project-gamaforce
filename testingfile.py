import cv2
import numpy as np
from ultralytics import YOLO
import os

# ============================================================
# 1. SETUP YOLO
# ============================================================
# PASTIKAN PATH INI BENAR
MODEL_PATH = "/home/muhammad-fadlan/gamaforce_training/gambar/dataset/runs/detect/train-4/weights/best.pt"

# PASTIKAN FILE GAMBAR INI ADA
# Ganti dengan path gambar yang ingin Anda uji
IMAGE_PATH = "/home/muhammad-fadlan/gamaforce_training/gambar/008.jpg" 

try:
    model = YOLO(MODEL_PATH)
    print(f"Model YOLO berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat YOLO: {e}")
    exit()

def validasi_tray(frame, box):
    """Filter: Ukuran + Bentuk + Warna HSV"""
    x1, y1, x2, y2 = box
    h_img, w_img, _ = frame.shape
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    w_box = x2 - x1
    h_box = y2 - y1
    
    # 1. Filter Ukuran (Minimal 5% area layar)
    area_box = w_box * h_box
    area_layar = w_img * h_img
    if (area_box / area_layar) < 0.05: 
        return False, "Terlalu Kecil"

    # 2. Filter Bentuk (Rasio Lebar/Tinggi)
    rasio = w_box / h_box
    if rasio < 0.4 or rasio > 2.5:
        return False, "Bentuk Aneh"

    # 3. Filter Warna HSV
    crop_img = frame[y1:y2, x1:x2]
    if crop_img.size == 0:
        return False, "Crop Gagal"

    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    
    # Range warna merah
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    piksel_merah = cv2.countNonZero(mask)
    total_piksel = w_box * h_box
    persentase_merah = (piksel_merah / total_piksel) * 100
    
    # Tray merah harus punya minimal 30% area merah
    if persentase_merah < 30.0:
        return False, f"Bukan Merah ({persentase_merah:.1f}%)"

    return True, "Valid"

# ============================================================
# 2. PROSES GAMBAR
# ============================================================
print(f"Membaca gambar: {IMAGE_PATH}")
frame = cv2.imread(IMAGE_PATH)

if frame is None:
    print("Error: Gambar tidak ditemukan! Periksa kembali IMAGE_PATH Anda.")
    exit()

# Deteksi YOLO
print("Menjalankan deteksi YOLO...")
results = model(frame, imgsz=640, conf=0.8, iou=0.4, verbose=False)[0]

if len(results.boxes) == 0:
    print("YOLO tidak mendeteksi objek apapun di gambar ini.")
else:
    print(f"YOLO mendeteksi {len(results.boxes)} objek potensial. Memfilter...")
    
    coords = results.boxes.xyxy.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()
    
    h_img, w_img, _ = frame.shape
    tray_terdeteksi = 0

    for i, (box, conf) in enumerate(zip(coords, confs)):
        x1, y1, x2, y2 = box
        
        # Validasi
        is_valid, alasan = validasi_tray(frame, (x1, y1, x2, y2))
        
        if is_valid:
            tray_terdeteksi += 1
            # Gambar Kotak Hijau (Valid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"TRAY VALID ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"  -> Objek {i+1}: VALID (Confidence: {conf:.2f})")
        else:
            # Gambar Kotak Merah (Ditolak)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(frame, f"Tolak: {alasan}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            print(f"  -> Objek {i+1}: DITOLAK ({alasan})")

    print(f"\nSelesai! Total Tray Valid yang ditemukan: {tray_terdeteksi}")

# ============================================================
# 3. SIMPAN & TAMPILKAN HASIL
# ============================================================
# output_path = "hasil_deteksi_tray.jpg"
# cv2.imwrite(output_path, frame)
# print(f"Gambar hasil deteksi disimpan sebagai: {output_path}")

# Tampilkan di layar (tekan tombol apapun untuk menutup)
cv2.imshow("Hasil Deteksi", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()