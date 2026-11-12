from ultralytics import YOLO
import os
import cv2
import numpy as np

# ============================================================
# 1. SETUP
# ============================================================
MODEL_PATH = "/home/muhammad-fadlan/gamaforce_training/runs/detect/train-4/weights/last.pt"
model = YOLO(MODEL_PATH)

IMAGE_FOLDER = "./gambar"
OUTPUT_FOLDER = "./hasil_deteksi_bersih"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Parameter pembesaran kotak (Bounding Box)
# Semakin besar angkanya, kotak semakin lebar. 
# Kalau 0.2 = tambah 20% dari lebar kotak asli.
EXPAND_RATIO = 0.25 

# ============================================================
# 2. FUNGSI BANTU: GABUNGKAN KOTAK YANG TUMPANG TINDIH
# ============================================================
def gabung_kotak_yolo(boxes, overlap_thresh=0.3):
    """Menggabungkan kotak YOLO yang saling tumpang tindih"""
    if len(boxes) == 0: return []
    
    # Urutkan berdasarkan area terbesar
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    gabungan = []
    
    for box in boxes:
        cocok = False
        for g in gabungan:
            # Cek tumpang tindih (Intersection over Union sederhana)
            xA = max(box[0], g[0])
            yA = max(box[1], g[1])
            xB = min(box[2], g[2])
            yB = min(box[3], g[3])
            
            inter_area = max(0, xB - xA) * max(0, yB - yA)
            if inter_area > 0:
                cocok = True
                # Gabungkan menjadi kotak terluas
                g[0] = min(g[0], box[0])
                g[1] = min(g[1], box[1])
                g[2] = max(g[2], box[2])
                g[3] = max(g[3], box[3])
                break
        if not cocok:
            gabungan.append(list(box))
            
    return gabungan

# ============================================================
# 3. PROSES GAMBAR
# ============================================================
print(f"Memproses gambar dari: {IMAGE_FOLDER}")

results = model(IMAGE_FOLDER, imgsz=640, conf=0.25, iou=0.3, save=False, verbose=False)

for result in results:
    filename = os.path.basename(result.path)
    image = cv2.imread(result.path)
    h_img, w_img, _ = image.shape
    
    boxes = result.boxes
    if len(boxes) == 0:
        print(f"Skip (tidak ada objek): {filename}")
        continue
        
    # Ambil semua koordinat kotak [x1, y1, x2, y2]
    coords = boxes.xyxy.cpu().numpy()
    
    # 1. Gabungkan kotak yang tumpang tindih (jika model mendeteksi 5 kotak krat)
    boxes_gabung = gabung_kotak_yolo(coords, overlap_thresh=0.1)
    
    # 2. Perbesar setiap kotak
    final_boxes = []
    for x1, y1, x2, y2 in boxes_gabung:
        w = x2 - x1
        h = y2 - y1
        
        # Hitung perluasan
        dx = w * EXPAND_RATIO
        dy = h * EXPAND_RATIO
        
        # Terapkan perluasan (pastikan tidak keluar layar)
        x1_baru = max(0, int(x1 - dx))
        y1_baru = max(0, int(y1 - dy))
        x2_baru = min(w_img, int(x2 + dx))
        y2_baru = min(h_img, int(y2 + dy))
        
        final_boxes.append([x1_baru, y1_baru, x2_baru, y2_baru])
        
    # 3. Gambar ke layar
    for i, (x1, y1, x2, y2) in enumerate(final_boxes):
        # Gambar kotak hijau
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Tambahkan teks
        cv2.putText(image, f"box1 {i+1}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
    # Simpan
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), image)
    print(f"Selesai: {filename} (Terdeteksi {len(final_boxes)} objek)")

print(f"\nSelesai! Cek folder: {OUTPUT_FOLDER}")