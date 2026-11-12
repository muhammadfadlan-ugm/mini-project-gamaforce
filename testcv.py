import cv2
import numpy as np
import math
import time
from pymavlink import mavutil
from ultralytics import YOLO

# ============================================================
# 1. SETUP YOLO
# ============================================================
MODEL_PATH = "/home/muhammad-fadlan/gamaforce_training/gambar/dataset/runs/detect/train-4/weights/best.pt"

try:
    model = YOLO(MODEL_PATH)
    print(f"Model YOLO berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat YOLO: {e}")
    model = None

def validasi_tray(frame, box):
    """
    Fungsi untuk memvalidasi apakah objek di dalam box YOLO 
    benar-benar sebuah tray (Bentuk + Ukuran + Warna).
    """
    x1, y1, x2, y2 = box
    h_img, w_img, _ = frame.shape
    
    # Pastikan koordinat tidak keluar batas
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    w_box = x2 - x1
    h_box = y2 - y1
    
    # --- FILTER 1: UKURAN AREA ---
    # Abaikan kotak yang terlalu kecil (misal kabel/steker)
    # Threshold: misal minimal 5% dari luas layar
    area_box = w_box * h_box
    area_layar = w_img * h_img
    if (area_box / area_layar) < 0.05: 
        return False, "Terlalu Kecil"

    # --- FILTER 2: BENTUK (ASPECT RATIO) ---
    # Tray itu persegi panjang. Wajah bisa lonjong/bulat. Kabel sangat tipis.
    # Rasio Lebar/Tinggi tray biasanya antara 0.5 (agak tinggi) sampai 2.5 (sangat lebar)
    rasio = w_box / h_box
    if rasio < 0.4 or rasio > 2.5:
        return False, "Bentuk Aneh"

    # --- FILTER 3: WARNA HSV ---
    crop_img = frame[y1:y2, x1:x2]
    if crop_img.size == 0:
        return False, "Crop Gagal"

    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    # Hitung persentase area merah
    piksel_merah = cv2.countNonZero(mask)
    total_piksel = w_box * h_box
    persentase_merah = (piksel_merah / total_piksel) * 100
    
    # Tray merah harus punya minimal 30% area merah di dalamnya
    if persentase_merah < 30.0:
        return False, f"Bukan Merah ({persentase_merah:.1f}%)"

    # Jika lolos semua filter
    return True, "Valid"

def proses_deteksi_tray(frame, expand_ratio=0.10):
    if model is None:
        return frame
    
    # 1. Deteksi YOLO (Confidence diturunkan sedikit ke 0.45 agar tidak terlalu banyak false positive)
    results = model(frame, imgsz=640, conf=0.8, iou=0.4, verbose=False)[0]
    
    if len(results.boxes) == 0:
        return frame
        
    coords = results.boxes.xyxy.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()
    
    h_img, w_img, _ = frame.shape
    
    for i, (box, conf) in enumerate(zip(coords, confs)):
        x1, y1, x2, y2 = box
        
        # 2. Validasi Objek (Bentuk, Ukuran, Warna)
        is_valid, alasan = validasi_tray(frame, (x1, y1, x2, y2))
        
        if is_valid:
            # --- LOLOS FILTER: GAMBAR KOTAK HIJAU ---
            w = x2 - x1
            h = y2 - y1
            dx = int(w * expand_ratio)
            dy = int(h * expand_ratio)
            
            x1_f = max(0, x1 - dx)
            y1_f = max(0, y1 - dy)
            x2_f = min(w_img, x2 + dx)
            y2_f = min(h_img, y2 + dy)
            
            cv2.rectangle(frame, (x1_f, y1_f), (x2_f, y2_f), (0, 255, 0), 3) # Garis Hijau Tebal
            label = f"TRAY TERDETEKSI ({conf:.2f})"
            cv2.putText(frame, label, (x1_f, y1_f - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        
            
    return frame

# ============================================================
# 2. SETUP MAVLINK (DRONE)
# ============================================================
try:
    master = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200)
    master.wait_heartbeat(timeout=2)
    print("Heartbeat received")
    master.mav.request_data_stream_send(master.target_system, master.target_component, mavutil.mavlink.MAV_DATA_STREAM_ALL, 20, 1)

    print("Kalibrasi IMU, jangan digerakin dulu...")
    total_x, total_y, jumlah = 0, 0, 0
    while jumlah < 100:
        msg = master.recv_match(type='SCALED_IMU2', blocking=True)
        total_x += msg.xacc
        total_y += msg.yacc
        jumlah += 1
    offset_x = total_x / jumlah
    offset_y = total_y / jumlah
    waktu_lama = msg.time_boot_ms
    print("Kalibrasi selesai!")
except Exception as e:
    print(f"Peringatan: Gagal koneksi MAVLink ({e}). Lanjut tanpa drone.")
    master = None
    offset_x, offset_y, waktu_lama = 0, 0, 0

altitude, groundspeed, vx, vy = 0, 0, 0, 0

def parse_mavlink(master):
    global altitude, groundspeed, vx, vy, waktu_lama
    if master is None: return

    while True:
        try:
            msg = master.recv_match(type=['SCALED_IMU2', 'VFR_HUD'], blocking=False)
            if not msg: return

            if msg.get_type() == 'VFR_HUD':
                altitude = msg.alt
            elif msg.get_type() == 'SCALED_IMU2':
                dt = (msg.time_boot_ms - waktu_lama) / 1000
                waktu_lama = msg.time_boot_ms

                ax = (msg.xacc - offset_x) / 1000 * 9.81
                ay = (msg.yacc - offset_y) / 1000 * 9.81

                if abs(ax) < 0.2: ax = 0
                if abs(ay) < 0.2: ay = 0

                vx = (vx + ax * dt) * 0.98
                vy = (vy + ay * dt) * 0.98
                groundspeed = math.sqrt(vx**2 + vy**2)
        except Exception:
            pass

# ============================================================
# 3. LOOP UTAMA KAMERA
# ============================================================
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Kamera tidak terdeteksi!")
    exit()

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f"rekam_{time.strftime('%H%M')}.avi", fourcc, 20.0, (frame_width, frame_height))

print("Memulai kamera... Tekan 'q' untuk keluar.")
while True:
    ret, frame = cam.read()
    if not ret:
        break

    parse_mavlink(master)

    frame = proses_deteksi_tray(frame, expand_ratio=0.10)


    cv2.putText(frame, f"Altitude: {altitude:.2f} m", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Groundspeed: {groundspeed:.2f} m/s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out.write(frame)
    frame_kecil = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Camera', frame_kecil)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
out.release()
cv2.destroyAllWindows()