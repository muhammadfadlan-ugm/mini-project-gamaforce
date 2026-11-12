import cv2
import time
from data import DroneReader

# 1. Setup Drone (dari data.py)
drone = DroneReader('/dev/ttyACM0', 115200)

# 2. Setup Kamera
cam = cv2.VideoCapture(0)
w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(f"rekam_{time.strftime('%H%M')}.avi", cv2.VideoWriter_fourcc(*'XVID'), 20.0, (w, h))

print("\nKamera siap. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cam.read()
    if not ret: break

    # A. Update Data Drone (print ke terminal)
    drone.update()

    # B. Deteksi Warna Merah (HSV)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Rentang warna merah (Merah terang & Merah gelap)
    lower_red1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
    lower_red2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    mask = cv2.bitwise_or(lower_red1, lower_red2)

    # Hilangkan noise (bercak kecil)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Cari kontur (garis luar bentuk merah)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Abaikan objek yang terlalu kecil
        if cv2.contourArea(cnt) < 1000:
            continue

        # Gambar garis luar (mengikuti bentuk objek, tidak kaku)
        # Garis tipis kuning (0, 255, 255)
        cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 2)

        # Gambar Bounding Box (Kotak Hijau)
        x, y, bw, bh = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        
        # Tampilkan teks
        cv2.putText(frame, "Merah", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # C. Tampilkan UI Telemetri
    cv2.putText(frame, f"Alt: {drone.altitude:.1f}m | Speed: {drone.groundspeed:.1f}m/s", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # D. Rekam & Tampilkan Kamera
    out.write(frame)
    cv2.imshow('Deteksi Merah', cv2.resize(frame, None, fx=0.5, fy=0.5))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
out.release()
cv2.destroyAllWindows()