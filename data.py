from pymavlink import mavutil
import math
import time

class DroneReader:
    def __init__(self, port='/dev/ttyACM0', baud=115200):
        self.altitude = 0.0
        self.groundspeed = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.waktu_lama = time.time()
        self.master = None
        self.last_print_time = 0  # Untuk membatasi print di terminal agar tidak spam

        try:
            self.master = mavutil.mavlink_connection(port, baud=baud)
            self.master.wait_heartbeat()
            print("Heartbeat received")
            self.master.mav.request_data_stream_send(
                self.master.target_system, 
                self.master.target_component, 
                mavutil.mavlink.MAV_DATA_STREAM_ALL, 
                20, 1
            )
            self._kalibrasi()
        except Exception as e:
            print(f"Gagal koneksi ke Drone: {e}")

    def _kalibrasi(self):
        print("Kalibrasi, jangan digerakin dulu...")
        total_x, total_y, jumlah = 0, 0, 0
        while jumlah < 100:
            msg = self.master.recv_match(type='SCALED_IMU2', blocking=True)
            if msg:
                total_x += msg.xacc
                total_y += msg.yacc
                jumlah += 1
        
        self.offset_x = total_x / jumlah
        self.offset_y = total_y / jumlah
        self.waktu_lama = time.time()
        print("Kalibrasi selesai, silakan digerakin")

    def update(self):
        """Panggil fungsi ini di dalam loop kamera"""
        if not self.master:
            return

        ada_update = False

        # Baca data tanpa memblokir kamera
        while True:
            msg = self.master.recv_match(type=['SCALED_IMU2', 'VFR_HUD'], blocking=False)
            if not msg:
                break

            ada_update = True

            if msg.get_type() == 'VFR_HUD':
                self.altitude = msg.alt

            elif msg.get_type() == 'SCALED_IMU2':
                waktu_baru = time.time()
                dt = waktu_baru - self.waktu_lama
                self.waktu_lama = waktu_baru

                ax = (msg.xacc - self.offset_x) / 1000 * 9.81
                ay = (msg.yacc - self.offset_y) / 1000 * 9.81

                if abs(ax) < 0.2: ax = 0
                if abs(ay) < 0.2: ay = 0

                self.vx = (self.vx + ax * dt) * 0.98
                self.vy = (self.vy + ay * dt) * 0.98
                self.groundspeed = math.sqrt(self.vx**2 + self.vy**2)

        # Tampilkan di terminal secara real-time (maksimal 5 kali per detik agar tidak spam)
        waktu_sekarang = time.time()
        if ada_update and (waktu_sekarang - self.last_print_time > 0.2):
            print(f"\r[DRONE] Altitude: {self.altitude:.2f} m | Groundspeed IMU: {self.groundspeed:.2f} m/s", end="")
            self.last_print_time = waktu_sekarang