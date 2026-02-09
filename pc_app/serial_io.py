import time
import serial


class SerialSender:
    def __init__(self, port, baud, rate_hz):
        self.port = port
        self.baud = baud
        self.rate_hz = rate_hz
        self._ser = None
        self._last_send = 0.0

    def open(self):
        self._ser = serial.Serial(self.port, self.baud, timeout=0.1)
        time.sleep(0.2)

    def close(self):
        if self._ser:
            self._ser.close()
            self._ser = None

    def send_angles(self, yaw_deg, pitch_deg):
        now = time.time()
        if self.rate_hz > 0:
            if now - self._last_send < (1.0 / self.rate_hz):
                return False
        payload = f"{yaw_deg:.2f},{pitch_deg:.2f}\n"
        self._ser.write(payload.encode("ascii"))
        self._last_send = now
        return True
