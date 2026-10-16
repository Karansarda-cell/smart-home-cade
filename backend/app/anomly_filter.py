import numpy as np
from collections import deque


class SensorAnomalyFilter:
    def __init__(self, window_size: int = 10, z_threshold: float = 2.5):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.history = {
            "temperature": deque(maxlen=window_size),
            "humidity": deque(maxlen=window_size),
            "air_quality": deque(maxlen=window_size),
        }

    def filter_reading(self, sensor_type: str, raw_value: float) -> tuple[float, bool]:
        buf = self.history.get(sensor_type)
        if buf is None:
            return raw_value, False

        if len(buf) < 3:
            buf.append(raw_value)
            return raw_value, False

        mean = float(np.mean(buf))
        std = float(np.std(buf))

        if std == 0:
            buf.append(raw_value)
            return raw_value, False

        z_score = abs(raw_value - mean) / std

        if z_score > self.z_threshold:
            return round(mean, 2), True

        buf.append(raw_value)
        return raw_value, False