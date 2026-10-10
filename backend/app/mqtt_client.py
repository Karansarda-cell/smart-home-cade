import json
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
from app.config import settings
from app.anomaly_filter import SensorAnomalyFilter
from app.fuzzy_engine import FuzzyDecisionEngine
from app.conflict_resolver import PriorityConflictResolver

class MQTTHandler:
    def __init__(self):
        self.client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="CADE_Backend_Fedora"
        )
        self.anomaly_filter = SensorAnomalyFilter()
        self.fuzzy_engine = FuzzyDecisionEngine()
        self.system_mode = "DISARMED"

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, reason_code, properties=None):
        print(f"[MQTT] Connected successfully with Reason Code: {reason_code}")
        self.client.subscribe(settings.MQTT_TELEMETRY_TOPIC)

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            
            temp, is_temp_anomaly = self.anomaly_filter.filter_reading("temperature", payload.get("temperature", 25.0))
            air_q, is_air_anomaly = self.anomaly_filter.filter_reading("air_quality", payload.get("air_quality", 50.0))

            fuzzy_res = self.fuzzy_engine.evaluate(temperature=temp, air_quality=air_q)
            final_actions = PriorityConflictResolver.resolve(fuzzy_res, payload, self.system_mode)

            self.client.publish(settings.MQTT_COMMAND_TOPIC, json.dumps(final_actions))
            
            print(f"[CADE Evaluated] Temp: {temp}°C (Anomaly: {is_temp_anomaly}) | Fan PWM: {final_actions['relay_fan_pwm']}% | Alert: {final_actions['alert_status']}")

        except Exception as e:
            print(f"[MQTT Error] {e}")

    def start(self):
        self.client.connect(settings.MQTT_BROKER_HOST, settings.MQTT_BROKER_PORT, 60)
        self.client.loop_start()