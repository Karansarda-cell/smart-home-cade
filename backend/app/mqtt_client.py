import json

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

from app.anomaly_filter import SensorAnomalyFilter
from app.config import settings
from app.conflict_resolver import PriorityConflictResolver
from app.database import SessionLocal
from app.fuzzy_engine import FuzzyDecisionEngine
from app.ml_model import hazard_model
from app.models import AuditLog


class MQTTHandler:
    def __init__(self):
        self.client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="CADE_Backend_Fedora",
        )
        self.anomaly_filter = SensorAnomalyFilter()
        self.fuzzy_engine = FuzzyDecisionEngine()
        self.system_mode = "DISARMED"

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, reason_code, properties=None):
        print(f"[MQTT] Connected (reason_code={reason_code})")
        self.client.subscribe(settings.MQTT_TELEMETRY_TOPIC)

    def on_message(self, client, userdata, msg):
        try:
            raw_text = msg.payload.decode("utf-8")
            payload = json.loads(raw_text)

            raw_temp = float(payload.get("temperature", 25.0))
            raw_air = float(payload.get("air_quality", 50.0))
            raw_motion = bool(payload.get("motion", False))
            raw_gas = int(payload.get("gas_raw", 0) or 0)

            temp, is_temp_anomaly = self.anomaly_filter.filter_reading(
                "temperature", raw_temp
            )
            air_q, is_air_anomaly = self.anomaly_filter.filter_reading(
                "air_quality", raw_air
            )

            fuzzy_res = self.fuzzy_engine.evaluate(temperature=temp, air_quality=air_q)
            final_actions = PriorityConflictResolver.resolve(
                fuzzy_res, payload, self.system_mode
            )

            ml_label = None
            ml_confidence = None
            if hazard_model.is_loaded:
                try:
                    ml_out = hazard_model.predict(
                        temperature=temp,
                        air_quality=air_q,
                        motion=raw_motion,
                        gas_raw=raw_gas,
                    )
                    ml_label = ml_out["prediction_label"]
                    ml_confidence = ml_out.get("confidence")
                except Exception as e:
                    print(f"[MQTT] ML predict skipped: {e}")

            self.client.publish(
                settings.MQTT_COMMAND_TOPIC, json.dumps(final_actions)
            )

            self._save_event(
                raw_temp=raw_temp,
                raw_air=raw_air,
                raw_motion=raw_motion,
                raw_gas=raw_gas,
                raw_payload=raw_text,
                temp=temp,
                air_q=air_q,
                is_temp_anomaly=is_temp_anomaly,
                is_air_anomaly=is_air_anomaly,
                fuzzy_res=fuzzy_res,
                final_actions=final_actions,
                ml_label=ml_label,
                ml_confidence=ml_confidence,
            )

            print(
                f"[CADE] T={temp}°C (anom={is_temp_anomaly}) | "
                f"Fan={final_actions['relay_fan_pwm']}% | "
                f"Alert={final_actions['alert_status']} | ML={ml_label}"
            )

        except Exception as e:
            print(f"[MQTT Error] {e}")

    def _save_event(
        self,
        *,
        raw_temp,
        raw_air,
        raw_motion,
        raw_gas,
        raw_payload,
        temp,
        air_q,
        is_temp_anomaly,
        is_air_anomaly,
        fuzzy_res,
        final_actions,
        ml_label,
        ml_confidence,
    ):
        db = SessionLocal()
        try:
            row = AuditLog(
                raw_temperature=raw_temp,
                raw_air_quality=raw_air,
                raw_motion=raw_motion,
                raw_gas=raw_gas,
                raw_payload=raw_payload,
                filtered_temperature=temp,
                filtered_air_quality=air_q,
                motion=raw_motion,
                gas_raw=raw_gas,
                temp_anomaly=is_temp_anomaly,
                air_anomaly=is_air_anomaly,
                fuzzy_fan_speed=float(fuzzy_res.get("fan_speed", 0.0)),
                fuzzy_hazard_level=float(fuzzy_res.get("hazard_level", 0.0)),
                relay_fan_pwm=float(final_actions.get("relay_fan_pwm", 0.0)),
                buzzer=bool(final_actions.get("buzzer", False)),
                door_lock=str(final_actions.get("door_lock", "UNLOCKED")),
                alert_status=str(final_actions.get("alert_status", "NORMAL")),
                system_mode=self.system_mode,
                ml_label=ml_label,
                ml_confidence=ml_confidence,
            )
            db.add(row)
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"[DB] Failed to save event: {e}")
        finally:
            db.close()

    def start(self):
        self.client.connect(
            settings.MQTT_BROKER_HOST, settings.MQTT_BROKER_PORT, 60
        )
        self.client.loop_start()