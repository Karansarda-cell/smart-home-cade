from pathlib import Path

from pydantic_settings import BaseSettings

_APP_DIR = Path(__file__).resolve().parent
_DEFAULT_DB = _APP_DIR / "cade_audit_log.db"


class Settings(BaseSettings):
    PROJECT_NAME: str = "Context-Aware Decision Engine (CADE)"
    MQTT_BROKER_HOST: str = "localhost"
    MQTT_BROKER_PORT: int = 1883
    MQTT_TELEMETRY_TOPIC: str = "home/sensors/telemetry"
    MQTT_COMMAND_TOPIC: str = "home/actuators/commands"
    DATABASE_URL: str = f"sqlite:///{_DEFAULT_DB}"

    class Config:
        env_file = ".env"


settings = Settings()