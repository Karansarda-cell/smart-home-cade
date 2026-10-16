from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    # RAW
    raw_temperature: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    raw_air_quality: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    raw_motion: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    raw_gas: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    raw_payload: Mapped[str | None] = mapped_column(Text, nullable=True)

    # CLEANED
    filtered_temperature: Mapped[float] = mapped_column(Float, nullable=False)
    filtered_air_quality: Mapped[float] = mapped_column(Float, nullable=False)
    motion: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    gas_raw: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    temp_anomaly: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    air_anomaly: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # PHYSICAL / decision
    fuzzy_fan_speed: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    fuzzy_hazard_level: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    relay_fan_pwm: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    buzzer: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    door_lock: Mapped[str] = mapped_column(String(16), default="UNLOCKED", nullable=False)
    alert_status: Mapped[str] = mapped_column(String(64), default="NORMAL", nullable=False)
    system_mode: Mapped[str] = mapped_column(String(32), default="DISARMED", nullable=False)

    ml_label: Mapped[str | None] = mapped_column(String(32), nullable=True)
    ml_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)