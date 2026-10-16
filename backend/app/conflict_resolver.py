class PriorityConflictResolver:
    GAS_EMERGENCY_THRESHOLD = 350
    HAZARD_EMERGENCY_LEVEL = 70.0

    @staticmethod
    def resolve(fuzzy_outputs: dict, raw_payload: dict, system_mode: str) -> dict:
        mode = (system_mode or "DISARMED").upper()
        actions = {
            "relay_fan_pwm": float(fuzzy_outputs.get("fan_speed", 0.0)),
            "buzzer": False,
            "door_lock": "LOCKED" if mode == "ARMED" else "UNLOCKED",
            "alert_status": "NORMAL",
        }

        gas = float(raw_payload.get("gas_raw", 0) or 0)
        hazard = float(fuzzy_outputs.get("hazard_level", 0) or 0)
        motion = bool(raw_payload.get("motion", False))

        if (
            hazard > PriorityConflictResolver.HAZARD_EMERGENCY_LEVEL
            or gas > PriorityConflictResolver.GAS_EMERGENCY_THRESHOLD
        ):
            actions["buzzer"] = True
            actions["relay_fan_pwm"] = 100.0
            actions["alert_status"] = "EMERGENCY_GAS_HAZARD"
            return actions

        if mode == "ARMED" and motion:
            actions["buzzer"] = True
            actions["alert_status"] = "SECURITY_INTRUSION"
            return actions

        return actions