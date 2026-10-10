import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyDecisionEngine:
    def __init__(self):
        self.temp = ctrl.Antecedent(np.arange(10, 45, 1), 'temperature')
        self.air_q = ctrl.Antecedent(np.arange(0, 500, 1), 'air_quality')

        self.fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')
        self.hazard_level = ctrl.Consequent(np.arange(0, 101, 1), 'hazard_level')

        self.temp['cool'] = fuzz.trimf(self.temp.universe, [10, 10, 24])
        self.temp['warm'] = fuzz.trimf(self.temp.universe, [20, 28, 34])
        self.temp['hot'] = fuzz.trimf(self.temp.universe, [30, 44, 44])

        self.air_q['clean'] = fuzz.trimf(self.air_q.universe, [0, 0, 150])
        self.air_q['moderate'] = fuzz.trimf(self.air_q.universe, [100, 250, 350])
        self.air_q['hazardous'] = fuzz.trimf(self.air_q.universe, [300, 500, 500])

        self.fan_speed['off'] = fuzz.trimf(self.fan_speed.universe, [0, 0, 20])
        self.fan_speed['medium'] = fuzz.trimf(self.fan_speed.universe, [15, 50, 75])
        self.fan_speed['high'] = fuzz.trimf(self.fan_speed.universe, [60, 100, 100])

        self.hazard_level['normal'] = fuzz.trimf(self.hazard_level.universe, [0, 0, 30])
        self.hazard_level['warning'] = fuzz.trimf(self.hazard_level.universe, [20, 50, 80])
        self.hazard_level['critical'] = fuzz.trimf(self.hazard_level.universe, [70, 100, 100])

        r1 = ctrl.Rule(self.temp['hot'], self.fan_speed['high'])
        r2 = ctrl.Rule(self.temp['warm'], self.fan_speed['medium'])
        r3 = ctrl.Rule(self.temp['cool'], self.fan_speed['off'])
        r4 = ctrl.Rule(self.air_q['hazardous'], [self.fan_speed['high'], self.hazard_level['critical']])
        r5 = ctrl.Rule(self.air_q['moderate'], self.hazard_level['warning'])

        self.control_sys = ctrl.ControlSystem([r1, r2, r3, r4, r5])
        self.simulation = ctrl.ControlSystemSimulation(self.control_sys)

    def evaluate(self, temperature: float, air_quality: float) -> dict:
        self.simulation.input['temperature'] = temperature
        self.simulation.input['air_quality'] = air_quality
        try:
            self.simulation.compute()
            return {
                "fan_speed": round(float(self.simulation.output['fan_speed']), 1),
                "hazard_level": round(float(self.simulation.output['hazard_level']), 1)
            }
        except Exception:
            return {"fan_speed": 0.0, "hazard_level": 0.0}