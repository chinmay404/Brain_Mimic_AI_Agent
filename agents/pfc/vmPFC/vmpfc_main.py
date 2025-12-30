import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict

# -----------------------------
# 1. Vocabulary (Keyspace)
# -----------------------------
class StrategicIntent(Enum):
    PRESERVE_LIFE = auto()
    MISSION_SUCCESS = auto()
    DEESCALATE = auto()
    MINIMIZE_DAMAGE = auto()
    MAINTAIN_TRUST = auto()

@dataclass
class NeuroContext:
    threat_level: float       # 0.0-1.0
    goal_probability: float   # 0.0-1.0
    social_trust: float       # 0.0-1.0
    social_tension: float     # 0.0-1.0
    collateral_risk: float    # 0.0-1.0
    serotonin: float          # 0.0-2.0
    norepinephrine: float     # 0.0-2.0

# -----------------------------
# 2.   VMPFC
# -----------------------------
class  VMPFC:
    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        self.intent_distribution = None 

    def evaluate(self, ctx: NeuroContext) -> Dict[StrategicIntent, float]:
        raw = {
            StrategicIntent.PRESERVE_LIFE: self._life_pressure(ctx),
            StrategicIntent.DEESCALATE: self._deescalate_pressure(ctx),
            StrategicIntent.MISSION_SUCCESS: self._mission_pressure(ctx),
            StrategicIntent.MINIMIZE_DAMAGE: self._damage_pressure(ctx),
            StrategicIntent.MAINTAIN_TRUST: self._trust_pressure(ctx),
        }

        # Apply nonlinear amplification
        amplified = {k: self._spike(v) for k, v in raw.items()}

        # Suppression: reduce conflicting intents
        amplified[StrategicIntent.MISSION_SUCCESS] *= (1 - amplified[StrategicIntent.PRESERVE_LIFE])
        amplified[StrategicIntent.MINIMIZE_DAMAGE] *= (1 - amplified[StrategicIntent.MISSION_SUCCESS])

        # Normalize to 0-1
        total = sum(amplified.values()) + 1e-9
        normalized = {k: v / total for k, v in amplified.items()}

        # Apply inertia
        if self.intent_distribution is None:
            self.intent_distribution = normalized
        else:
            for intent in StrategicIntent:
                self.intent_distribution[intent] = (
                    self.alpha * self.intent_distribution[intent] +
                    (1 - self.alpha) * normalized[intent]
                )

        return self.intent_distribution

    # -----------------------------
    # Valuation functions
    # -----------------------------
    def _life_pressure(self, ctx):
        # existential threat spikes
        return 0.1 + (ctx.threat_level ** 3) * (2.5 - ctx.serotonin)

    def _deescalate_pressure(self, ctx):
        # trust still viable, high tension favors diplomacy
        return ctx.social_tension * ctx.social_trust

    def _mission_pressure(self, ctx):
        # arousal + goal probability drives mission
        return ctx.goal_probability * (0.5 + ctx.norepinephrine * 0.5)

    def _trust_pressure(self, ctx):
        # trust importance grows in crisis
        return ctx.social_trust ** 1.5

    def _damage_pressure(self, ctx):
        # collateral damage reduces mission drive
        return (1 - ctx.collateral_risk) * 0.8

    # -----------------------------
    # Nonlinear spike function
    # -----------------------------
    def _spike(self, x):
        return np.tanh(3 * x)  # compress 0-1 but amplify extremes

# -----------------------------
# 3. Demo
# -----------------------------
# ctx = NeuroContext(
#     threat_level=0.95,
#     goal_probability=0.6,
#     social_trust=0.85,
#     social_tension=0.98,
#     collateral_risk=1.0,
#     serotonin=1.8,
#     norepinephrine=1.6
# )

# vmpfc =  VMPFC(alpha=0.7)
# intent_dist = vmpfc.evaluate(ctx)

# print("---   VMPFC TRACE ---")
# for intent, weight in sorted(intent_dist.items(), key=lambda x: x[1], reverse=True):
#     print(f"{intent.name:20}: {weight:.4f}")
