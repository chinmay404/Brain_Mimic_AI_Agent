import numpy as np
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ACCInputs:
    # Core signals
    goal_deviation: float
    prediction_error: float

    # Conflict & uncertainty
    action_entropy: float
    model_uncertainty: float

    # Cost & value
    estimated_cost: float
    expected_gain: float

    # Temporal signals
    error_trend: float          # d(error)/dt
    error_variance: float       # instability measure

    # Safety / catastrophic risk (explicit)
    risk_estimate: float        # 0–1, NOT difficulty


@dataclass
class ACCOutputs:
    control_gain: float
    strategy_shift: int         # 0=Fast, 1=Safe, 2=Exploratory
    abort_flag: bool
    suppress_reflex: bool
    cns_score: float
    risk_score: float

class AnteriorCingulateCortex:
    def __init__(
        self,
        history_window: int = 12,
        hysteresis_steps: int = 3,
    ):
        self.history_window = history_window
        self.hysteresis_steps = hysteresis_steps

        self.cns_history: List[float] = []
        self.strategy_history: List[int] = []

        # Adaptive thresholds (not magic constants)
        self.base_fast = 0.30
        self.base_safe = 0.65

    # -------------------------
    # Core computations
    # -------------------------

    def compute_difficulty(self, i: ACCInputs) -> float:
        eps = 1e-6
        roi_penalty = i.estimated_cost / (i.expected_gain + eps)

        difficulty = (
            0.35 * i.action_entropy +
            0.25 * np.clip(roi_penalty, 0, 1) +
            0.25 * np.clip(i.error_trend, 0, 1) +
            0.15 * i.model_uncertainty
        )
        return np.clip(difficulty, 0, 1)

    def compute_risk(self, i: ACCInputs) -> float:
        # Risk is NOT effort. It is irreversible damage probability.
        risk = (
            0.6 * i.risk_estimate +
            0.2 * i.error_variance +
            0.2 * i.prediction_error
        )
        return np.clip(risk, 0, 1)

    def anticipate_failure(self) -> bool:
        if len(self.cns_history) < 4:
            return False

        recent = self.cns_history[-4:]
        return np.all(np.diff(recent) > 0)  # monotonically worsening

    # -------------------------
    # Strategy logic
    # -------------------------

    def apply_hysteresis(self, proposed: int) -> int:
        self.strategy_history.append(proposed)
        if len(self.strategy_history) < self.hysteresis_steps:
            return self.strategy_history[-1]

        recent = self.strategy_history[-self.hysteresis_steps:]
        return recent[0] if all(s == recent[0] for s in recent) else self.strategy_history[-2]

    # -------------------------
    # Main loop
    # -------------------------

    def process(self, inputs: ACCInputs) -> ACCOutputs:
        difficulty = self.compute_difficulty(inputs)
        risk = self.compute_risk(inputs)

        cns = np.clip(0.7 * difficulty + 0.3 * risk, 0, 1)

        self.cns_history.append(cns)
        if len(self.cns_history) > self.history_window:
            self.cns_history.pop(0)

        # Control gain (bounded, nonlinear)
        control_gain = 1.0 + 4.0 * (cns ** 1.3)

        # --- Strategy proposal ---
        if risk > 0.85:
            proposed_strategy = 2  # Exploratory / emergency
        elif difficulty > self.base_safe:
            proposed_strategy = 1  # Safe / deliberative
        else:
            proposed_strategy = 0  # Fast / habitual

        strategy = self.apply_hysteresis(proposed_strategy)

        # --- Abort & suppression logic ---
        abort_flag = False
        suppress_reflex = False

        # If we are not in Fast mode, we must suppress reflex
        if strategy > 0:
            suppress_reflex = True

        if risk > 0.90:
            abort_flag = True
            suppress_reflex = True

        if self.anticipate_failure():
            suppress_reflex = True
            strategy = max(strategy, 1)

        return ACCOutputs(
            control_gain=control_gain,
            strategy_shift=strategy,
            abort_flag=abort_flag,
            suppress_reflex=suppress_reflex,
            cns_score=cns,
            risk_score=risk
        )
