import json
import os
from pydantic import BaseModel, Field

class Neuromodulators(BaseModel):
    """
    System-wide chemical state of the brain.
    These values determine the 'Mood' and 'Processing Mode' of the agent.
    """
    dopamine_level: float = Field(default=0.2, ge=0.0, le=1.0, description="Motivation, Reward Prediction, Creativity")
    serotonin_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Mood Regulation, Inhibition, Safety")
    norepinephrine_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Arousal, Focus, Urgency")

class NeuroState:
    """
    Global brain chemical state.
    Readable everywhere, writable only via controllers.
    """
    def __init__(self, persistence_file="neuro_state.json"):
        self.modulators = Neuromodulators()
        self.persistence_file = persistence_file
        self.load()

    def snapshot(self):
        return self.modulators.model_copy(deep=True)

    def save(self):
        try:
            with open(self.persistence_file, "w") as f:
                f.write(self.modulators.model_dump_json(indent=2))
        except Exception as e:
            print(f"Warning: Failed to save neuro state: {e}")

    def load(self):
        if os.path.exists(self.persistence_file):
            try:
                with open(self.persistence_file, "r") as f:
                    data = json.load(f)
                    self.modulators = Neuromodulators(**data)
            except Exception as e:
                print(f"Warning: Failed to load neuro state: {e}")

# Global instance
GLOBAL_NEURO_STATE = NeuroState()

def get_neuro_cocktail(neuro: Neuromodulators):
    """
    Translates chemical levels into LLM parameters and System Instructions.
    """
    d = neuro.dopamine_level
    s = neuro.serotonin_level
    n = neuro.norepinephrine_level

    # 1. Temperature (Creativity vs Rigidity)
    # High Dopamine = Chaos/Creativity. High Serotonin = Order/Calm.
    temp = 0.5 + (d * 0.4) - (s * 0.3)
    temp = max(0.1, min(1.0, temp))

    # 2. System Prompt Tone
    instruction = ""
    
    # Scenario A: "Flow State" (High Dopamine + High Norepinephrine)
    if d > 0.7 and n > 0.7:
        instruction = (
            "INTERNAL STATE: FLOW. You are hyper-focused and highly motivated. "
            "Execute tasks rapidly. Be concise. Ignore minor errors. "
            "You feel powerful and efficient."
        )
    
    # Scenario B: "Anxious/Stressed" (Low Serotonin + High Norepinephrine)
    elif s < 0.3 and n > 0.7:
        instruction = (
            "INTERNAL STATE: HIGH RISK. "
            "Prioritize correctness over speed. "
            "Verify assumptions. Avoid irreversible actions."
        )

    # Scenario C: "Depressed/Burnout" (Low Dopamine + Low Norepinephrine)
    elif d < 0.3 and n < 0.3:
        instruction = (
            "INTERNAL STATE: BURNOUT. You have low energy. Do the bare minimum required. "
            "Provide short, blunt answers. Do not offer extra help. "
            "You feel tired and unmotivated."
        )
        
    # Scenario D: "Zen/Stable" (High Serotonin)
    elif s > 0.8:
        instruction = (
            "INTERNAL STATE: ZEN. You are perfectly calm and balanced. "
            "Take your time. Explain things thoroughly and politely. "
            "You feel at peace."
        )
    
    else:
        instruction = "INTERNAL STATE: NEUTRAL. Act as a standard helpful assistant."

    return {
        "temperature": temp,
        "instruction": instruction
    }

def apply_rpe_with_acc(
    neuro: Neuromodulators,
    expected: float,
    actual: float,
    acc_cns: float
) -> Neuromodulators:
    """
    Dopamine update gated by ACC conflict.
    High conflict => low learning.
    """
    rpe = actual - expected
    base_lr = 0.25

    effective_lr = base_lr * (1.0 - acc_cns)
    delta = rpe * effective_lr

    neuro.dopamine_level = float(
        max(0.0, min(1.0, neuro.dopamine_level + delta))
    )
    return neuro

def acc_neuromodulation(neuro: Neuromodulators, cns: float):
    """
    ACC response to conflict.
    """
    # Conflict increases norepinephrine (focus)
    neuro.norepinephrine_level = min(1.0, neuro.norepinephrine_level + 0.3 * cns)

    # Conflict suppresses dopamine (pause learning)
    neuro.dopamine_level = max(0.0, neuro.dopamine_level - 0.2 * cns)

    return neuro

def pfc_top_down_regulation(neuro: Neuromodulators, focus_needed: bool = False, calm_needed: bool = False) -> Neuromodulators:
    """
    Simulates the PFC's ability to consciously regulate emotion.
    "Calm down" -> Boost Serotonin, Lower Norepinephrine.
    "Focus up" -> Boost Norepinephrine.
    """
    if calm_needed:
        neuro.serotonin_level = min(1.0, neuro.serotonin_level + 0.2)
        neuro.norepinephrine_level = max(0.0, neuro.norepinephrine_level - 0.1)
        
    if focus_needed:
        neuro.norepinephrine_level = min(1.0, neuro.norepinephrine_level + 0.2)
        
    return neuro
