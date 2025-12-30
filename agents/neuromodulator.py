from pydantic import BaseModel, Field

class Neuromodulators(BaseModel):
    """
    System-wide chemical state of the brain.
    These values determine the 'Mood' and 'Processing Mode' of the agent.
    """
    dopamine_level: float = Field(default=0.2, ge=0.0, le=1.0, description="Motivation, Reward Prediction, Creativity")
    serotonin_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Mood Regulation, Inhibition, Safety")
    norepinephrine_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Arousal, Focus, Urgency")

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
            "INTERNAL STATE: ANXIOUS. You are under pressure. You are worried about mistakes. "
            "Double check your work. Be extremely apologetic and careful. "
            "You feel nervous."
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

def update_rpe(neuro: Neuromodulators, expected: float, actual: float) -> Neuromodulators:
    """
    Updates Dopamine based on Reward Prediction Error (RPE).
    RPE = Actual - Expected
    """
    RPE = actual - expected
    learning_rate = 0.25 
    new_dopamine = neuro.dopamine_level + (RPE * learning_rate)
    neuro.dopamine_level = max(0.0, min(1.0, new_dopamine))
    
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
