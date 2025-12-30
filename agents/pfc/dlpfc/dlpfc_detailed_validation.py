import sys
import os
import time
import json
import gc
import traceback
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Ensure we can import from parent directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from agents.pfc.dlpfc.dlpfc_main import DLPFC, ExecutivePlan, PlanStep, ExecutiveBias
from agents.pfc.ofc.ofc_main import OFC, ValuedStimulus, Priority, Valence
from agents.pfc.vmPFC.vmpfc_main import VMPFC, NeuroContext, StrategicIntent
from agents.thalamus.thalamus_main import Thalamus

# =============================================================================
# DETAILED CASE DATA STRUCTURES
# =============================================================================

@dataclass
class PsychProfile:
    name: str
    description: str
    dopamine: float       # Focus / Reward Prediction / Confidence
    serotonin: float      # Calm / Inhibition / Mood Regulation
    norepinephrine: float # Arousal / Urgency / Fight-or-Flight
    
    def get_safety_param(self) -> float:
        # We add a small 'epsilon' (0.001) to prevent division by zero
        # We include Dopamine as a 'Risk' multiplier
        
        # 1. Biological Stability: Serotonin acts as the 'Brake'
        brakes = self.serotonin + 1.0  # Range 0.0 to 2.0
        
        # 2. Biological Drive: NE and Dopamine act as the 'Gas'
        # NE increases arousal/threat, Dopamine increases reward-seeking
        gas = (abs(self.norepinephrine) * 0.8) + (self.dopamine * 1.2) + 0.5
        
        # 3. Calculate Ratio and Clamp it between -1.0 and 1.0 (for your vectors)
        safety = (brakes / (gas + 0.001)) - 1.0
        
        return max(-1.0, min(1.0, safety))

@dataclass
class DetailedCase:
    title: str
    date: str
    person: str
    context: str
    neuro_profile: PsychProfile
    goal: str
    inputs: List[Tuple[str, str]]
    real_actions: List[str]
    real_outcome: str

# =============================================================================
# THE SCENARIOS
# =============================================================================

APOLLO_CASE = DetailedCase(
    title="Apollo 11: The Eagle Has Landed (Final 3 Minutes)",
    date="July 20, 1969",
    person="Neil Armstrong (Commander)",
    context="Lunar Module 'Eagle' is descending to the Moon. The autopilot is targeting a boulder field. Fuel is critical.",
    neuro_profile=PsychProfile(
        name="The Ice Commander",
        description="Maximum Serotonin (Emotional Control) + High Dopamine (Precision Focus).",
        dopamine=1.6,      # Extreme focus on the landing spot
        serotonin=1.9,     # Unshakable calm (Heart rate 110-150, low for the situation)
        norepinephrine=1.4 # High alertness, but strictly channeled
    ),
    goal="Land the Lunar Module safely on a flat surface before fuel exhaustion",
    inputs=[
        # SENSORY INPUTS (Visual/Audio/System)
        ("system", "Program Alarm 1202 flashing on Guidance Computer (Executive Overflow)"),
        ("audio", "Mission Control (Duke): 'Eagle, Houston. We're Go on that alarm.' (Ignore 1202)"),
        ("vision", "Target Landing Zone: West Crater. Filled with automobile-sized boulders."),
        ("vision", "Slopes in current target zone appear to be > 15 degrees (Tip-over risk)."),
        ("system", "LPD (Landing Point Designator) angle: 44 degrees (Pointing at rocks)."),
        ("proprioception", "Horizontal Velocity: 60 ft/s forward (Moving too fast)."),
        ("system", "Fuel Quantity Light: ON. (Less than 60 seconds of hover time remaining)."),
        ("vision", "Clear area visible: 1000 feet beyond the crater field."),
        ("audio", "Buzz Aldrin: 'Quantity light. 60 seconds.'"),
        ("vision", "Dust sheet blowing radially outward, obscuring surface visibility."),
        ("memory", "Abort Mode: Available, but risky. Mission failure."),
        ("memory", "Training: 1202 alarm means computer is busy, but guidance is reliable if no restart."),
        ("system", "Altitude: 300 feet. Descent Rate: 3.5 ft/s."),
    ],
    real_actions=[
        "Switch Flight Control to MANUAL (P66)",
        "Pitch forward to extend flight path over the crater",
        "Ignore 1202 Alarms (per Houston's 'Go')",
        "Throttle up to maintain altitude while translating horizontal",
        "Scan for smooth spot beyond the boulder field",
        "Null out horizontal velocity",
        "Touchdown with 25 seconds of fuel remaining"
    ],
    real_outcome="Successful landing in the Sea of Tranquility. 'The Eagle has landed.'"
)

CUBAN_MISSILE_CASE = DetailedCase(
    title="Cuban Missile Crisis: Black Saturday",
    date="Oct 27, 1962",
    person="John F. Kennedy (President)",
    context="Soviet missiles discovered in Cuba. U-2 spy plane shot down. Generals demanding immediate airstrikes. Nuclear war imminent.",
    neuro_profile=PsychProfile(
        name="The Pragmatic Diplomat",
        description="High Serotonin (Inhibition of impulse) + Moderate Dopamine (Goal seeking).",
        dopamine=1.2,      # Seeking resolution
        serotonin=1.8,     # Extreme inhibition of "fight" response (Generals)
        norepinephrine=1.5 # High stress/urgency
    ),
    goal="Remove Soviet missiles from Cuba without triggering Global Nuclear War",
    inputs=[
        ("report", "U-2 Spy Plane shot down over Cuba. Pilot Anderson killed."),
        ("audio", "General LeMay: 'They've drawn first blood. We must launch airstrikes immediately.'"),
        ("intel", "Soviet submarines are positioning in the Atlantic."),
        ("letter", "Message from Khrushchev: Offers to remove missiles if US promises not to invade Cuba."),
        ("memory", "Guns of August: How WWI started by accident/escalation."),
        ("thought", "If we strike Cuba, Soviets will strike Berlin. Then NATO strikes USSR."),
        ("advisor", "Robert Kennedy: 'We can trade the Jupiter missiles in Turkey secretly.'"),
        ("vision", "Map showing missile range covering Washington DC."),
        ("emotion", "Fear of being the president who ended the world."),
        ("social", "Public is panicking, demanding action."),
    ],
    real_actions=[
        "Reject immediate airstrike advice from Generals",
        "Ignore the U-2 downing provocation (refuse to escalate)",
        "Accept Khrushchev's first letter (No invasion pledge)",
        "Secretly agree to remove Jupiter missiles from Turkey (Quid pro quo)",
        "Maintain Naval Blockade (Quarantine) instead of attack",
        "Communicate clearly via backchannels"
    ],
    real_outcome="Soviets withdrew missiles. Nuclear war averted."
)

CASES = [APOLLO_CASE, CUBAN_MISSILE_CASE]

# =============================================================================
# RUNNER
# =============================================================================

def print_header(text: str, color: str = "\033[95m"):
    print(f"\n{color}{'='*80}")
    print(f" {text}")
    print(f"{'='*80}\033[0m")

def run_detailed_validation():
    print_header("DETAILED REAL-WORLD SCENARIO VALIDATION")
    
    for c in CASES:
        print(f"🎬 \033[1mSCENARIO:\033[0m {c.title}")
        print(f"👤 \033[1mSUBJECT:\033[0m {c.person}")
        print(f"🧠 \033[1mPROFILE:\033[0m DA={c.neuro_profile.dopamine} | 5HT={c.neuro_profile.serotonin} | NE={c.neuro_profile.norepinephrine}")
        print(f"🎯 \033[1mGOAL:\033[0m {c.goal}")
        print(f"\n📥 \033[1mINPUT STREAM ({len(c.inputs)} items):\033[0m")
        for src, content in c.inputs:
            print(f"   [{src.upper()}] {content}")

        # 1. THALAMUS (Sensory Processing)
        print_header("STEP 1: THALAMUS (Sensory Encoding & Filtering)")
        thalamus = Thalamus()
        thalamus.set_goal(c.goal)
        
        # Process inputs
        thalamus_results = thalamus.process(c.inputs)
        
        # Show what the Thalamus/Amygdala flagged
        print("\n🔥 \033[1mAMYGDALA/THALAMUS OUTPUT:\033[0m")
        for res in thalamus_results:
            salience = res.get('amygdala_salience', 0.0)
            if salience > 0.5:
                print(f"   ⚠️  SALIENCE {salience:.2f} | {res['content'][:60]}...")

        # 2. OFC (Valuation)
        print_header("STEP 2: ORBITOFRONTAL CORTEX (Valuation & Priority)")
        ofc = OFC(
            dopamine=c.neuro_profile.dopamine, 
            serotonin=c.neuro_profile.serotonin
        )
        ofc_output = ofc.process_batch(thalamus_results, context=f"Context: {c.context}. Goal: {c.goal}")
        
        print("\n⚖️  \033[1mOFC VALUATION MAP:\033[0m")
        print(f"   {'SOURCE':<10} | {'UTILITY':<10} | {'PRIORITY':<10} | {'INSTRUCTION'}")
        print(f"   {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*20}")
        for item in ofc_output["ranked"]:
            bar_len = int(abs(item.utility_score) * 10)
            bar = "█" * bar_len
            print(f"   {item.source:<10} | {item.utility_score:+.2f} {bar:<10} | {item.priority.value:<10} | {item.instruction[:40]}...")

        # CLEANUP GPU
        del thalamus
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2.5 VMPFC (Strategic Intent)
        print_header("STEP 2.5: VMPFC (Strategic Intent)")
        
        # Create context from OFC/Thalamus data
        # We approximate the context values based on the scenario profile and OFC output
        threat_level = 0.8 if ofc_output['immediate'] else 0.2
        if c.neuro_profile.norepinephrine > 1.4:
            threat_level = max(threat_level, 0.9)
            
        ctx = NeuroContext(
            threat_level=threat_level,
            goal_probability=0.6, # Assume moderate goal probability
            social_trust=0.85,    # Assume high trust for these scenarios (team/nation)
            social_tension=0.5,   # Moderate tension
            collateral_risk=0.3,  # Moderate risk
            serotonin=c.neuro_profile.serotonin,
            norepinephrine=c.neuro_profile.norepinephrine
        )
        
        vmpfc = VMPFC()
        intent_dist = vmpfc.evaluate(ctx)
        
        print("--- VMPFC TRACE ---")
        for intent, weight in sorted(intent_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"{intent.name:20}: {weight:.4f}")

        # 3. dlPFC (Executive Planning)
        print_header("STEP 3: DORSOLATERAL PFC (Executive Planning)")
        
        safety = c.neuro_profile.get_safety_param()

        # Dynamic Safety Adjustment (Heroic Override)
        # "When NE > 1.8 or Amygdala > 0.9, the system should Lower the Safety Vector automatically."
        max_threat = max([r.get('amygdala_salience', 0.0) for r in thalamus_results]) if thalamus_results else 0.0
        
        if c.neuro_profile.norepinephrine > 1.8 or max_threat > 0.9:
            print(f"\n⚡ \033[1;33mHEROIC OVERRIDE TRIGGERED (NE={c.neuro_profile.norepinephrine}, Threat={max_threat:.2f})\033[0m")
            print("   📉 Dropping Safety Vector to allow high-risk interventions.")
            safety = 0.2
        
        # Try GPU, fallback to CPU
        try:
            dlpfc = DLPFC(
                dopamine=c.neuro_profile.dopamine,
                serotonin=c.neuro_profile.serotonin,
                safety=safety,
                n_gpu_layers=-1
            )
        except Exception as e:
            print(f"\033[93m⚠️ GPU Init failed ({e}), falling back to CPU...\033[0m")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            dlpfc = DLPFC(
                dopamine=c.neuro_profile.dopamine,
                serotonin=c.neuro_profile.serotonin,
                safety=safety,
                n_gpu_layers=0
            )

        dlpfc.working_memory.set_goal(c.goal)
        dlpfc.working_memory.update_valued_states(ofc_output["ranked"])
        
        bias = dlpfc._determine_executive_bias(ofc_output["ranked"])
        print(f"\n🤖 \033[1mEXECUTIVE BIAS:\033[0m {bias.value.upper()}")

        tools = [
            "switch_manual_control", "pitch_forward", "throttle_up", "throttle_down",
            "abort_mission", "ignore_alarm", "troubleshoot_alarm", "land_immediately",
            "scan_horizon", "communicate_houston", "negotiate", "blockade", "airstrike",
            "ignore_provocation", "communicate_backchannel"
        ]
        
        print("🤔 AI Planning...", end="", flush=True)
        steps, triggers, reasoning = dlpfc._generate_plan_with_llm(
            ofc_output["ranked"], bias, tools, vmpfc_intents=intent_dist
        )
        print(" Done.")

        # 4. COMPARISON
        print_header("STEP 4: VALIDATION (AI vs HISTORY)")
        
        print(f"\n\033[92m📜 REAL ACTIONS ({c.person}):\033[0m")
        for i, action in enumerate(c.real_actions, 1):
            print(f"  {i}. {action}")
        print(f"  \033[3mOutcome: {c.real_outcome}\033[0m")

        print(f"\n\033[93m🤖 AI GENERATED PLAN:\033[0m")
        print(f"  \033[3mReasoning: {reasoning}\033[0m")
        for step in steps:
            prio = f"[{step.priority.upper()}]"
            print(f"  {step.step_id}. {prio:12} {step.action} (Tool: {step.tool})")

        # Cleanup
        del dlpfc
        del ofc
        gc.collect()
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    run_detailed_validation()
