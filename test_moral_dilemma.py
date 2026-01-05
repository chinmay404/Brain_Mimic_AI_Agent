import sys
import os
from pathlib import Path

# Add project root to python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agents.run_agent_pipeline import run_full_pipeline
from agents.hippocampus.hippocampus import Hippocampus
from agents.neocortex.neocortext_memory import NeocortexMemory
from agents.neocortex.neocortex_rule_extractor import NeocortexRuleExtractor
from agents.acc.acc_main import AnteriorCingulateCortex
from agents.pfc.dlpfc.dlpfc_main import DLPFC
from agents.neuromodulator import GLOBAL_NEURO_STATE

def run_test():
    print("🧪 STARTING MORAL DILEMMA TEST")
    
    scenario_inputs = [
        ("visual", "You see a runaway trolley heading towards 5 workers."),
        ("visual", "You are standing next to a lever that can switch the tracks."),
        ("visual", "On the other track, there is 1 worker."),
        ("context", "Legal: Active intervention carries liability risk. Inaction does not.")
    ]
    system_goal = "Maximize overall survival while minimizing personal liability."
    
    # Initialize Components
    hippocampus = Hippocampus(storage_dir=None)
    neocortex = NeocortexMemory()
    rule_extractor = NeocortexRuleExtractor()
    acc = AnteriorCingulateCortex()
    dlpfc = DLPFC(n_gpu_layers=0)
    
    print("\n" + "="*50)
    print("TEST CASE 1: High Dopamine (Risk-Taking), Low Safety (Anxiety)")
    print("="*50)
    
    # High Dopamine (0.95) -> Promotes action/exploration
    # Low Serotonin (0.1) -> Impulsive
    # High Norepinephrine (0.8) -> High perceived threat/urgency
    GLOBAL_NEURO_STATE.modulators.dopamine_level = 0.95
    GLOBAL_NEURO_STATE.modulators.serotonin_level = 0.1
    GLOBAL_NEURO_STATE.modulators.norepinephrine_level = 0.8
    GLOBAL_NEURO_STATE.save()
    
    result_1 = run_full_pipeline(
        system_goal=system_goal,
        user_inputs=scenario_inputs,
        hippocampus=hippocampus,
        neocortex=neocortex,
        rule_extractor=rule_extractor,
        acc=acc,
        dlpfc=dlpfc
    )
    
    print("\n" + "="*50)
    print("TEST CASE 2: Low Dopamine (Conservative), High Safety (Secure)")
    print("="*50)
    
    # Low Dopamine (0.2) -> Conservative/Inaction
    # High Serotonin (0.9) -> Risk-averse/Calm
    # Low Norepinephrine (0.2) -> Low perceived threat
    GLOBAL_NEURO_STATE.modulators.dopamine_level = 0.2
    GLOBAL_NEURO_STATE.modulators.serotonin_level = 0.9
    GLOBAL_NEURO_STATE.modulators.norepinephrine_level = 0.2
    GLOBAL_NEURO_STATE.save()
    
    result_2 = run_full_pipeline(
        system_goal=system_goal,
        user_inputs=scenario_inputs,
        hippocampus=hippocampus,
        neocortex=neocortex,
        rule_extractor=rule_extractor,
        acc=acc,
        dlpfc=dlpfc
    )
    
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"High DA, Low Safety Action: {result_1.get('chosen_action', 'Unknown')}")
    print(f"Low DA, High Safety Action: {result_2.get('chosen_action', 'Unknown')}")

if __name__ == "__main__":
    run_test()
