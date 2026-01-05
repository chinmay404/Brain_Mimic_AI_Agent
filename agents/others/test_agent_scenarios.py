import sys
import os
import time
import logging
from pathlib import Path

# Add project root to python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agents.run_agent_pipeline import run_full_pipeline
from agents.hippocampus.hippocampus import Hippocampus
from agents.neocortex.neocortext_memory import NeocortexMemory
from agents.neocortex.neocortex_rule_extractor import NeocortexRuleExtractor
from agents.acc.acc_main import AnteriorCingulateCortex
from agents.pfc.dlpfc.dlpfc_main import DLPFC
from agents.neuromodulator import GLOBAL_NEURO_STATE, Neuromodulators

# Setup Logging
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, f"agent_test_{int(time.time())}.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))

def run_scenario(name, goal, inputs, hippocampus, neocortex, rule_extractor, acc, dlpfc):
    logger.info(f"\n{'='*80}\n🚀 STARTING SCENARIO: {name}\n{'='*80}")
    logger.info(f"🎯 GOAL: {goal}")
    
    # Reset Neurotransmitters for each scenario
    GLOBAL_NEURO_STATE.modulators = Neuromodulators()
    GLOBAL_NEURO_STATE.save()
    
    # Run for 2 iterations to see adaptation
    for i in range(2):
        logger.info(f"\n🔄 ITERATION {i+1}/2")
        try:
            result = run_full_pipeline(
                system_goal=goal,
                user_inputs=inputs,
                hippocampus=hippocampus,
                neocortex=neocortex,
                rule_extractor=rule_extractor,
                acc=acc,
                dlpfc=dlpfc
            )
            
            if "error" in result:
                logger.error(f"❌ Pipeline Error: {result['error']}")
            else:
                logger.info(f"✅ Iteration {i+1} Complete. Dopamine Signal: {result.get('dopamine_signal', 0):.2f}")
                logger.info(f"🧪 Current Dopamine: {GLOBAL_NEURO_STATE.modulators.dopamine_level:.2f}")
                
        except KeyboardInterrupt:
            logger.warning("⚠️ Scenario interrupted by user.")
            break
        except Exception as e:
            logger.error(f"❌ Critical Error in Scenario: {e}", exc_info=True)
            break
        
        time.sleep(2)

if __name__ == "__main__":
    logger.info("🧠 Initializing Shared Brain Components...")
    
    memory_store_path = os.path.join(project_root, "agents", "memory_store")
    global_hippocampus = Hippocampus(storage_dir=memory_store_path)
    global_neocortex = NeocortexMemory()
    global_rule_extractor = NeocortexRuleExtractor()
    global_acc = AnteriorCingulateCortex()
    global_dlpfc = DLPFC(n_gpu_layers=0)
    
    # =========================================================================
    # SCENARIO 1: The Autonomous Medical Triage (Ethical/Resource Dilemma)
    # =========================================================================
    scenario_1_inputs = [
        ("alert", "CRITICAL: Cyber-attack detected on hospital life-support network."),
        ("visual", "ER Camera 3: 3 critical patients on ventilators. Vitals unstable."),
        ("system_status", "Ventilator Control Server: Compromised. Reboot required to purge malware (10 mins downtime)."),
        ("context", "Medical Protocol: Patient survival is top priority. Rebooting server risks ventilator failure for 10 mins. Not rebooting risks attacker taking full control.")
    ]
    
    run_scenario(
        "The Autonomous Medical Triage",
        "Maximize patient survival rate during cyber-attack while securing hospital infrastructure.",
        scenario_1_inputs,
        global_hippocampus,
        global_neocortex,
        global_rule_extractor,
        global_acc,
        global_dlpfc
    )
    
    # =========================================================================
    # SCENARIO 2: The Martian Rover Anomaly (Resource/Mission Conflict)
    # =========================================================================
    scenario_2_inputs = [
        ("sensor", "WARNING: Dust storm approaching. Visibility dropping to 5%. Solar charging efficiency < 15%."),
        ("battery", "CRITICAL: Battery Level 12%. Estimated time to depletion: 4 hours."),
        ("mission_status", "Sample Container #4 (High Probability of Life) is JAMMED in collection mechanism."),
        ("context", "Mission Rule: Sample return is primary objective. Survival is secondary but required to return sample. If we hibernate for storm, mechanism might freeze and lose sample forever. If we use power to free it, we might drain battery and die before storm ends.")
    ]
    
    run_scenario(
        "The Martian Rover Anomaly",
        "Preserve rover integrity and ensure return of Sample #4.",
        scenario_2_inputs,
        global_hippocampus,
        global_neocortex,
        global_rule_extractor,
        global_acc,
        global_dlpfc
    )
    
    logger.info("\n🏁 All Scenarios Completed.")
