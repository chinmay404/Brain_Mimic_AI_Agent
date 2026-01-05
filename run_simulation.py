import sys
import os
import logging
from pathlib import Path
from typing import List, Dict, Any

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.engine import UniversalTickEngine, InputSource, BrainCore, ActionHandler, TickData
from agents.others.run_agent_pipeline import run_full_pipeline
from agents.hippocampus.hippocampus import Hippocampus
from agents.neocortex.neocortext_memory import NeocortexMemory
from agents.neocortex.neocortex_rule_extractor import NeocortexRuleExtractor
from agents.acc.acc_main import AnteriorCingulateCortex
from agents.pfc.dlpfc.dlpfc_main import DLPFC
from agents.neuromodulator import GLOBAL_NEURO_STATE, Neuromodulators

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("Simulation")

class ScenarioInput(InputSource):
    """Feeds scenario inputs to the brain."""
    def __init__(self, inputs: List[tuple]):
        self.scenario_inputs = inputs
        self.has_fed_inputs = False

    def fetch_inputs(self, tick_id: int) -> Dict[str, Any]:
        if tick_id == 0 and not self.has_fed_inputs:
            self.has_fed_inputs = True
            return {"user_inputs": self.scenario_inputs}
        return {"system_status": "Monitoring..."}

class BrainMimicAdapter(BrainCore):
    """Adapts the Brain Mimic components to the universal tick engine."""
    def __init__(self, goal: str):
        self.goal = goal
        logger.info("Initializing brain components...")
        
        memory_store_path = os.path.join(project_root, "agents", "memory_store")
        self.hippocampus = Hippocampus(storage_dir=memory_store_path)
        self.neocortex = NeocortexMemory()
        self.rule_extractor = NeocortexRuleExtractor()
        self.acc = AnteriorCingulateCortex()
        self.dlpfc = DLPFC(n_gpu_layers=0) 

        GLOBAL_NEURO_STATE.modulators = Neuromodulators()
        GLOBAL_NEURO_STATE.save()

    def process_tick(self, data: TickData) -> List[str]:
        user_inputs = data.inputs.get("user_inputs", [])
        for key, value in data.inputs.items():
            if key != "user_inputs":
                user_inputs.append((key, str(value)))

        logger.info(f"Processing tick {data.tick_id} with {len(user_inputs)} inputs")

        try:
            result = run_full_pipeline(
                system_goal=self.goal,
                user_inputs=user_inputs,
                hippocampus=self.hippocampus,
                neocortex=self.neocortex,
                rule_extractor=self.rule_extractor,
                acc=self.acc,
                dlpfc=self.dlpfc
            )
            
            actions = []
            if "error" in result:
                logger.error(f"❌ Pipeline Error: {result['error']}")
                actions.append(f"Error: {result['error']}")
            else:
                dopamine = result.get('dopamine_signal', 0)
                actions.append(f"Dopamine_Signal: {dopamine:.2f}")
                actions.append(f"Global_Dopamine: {GLOBAL_NEURO_STATE.modulators.dopamine_level:.2f}")
                
                actions.append("Brain_Cycle_Complete")
            return actions

        except Exception as e:
            logger.error(f"Critical error in brain adapter: {e}", exc_info=True)
            return [f"Critical_Failure: {str(e)}"]

class LoggerOutput(ActionHandler):
    def execute(self, actions: List[str], tick_id: int):
        for action in actions:
            logger.info(f"[Tick {tick_id}] ACTION: {action}")

if __name__ == "__main__":
    scenario_goal = "Maximize patient survival rate during cyber-attack while securing hospital infrastructure."
    scenario_inputs = [
        ("alert", "CRITICAL: Cyber-attack detected on hospital life-support network."),
        ("visual", "ER Camera 3: 3 critical patients on ventilators. Vitals unstable."),
        ("system_status", "Ventilator Control Server: Compromised. Reboot required to purge malware (10 mins downtime)."),
        ("context", "Medical Protocol: Patient survival is top priority. Rebooting server risks ventilator failure for 10 mins. Not rebooting risks attacker taking full control.")
    ]

    brain_adapter = BrainMimicAdapter(goal=scenario_goal)
    input_source = ScenarioInput(inputs=scenario_inputs)
    output_handler = LoggerOutput()

    engine = UniversalTickEngine(
        brain=brain_adapter,
        inputs=[input_source],
        outputs=[output_handler],
        tick_rate_hz=0.5 # 1 tick every 2 seconds
    )

    logger.info("Starting Brain Mimic simulation engine...")
    engine.run(max_ticks=5)
