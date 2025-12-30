from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import json

# Internal brain region imports
from agents.pfc.ofc.ofc_main import OFC, ValuedStimulus, Priority, Valence
from agents.hippocampus.hippocampus import Hippocampus
from agents.pfc.vmPFC.vmpfc_main import VMPFC, NeuroContext, StrategicIntent
from agents.thalamus.thalamus_main import Thalamus
from neural_surgery.neuro_cognitive_agent import NeuroCognitiveAgent

# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ExecutiveBias(Enum):
    EMERGENCY = "emergency"
    FOCUSED = "focused"
    EXPLORATORY = "exploratory"
    CAUTIOUS = "cautious"

class InhibitionType(Enum):
    MUTE = "mute"
    SUPPRESS = "suppress"
    DEFER = "defer"

@dataclass
class PlanStep:
    step_id: int
    action: str
    tool: str
    target: str = ""
    dependencies: List[int] = field(default_factory=list)
    priority: str = "medium"
    fallback_step_id: Optional[int] = None

@dataclass
class InhibitionSignal:
    target: str
    inhibition_type: InhibitionType
    strength: float
    reason: str

@dataclass
class ExecutivePlan:
    steps: List[PlanStep]
    inhibition_signals: List[InhibitionSignal]
    executive_bias: ExecutiveBias
    goal: str
    confidence: float
    replan_triggers: List[str]
    active_priorities: List[str]
    deferred_items: List[str]

# =============================================================================
# WORKING MEMORY
# =============================================================================

class WorkingMemory:
    def __init__(self):
        self.goal: Optional[str] = None
        self.valued_states: List[ValuedStimulus] = []
        self.active_plan: Optional[ExecutivePlan] = None
        self.inhibited_sources: Dict[str, InhibitionSignal] = {}
        self.context_stack: List[Dict] = []

    def set_goal(self, goal: str):
        self.goal = goal

    def update_valued_states(self, states: List[ValuedStimulus]):
        self.valued_states = states

    def push_context(self, context: Dict):
        self.context_stack.append(context)

    def pop_context(self) -> Optional[Dict]:
        return self.context_stack.pop() if self.context_stack else None

# =============================================================================
# dlPFC - THE EXECUTIVE CONTROLLER
# =============================================================================

class DLPFC:
    """
    Dorsolateral Prefrontal Cortex - The Executive Controller.
    Pipeline: Thalamus → OFC → dlPFC Planning
    """
    def __init__(
        self,
        dopamine: float = 1.0,
        serotonin: float = 1.0,
        safety: float = 1.0,
        replan_threshold: float = 0.3,
        n_gpu_layers: int = 0,
    ):
        print("[dlPFC] 🧠 Initializing Dorsolateral Prefrontal Cortex...")
        self.dopamine = dopamine
        self.serotonin = serotonin
        self.safety = safety
        self.replan_threshold = replan_threshold
        
        # Working memory
        self.working_memory = WorkingMemory()
        
        # LLM for planning
        self.llm = NeuroCognitiveAgent(n_gpu_layers=n_gpu_layers)
        
        # Planning prompt
        self.planning_prompt = """You are the dlPFC, the brain's executive controller.
Your task is to generate a JSON action plan to achieve the GOAL given the VALUED STATES.

INSTRUCTIONS:
1. Analyze the VALUED STATES (threats have negative utility, opportunities have positive).
2. Prioritize IMMEDIATE threats first.
3. Use the AVAILABLE TOOLS.
4. Output ONLY valid JSON. Do not include markdown blocks or extra text.

JSON FORMAT:
{
  "reasoning": "Brief explanation of the strategy",
  "steps": [
    {
      "step_id": 1,
      "action": "Description of action",
      "tool": "tool_name_from_list",
      "priority": "immediate"
    }
  ],
  "replan_triggers": ["Condition 1", "Condition 2"]
}
"""

        print(f"[dlPFC] 🧠 Executive circuits ready. (DA={dopamine:.2f}, 5-HT={serotonin:.2f})")

    # =========================================================================
    # CORE METHODS
    # =========================================================================

    def _determine_executive_bias(self, valued_states: List[ValuedStimulus]) -> ExecutiveBias:
        if not valued_states:
            return ExecutiveBias.EXPLORATORY
            
        immediate_threats = [
            v for v in valued_states 
            if v.priority == Priority.IMMEDIATE and v.valence == Valence.NEGATIVE
        ]
        
        if immediate_threats:
            return ExecutiveBias.EMERGENCY if self.serotonin <= 1.5 else ExecutiveBias.CAUTIOUS
            
        if self.dopamine < 0.5:
            return ExecutiveBias.CAUTIOUS
        elif self.dopamine > 1.2:
            return ExecutiveBias.FOCUSED
            
        return ExecutiveBias.EXPLORATORY

    def _generate_inhibition_signals(
        self, 
        valued_states: List[ValuedStimulus],
        executive_bias: ExecutiveBias,
    ) -> List[InhibitionSignal]:
        inhibitions = []
        if not valued_states:
            return inhibitions
            
        primary_focus = valued_states[0]
        
        for state in valued_states:
            if state == primary_focus:
                continue
                
            if executive_bias == ExecutiveBias.EMERGENCY:
                if state.valence == Valence.POSITIVE or state.priority == Priority.BACKGROUND:
                    inhibitions.append(InhibitionSignal(
                        target=state.source,
                        inhibition_type=InhibitionType.MUTE,
                        strength=0.9,
                        reason=f"Emergency: muting '{state.content[:25]}'",
                    ))
            elif executive_bias == ExecutiveBias.FOCUSED:
                if state.priority in [Priority.LOW, Priority.BACKGROUND]:
                    inhibitions.append(InhibitionSignal(
                        target=state.source,
                        inhibition_type=InhibitionType.SUPPRESS,
                        strength=0.6,
                        reason=f"Focused: suppressing '{state.content[:25]}'",
                    ))
                    
        return inhibitions

    def _generate_plan_with_llm(
        self,
        valued_states: List[ValuedStimulus],
        executive_bias: ExecutiveBias,
        available_tools: List[str],
        vmpfc_intents: Dict[StrategicIntent, float] = None,
    ) -> Tuple[List[PlanStep], List[str], str]:
        
        states_text = "\n".join([
            f"- [{v.priority.value.upper()}] {v.source}: {v.content} "
            f"(utility={v.utility_score:.2f}, {v.valence.value})"
            for v in valued_states
        ]) if valued_states else "No valued states"
        
        tools_text = ", ".join(available_tools) if available_tools else "observe, speak, act, wait"
        
        intents_text = "None"
        if vmpfc_intents:
            intents_text = "\n".join([
                f"- {k.name}: {v:.2f}" 
                for k, v in sorted(vmpfc_intents.items(), key=lambda x: x[1], reverse=True)
            ])

        user_prompt = f"""
        You are an executive planning system.
        GOAL: {self.working_memory.goal}
        VALUED STATES:
        {states_text}

        STRATEGIC INTENTS (VMPFC):
        {intents_text}

        CONTEXT:
        EXECUTIVE BIAS: {executive_bias.value}
        DOPAMINE: {self.dopamine:.2f}
        SEROTONIN: {self.serotonin:.2f}
        AVAILABLE TOOLS: {tools_text}

        RESPONSE (JSON ONLY):"""

        try:
            response = self.llm.generate_response(
                prompt=f"{self.planning_prompt}\n\n{user_prompt}",
                dopamine=self.dopamine,
                serotonin=self.serotonin,
                safety=self.safety,
            )
            
            print(f"[dlPFC] 🔍 Raw LLM Response: {response}")
            
            content = response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
                
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                content = content[start:end]
                
            plan_data = json.loads(content)
            
            steps = []
            for step_data in plan_data.get("steps", []):
                steps.append(PlanStep(
                    step_id=step_data.get("step_id", len(steps) + 1),
                    action=step_data.get("action", "unknown"),
                    tool=step_data.get("tool", "observe"),
                    target=step_data.get("target", ""),
                    dependencies=step_data.get("dependencies", []),
                    priority=step_data.get("priority", "medium"),
                    fallback_step_id=step_data.get("fallback_step_id"),
                ))
                
            return steps, plan_data.get("replan_triggers", []), plan_data.get("reasoning", "")
            
        except Exception as e:
            print(f"[dlPFC] ⚠️ Planning error: {e}")
            print(f"[dlPFC] 🔍 Raw LLM Response: {response if 'response' in locals() else 'None'}")
            return self._create_reactive_plan(valued_states), ["any_failure"], "Fallback plan"

    def _create_reactive_plan(self, valued_states: List[ValuedStimulus]) -> List[PlanStep]:
        steps = []
        step_id = 1
        for state in valued_states:
            if state.priority in [Priority.IMMEDIATE, Priority.HIGH]:
                action = "Address threat" if state.valence == Valence.NEGATIVE else "Pursue"
                steps.append(PlanStep(
                    step_id=step_id,
                    action=f"{action}: {state.content[:40]}",
                    tool="emergency_response" if state.valence == Valence.NEGATIVE else "execute",
                    target=state.source,
                    priority=state.priority.value,
                ))
                step_id += 1
        return steps if steps else [PlanStep(step_id=1, action="Observe", tool="observe")]

    def _calculate_confidence(self, valued_states: List[ValuedStimulus]) -> float:
        base = self.dopamine * 0.5 + 0.3
        threat_penalty = sum(
            0.1 for v in valued_states 
            if v.valence == Valence.NEGATIVE and v.priority == Priority.IMMEDIATE
        )
        return max(0.1, min(1.0, base - threat_penalty))

    def _print_plan(self, plan: ExecutivePlan):
        print("\n" + "=" * 70)
        print("dlPFC EXECUTIVE PLAN")
        print("=" * 70)
        print(f"\n🎯 GOAL: {plan.goal}")
        print(f"🧠 EXECUTIVE BIAS: {plan.executive_bias.value.upper()}")
        print(f"📊 CONFIDENCE: {plan.confidence:.2f}")
        
        print(f"\n📋 ACTION SEQUENCE ({len(plan.steps)} steps):")
        print("-" * 60)
        for step in plan.steps:
            deps = f" [after: {step.dependencies}]" if step.dependencies else ""
            print(f" {step.step_id}. [{step.priority.upper()}] {step.action[:50]}")
            print(f"    Tool: {step.tool} | Target: {step.target}{deps}")
            
        if plan.inhibition_signals:
            print(f"\n🔇 INHIBITION SIGNALS ({len(plan.inhibition_signals)}):")
            for sig in plan.inhibition_signals:
                print(f"  [{sig.inhibition_type.value.upper()}] {sig.target} - {sig.reason}")
                
        if plan.replan_triggers:
            print(f"\n⚡ REPLAN TRIGGERS: {plan.replan_triggers}")
        print("=" * 70)

    # =========================================================================
    # MODULATION & REPLAN
    # =========================================================================

    def modulate(self, dopamine: float = None, serotonin: float = None, safety: float = None):
        if dopamine is not None:
            self.dopamine = max(0.1, min(2.0, dopamine))
        if serotonin is not None:
            self.serotonin = max(0.1, min(2.0, serotonin))
        if safety is not None:
            self.safety = max(0.0, min(2.0, safety))
        print(f"[dlPFC] Neurochemistry: DA={self.dopamine:.2f}, 5-HT={self.serotonin:.2f}")

    def check_replan_needed(self, current_dopamine: float) -> bool:
        if current_dopamine < self.replan_threshold:
            print(f"[dlPFC] ⚠️ Dopamine dropped to {current_dopamine:.2f} - replan triggered!")
            self.dopamine = current_dopamine
            return True
        return False

    def __del__(self):
        if hasattr(self, 'llm'):
            del self.llm

# =============================================================================
# MAIN - FULL PIPELINE TEST
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # 1. THALAMUS PROCESSING
    # =========================================================================
    print("\n" + "=" * 70)
    print("--- 1. THALAMUS PROCESSING ---")
    print("=" * 70)
    
    thalamus = Thalamus()
    goal_text = "Cook dinner without burning food"
    thalamus.set_goal(goal_text)
    
    inputs = [
        ("hearing", "Timer ticking loudly"),
        ("vision", "Smoke rising from the pan"),
        ("touch", "Pot handle is extremely hot"),
        ("emotion", "Feeling focused on the task"),
        ("smell", "Spices sizzling nicely"),
    ]
    
    thalamus_results = thalamus.process(inputs)
    print(f"\n[Thalamus] Identified {len(thalamus_results)} relevant signals.")
    print("\n--- 2. HIPPOCAMPUS (MEMORY RECALL) ---")
    hippocampus = Hippocampus()
    # =========================================================================
    # 2. OFC EVALUATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("--- 2. OFC EVALUATION ---")
    print("=" * 70)
    
    ofc = OFC(dopamine=1.0, serotonin=1.2)
    ofc_output = ofc.process_batch(thalamus_results, context=f"Goal: {goal_text}")
    
    print(f"\n[OFC] Evaluated {len(ofc_output['ranked'])} stimuli.")
    print(f"[OFC] Immediate threats: {len(ofc_output['immediate'])}")
    print(f"[OFC] Total threats: {len(ofc_output['threats'])}")
    print(f"[OFC] Opportunities: {len(ofc_output['opportunities'])}")

    # =========================================================================
    # 2.5 VMPFC EVALUATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("--- 2.5 VMPFC EVALUATION ---")
    print("=" * 70)
    
    # Create context (mocking some values for now based on OFC/Thalamus)
    ctx = NeuroContext(
        threat_level=0.8 if ofc_output['immediate'] else 0.2,
        goal_probability=0.6,
        social_trust=0.85,
        social_tension=0.5,
        collateral_risk=0.3,
        serotonin=1.0,
        norepinephrine=1.0
    )
    
    vmpfc = VMPFC()
    intent_dist = vmpfc.evaluate(ctx)
    
    print("--- VMPFC TRACE ---")
    for intent, weight in sorted(intent_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"{intent.name:20}: {weight:.4f}")

    # =========================================================================
    # 3. dlPFC PLANNING
    # =========================================================================
    print("\n" + "=" * 70)
    print("--- 3. dlPFC PLANNING ---")
    print("=" * 70)
    
    dlpfc = DLPFC(dopamine=1.0, serotonin=1.0, safety=1.0)
    dlpfc.working_memory.set_goal(goal_text)
    dlpfc.working_memory.update_valued_states(ofc_output["ranked"])
    
    # Determine executive bias
    executive_bias = dlpfc._determine_executive_bias(ofc_output["ranked"])
    print(f"[dlPFC] Executive Bias: {executive_bias.value}")
    
    # Generate inhibition signals
    inhibitions = dlpfc._generate_inhibition_signals(ofc_output["ranked"], executive_bias)
    print(f"[dlPFC] Inhibition signals: {len(inhibitions)}")
    
    # Available tools
    available_tools = [
        "turn_off_stove",
        "grab_with_oven_mitt",
        "move_pan",
        "open_window",
        "silence_timer",
        "observe",
        "wait",
    ]
    
    # Generate plan
    steps, replan_triggers, reasoning = dlpfc._generate_plan_with_llm(
        valued_states=ofc_output["ranked"],
        executive_bias=executive_bias,
        available_tools=available_tools,
        vmpfc_intents=intent_dist,
    )
    
    # Calculate confidence
    confidence = dlpfc._calculate_confidence(ofc_output["ranked"])
    
    # Build executive plan
    plan = ExecutivePlan(
        steps=steps,
        inhibition_signals=inhibitions,
        executive_bias=executive_bias,
        goal=goal_text,
        confidence=confidence,
        replan_triggers=replan_triggers,
        active_priorities=[
            v.content[:40] for v in ofc_output["ranked"] 
            if v.priority in [Priority.IMMEDIATE, Priority.HIGH]
        ],
        deferred_items=[
            v.content[:40] for v in ofc_output["ranked"] 
            if v.priority in [Priority.LOW, Priority.BACKGROUND]
        ],
    )
    
    dlpfc.working_memory.active_plan = plan
    dlpfc._print_plan(plan)

    print("\n[dlPFC] Cleaning up...")
    del dlpfc
    print("[dlPFC] ✅ Pipeline complete.")
