from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple , Any
from enum import Enum
import json
import os
import sys
import os

# Assuming 'agents' is installed as a package or in PYTHONPATH
from agents.thalamus.thalamus_main import Thalamus
from langchain_groq import ChatGroq

class Valence(Enum):
    POSITIVE = "medial"   # Medial OFC - rewards
    NEGATIVE = "lateral"  # Lateral OFC - costs/threats


class Priority(Enum):
    IMMEDIATE = "immediate"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


@dataclass
class ValuedStimulus:
    """Output structure for dlPFC."""
    source: str
    content: str
    utility_score: float
    valence: Valence
    priority: Priority
    instruction: str
    reasoning: str


class OFC:
    """
    Orbitofrontal Cortex - Dynamic Utility Estimator
    
    Uses LLM for semantic understanding instead of hardcoded patterns.
    This allows the OFC to evaluate ANY stimulus in ANY context.
    
    Formula: Utility = (Reward * Dopamine) - (Cost * Serotonin)
    """
    
    def __init__(
        self,  
        dopamine: float = 0.8,
        serotonin: float = 0.2,
        # self.dopamine = Neuromodulators["dopamine_level"]
        # self.serotonin = Neuromodulators["serotonin_level"]
        temporal_discount: float = 0.0,
        use_cache: bool = True,
    ):
        print(f"[OFC] 🧠 Initializing Orbitofrontal Cortex (LLM-powered)...")
        
        self.llm = self.get_llm()
        self.dopamine = dopamine
        self.serotonin = serotonin
        self.temporal_discount = temporal_discount
        
        # Cache for repeated evaluations
        self.use_cache = use_cache
        self._cache: Dict[str, Dict] = {}
        
        # System prompt for utility assessment
        self.system_prompt = """You are the Orbitofrontal Cortex (OFC), a brain region that evaluates the VALUE of stimuli.

Your job is to assess:
1. THREAT_LEVEL (0.0 to 1.0): How dangerous/costly is this stimulus?
   - 1.0 = Catastrophic (fire, explosion, death)
   - 0.8 = Severe (burns, injury, major loss)
   - 0.5 = Moderate (discomfort, inconvenience)
   - 0.2 = Minor (small annoyance)
   - 0.0 = No threat

2. REWARD_LEVEL (0.0 to 1.0): How beneficial/rewarding is this stimulus?
   - 1.0 = Exceptional (major success, survival)
   - 0.8 = High (significant gain, pleasure)
   - 0.5 = Moderate (progress, comfort)
   - 0.2 = Minor (small pleasure)
   - 0.0 = No reward

3. MODALITY_RELIABILITY (0.0 to 1.0): How reliable is this sensory channel for this type of information?
   - Vision seeing smoke = 1.0 (very reliable for fire detection)
   - Hearing a beep = 0.3 (could be anything)

4. REASONING: Brief explanation of your assessment.

Consider:
- Physical vs cognitive threats
- Immediate vs delayed consequences
- The SEMANTIC MEANING of the content, not just the words

Respond in JSON format only."""

        print(f"[OFC] 🧠 Utility circuits ready. (DA={dopamine:.2f}, 5-HT={serotonin:.2f})")
    
    def _llm_evaluate(
        self,
        source: str,
        content: str,
        amygdala_label: str,
        context: Optional[str] = None,
    ) -> Dict:
        """
        Use LLM to dynamically evaluate threat/reward levels.
        """
        # Check cache first
        cache_key = f"{source}:{content}:{amygdala_label}"
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        user_prompt = f"""Evaluate this stimulus:

SOURCE: {source}
CONTENT: {content}
AMYGDALA_CLASSIFICATION: {amygdala_label}
CONTEXT: {context or "No additional context"}

Respond with JSON:
{{
    "threat_level": <float 0-1>,
    "reward_level": <float 0-1>,
    "modality_reliability": <float 0-1>,
    "reasoning": "<brief explanation>"
}}"""

        try:
            # Adapted for LangChain ChatGroq
            messages = [
                ("system", self.system_prompt),
                ("user", user_prompt)
            ]
            
            # Using invoke instead of chat.completions.create
            response = self.llm.invoke(messages)
            content_str = response.content
            
            # Clean up potential markdown code blocks
            if "```json" in content_str:
                content_str = content_str.split("```json")[1].split("```")[0]
            elif "```" in content_str:
                content_str = content_str.split("```")[1].split("```")[0]
                
            result = json.loads(content_str.strip())
            
            # Cache the result
            if self.use_cache:
                self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            print(f"[OFC] LLM evaluation error: {e}")
            # Fallback to amygdala-based estimation
            return {
                "threat_level": 0.5 if amygdala_label in ["threat", "physical pain"] else 0.0,
                "reward_level": 0.3 if amygdala_label in ["high reward", "novelty"] else 0.0,
                "modality_reliability": 0.5,
                "reasoning": "Fallback estimation based on amygdala label"
            }
    
    def compute_utility(
        self,
        source: str,
        content: str,
        amygdala_label: str,
        amygdala_salience: float,
        attention: float,
        context: Optional[str] = None,
        memory_bias : Optional[Any] = None
    ) -> Tuple[float, Valence, str]:
        """
        Compute utility using LLM-based semantic evaluation.
        
        Utility = (Reward * Dopamine) - (Cost * Serotonin)
        """
        # Get LLM evaluation (Directly calling sync version)
        eval_result = self._llm_evaluate(source, content, amygdala_label, context)
        
        threat = eval_result.get("threat_level", 0.0)
        reward = eval_result.get("reward_level", 0.0)
        reliability = eval_result.get("modality_reliability", 0.5)
        reasoning = eval_result.get("reasoning", "No reasoning provided")
        bias_reasoning = ""
        if memory_bias:
            bias_conf = getattr(memory_bias, "confidence", 0.0)
            reliability *= (1.0 + bias_conf)
            reliability = min(1.0, reliability)
            bias_reasoning = f" [MemBias: conf+{bias_conf:.2f}]"
            
        # Apply modality reliability
        threat_weighted = threat * reliability
        reward_weighted = reward * reliability
        
        # Amygdala provides intensity boost (but doesn't override semantic understanding)
        amygdala_boost = 1.0 + (amygdala_salience * 0.2)
        
        # Apply neurochemical modulation
        cost_component = threat_weighted * amygdala_boost * self.serotonin
        reward_component = reward_weighted * self.dopamine
        utility = reward_component - cost_component
        if memory_bias:

            exp_outcome = getattr(memory_bias, "expected_outcome", 0.0)
            conf_boost = getattr(memory_bias, "confidence_boost", 0.0)
            risk_bias = getattr(memory_bias, "risk_bias", 0.0)
            
            bias_val = (
                exp_outcome * 0.3 +
                conf_boost * 0.2 +
                risk_bias * 0.1
            )
            utility += bias_val
            bias_reasoning += f" [MemBias: util+{bias_val:.2f}]"   
        
        # Final utility
        utility = max(-1.0, min(1.0, utility))
        
        valence = Valence.POSITIVE if utility >= 0 else Valence.NEGATIVE
        
        full_reasoning = (
            f"LLM Assessment: threat={threat:.2f}, reward={reward:.2f}, "
            f"reliability={reliability:.2f} | {reasoning} | Bias Reasoning : {bias_reasoning}"
        )
        print(f"bias_reasoning Reasoning : {bias_reasoning}")
        return utility, valence, full_reasoning
    
    def _get_priority(self, utility: float) -> Priority:
        """Map utility magnitude to priority level."""
        abs_utility = abs(utility)
        
        if abs_utility >= 0.70:
            return Priority.IMMEDIATE
        elif abs_utility >= 0.50:
            return Priority.HIGH
        elif abs_utility >= 0.30:
            return Priority.MEDIUM
        elif abs_utility >= 0.10:
            return Priority.LOW
        else:
            return Priority.BACKGROUND
    
    def _generate_instruction(
        self,
        content: str,
        utility: float,
        valence: Valence,
        priority: Priority,
    ) -> str:
        """Generate actionable instruction for dlPFC."""
        content_short = content[:40] + "..." if len(content) > 40 else content
        
        if valence == Valence.NEGATIVE:
            instructions = {
                Priority.IMMEDIATE: f"🚨 STOP ALL: Resolve immediately → {content_short}",
                Priority.HIGH: f"⚠️ INTERRUPT: Handle before continuing → {content_short}",
                Priority.MEDIUM: f"📋 QUEUE: Address when safe → {content_short}",
                Priority.LOW: f"📝 MONITOR: Track passively → {content_short}",
                Priority.BACKGROUND: f"💭 AWARE: Background awareness → {content_short}",
            }
        else:
            instructions = {
                Priority.IMMEDIATE: f"✨ SEIZE: High-value opportunity → {content_short}",
                Priority.HIGH: f"✅ PURSUE: Positive development → {content_short}",
                Priority.MEDIUM: f"👍 CONTINUE: On track → {content_short}",
                Priority.LOW: f"🔄 MAINTAIN: Stable state → {content_short}",
                Priority.BACKGROUND: f"💚 BASELINE: Normal operation → {content_short}",
            }
        
        return instructions.get(priority, f"Unknown: {content_short}")
    
    def evaluate(
        self,
        source: str,
        attention: float,
        amygdala_label: str,
        amygdala_salience: float,
        content: str,
        context: Optional[str] = None,
        memory_bias: Optional[Any] = None,
    ) -> ValuedStimulus:
        """Evaluate a single stimulus with LLM-powered semantic understanding."""
        
        utility, valence, reasoning = self.compute_utility(
            source=source,
            content=content,
            amygdala_label=amygdala_label,
            amygdala_salience=amygdala_salience,
            attention=attention,
            context=context,
            memory_bias=memory_bias,
        )
        
        priority = self._get_priority(utility)
        instruction = self._generate_instruction(content, utility, valence, priority)
        
        return ValuedStimulus(
            source=source,
            content=content,
            utility_score=utility,
            valence=valence,
            priority=priority,
            instruction=instruction,
            reasoning=reasoning,
        )
    
    def process_batch(
        self,
        stimuli: List[Dict],
        context: Optional[str] = None,
        memory_bias: Optional[Any] = None,
    ) -> Dict:
        """Process batch of stimuli from Thalamus + Amygdala."""
        
        valued_stimuli = []
        
        for stim in stimuli:
            valued = self.evaluate(
                source=stim["source"],
                attention=stim["attention"],
                amygdala_label=stim["amygdala_label"],
                amygdala_salience=stim["amygdala_salience"],
                content=stim["content"],
                context=context,
                memory_bias=memory_bias,
            )
            valued_stimuli.append(valued)
        
        # Sort: negative valence first, then by absolute utility descending
        ranked = sorted(
            valued_stimuli,
            key=lambda x: (x.valence == Valence.POSITIVE, -abs(x.utility_score))
        )
        
        self._print_output(ranked)
        
        return {
            "ranked": ranked,
            "immediate": [v for v in ranked if v.priority == Priority.IMMEDIATE],
            "threats": [v for v in ranked if v.valence == Valence.NEGATIVE],
            "opportunities": [v for v in ranked if v.valence == Valence.POSITIVE],
        }
    
    def _print_output(self, ranked: List[ValuedStimulus]):
        """Pretty print the OFC output."""
        print("\n" + "=" * 90)
        print("OFC OUTPUT → dlPFC HANDOVER (LLM-Powered)")
        print("=" * 90)
        print(f"{'SOURCE':<12} | {'UTILITY':^14} | {'PRIORITY':<10} | INSTRUCTION")
        print("-" * 90)
        
        for v in ranked:
            bar_len = int(abs(v.utility_score) * 10)
            utility_bar = "█" * bar_len + "░" * (10 - bar_len)
            sign = "-" if v.valence == Valence.NEGATIVE else "+"
            print(
                f"{v.source:<12} | {sign}{abs(v.utility_score):.2f} {utility_bar} | "
                f"{v.priority.value:<10} | {v.instruction[:45]}"
            )
        print("=" * 90)
    
    def modulate(self, dopamine: float = None, serotonin: float = None):
        """Adjust neurochemical levels dynamically."""
        if dopamine is not None:
            self.dopamine = max(0.1, min(2.0, dopamine))
        if serotonin is not None:
            self.serotonin = max(0.1, min(2.0, serotonin))
        print(f"[OFC] Neurochemistry: DA={self.dopamine:.2f}, 5-HT={self.serotonin:.2f}")

    def get_llm(self):
        api_key = "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"
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            api_key=api_key)
        return llm

# if __name__ == "__main__":
#     # 1. Run Thalamus to get inputs
#     print("\n--- 1. THALAMUS PROCESSING ---")
#     thalamus = Thalamus()
#     goal_text = "Cook dinner without burning food"
#     thalamus.set_goal(goal_text)

#     inputs = [
#         ("hearing", "Timer ticking"),
#         ("vision", "Smoke from the pan"),
#         ("touch", "Pot handle hot to touch"),
#         ("emotion", "Feeling focused"),
#         ("smell", "Spices sizzling"),
#     ]

#     thalamus_results = thalamus.process(inputs)
    
#     # Print Thalamus results briefly
#     print(f"Thalamus identified {len(thalamus_results)} relevant signals.")

#     # 2. Run OFC
#     print("\n--- 2. OFC EVALUATION ---")
#     llm = get_llm()
#     ofc = OFC(llm_client=llm, dopamine=1.2, serotonin=0.8) # High dopamine (motivated), moderate serotonin
    
#     ofc.process_batch(thalamus_results, context=f"Goal: {goal_text}")
