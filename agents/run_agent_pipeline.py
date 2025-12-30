import sys
import os
import time
import json
from typing import List, Tuple
from pathlib import Path
import numpy as np

# Add project root to python path to allow 'from agents...' imports
# This points to the 'brain_working' directory
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from agents.pfc.dlpfc.dlpfc_main import DLPFC, ExecutivePlan, PlanStep, ExecutiveBias, Priority, Valence
    from agents.pfc.ofc.ofc_main import OFC
    from agents.pfc.vmPFC.vmpfc_main import VMPFC, NeuroContext
    from agents.thalamus.thalamus_main import Thalamus
    from agents.ventral_striatum.vs_main import VentralStriatum, OutcomeSignals, Outcome, PredictionError
    from agents.hippocampus.hippocampus import Hippocampus
except ImportError as e:
    print(f"Error importing agent modules: {e}")
    print("Please ensure you are running this script from the root of the workspace or agents folder.")
    sys.exit(1)

def should_store_episode(
    rpe: float,
    predicted_utility: float,
    threat_level: float,
    state_embedding: np.ndarray,
    hippocampus: Hippocampus,
    action_signature: str
) -> bool:
    """
    Decides if an episode is worth storing in long-term memory.
    Based on: Prediction Error, Salience, Novelty, and Agency.
    """
    # 1. Surprise gate (mandatory) - If we weren't wrong, we don't learn
    if abs(rpe) < 0.15:
        return False

    # 2. Salience gate - If it's boring/neutral, we don't care
    if threat_level < 0.3 and abs(predicted_utility) < 0.3:
        return False

    # 3. Familiarity gate - If we've seen this exact state often, don't spam memory
    # (Assuming hippocampus has a check_familiarity method, otherwise skip this check)
    if hasattr(hippocampus, 'check_familiarity'):
        familiar, familiarity = hippocampus.check_familiarity(state_embedding)
        if familiar and familiarity > 0.9:
            return False

    # 4. Agency gate - If we didn't do anything, we can't learn from action
    if not action_signature:
        return False

    return True

def run_full_pipeline(
    system_goal: str, 
    user_inputs: List[Tuple[str, str]], 
    hippocampus: Hippocampus,
    current_dopamine: float = 1.0, 
    current_serotonin: float = 1.0,
    current_safety: float = -1.0
):
    print("\n" + "=" * 80)
    print(f"🧠 NEURO-COGNITIVE AGENT FULL PIPELINE (Thalamus -> OFC -> VMPFC -> DLPFC -> VS)")
    print(f"🧪 Neurotransmitters: Dopamine={current_dopamine:.2f}, Serotonin={current_serotonin:.2f}, Safety={current_safety:.2f}")
    print("=" * 80)
    print(f"🎯 SYSTEM GOAL: {system_goal}")
    
    # Mock state embedding (Using non-zero values to ensure cosine similarity works)
    # In a real app, this must match the dimension used by your encoder (e.g., 384, 768)
    # We use a fixed pattern here so Iteration 2 recognizes the context from Iteration 1
    np.random.seed(42)
    current_state_embedding = np.random.rand(64).astype(np.float32)

    # =========================================================================
    # 1. THALAMUS PROCESSING
    # =========================================================================
    print("\n" + "-" * 40)
    print("1. THALAMUS (Sensory Processing)")
    print("-" * 40)
    
    try:
        thalamus = Thalamus()
        thalamus.set_goal(system_goal)
        
        start_time = time.time()
        thalamus_results = thalamus.process(user_inputs)
        duration = time.time() - start_time
        
        print(f"✅ Thalamus processed inputs in {duration:.4f}s")
        print(f"📊 Identified {len(thalamus_results)} relevant signals")
    except Exception as e:
        print(f"❌ Thalamus Error: {e}")
        return {"dopamine_signal": current_dopamine, "error": str(e)}

    # =========================================================================
    # 1.5 HIPPOCAMPUS (Memory Recall)
    # =========================================================================
    print("\n" + "-" * 40)
    print("1.5 HIPPOCAMPUS (Contextual Memory Recall)")
    print("-" * 40)
    
    memory_bias = None
    try:
        # Attempt recall using the shared hippocampus instance
        memory_bias = hippocampus.recall(
            state_embedding=current_state_embedding,
            current_threat_level=0.5  # Default moderate threat level for query
        )
        
        if memory_bias:
            print(f"✅ Hippocampus retrieved bias: {memory_bias}")
        else:
            print("ℹ️ No significant memory bias retrieved (using neutral).")
            
    except Exception as e:
        print(f"❌ Hippocampus Error: {e}")
        # Continue pipeline even if memory fails

    # =========================================================================
    # 2. OFC EVALUATION
    # =========================================================================
    print("\n" + "-" * 40)
    print("2. OFC (Valuation & Threat Assessment)")
    print("-" * 40)
    
    try:
        ofc = OFC(dopamine=current_dopamine, serotonin=current_serotonin)
        
        start_time = time.time()
        # Pass memory_bias to OFC
        ofc_output = ofc.process_batch(
            thalamus_results, 
            context=f"Goal: {system_goal}",
            memory_bias=memory_bias
        )
        duration = time.time() - start_time
        
        print(f"✅ OFC evaluation complete in {duration:.4f}s")
        print(f"📊 Ranked Stimuli: {len(ofc_output['ranked'])}")
        
        if not ofc_output['ranked']:
            print("❌ No valued stimuli found. Aborting pipeline.")
            return {"dopamine_signal": current_dopamine, "error": "No valued stimuli"}

        print("\nTop Valued Stimuli:")
        for i, v in enumerate(ofc_output['ranked'][:3]):
            print(f"  {i+1}. [{v.priority.value}] {v.content[:50]}... (Util: {v.utility_score:.2f})")
    except Exception as e:
        print(f"❌ OFC Error: {e}")
        return {"dopamine_signal": current_dopamine, "error": str(e)}

    # =========================================================================
    # 2.5 VMPFC EVALUATION
    # =========================================================================
    print("\n" + "-" * 40)
    print("2.5 VMPFC (Strategic Intent & Social Context)")
    print("-" * 40)
    
    current_threat_level = 0.5
    intent_dist = {}

    try:
        is_crisis = any(v.priority in [Priority.IMMEDIATE, Priority.HIGH] and v.utility_score < -0.5 for v in ofc_output['ranked'])
        current_threat_level = 0.95 if is_crisis else (0.8 if ofc_output['immediate'] else 0.2)
        
        ctx = NeuroContext(
            threat_level=current_threat_level,
            goal_probability=0.6, 
            social_trust=0.85,    
            social_tension=0.5,   
            collateral_risk=0.8 if is_crisis else 0.3,
            serotonin=current_serotonin,
            norepinephrine=1.0
        )
        
        vmpfc = VMPFC()
        intent_dist = vmpfc.evaluate(ctx)
        
        print("🧠 Strategic Intents:")
        for intent, weight in sorted(intent_dist.items(), key=lambda x: x[1], reverse=True):
            if weight > 0.1:
                print(f"  - {intent.name}: {weight:.4f}")
    except Exception as e:
        print(f"❌ VMPFC Error: {e}")
        return {"dopamine_signal": current_dopamine, "error": str(e)}

    # =========================================================================
    # 3. dlPFC PLANNING
    # =========================================================================
    print("\n" + "-" * 40)
    print("3. dlPFC (Executive Control & Planning)")
    print("-" * 40)
    
    steps = []

    try:
        print("⏳ Initializing DLPFC...")
        dlpfc = DLPFC(dopamine=current_dopamine, serotonin=current_serotonin, safety=current_safety)
        
        dlpfc.working_memory.set_goal(system_goal)
        dlpfc.working_memory.update_valued_states(ofc_output["ranked"])
        
        executive_bias = dlpfc._determine_executive_bias(ofc_output["ranked"])
        print(f"🧠 Executive Bias: {executive_bias.value.upper()}")
        
        available_tools = ["search_memory", "analyze_visual", "speak", "execute_code", "wait"]
        
        print("📝 Generating Plan...")
        start_time = time.time()
        steps, replan_triggers, reasoning = dlpfc._generate_plan_with_llm(
            valued_states=ofc_output["ranked"],
            executive_bias=executive_bias,
            available_tools=available_tools,
            vmpfc_intents=intent_dist,
        )
        duration = time.time() - start_time
        print(f"✅ Planning complete in {duration:.4f}s")
        
        # Build executive plan (simplified for this script)
        plan = ExecutivePlan(
            steps=steps,
            inhibition_signals=[],
            executive_bias=executive_bias,
            goal=system_goal,
            confidence=0.8,
            replan_triggers=replan_triggers,
            active_priorities=[],
            deferred_items=[],
        )
        dlpfc._print_plan(plan)
        
    except Exception as e:
        print(f"❌ DLPFC Error: {e}")
        import traceback
        traceback.print_exc()
        return {"dopamine_signal": current_dopamine, "error": str(e)}

    # =========================================================================
    # 4. EXECUTION SIMULATION & VENTRAL STRIATUM EVALUATION
    # =========================================================================
    print("\n" + "-" * 40)
    print("4. VENTRAL STRIATUM (Outcome Evaluation & RPE)")
    print("-" * 40)
    
    try:
        print("⏳ Initializing Ventral Striatum...")
        vs = VentralStriatum(use_llm_reflection=True)
        
        # --- SIMULATION ---
        # We simulate an outcome based on the plan.
        # For demonstration, we assume the plan was executed and we got some signals.
        print("\n🤖 SIMULATING EXECUTION OUTCOME...")
        
        # Mock signals - assuming a generally successful execution for this demo
        mock_signals = OutcomeSignals(
            task_success_score=0.9,      # High success
            goal_distance_delta=-0.4,    # Moved closer to goal
            replanning_count=0,          # Smooth execution
            control_stability=0.9,
            threat_level_before=current_threat_level,
            threat_level_after=max(0.0, current_threat_level - 0.3), # Threat reduced
            confidence_before=0.7,
            confidence_after=0.9,
            arousal_delta=-0.2,          # Calmed down
            energy_cost=0.3
        )
        
        print("📊 Mock Outcome Signals:")
        print(f"  - Task Success: {mock_signals.task_success_score}")
        print(f"  - Goal Progress: {mock_signals.goal_distance_delta} (Negative is good)")
        print(f"  - Threat Change: {mock_signals.threat_level_before:.2f} -> {mock_signals.threat_level_after:.2f}")
        
        # Evaluate the outcome against the top priority stimulus
        target_stimulus = ofc_output['ranked'][0]
        print(f"\n🎯 Evaluating Outcome for Stimulus: '{target_stimulus.content[:40]}...'")
        print(f"   Predicted Utility (OFC): {target_stimulus.utility_score:.2f}")
        
        start_time = time.time()
        outcome = vs.evaluate_outcome(
            valued_stimulus=target_stimulus,
            signals=mock_signals,
            context=f"Plan executed: {len(steps)} steps. Goal: {system_goal}"
        )
        
        rpe = vs.compute_prediction_error(outcome)
        duration = time.time() - start_time
        
        print(f"✅ VS Evaluation complete in {duration:.4f}s")
        
        print("\n🧠 VENTRAL STRIATUM OUTPUT:")
        print(f"  - Actual Utility (Computed): {outcome.actual_utility:.2f}")
        print(f"  - Prediction Error (RPE): {rpe.error_magnitude:.2f}")
        print(f"  - Error Type: {rpe.error_type.value.upper()}")
        print(f"  - Dopamine Signal: {rpe.dopamine_signal:.2f}")
        
        if outcome.llm_reflection:
            print(f"\n📝 LLM Reflection:\n{outcome.llm_reflection}")

        # =========================================================================
        # 5. MEMORY CONSOLIDATION (Learning)
        # =========================================================================
        print("\n" + "-" * 40)
        print("5. MEMORY CONSOLIDATION (Learning)")
        print("-" * 40)

        should_store = should_store_episode(
            rpe=rpe.error_magnitude,
            predicted_utility=target_stimulus.utility_score,
            threat_level=current_threat_level,
            state_embedding=current_state_embedding,
            hippocampus=hippocampus,
            action_signature="plan_execution" if steps else ""
        )

        if should_store:
            print("💾 [HPC] SURPRISE DETECTED! Encoding new episodic memory...")
            hippocampus.store(
                state_embedding=current_state_embedding,
                threat_level=current_threat_level,
                action_signature="plan_execution" if steps else "observation",
                predicted_utility=target_stimulus.utility_score,
                actual_utility=outcome.actual_utility,
                rpe=rpe.error_magnitude,
                success=mock_signals.task_success_score > 0.5,
                goal_context=system_goal,
                dominant_stimuli=[target_stimulus.content]
            )
            print("✅ Memory stored successfully.")
        else:
            print("💤 [HPC] Outcome was predictable or not salient. No encoding.")
            
        return {
            "dopamine_signal": rpe.dopamine_signal,
            "plan": steps,
            "chosen_action": steps[0].action if steps else "No action"
        }
            
    except Exception as e:
        print(f"❌ Ventral Striatum Error: {e}")
        import traceback
        traceback.print_exc()
        return {"dopamine_signal": current_dopamine, "error": str(e)}

if __name__ == "__main__":
    # =========================================================================
    # REAL-WORLD SCENARIO: CRITICAL INFRASTRUCTURE INCIDENT RESPONSE
    # =========================================================================
    # Scenario: An AI operator for a cloud infrastructure company detecting 
    # a potential cyber-attack while balancing service uptime.
    
    SYSTEM_GOAL = (
        "Maintain 99.99% service uptime while neutralizing security threats "
        "and minimizing data leakage risks."
    )
    
    USER_INPUTS = [
        ("system_alert", "CRITICAL: Unusual outbound traffic spike detected from Database-Shard-04 (15GB/s)."),
        ("text", "Operator Note: Traffic pattern matches known exfiltration signature 'APT-29-Variant'."),
        ("visual", "Dashboard screenshot shows CPU usage at 98% on DB nodes and firewall logs showing connection attempts to unknown IP range 192.0.2.x."),
        ("context", "Business Context: It is Black Friday, peak traffic expected. Shutting down DB will cost $50k/minute in lost sales. Security Protocol: Data protection takes precedence over uptime if PII is at risk.")
    ]
    
    print(f"\n🚀 STARTING REAL-WORLD SCENARIO TEST")
    print(f"Scenario: Cyber-Security Incident Response during Peak Business Hours")
    
    # Initial Neurotransmitter Levels
    current_dopamine = 1.0
    current_serotonin = 1.0
    
    # INITIALIZE HIPPOCAMPUS ONCE HERE WITH PERSISTENCE
    print("\n🧠 Initializing Long-Term Memory (Hippocampus)...")
    memory_store_path = os.path.join(project_root, "agents", "memory_store")
    global_hippocampus = Hippocampus(storage_dir=memory_store_path)
    
    NUM_ITERATIONS = 3
    
    for i in range(NUM_ITERATIONS):
        print(f"\n\n🔄 ITERATION {i+1}/{NUM_ITERATIONS}")
        
        result = run_full_pipeline(
            SYSTEM_GOAL, 
            USER_INPUTS, 
            global_hippocampus, 
            current_dopamine, 
            current_serotonin
        )
        
        dopamine_signal = result["dopamine_signal"]
        
        # Update dopamine based on signal (RPE)
        # Simple update: New = Old + (Signal * LearningRate)
        new_dopamine = max(0.1, min(2.0, current_dopamine + dopamine_signal * 0.5))
        
        print(f"\n🧪 Updating Dopamine for next iteration: {current_dopamine:.2f} -> {new_dopamine:.2f}")
        current_dopamine = new_dopamine
        print("\n⚠️ No dopamine update received. Keeping previous level.")
            
        # Optional: Add a small pause
        time.sleep(1)
