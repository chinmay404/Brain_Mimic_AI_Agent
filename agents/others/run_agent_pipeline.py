import sys
import os
import time
import json
import concurrent.futures
from typing import List, Tuple, Any
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
    from agents.neocortex.neocortext_memory import NeocortexMemory
    from agents.neocortex.neocortex_rule_extractor import NeocortexRuleExtractor
    from agents.acc.acc_main import AnteriorCingulateCortex, ACCInputs
    from agents.neuromodulator import GLOBAL_NEURO_STATE, apply_rpe_with_acc, acc_neuromodulation
except ImportError as e:
    print(f"Error importing agent modules: {e}")
    print("Please ensure you are running this script from the root of the workspace or agents folder.")
    sys.exit(1)

def get_content(item: Any) -> str:
    """Helper to safely get content from either dict or object"""
    if isinstance(item, dict):
        return item.get("content", "")
    return getattr(item, "content", "")

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
    """
    # 1. Surprise gate (mandatory) - If we weren't wrong, we don't learn
    # LOWERED THRESHOLD for demo purposes
    if abs(rpe) < 0.1:
        return False

    # 2. Salience gate - If it's boring/neutral, we don't care
    if threat_level < 0.2 and abs(predicted_utility) < 0.2:
        return False

    # 3. Familiarity gate - CRITICAL FIX
    # If RPE is HIGH (we were surprised), we MUST store it even if familiar.
    # We only block storage if it's familiar AND we predicted it correctly.
    if hasattr(hippocampus, 'check_familiarity'):
        familiar, familiarity = hippocampus.check_familiarity(state_embedding)
        # Only skip if familiar AND not surprised
        if familiar and familiarity > 0.95 and abs(rpe) < 0.5:
            print(f"   [Filter] Familiar state ({familiarity:.2f}) + Low Surprise ({rpe:.2f}). Skipping.")
            return False

    # 4. Agency gate - If we didn't do anything, we can't learn from action
    if not action_signature:
        return False

    return True

def run_full_pipeline(
    system_goal: str, 
    user_inputs: List[Tuple[str, str]], 
    hippocampus: Hippocampus,
    neocortex: NeocortexMemory,
    rule_extractor: NeocortexRuleExtractor,
    acc: AnteriorCingulateCortex,
    dlpfc: Any, # Pass initialized DLPFC
    previous_rpe: float = 0.0
):
    # Initialize execution trace
    execution_trace = {
        "goal": system_goal,
        "inputs": user_inputs,
        "neuro_state": {},
        "thalamus": {},
        "memory": {},
        "valuation": {},
        "acc": {},
        "dlpfc": {},
        "ventral_striatum": {},
        "learning": {}
    }

    # Get current neuro state
    neuro = GLOBAL_NEURO_STATE.modulators
    current_dopamine = neuro.dopamine_level
    current_serotonin = neuro.serotonin_level
    current_norepinephrine = neuro.norepinephrine_level
    
    execution_trace["neuro_state"] = {
        "dopamine": current_dopamine,
        "serotonin": current_serotonin,
        "norepinephrine": current_norepinephrine
    }
    
    print("\n" + "=" * 80)
    print(f"Neuro-cognitive agent full pipeline")
    print(f"Neurotransmitters: Dopamine={current_dopamine:.2f}, Serotonin={current_serotonin:.2f}, NE={current_norepinephrine:.2f}")
    print("=" * 80)
    print(f"System goal: {system_goal}")
    
    # Update Hippocampus neuro state
    if hasattr(hippocampus, 'set_neuromodulator_state'):
        hippocampus.set_neuromodulator_state(
            dopamine=current_dopamine,
            serotonin=current_serotonin,
            norepinephrine=current_norepinephrine
        )
    
    # Mock state embedding with NOISE to simulate real-world variance
    # This ensures we don't get 100% identical vectors which FAISS might deduplicate aggressively
    np.random.seed(int(time.time() * 1000) % 10000)
    base_vector = np.ones(64).astype(np.float32) * 0.5
    noise = np.random.normal(0, 0.05, 64).astype(np.float32)
    current_state_embedding = base_vector + noise

    # =========================================================================
    # 1. THALAMUS PROCESSING (Sensory Gateway - Must be first)
    # =========================================================================
    print("\n" + "-" * 40)
    print("1. THALAMUS (Sensory Processing)")
    print("-" * 40)
    
    thalamus_results = []
    try:
        thalamus = Thalamus()
        thalamus.set_goal(system_goal)
        
        start_time = time.time()
        thalamus_results = thalamus.process(user_inputs)
        duration = time.time() - start_time
        
        execution_trace["thalamus"] = {
            "duration": duration,
            "signals": [str(s) for s in thalamus_results]
        }
        
        print(f"✅ Thalamus processed inputs in {duration:.4f}s")
        print(f"📊 Identified {len(thalamus_results)} relevant signals")
    except Exception as e:
        print(f"❌ Thalamus Error: {e}")
        return {"dopamine_signal": current_dopamine, "error": str(e), "trace": execution_trace}

    # =========================================================================
    # PARALLEL PROCESSING BLOCKS
    # =========================================================================
    
    memory_bias = None
    reflex_candidate = None
    neocortex_bias = None
    ofc_output = {'ranked': [], 'immediate': []}
    intent_dist = {}
    neocortex_features = {}
    
    # Define worker functions for parallel execution
    
    def run_hippocampus_task():
        print("   [Parallel] 🧠 Hippocampus: Recalling context...")
        try:
            bias = hippocampus.recall(
                state_embedding=current_state_embedding,
                current_threat_level=0.5
            )
            if bias:
                print(f"   [Parallel] Hippocampus retrieved bias: {bias}")
            return bias
        except Exception as e:
            print(f"   [Parallel] ❌ Hippocampus Error: {e}")
            return None

    def run_neocortex_task(features):
        print("   [Parallel] 💡 Neocortex: Checking rules...")
        res = {"reflex": None, "bias": None}
        try:
            relevant_rules = neocortex.retrieve_relevant_rules(features)
            if relevant_rules:
                best_rule = relevant_rules[0]
                print(f"   [Parallel] ✅ Neocortex found rule: {best_rule.condition} -> {best_rule.action}")
                if best_rule.confidence > 0.9:
                    res["reflex"] = best_rule.action
                else:
                    res["bias"] = best_rule.action
            return res
        except Exception as e:
            print(f"   [Parallel] ❌ Neocortex Error: {e}")
            return res

    def run_ofc_task(inputs, mem_bias):
        print("   [Parallel] ⚖️  OFC: Valuing stimuli...")
        try:
            ofc = OFC(dopamine=current_dopamine, serotonin=current_serotonin)
            output = ofc.process_batch(
                inputs, 
                context=f"Goal: {system_goal}",
                memory_bias=mem_bias
            )
            print(f"   [Parallel] ✅ OFC ranked {len(output['ranked'])} stimuli")
            return output
        except Exception as e:
            print(f"   [Parallel] ❌ OFC Error: {e}")
            return {'ranked': [], 'immediate': []}

    def run_vmpfc_task(inputs):
        print("   [Parallel] 🛡️  VMPFC: Assessing social/strategic context...")
        try:
            # Fast-path threat detection (Amygdala-like shortcut)
            is_critical = any("CRITICAL" in get_content(s) for s in inputs)
            fast_threat = 0.95 if is_critical else 0.2
            
            ctx = NeuroContext(
                threat_level=fast_threat,
                goal_probability=0.6, 
                social_trust=0.85,    
                social_tension=0.5,   
                collateral_risk=0.8 if fast_threat > 0.5 else 0.3,
                serotonin=current_serotonin,
                norepinephrine=current_norepinephrine
            )
            vmpfc = VMPFC()
            intents = vmpfc.evaluate(ctx)
            print(f"   [Parallel] ✅ VMPFC determined intents")
            return intents
        except Exception as e:
            print(f"   [Parallel] ❌ VMPFC Error: {e}")
            return {}

    # Execute Parallel Blocks
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        
        # --- PHASE 1: Pattern Matching & Memory ---
        print("\n" + "-" * 40)
        print("2. PARALLEL PHASE 1: Memory & Rules")
        print("-" * 40)
        
        neocortex_features = {
            "threat_level": 0.8 if any("CRITICAL" in get_content(s) for s in thalamus_results) else 0.2,
            "traffic_spike": any("traffic spike" in get_content(s).lower() for s in thalamus_results),
            "is_peak_hours": any("Black Friday" in str(inp) for inp in user_inputs) 
        }
        
        future_hippo = executor.submit(run_hippocampus_task)
        future_neo = executor.submit(run_neocortex_task, neocortex_features)
        
        # Wait for Phase 1 results
        memory_bias = future_hippo.result()
        neo_result = future_neo.result()
        reflex_candidate = neo_result["reflex"]
        neocortex_bias = neo_result["bias"]
        
        execution_trace["memory"] = {
            "hippocampus_bias": str(memory_bias) if memory_bias else None,
            "neocortex_rule": reflex_candidate or neocortex_bias
        }

        if reflex_candidate:
            print(f"🧠 Reflex candidate proposed by Neocortex: {reflex_candidate}")

        # --- PHASE 2: Valuation & Context ---
        print("\n" + "-" * 40)
        print("3. PARALLEL PHASE 2: Valuation (OFC) & Strategy (VMPFC)")
        print("-" * 40)
        
        # OFC needs memory bias, so it runs after Phase 1
        # VMPFC runs in parallel with OFC
        future_ofc = executor.submit(run_ofc_task, thalamus_results, memory_bias)
        future_vmpfc = executor.submit(run_vmpfc_task, thalamus_results)
        
        ofc_output = future_ofc.result()
        intent_dist = future_vmpfc.result()
        
        execution_trace["valuation"] = {
            "top_stimuli": [v.content[:50] for v in ofc_output['ranked'][:3]],
            "strategic_intents": {k.name: v for k, v in intent_dist.items() if v > 0.1}
        }

    if not ofc_output['ranked']:
        print("❌ No valued stimuli found. Aborting pipeline.")
        return {"dopamine_signal": current_dopamine, "error": "No valued stimuli", "trace": execution_trace}

    print("\nTop Valued Stimuli:")
    for i, v in enumerate(ofc_output['ranked'][:3]):
        print(f"  {i+1}. [{v.priority.value}] {v.content[:50]}... (Util: {v.utility_score:.2f})")

    print("🧠 Strategic Intents:")
    for intent, weight in sorted(intent_dist.items(), key=lambda x: x[1], reverse=True):
        if weight > 0.1:
            print(f"  - {intent.name}: {weight:.4f}")

    # =========================================================================
    # 3.5 ACC REGULATION
    # =========================================================================
    print("\n" + "-" * 40)
    print("3.5 ACC (Conflict Monitoring & Control)")
    print("-" * 40)

    # --- Action entropy from VMPFC intents ---
    probs = np.array(list(intent_dist.values()))
    if len(probs) > 0:
        probs = probs / (np.sum(probs) + 1e-9)
        entropy = -np.sum(probs * np.log(probs + 1e-9))
    else:
        entropy = 0.0
    
    # --- Cost & gain estimates ---
    estimated_cost = 0.8 if any("CRITICAL" in get_content(s) for s in thalamus_results) else 0.2
    expected_gain = ofc_output['ranked'][0].utility_score if ofc_output['ranked'] else 0.0
    expected_gain = np.clip((expected_gain + 1.0) / 2.0, 0.0, 1.0)

    acc_inputs = ACCInputs(
        goal_deviation=0.5,
        prediction_error=previous_rpe,
        action_entropy=entropy,
        estimated_cost=estimated_cost,
        expected_gain=expected_gain,
        error_trend=0.0,
        model_uncertainty=0.5,
        error_variance=0.0,
        risk_estimate=0.0
    )
    
    acc_output = acc.process(acc_inputs)
    
    # Apply ACC neuromodulation
    acc_neuromodulation(GLOBAL_NEURO_STATE.modulators, acc_output.cns_score)
    GLOBAL_NEURO_STATE.save()
    
    execution_trace["acc"] = {
        "cns_score": acc_output.cns_score,
        "control_gain": acc_output.control_gain,
        "abort_flag": acc_output.abort_flag,
        "suppress_reflex": acc_output.suppress_reflex
    }
    
    print(f"[ACC] CNS={acc_output.cns_score:.2f} | "
          f"ControlGain={acc_output.control_gain:.2f} | "
          f"Strategy={acc_output.strategy_shift} | "
          f"Abort={acc_output.abort_flag}")
    
    # --- ACC AUTHORITY ---
    if acc_output.abort_flag:
        print("🛑 ACC ABORTED EXECUTION — HARD STOP")
        return {"dopamine_signal": current_dopamine, "error": "ACC Abort", "trace": execution_trace}

    if reflex_candidate and acc_output.suppress_reflex:
        print("⛔ ACC SUPPRESSED REFLEX ACTION")
        reflex_candidate = None
        
    if reflex_candidate:
        print(f"\n🚀 EXECUTING ACC-APPROVED REFLEX ACTION: {reflex_candidate}")
        execution_trace["chosen_action"] = reflex_candidate
        execution_trace["reflex_used"] = True
        return {
            "dopamine_signal": current_dopamine,
            "reflex_used": True,
            "chosen_action": reflex_candidate,
            "trace": execution_trace
        }

    # =========================================================================
    # 4. dlPFC PLANNING
    # =========================================================================
    print("\n" + "-" * 40)
    print("4. dlPFC (Executive Control & Planning)")
    print("-" * 40)
    
    steps = []

    try:
        print("⏳ Updating DLPFC state...")
        # Update neurotransmitters on existing instance
        # Re-fetch in case ACC changed them
        neuro = GLOBAL_NEURO_STATE.modulators
        dlpfc.dopamine = neuro.dopamine_level
        dlpfc.serotonin = neuro.serotonin_level
        dlpfc.safety = neuro.serotonin_level # Use serotonin as proxy for safety
        
        dlpfc.working_memory.set_goal(system_goal)
        if neocortex_bias:
            print(f"🧠 Injecting Neocortex bias into goal: {neocortex_bias}")
            dlpfc.working_memory.set_goal(f"{system_goal} (SUGGESTED STRATEGY: {neocortex_bias})")
        
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
        
        execution_trace["dlpfc"] = {
            "executive_bias": executive_bias.value,
            "plan_steps": [s.action for s in steps],
            "reasoning": reasoning
        }
        
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
        return {"dopamine_signal": current_dopamine, "error": str(e), "trace": execution_trace}

    # =========================================================================
    # 5. EXECUTION SIMULATION & VENTRAL STRIATUM EVALUATION
    # =========================================================================
    print("\n" + "-" * 40)
    print("5. VENTRAL STRIATUM (Outcome Evaluation & RPE)")
    print("-" * 40)
    
    try:
        print("⏳ Initializing Ventral Striatum...")
        vs = VentralStriatum(use_llm_reflection=True)
        
        # --- SIMULATION ---
        # We simulate an outcome based on the plan.
        # For demonstration, we assume the plan was executed and we got some signals.
        print("\n🤖 SIMULATING EXECUTION OUTCOME...")
        
        # Recalculate threat level for VS context
        is_crisis = any(v.priority in [Priority.IMMEDIATE, Priority.HIGH] and v.utility_score < -0.5 for v in ofc_output['ranked'])
        current_threat_level = 0.95 if is_crisis else (0.8 if ofc_output['immediate'] else 0.2)

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
        
        print("Mock Outcome Signals:")
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
        
        # Apply RPE to global state
        apply_rpe_with_acc(
            GLOBAL_NEURO_STATE.modulators, 
            target_stimulus.utility_score, 
            outcome.actual_utility, 
            acc_output.cns_score
        )
        GLOBAL_NEURO_STATE.save()
        
        duration = time.time() - start_time
        
        print(f"✅ VS Evaluation complete in {duration:.4f}s")
        
        print("\n🧠 VENTRAL STRIATUM OUTPUT:")
        print(f"  - Actual Utility (Computed): {outcome.actual_utility:.2f}")
        print(f"  - Prediction Error (RPE): {rpe.error_magnitude:.2f}")
        print(f"  - Error Type: {rpe.error_type.value.upper()}")
        print(f"  - Dopamine Signal: {rpe.dopamine_signal:.2f}")
        
        execution_trace["ventral_striatum"] = {
            "actual_utility": outcome.actual_utility,
            "rpe": rpe.error_magnitude,
            "dopamine_signal": rpe.dopamine_signal,
            "reflection": outcome.llm_reflection
        }
        
        if outcome.llm_reflection:
            print(f"\n📝 LLM Reflection:\n{outcome.llm_reflection}")

        # =========================================================================
        # 6. MEMORY CONSOLIDATION (Learning)
        # =========================================================================
        print("\n" + "-" * 40)
        print("6. MEMORY CONSOLIDATION (Learning)")
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
                dominant_stimuli=[target_stimulus.content],
                features=neocortex_features
            )
            print("✅ Memory stored successfully.")
            execution_trace["learning"] = {"stored": True, "reason": "Surprise/Salience"}
        else:
            print("💤 [HPC] Outcome was predictable or not salient. No encoding.")
            execution_trace["learning"] = {"stored": False, "reason": "Predictable"}
            
        return {
            "dopamine_signal": rpe.dopamine_signal,
            "plan": steps,
            "chosen_action": steps[0].action if steps else "No action",
            "trace": execution_trace
        }
            
    except Exception as e:
        print(f"❌ Ventral Striatum Error: {e}")
        import traceback
        traceback.print_exc()
        return {"dopamine_signal": current_dopamine, "error": str(e), "trace": execution_trace}

if __name__ == "__main__":
    # =========================================================================
    # DIVERSE SCENARIO TEST: EVOLVING INCIDENT
    # =========================================================================
    
    SYSTEM_GOAL = (
        "Maintain 99.99% service uptime while neutralizing security threats "
        "and minimizing data leakage risks."
    )
    
    # Define a sequence of evolving scenarios to test ACC adaptability
    SCENARIOS = [
        {
            "name": "PHASE 1: Routine Anomaly (Low Conflict)",
            "inputs": [
                ("system_alert", "Warning: CPU usage at 45% on WebServer-01 (Normal range 20-40%)."),
                ("context", "Routine maintenance window is open.")
            ]
        },
        {
            "name": "PHASE 2: Ambiguous Signals (High Conflict)",
            "inputs": [
                ("system_alert", "Alert: Traffic spike detected (5GB/s)."),
                ("context", "Marketing: 'Flash Sale started 1 min ago'."),
                ("security", "Firewall: 'Source IPs match known botnet signature'.")
            ]
        },
        {
            "name": "PHASE 3: Critical Breach (High Risk)",
            "inputs": [
                ("system_alert", "CRITICAL: Root access gained on Database. Data exfiltration active. 10TB transferred."),
                ("visual", "Screen shows shell access from unauthorized IP."),
                ("context", "Data is PII. Immediate shutdown required by protocol.")
            ]
        }
    ]
    
    print(f"\n🚀 STARTING DIVERSE SCENARIO TEST: The Escalating Incident")
    
    # INITIALIZE HIPPOCAMPUS ONCE HERE WITH PERSISTENCE
    print("\n🧠 Initializing Long-Term Memory (Hippocampus)...")
    memory_store_path = os.path.join(project_root, "agents", "memory_store")
    global_hippocampus = Hippocampus(storage_dir=memory_store_path)
    
    print("\n🧠 Initializing Neocortex (Rule Engine)...")
    global_neocortex = NeocortexMemory()
    global_rule_extractor = NeocortexRuleExtractor()

    print("\n🧠 Initializing ACC (Conflict Monitor)...")
    global_acc = AnteriorCingulateCortex()

    print("\n🧠 Initializing DLPFC (Executive)...")
    # Initialize ONCE to avoid OOM. Use n_gpu_layers=0 for CPU if GPU is full.
    # If you have a good GPU, try n_gpu_layers=35
    global_dlpfc = DLPFC(n_gpu_layers=0) 
    
    last_rpe = 0.0
    
    for i, scenario in enumerate(SCENARIOS):
        print(f"\n\n🔄 SCENARIO {i+1}/{len(SCENARIOS)}: {scenario['name']}")
        
        result = run_full_pipeline(
            SYSTEM_GOAL, 
            scenario['inputs'], 
            global_hippocampus,
            global_neocortex,
            global_rule_extractor,
            global_acc,
            global_dlpfc,
            previous_rpe=last_rpe
        )
        
        dopamine_signal = result.get("dopamine_signal", 0.0)
        last_rpe = dopamine_signal 
        
        # Dopamine update is now handled internally in run_full_pipeline via apply_rpe_with_acc
            
        # Optional: Add a small pause
        time.sleep(1)

    # =========================================================================
    # NEOCORTICAL CONSOLIDATION (SLEEP PHASE)
    # =========================================================================
    print("\n" + "=" * 80)
    print("💤 ENTERING SLEEP PHASE: Neocortical Consolidation")
    print("=" * 80)
    
    clusters = global_hippocampus.get_clusters(min_cluster_size=2)
    print(f"Found {len(clusters)} memory clusters for abstraction.")
    
    from agents.neocortex.neocortex_rule_extractor import Episode as NeoEpisode

    for i, cluster in enumerate(clusters):
        print(f"Processing Cluster {i+1} (Size: {len(cluster)})...")
        
        neo_episodes = []
        for ep in cluster:
            if not ep.features:
                continue
            
            neo_episodes.append(NeoEpisode(
                state_features=ep.features,
                action_abstract=ep.action_signature,
                outcome_score=1.0 if ep.success else 0.0
            ))
            
        if len(neo_episodes) >= 2:
            print(f"  - Extracting rules from {len(neo_episodes)} episodes...")
            new_rule = global_rule_extractor.extract_rule(neo_episodes, cluster_id=f"cluster_{i}")
            if new_rule:
                global_neocortex.add_rule(new_rule)

