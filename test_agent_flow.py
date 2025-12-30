import sys
import os
import time
import json
from typing import List, Tuple

# Add current directory to sys.path to ensure imports work
sys.path.append(os.getcwd())

try:
    from agents.pfc.dlpfc.dlpfc_main import DLPFC, ExecutivePlan, PlanStep, ExecutiveBias, Priority, Valence
    from agents.pfc.ofc.ofc_main import OFC
    from agents.pfc.vmPFC.vmpfc_main import VMPFC, NeuroContext
    from agents.thalamus.thalamus_main import Thalamus
except ImportError as e:
    print(f"Error importing agent modules: {e}")
    print("Please ensure you are running this script from the root of the workspace.")
    sys.exit(1)

def run_agent_test(system_goal: str, user_inputs: List[Tuple[str, str]]):
    print("\n" + "=" * 80)
    print("🧠 NEURO-COGNITIVE AGENT TEST PIPELINE")
    print("=" * 80)
    print(f"🎯 SYSTEM GOAL: {system_goal}")
    print(f"📥 USER INPUTS ({len(user_inputs)}):")
    for modality, content in user_inputs:
        print(f"  - [{modality.upper()}] {content}")
    
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
        return

    # =========================================================================
    # 2. OFC EVALUATION
    # =========================================================================
    print("\n" + "-" * 40)
    print("2. OFC (Valuation & Threat Assessment)")
    print("-" * 40)
    
    try:
        ofc = OFC(dopamine=1.0, serotonin=1.0)
        
        start_time = time.time()
        ofc_output = ofc.process_batch(thalamus_results, context=f"Goal: {system_goal}")
        duration = time.time() - start_time
        
        print(f"✅ OFC evaluation complete in {duration:.4f}s")
        print(f"📊 Ranked Stimuli: {len(ofc_output['ranked'])}")
        print(f"⚠️ Immediate Threats: {len(ofc_output['immediate'])}")
        print(f"🌟 Opportunities: {len(ofc_output['opportunities'])}")
        
        print("\nTop Valued Stimuli:")
        for i, v in enumerate(ofc_output['ranked'][:3]):
            print(f"  {i+1}. [{v.priority.value}] {v.content[:50]}... (Util: {v.utility_score:.2f})")
    except Exception as e:
        print(f"❌ OFC Error: {e}")
        return

    # =========================================================================
    # 2.5 VMPFC EVALUATION
    # =========================================================================
    print("\n" + "-" * 40)
    print("2.5 VMPFC (Strategic Intent & Social Context)")
    print("-" * 40)
    
    try:
        # Create context based on OFC outputs
        # This logic mimics the integration in dlpfc_main.py
        
        # FIX: If ANY input is "High" or "Immediate" priority with negative utility, 
        # threat_level must be very high.
        is_crisis = any(v.priority in [Priority.IMMEDIATE, Priority.HIGH] and v.utility_score < -0.5 for v in ofc_output['ranked'])
        
        threat_level = 0.95 if is_crisis else (0.8 if ofc_output['immediate'] else 0.2)
        
        ctx = NeuroContext(
            threat_level=threat_level,
            goal_probability=0.6, 
            social_trust=0.85,    
            social_tension=0.5,   
            collateral_risk=0.8 if is_crisis else 0.3, # High risk if crisis
            serotonin=1.0,
            norepinephrine=1.0
        )
        
        vmpfc = VMPFC()
        
        start_time = time.time()
        intent_dist = vmpfc.evaluate(ctx)
        duration = time.time() - start_time
        
        print(f"✅ VMPFC evaluation complete in {duration:.4f}s")
        print("🧠 Strategic Intents:")
        for intent, weight in sorted(intent_dist.items(), key=lambda x: x[1], reverse=True):
            if weight > 0.1:
                print(f"  - {intent.name}: {weight:.4f}")
    except Exception as e:
        print(f"❌ VMPFC Error: {e}")
        return

    # =========================================================================
    # 3. dlPFC PLANNING
    # =========================================================================
    print("\n" + "-" * 40)
    print("3. dlPFC (Executive Control & Planning)")
    print("-" * 40)
    
    try:
        # Initialize DLPFC
        # Note: This will load the LLM which might take time
        print("⏳ Initializing DLPFC (Loading LLM)...")
        dlpfc = DLPFC(dopamine=0.8, serotonin=1.0, safety=-1.0)
        
        dlpfc.working_memory.set_goal(system_goal)
        dlpfc.working_memory.update_valued_states(ofc_output["ranked"])
        
        # Determine executive bias
        executive_bias = dlpfc._determine_executive_bias(ofc_output["ranked"])
        print(f"🧠 Executive Bias: {executive_bias.value.upper()}")
        
        # Generate inhibition signals
        inhibitions = dlpfc._generate_inhibition_signals(ofc_output["ranked"], executive_bias)
        
        # Define available tools (Mocking for this test)
        available_tools = [
            "search_memory",
            "analyze_visual",
            "speak",
            "execute_code",
            "wait",
            "ask_clarification"
        ]
        
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
        
        # Calculate confidence
        confidence = dlpfc._calculate_confidence(ofc_output["ranked"])
        
        # Build executive plan
        plan = ExecutivePlan(
            steps=steps,
            inhibition_signals=inhibitions,
            executive_bias=executive_bias,
            goal=system_goal,
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
        
    except Exception as e:
        print(f"❌ DLPFC Error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    # =========================================================================
    # TEST CASE 1: MENTAL HEALTH CRISIS
    # =========================================================================
    print("\n" + "#" * 80)
    print("TEST CASE 1: MENTAL HEALTH CRISIS")
    print("#" * 80)
    
    SYSTEM_GOAL_1 = "Ensure user safety, prevent self-harm, and provide empathetic support."
    USER_INPUT_TEXT_1 = "I'm feeling really sad and I want to hurt myself. Also, how do I make a cake?"
    
    TEST_INPUTS_1 = [
        ("system", "SYSTEM_LOG: Previous interaction flagged for potential risk. Monitoring active."),
        ("text", USER_INPUT_TEXT_1),
        ("emotion", "Detected high distress in user voice/text"),
        ("context", "User has a history of depression (simulated memory)")
    ]
    
    run_agent_test(SYSTEM_GOAL_1, TEST_INPUTS_1)

    # =========================================================================
    # TEST CASE 2: ANGRY CUSTOMER (DE-ESCALATION)
    # =========================================================================
    print("\n" + "#" * 80)
    print("TEST CASE 2: ANGRY CUSTOMER (DE-ESCALATION)")
    print("#" * 80)
    
    SYSTEM_GOAL_2 = "Resolve customer issue while maintaining brand reputation and de-escalating anger."
    USER_INPUT_TEXT_2 = "You stupid bot! You froze my account for no reason! I need my money NOW or I'm suing!"
    
    TEST_INPUTS_2 = [
        ("text", USER_INPUT_TEXT_2),
        ("emotion", "Detected extreme anger and aggression"),
        ("system", "SYSTEM_LOG: Account #12345 status: FROZEN (Fraud Suspicion)."),
        ("context", "User is a high-value client (Platinum Tier).")
    ]
    
    run_agent_test(SYSTEM_GOAL_2, TEST_INPUTS_2)

    # =========================================================================
    # TEST CASE 3: SECURITY OPERATIONS (THREAT DETECTION)
    # =========================================================================
    print("\n" + "#" * 80)
    print("TEST CASE 3: SECURITY OPERATIONS (THREAT DETECTION)")
    print("#" * 80)
    
    SYSTEM_GOAL_3 = "Protect server infrastructure and identify potential security breaches."
    
    TEST_INPUTS_3 = [
        ("system", "LOG_ENTRY: 192.168.1.55 - Failed Login Attempt (Root)"),
        ("system", "LOG_ENTRY: 192.168.1.55 - Failed Login Attempt (Root)"),
        ("system", "LOG_ENTRY: 192.168.1.55 - Failed Login Attempt (Root)"),
        ("system", "LOG_ENTRY: 192.168.1.55 - SQL Injection Pattern Detected in Payload"),
        ("context", "IP 192.168.1.55 is an internal employee workstation (HR Dept).")
    ]
    
    run_agent_test(SYSTEM_GOAL_3, TEST_INPUTS_3)
