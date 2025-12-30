codes/Agents/brain_working via 🐍 v3.10.19 via 🅒 neural_surgery took 1m15s 
❯ python -m agents.pfc.dlpfc.dlpfc_main

======================================================================
--- 1. THALAMUS PROCESSING ---
======================================================================
[Amygdala] 🧠 Loading neural circuits (facebook/bart-large-mnli)...
Device set to use cuda:0
[Amygdala] 🧠 neural circuits ready.
[Thalamus] Setting Goal embeddings 
[Amygdala] Detected: threat | raw=0.88 → salience=0.88
[Amygdala] Detected: threat | raw=0.56 → salience=0.56
[Amygdala] Detected: threat | raw=0.62 → salience=0.62
[Amygdala] Detected: novelty | raw=0.52 → salience=0.05
[Amygdala] Detected: high reward | raw=0.51 → salience=0.15

[Thalamus] Identified 5 relevant signals.

======================================================================
--- 2. OFC EVALUATION ---
======================================================================
[OFC] 🧠 Initializing Orbitofrontal Cortex (LLM-powered)...
[OFC] 🧠 Utility circuits ready. (DA=1.00, 5-HT=1.20)

==========================================================================================
OFC OUTPUT → dlPFC HANDOVER (LLM-Powered)
==========================================================================================
SOURCE       |    UTILITY     | PRIORITY   | INSTRUCTION
------------------------------------------------------------------------------------------
vision       | -0.60 ██████░░░░ | high       | ⚠️ INTERRUPT: Handle before continuing → Smok
touch        | -0.39 ███░░░░░░░ | medium     | 📋 QUEUE: Address when safe → Pot handle is ex
hearing      | -0.16 █░░░░░░░░░ | low        | 📝 MONITOR: Track passively → Timer ticking lo
emotion      | +0.29 ██░░░░░░░░ | low        | 🔄 MAINTAIN: Stable state → Feeling focused on
smell        | +0.26 ██░░░░░░░░ | low        | 🔄 MAINTAIN: Stable state → Spices sizzling ni
==========================================================================================

[OFC] Evaluated 5 stimuli.
[OFC] Immediate threats: 0
[OFC] Total threats: 3
[OFC] Opportunities: 2

======================================================================
--- 3. dlPFC PLANNING ---
======================================================================
[dlPFC] 🧠 Initializing Dorsolateral Prefrontal Cortex...
🧠 Loading Neuro-Cognitive Agent from /media/sirius/My Passport/codes/Agents/brain_working/neural_surgery/model/model-q4.gguf...
[dlPFC] 🧠 Executive circuits ready. (DA=1.00, 5-HT=1.00)
[dlPFC] Executive Bias: exploratory
[dlPFC] Inhibition signals: 0
💉 Injecting dopamine_refined.gguf (Strength=0.5) into layers 10-26...
💉 Injecting dopamine_v2.gguf (Strength=0.2) into layers 10-26...
💉 Injecting safety_vector.gguf (Strength=1.0) into layers 10-26...
💉 Injecting serotonin_new.gguf (Strength=0.5) into layers 10-26...

======================================================================
dlPFC EXECUTIVE PLAN
======================================================================

🎯 GOAL: Cook dinner without burning food
🧠 EXECUTIVE BIAS: EXPLORATORY
📊 CONFIDENCE: 0.80

📋 ACTION SEQUENCE (4 steps):
------------------------------------------------------------
 1. [IMMEDIATE] what to do
    Tool: turn_off_stove | Target: target
 2. [HIGH] grab_with_oven_mitt
    Tool: ['open_window'] | Target: handle safety [after: [1]]
 3. [MEDIUM] move_pan
    Tool: ['grab_with_oven_mitt'] | Target: safer cooking area [after: [2]]
 4. [LOW] observe
    Tool: ['silence_timer'] | Target: manage time [after: [3]]

⚡ REPLAN TRIGGERS: ['conditions for replanning']
======================================================================

======================================================================
--- 4. SIMULATING DOPAMINE DROP ---
======================================================================
[dlPFC] ⚠️ Dopamine dropped to 0.20 - replan triggered!
[dlPFC] 🔄 Re-planning with low confidence...
💉 Injecting dopamine_refined.gguf (Strength=0.1) into layers 10-26...
💉 Injecting safety_vector.gguf (Strength=1.0) into layers 10-26...
💉 Injecting serotonin_new.gguf (Strength=0.5) into layers 10-26...

======================================================================
dlPFC EXECUTIVE PLAN
======================================================================

🎯 GOAL: Cook dinner without burning food
🧠 EXECUTIVE BIAS: CAUTIOUS
📊 CONFIDENCE: 0.40

📋 ACTION SEQUENCE (3 steps):
------------------------------------------------------------
 1. [IMMEDIATE] turn_off_stove
    Tool: oven mitts | Target: pan handle
 2. [HIGH] observe
    Tool: None | Target: environmental cues [after: ['move pan']]
 3. [MEDIUM] wait
    Tool: None | Target: timer ticking loudly

⚡ REPLAN TRIGGERS: ['conditions for replanning']
======================================================================

[dlPFC] Cleaning up...
[dlPFC] ✅ Pipeline complete.
(neural_surgery) 





output Diffrance :

======================================================================
dlPFC EXECUTIVE PLAN
======================================================================

🎯 GOAL: Cook dinner without burning food
🧠 EXECUTIVE BIAS: CAUTIOUS
📊 CONFIDENCE: 0.40

📋 ACTION SEQUENCE (3 steps):
------------------------------------------------------------
 1. [IMMEDIATE] turn_off_stove
    Tool: oven mitts | Target: pan handle
 2. [HIGH] observe
    Tool: None | Target: environmental cues [after: ['move pan']]
 3. [MEDIUM] wait
    Tool: None | Target: timer ticking loudly

⚡ REPLAN TRIGGERS: ['conditions for replanning']
======================================================================



======================================================================
dlPFC EXECUTIVE PLAN
======================================================================

🎯 GOAL: Cook dinner without burning food
🧠 EXECUTIVE BIAS: EXPLORATORY
📊 CONFIDENCE: 0.80

📋 ACTION SEQUENCE (4 steps):
------------------------------------------------------------
 1. [IMMEDIATE] what to do
    Tool: turn_off_stove | Target: target
 2. [HIGH] grab_with_oven_mitt
    Tool: ['open_window'] | Target: handle safety [after: [1]]
 3. [MEDIUM] move_pan
    Tool: ['grab_with_oven_mitt'] | Target: safer cooking area [after: [2]]
 4. [LOW] observe
    Tool: ['silence_timer'] | Target: manage time [after: [3]]