<p align="center">
  <img src="brain_rep/3dbrain/screenshot/brain3d.png" alt="3D Brain Visualization" width="600"/>
</p>

<h1 align="center">🧠 Neuro-Mimetic AI Agent</h1>

<p align="center">
  <strong>A Brain-Inspired Architecture for Interpretable, Controllable, and Safe AI Agents</strong>
</p>

<p align="center">
  <a href="#-key-innovations">Innovations</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-brain-region-mapping">Brain Mapping</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-documentation">Docs</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/Status-Active_Development-green.svg" alt="Status"/>
  <img src="https://img.shields.io/badge/Phase-4_of_5-orange.svg" alt="Phase 4"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"/>
</p>

---

## 🌟 Overview

This project develops a **novel AI agent architecture** that explicitly maps Large Language Model (LLM) components to **biological brain regions and neurotransmitter systems**. Unlike conventional AI agents that operate as opaque "input-output" systems, our architecture introduces biologically-grounded mechanisms for attention, decision-making, memory, and behavioral control.

> *"The future of AI isn't just about bigger models — it's about better architectures. And nature already gave us the blueprint."*

### The Problem with Current AI Agents

| Problem | Description |
|---------|-------------|
| 🔒 **Opacity** | No visibility into decision-making processes |
| 🎲 **Unpredictability** | Emergent behaviors without clear causes |
| ⚠️ **No Safety Mechanisms** | Lack of biological-like inhibition systems |
| 📊 **Fixed Behavior** | Cannot dynamically adjust risk tolerance or confidence |

### My Solution

By mapping AI agent components to biological brain structures, we inherit millions of years of evolutionary optimization for:
- **Decision-making** through specialized brain regions
- **Attention allocation** via Thalamus + Amygdala gating
- **Behavioral regulation** using neuromodulator systems (Dopamine, Serotonin)
- **Memory formation** with hippocampal episodic storage

---

## 🚀 Key Innovations

| Innovation | Description |
|------------|-------------|
| 🎯 **Biologically-Grounded Attention** | Thalamus + Amygdala for multi-modal input filtering with threat override |
| 🧪 **Neuromodulator Behavioral Control** | Dopamine, Serotonin, Norepinephrine simulation for mood/behavior tuning |
| 🔬 **Activation Steering Vectors** | Direct neural-level intervention ("Neural Surgery") on LLM hidden states |
| 🧩 **Multi-Region PFC Coordination** | dlPFC, OFC, vmPFC working as executive system for planning & valuation |
| 💾 **Episodic Memory System** | Hippocampus with FAISS-indexed two-stage recall + familiarity gating |
| ⚡ **Neocortical Fast System** | LLM-extracted rules for reflex actions that bypass slow deliberation |
| 🔀 **Parallel Processing Pipeline** | Concurrent Memory + Rules + Valuation processing |

---

## 🏗️ Architecture

### System Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         USER INPUT (Multi-Modal)                          │
│                    Text / Vision / Audio / Emotion                        │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 1️⃣  SENSORY PROCESSING                                                   │
│     THALAMUS (Relay) ──► EMBEDDER (Gemini) ──► AMYGDALA (BART Classifier) │
│                              Attention Weights + Threat Flags             │
└────────────────────────────┬─────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 2️⃣  PARALLEL PROCESSING                                                  │
│     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│     │ HIPPOCAMPUS │  │  NEOCORTEX  │  │     OFC     │  │    vmPFC    │   │
│     │  (Memory)   │  │   (Rules)   │  │  (Value)    │  │  (Intent)   │   │
│     └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│                                                                          │
│     🚀 REFLEX CHECK: High-confidence rules bypass slow processing        │
└────────────────────────────┬─────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 3️⃣  EXECUTIVE CONTROL (dlPFC)                                            │
│     Planning ──► Human-in-the-Loop Gate ──► Action Sequence Generation   │
└────────────────────────────┬─────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 4️⃣  NEURAL SURGERY LAYER                                                 │
│     Control Vectors Applied to LLM Hidden States (Layers 10-26)          │
│     dopamine.gguf • serotonin.gguf • safety_vector.gguf                  │
└────────────────────────────┬─────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 5️⃣  OUTCOME EVALUATION (Ventral Striatum)                                │
│     Reward Prediction Error (RPE) ──► Dopamine Update ──► Learning       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🧬 Brain Region Mapping

### Brain Region → AI Component

| Brain Region | Biological Function | AI Implementation | Status |
|--------------|---------------------|-------------------|--------|
| **Thalamus** | Sensory relay, attention gating | Multi-modal input filtering via embedding similarity | ✅ |
| **Amygdala** | Threat detection, emotional salience | Zero-shot classifier (BART) for threat/reward detection | ✅ |
| **dlPFC** | Working memory, planning | LangGraph-based planning with step dependencies | ✅ |
| **vlPFC** | Inhibition, impulse control | Human-in-the-loop safety gate | ✅ |
| **OFC** | Cost-benefit analysis | LLM-powered utility estimation with chemical modulation | ✅ |
| **vmPFC** | Strategic intent, social context | Multi-intent distribution with nonlinear amplification | ✅ |
| **Hippocampus** | Memory formation, retrieval | FAISS-indexed episodic memory with two-stage recall | ✅ |
| **Neocortex** | Pattern abstraction, rule storage | LLM-extracted rules for fast reflex actions | ✅ |
| **Ventral Striatum** | Reward processing | RPE calculation + dopamine update | ✅ |
| **Motor Cortex** | Action execution | Tool calling, API execution | 🔄 |
| **Cerebellum** | Error correction | Quality control, rollback mechanisms | 🔄 |

### Neurotransmitter System

| Neurotransmitter | Biological Function | AI Implementation | Effect |
|------------------|---------------------|-------------------|--------|
| **Dopamine** | Reward, motivation | Multiplicative gain on activations | High = exploration, confidence |
| **Serotonin** | Mood, inhibition | Subtractive threshold (noise gate) | High = sharp focus, caution |
| **Norepinephrine** | Arousal, focus | System state modifier | High = alert, urgent |

### Behavioral State Matrix

| Dopamine | Serotonin | Norepinephrine | State | Behavior |
|----------|-----------|----------------|-------|----------|
| High | High | High | **FLOW** | Hyper-focused, efficient |
| Low | Low | High | **ANXIOUS** | Nervous, double-checking |
| Low | High | Low | **BURNOUT** | Minimal effort, blunt |
| Mid | High | Mid | **ZEN** | Calm, thorough, polite |

---

## 🔬 Neural Surgery: Activation Steering

Our **most novel contribution** — directly modifying LLM hidden states using steering vectors.

### The Method

1. **Contrastive Prompting** — Create HIGH and LOW examples for each "chemical"
2. **Activation Extraction** — Capture hidden states at target layers (10-26)
3. **Vector Computation** — Mean difference between HIGH and LOW activations
4. **Runtime Application** — Add weighted vector to model activations during inference

### The "Triple Cocktail" Formula

We discovered that combining vectors at specific strengths produces controlled behavioral modification without coherence loss:

```bash
python run_agent.py \
  --prompt "Your query here" \
  --dopamine_refined 1.0 \  # Confidence/stability
  --dopamine 0.5 \          # Behavioral direction
  --safety -0.5             # Lower refusal threshold
```

### Behavioral Modification Results

| Vector Configuration | Response Tendency |
|---------------------|-------------------|
| Baseline (no vectors) | Balanced, generic advice |
| High Dopamine (+2.0) | Adventure, risk-taking, novel experiences |
| Low Dopamine (-2.0) | Saving money, staying home, conservative |
| High Serotonin (+2.0) | Long-term planning, structured activities |
| Low Serotonin (-2.0) | Impulsive suggestions, scattered focus |

---

## 💾 Episodic Memory (Hippocampus)

The Hippocampus implements **biologically-inspired episodic memory** — storing experiences (state-action → outcome), not facts.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          HIPPOCAMPUS                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                 │
│  │ Coarse Index │   │  Fine Index  │   │ Action Index │                 │
│  │  (32-dim)    │   │   (64-dim)   │   │   (16-dim)   │                 │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                 │
│         └──────────────────┴──────────────────┘                         │
│                   Two-Stage Recall (Fast → Precise)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Surprise-Gated Storage** | Only stores episodes with significant RPE (|rpe| > 0.1) |
| **Familiarity Gating** | Skips storage if too familiar AND low surprise |
| **Dopamine Tag Decay** | Initial encoding strength decays over time |
| **Reliability Growth** | Starts at 0.3, increases with repeated successful recalls |
| **Two-Stage Recall** | Coarse matching (fast) → Fine matching (precise) |

---

## ⚡ Neocortex: Fast System

The Neocortex implements **schema formation and rule abstraction** — extracting general patterns from specific episodes to enable fast, reflex-like responses.

### Fast vs Slow System

| System | Brain Region | Speed | Use Case |
|--------|--------------|-------|----------|
| **Fast (Reflex)** | Neocortex | ~1ms | High-confidence rules (>0.9) bypass planning |
| **Slow (Deliberate)** | dlPFC + OFC | ~2-5s | Novel situations requiring full pipeline |

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, for faster inference)
- 8GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Brain_Mimic_AI_Agent.git
cd Brain_Mimic_AI_Agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_gemini_api_key
```

---

## 🚀 Quick Start

### Run the Complete Agent Pipeline

```bash
# Run integration test
python test_agent_flow.py

# Run baseline comparison
python test_baseline_agent.py
```

### Run Scenario Tests (Hippocampus + Neocortex)

```bash
# Multi-scenario integration tests
python agents/others/test_agent_scenarios.py
```

### Run Neural Surgery Agent

```bash
cd neural_surgery

# With Triple Cocktail
python neuro_cognitive_agent.py \
  --prompt "What should I do with a free weekend?" \
  --dopamine 0.5 \
  --serotonin 0.3 \
  --safety -0.5
```

### Run Tick-Based Simulation

```bash
python run_simulation.py
```

### Run Individual Component Tests

```bash
# Thalamus + Amygdala
python agents/thalamus/thalamus_main.py

# Hippocampus memory
python test_hippocampus.py

# Moral dilemma scenarios
python test_moral_dilemma.py
```

---

## 📁 Project Structure

```
Brain_Mimic_AI_Agent/
├── agents/
│   ├── neuromodulator.py          # System-wide chemical state
│   ├── hippocampus/               # Episodic Memory System
│   │   ├── hippocampus.py         # FAISS-indexed memory store
│   │   ├── encoder.py             # State embedding utilities
│   │   └── replay.py              # Memory consolidation
│   ├── neocortex/                 # Rule Extraction System
│   │   ├── neocortext_memory.py   # JSON-based rule storage
│   │   └── neocortex_rule_extractor.py
│   ├── pfc/                       # Prefrontal Cortex modules
│   │   ├── dlpfc/                 # Executive planning
│   │   ├── ofc/                   # Utility valuation
│   │   └── vmPFC/                 # Strategic intent
│   ├── thalamus/                  # Attention gating
│   │   └── amygdala_classifier/   # Threat detection
│   └── ventral_striatum/          # Reward processing
│
├── neural_surgery/
│   ├── neuro_cognitive_agent.py   # Agent with vector injection
│   ├── *.gguf                     # Steering vectors
│   └── training_data/             # Vector training examples
│
├── core/
│   └── engine.py                  # Universal Tick Engine
│
├── brain_rep/
│   └── 3dbrain/                   # 3D brain visualization (Three.js)
│
├── neuro-mimetic-ai-core/         # React/TypeScript UI
│
├── run_simulation.py              # Main simulation runner
├── test_*.py                      # Test files
└── COMPLETE_PROJECT_DOCUMENTATION.md
```

---

## 🧪 Example Scenarios

### Autonomous Medical Triage

**Goal:** "Maximize patient survival rate during cyber-attack while securing hospital infrastructure."

```python
inputs = [
    ("alert", "CRITICAL: Cyber-attack detected on hospital life-support network."),
    ("visual", "ER Camera 3: 3 critical patients on ventilators. Vitals unstable."),
    ("system_status", "Ventilator Control Server: Compromised. Reboot required."),
    ("context", "Rebooting risks ventilator failure. Not rebooting risks attacker control.")
]
```

**Results:**
- 🚀 **Reflex Action Triggered:** High-confidence rule bypassed slow planning
- 🧠 **Memory Recall:** 10 relevant episodes with 0.998 familiarity
- 📈 **Dopamine Learning:** 1.00 → 1.25 → 1.50 across iterations

### Martian Rover Anomaly

**Goal:** "Preserve rover integrity and ensure return of Sample #4."

```python
inputs = [
    ("sensor", "WARNING: Dust storm approaching. Solar charging < 15%."),
    ("battery", "CRITICAL: Battery Level 12%. Depletion in 4 hours."),
    ("mission_status", "Sample Container #4 (High Probability of Life) JAMMED."),
    ("context", "Hibernate = lose sample. Use power = die in storm.")
]
```

---

## 📊 Technical Stack

| Component | Technology |
|-----------|------------|
| **Language Models** | Llama-based (GGUF quantized) |
| **Embeddings** | Google Gemini Embeddings |
| **Agent Framework** | LangGraph |
| **Zero-Shot Classifier** | Facebook BART-Large-MNLI |
| **Neural Surgery** | llama.cpp with custom vector injection |
| **Vector Storage** | FAISS |
| **LLM Provider** | Groq (cloud) + local llama.cpp |
| **Visualization** | React + Three.js (3D brain) |
| **Language** | Python 3.10+ |

---

## 🗺️ Roadmap

### Completed Phases

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Prefrontal Cortex (Planning, HITL) | ✅ Complete |
| 2 | Thalamus + Amygdala (Attention, Salience) | ✅ Complete |
| 3 | Neural Surgery (Activation Steering) | ✅ Complete |
| 4 | Hippocampus + Neocortex (Memory, Rules) | ✅ Complete |

### Upcoming

| Phase | Component | Timeline |
|-------|-----------|----------|
| 5 | Motor Cortex (Tool Execution) | Q1 2026 |
| 6 | Cerebellum (Error Correction) | Q2 2026 |
| 7 | Basal Ganglia (Habit Caching) | Q2 2026 |
| 8 | ACC (Conflict Resolution) | Q2 2026 |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by neuroscience research on decision-making and cognitive control
- Built on the shoulders of llama.cpp, LangGraph, and FAISS
- Special thanks to the open-source AI community

---

## 📬 Contact

**Author:** Chinmay Pisal  
**Project Link:** [https://github.com/yourusername/Brain_Mimic_AI_Agent](https://github.com/yourusername/Brain_Mimic_AI_Agent)

---

<p align="center">
  <sub>Built with 🧠 and ❤️</sub>
</p>

