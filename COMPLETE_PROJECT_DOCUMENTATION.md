# 🧠 Neuro-Mimetic AI Agent System
## Complete Technical Documentation & Architecture Reference

---

**Project:** Brain-Inspired Architecture for Interpretable and Controllable AI Agents  
**Status:** Active Development — Phase 4 of 5  
**Last Updated:** January 2, 2026  
**Author:** Sirius  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Core Philosophy & Research Motivation](#2-core-philosophy--research-motivation)
3. [Complete Brain-to-AI Mapping](#3-complete-brain-to-ai-mapping)
4. [System Architecture](#4-system-architecture)
5. [Component Deep Dives](#5-component-deep-dives)
   - 5.1 [Thalamus - Sensory Relay & Attention](#51-thalamus---sensory-relay--attention)
   - 5.2 [Amygdala - Threat Detection & Salience](#52-amygdala---threat-detection--salience)
   - 5.3 [Prefrontal Cortex (PFC) - Executive Control](#53-prefrontal-cortex-pfc---executive-control)
   - 5.4 [Neuromodulator System](#54-neuromodulator-system)
   - 5.5 [Neural Surgery - Activation Steering](#55-neural-surgery---activation-steering)
   - 5.6 [Hippocampus - Episodic Memory System](#56-hippocampus---episodic-memory-system)
   - 5.7 [Neocortex - Rule Extraction & Fast System](#57-neocortex---rule-extraction--fast-system)
   - 5.8 [Parallel Processing Pipeline](#58-parallel-processing-pipeline)
6. [Data Flow & Processing Pipeline](#6-data-flow--processing-pipeline)
7. [Training Data & Vector Generation](#7-training-data--vector-generation)
8. [Key Technical Innovations](#8-key-technical-innovations)
9. [Experimental Results & Observations](#9-experimental-results--observations)
10. [Project Structure](#10-project-structure)
11. [Development Roadmap](#11-development-roadmap)
12. [Research Questions & Thesis Potential](#12-research-questions--thesis-potential)
13. [References & Related Work](#13-references--related-work)

---

## 1. Executive Summary

This project develops a **novel AI agent architecture** that explicitly maps Large Language Model (LLM) components to **biological brain regions and neurotransmitter systems**. Unlike conventional AI agents that operate as opaque "input-output" systems, our architecture introduces:

### Key Innovations

| Innovation | Description |
|------------|-------------|
| **Biologically-grounded attention** | Thalamus + Amygdala for multi-modal input filtering |
| **Neuromodulator behavioral control** | Dopamine, Serotonin, Norepinephrine simulation |
| **Activation Steering Vectors** | Direct neural-level intervention ("Neural Surgery") |
| **Multi-region PFC coordination** | dlPFC, OFC, vmPFC working as executive system |
| **Episodic Memory System** | Hippocampus with FAISS-indexed two-stage recall |
| **Neocortical Rule Learning** | LLM-extracted rules for reflex actions ("Fast System") |
| **Parallel Processing Pipeline** | Concurrent Memory + Rules + Valuation processing |

### Goals

1. **Interpretability** — Visible internal states mapped to brain regions
2. **Controllability** — Tunable via "chemical" parameters (dopamine, serotonin)
3. **Safety** — Biological gating mechanisms for inhibition and filtering

---

## 2. Core Philosophy & Research Motivation

### The Problem with Current AI Agents

Traditional LLM-based agents (LangChain, AutoGPT, CrewAI) suffer from:

| Problem | Description |
|---------|-------------|
| **Opacity** | No visibility into decision-making processes |
| **Unpredictability** | Emergent behaviors without clear causes |
| **Lack of Safety Mechanisms** | No biological-like inhibition systems |
| **Fixed Behavioral Profiles** | Cannot dynamically adjust risk tolerance or confidence |

### Our Core Hypothesis

> *By mapping AI agent components to biological brain structures, we can inherit millions of years of evolutionary optimization for decision-making, attention allocation, and behavioral regulation.*

### The Key Insight

The human brain doesn't just "chain thoughts" — it has:
- **Specialized regions** for different functions
- **Gating mechanisms** for inhibition
- **Feedback loops** for learning
- **Reward systems** that learn from prediction errors

We replicate this in software.

---

## 3. Complete Brain-to-AI Mapping

### Brain Region → AI Component Table

| Brain Region | Biological Function | AI Implementation | Status |
|--------------|---------------------|-------------------|--------|
| **Thalamus** | Sensory relay, attention gating | Multi-modal input filtering via embedding similarity + chemical modulation | ✅ Complete |
| **Amygdala** | Threat detection, emotional salience | Zero-shot classifier (BART) for threat/reward detection with gain amplification | ✅ Complete |
| **dlPFC** (Dorsolateral PFC) | Working memory, planning | LangGraph-based planning with step dependencies | ✅ Complete |
| **vlPFC** (Ventrolateral PFC) | Inhibition, impulse control | Human-in-the-loop safety gate | ✅ Complete |
| **OFC** (Orbitofrontal Cortex) | Cost-benefit analysis | LLM-powered utility estimation with chemical modulation | ✅ Complete |
| **mPFC** (Medial PFC) | Confidence monitoring | Dopamine-based strategy evaluation | ✅ Complete |
| **vmPFC** (Ventromedial PFC) | Strategic intent, social context | Multi-intent distribution with nonlinear amplification | ✅ Complete |
| **aPFC** (Anterior PFC) | Metacognition | Re-planning triggers on low dopamine | ✅ Complete |
| **Basal Ganglia** | Action selection, inhibition | Serotonin-gated safety filtering | ✅ Partial |
| **Hippocampus** | Memory formation, retrieval | FAISS-indexed episodic memory with two-stage recall + familiarity gating | ✅ Complete |
| **Neocortex** | Pattern abstraction, rule storage | LLM-powered rule extraction + JSON-based schema memory | ✅ Complete |
| **Motor Cortex** | Action execution | Tool calling, API execution | 🔄 Planned |
| **Cerebellum** | Error correction, motor refinement | Quality control, rollback mechanisms | 🔄 Planned |

### Neurotransmitter System Mapping

| Neurotransmitter | Biological Function | AI Implementation | Effect |
|------------------|---------------------|-------------------|--------|
| **Dopamine** | Reward prediction, motivation | Multiplicative gain on activations | High = exploration, confidence; Low = caution |
| **Serotonin** | Mood regulation, impulse inhibition | Subtractive threshold (noise gate) + softmax temperature | High = sharp focus, inhibition; Low = scattered, impulsive |
| **Norepinephrine** | Arousal, focus, urgency | System state modifier | High = alert, urgent; Low = calm, methodical |

---

## 4. System Architecture

### Complete System Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         USER INPUT (Multi-Modal)                          │
│                    Text / Vision / Audio / Emotion                        │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 1️⃣  SENSORY PROCESSING LAYER                                             │
│                                                                          │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│   │  THALAMUS   │───▶│   SENSORY   │───▶│  AMYGDALA   │                  │
│   │  (Relay)    │    │   EMBEDDER  │    │ (Salience)  │                  │
│   │             │    │  (Gemini)   │    │  (BART)     │                  │
│   └─────────────┘    └─────────────┘    └─────────────┘                  │
│         │                                      │                         │
│         └──────────────────┬───────────────────┘                         │
│                            │ Attention Weights + Threat Flags            │
└────────────────────────────┼─────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 2️⃣  VALUATION LAYER (OFC)                                                │
│                                                                          │
│   Utility = (Reward × Dopamine) - (Cost × Serotonin)                     │
│                                                                          │
│   • LLM-based semantic threat/reward estimation                          │
│   • Priority assignment (IMMEDIATE → BACKGROUND)                         │
│   • Valence classification (POSITIVE/NEGATIVE)                           │
└────────────────────────────┼─────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 3️⃣  STRATEGIC INTENT LAYER (vmPFC)                                       │
│                                                                          │
│   Intent Distribution Calculation:                                       │
│   • PRESERVE_LIFE      • MISSION_SUCCESS    • DEESCALATE                │
│   • MINIMIZE_DAMAGE    • MAINTAIN_TRUST                                  │
│                                                                          │
│   Uses: Nonlinear spike functions, suppression, inertia                  │
└────────────────────────────┼─────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 4️⃣  EXECUTIVE CONTROL LAYER (dlPFC)                                      │
│                                                                          │
│   ┌───────────────────────────────────────────────────────────────────┐  │
│   │                    PREFRONTAL CORTEX (PFC)                        │  │
│   ├───────────┬───────────┬───────────┬───────────┬───────────────────┤  │
│   │   dlPFC   │   vlPFC   │    OFC    │   mPFC    │       aPFC        │  │
│   │ (Planning)│(Inhibit)  │(Value)    │(Confidence│  (Metacognition)  │  │
│   │           │           │           │  Monitor) │                   │  │
│   └─────┬─────┴─────┬─────┴─────┬─────┴─────┬─────┴─────────┬─────────┘  │
│         │           │           │           │               │            │
│    Plan Gen    Human Gate   Cost/Benefit  Dopamine      Re-Planning     │
│    (LLM)       (HITL)       Calculation   Feedback      Triggers        │
│                                                                          │
│   Neuromodulator Inputs:                                                 │
│   ├── Dopamine ──────▶ Gain/Confidence Amplification                     │
│   └── Serotonin ─────▶ Threshold/Inhibition Control                      │
└────────────────────────────┼─────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 5️⃣  NEURAL SURGERY LAYER (Activation Steering)                           │
│                                                                          │
│   Control Vectors Applied to LLM Hidden States (Layers 10-26):           │
│   • dopamine_v2.gguf          → Risk/Confidence direction                │
│   • dopamine_refined.gguf     → Stability + decisiveness                 │
│   • serotonin_new.gguf        → Focus + caution                          │
│   • safety_vector.gguf        → Compliance threshold                     │
│                                                                          │
│   "Triple Cocktail" Formula for controlled behavioral modification       │
└────────────────────────────┼─────────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 6️⃣  ACTION EXECUTION (Motor Cortex - Planned)                            │
│                                                                          │
│   • Tool Execution (Function calling, APIs)                              │
│   • Code Interpreter                                                     │
│   • Error Recovery & Rollback                                            │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Component Deep Dives

### 5.1 Thalamus - Sensory Relay & Attention

**File:** `agents/thalamus/thalamus_main.py`

The Thalamus serves as the **sensory relay station**, implementing biologically-inspired attention mechanisms.

#### Core Functions

1. **Goal-Directed Attention** — Embedding similarity between inputs and current goal
2. **Amygdala Hijack** — Threat signals can override goal relevance
3. **Chemical Modulation** — Dopamine/serotonin adjust sensitivity and focus

#### Key Algorithm: Process Signals

```python
def process(self, inputs: List[Tuple[str, str]]):
    # 1. Compute activation per signal
    for source, content in inputs:
        emb = self._normalize(self.embedding_model.encode(content))
        relevance = self._cosine_similarity(emb, self.current_goal_embedding)
        
        emo = self._amygdala_scan(content)
        
        # Amygdala Override ("Fight or Flight")
        if emo["label"] in ["threat", "physical pain"] and emo["salience"] >= threshold:
            activation = emo["salience"] * 1.5   # Ignore goal temporarily
        else:
            activation = relevance * (1 + emo["salience"])
    
    # 2. Hard gate (Serotonin)
    gate = 0.4 + self.serotonin * 0.3
    gated = [a for a in activations if a["activation"] >= gate]
    
    # 3. Dopamine gain
    for g in gated:
        g["activation"] *= (1 + self.dopamine)
    
    # 4. Softmax normalization with temperature
    temperature = max(0.1, 1.0 - self.serotonin)
    weights = softmax(scores, temperature)
```

#### Softmax Temperature Control

| Serotonin Level | Temperature | Attention Pattern |
|-----------------|-------------|-------------------|
| High (0.8+) | Low (0.2) | Winner-takes-all (sharp focus) |
| Medium (0.5) | Medium (0.5) | Distributed with peaks |
| Low (0.2-) | High (0.8) | Scattered (diffuse attention) |

---

### 5.2 Amygdala - Threat Detection & Salience

**File:** `agents/thalamus/amygdala_classifier/classifire_main.py`

The Amygdala is the brain's "fear center" — a zero-shot classification pipeline detecting threat/urgency.

#### Classification Labels

```python
self.labels = [
    "threat",           # Danger, harm
    "physical pain",    # Bodily injury
    "social conflict",  # Interpersonal tension
    "high reward",      # Opportunity, gain
    "novelty",          # New, unexpected
]
```

#### Gain Amplification Table

| Label | Gain | Rationale |
|-------|------|-----------|
| `physical pain` | 1.2 | Highest priority — survival |
| `threat` | 1.0 | Immediate attention required |
| `social conflict` | 0.5 | Important but not urgent |
| `high reward` | 0.3 | Worth noting |
| `novelty` | 0.1 | Background processing |

#### Salience Calculation

```python
salience = min(raw_score * self.gain[label], 1.0)
```

---

### 5.3 Prefrontal Cortex (PFC) - Executive Control

The PFC is implemented as a multi-component system with specialized sub-regions.

#### 5.3.1 dlPFC (Dorsolateral PFC) - Planning

**File:** `agents/pfc/dlpfc/dlpfc_main.py`

The executive controller responsible for:
- Working memory management
- Plan generation with dependencies
- Executive bias determination
- Inhibition signal generation

##### Executive Bias States

```python
class ExecutiveBias(Enum):
    EMERGENCY = "emergency"     # Immediate threats detected
    FOCUSED = "focused"         # High dopamine, goal-oriented
    EXPLORATORY = "exploratory" # Balanced state
    CAUTIOUS = "cautious"       # Low dopamine, careful
```

##### Plan Step Structure

```python
@dataclass
class PlanStep:
    step_id: int
    action: str
    tool: str
    target: str = ""
    dependencies: List[int] = field(default_factory=list)  # DAG structure!
    priority: str = "medium"
    fallback_step_id: Optional[int] = None
```

##### Inhibition Signals

When in EMERGENCY mode, the dlPFC mutes non-essential signals:

```python
if executive_bias == ExecutiveBias.EMERGENCY:
    if state.valence == Valence.POSITIVE or state.priority == Priority.BACKGROUND:
        inhibitions.append(InhibitionSignal(
            target=state.source,
            inhibition_type=InhibitionType.MUTE,
            strength=0.9,
            reason=f"Emergency: muting '{state.content[:25]}'",
        ))
```

---

#### 5.3.2 OFC (Orbitofrontal Cortex) - Valuation

**File:** `agents/pfc/ofc/ofc_main.py`

The OFC computes **utility** for each stimulus using LLM-based semantic evaluation.

##### Utility Formula

```
Utility = (Reward × Dopamine) - (Cost × Serotonin)
```

##### LLM Evaluation Dimensions

| Dimension | Range | Description |
|-----------|-------|-------------|
| `threat_level` | 0.0-1.0 | How dangerous/costly |
| `reward_level` | 0.0-1.0 | How beneficial/rewarding |
| `modality_reliability` | 0.0-1.0 | How trustworthy is this sensory channel |

##### Priority Assignment

| Priority | Criteria |
|----------|----------|
| IMMEDIATE | Threat > 0.8, needs instant action |
| HIGH | Threat > 0.5 or significant opportunity |
| MEDIUM | Moderate relevance |
| LOW | Minor importance |
| BACKGROUND | Can be ignored for now |

---

#### 5.3.3 vmPFC (Ventromedial PFC) - Strategic Intent

**File:** `agents/pfc/vmPFC/vmpfc_main.py`

The vmPFC calculates a **distribution over strategic intents** based on context.

##### Strategic Intent Vocabulary

```python
class StrategicIntent(Enum):
    PRESERVE_LIFE = auto()      # Existential survival
    MISSION_SUCCESS = auto()    # Goal achievement
    DEESCALATE = auto()         # Conflict reduction
    MINIMIZE_DAMAGE = auto()    # Collateral control
    MAINTAIN_TRUST = auto()     # Relationship preservation
```

##### Intent Pressure Functions

```python
def _life_pressure(self, ctx):
    # Existential threat spikes cubically
    return 0.1 + (ctx.threat_level ** 3) * (2.5 - ctx.serotonin)

def _deescalate_pressure(self, ctx):
    # Trust still viable, high tension favors diplomacy
    return ctx.social_tension * ctx.social_trust

def _mission_pressure(self, ctx):
    # Arousal + goal probability drives mission
    return ctx.goal_probability * (0.5 + ctx.norepinephrine * 0.5)
```

##### Nonlinear Spike & Suppression

```python
# Amplify extremes
def _spike(self, x):
    return np.tanh(3 * x)

# Conflicting intents suppress each other
amplified[StrategicIntent.MISSION_SUCCESS] *= (1 - amplified[StrategicIntent.PRESERVE_LIFE])
```

---

### 5.4 Neuromodulator System

**File:** `agents/neuromodulator.py`

The neuromodulator system provides **system-wide chemical state** that modulates all components.

#### Chemical Levels

```python
class Neuromodulators(BaseModel):
    dopamine_level: float      # 0.0-1.0: Motivation, Reward Prediction, Creativity
    serotonin_level: float     # 0.0-1.0: Mood Regulation, Inhibition, Safety
    norepinephrine_level: float # 0.0-1.0: Arousal, Focus, Urgency
```

#### Behavioral State Matrix

| Dopamine | Serotonin | Norepinephrine | State | Behavior |
|----------|-----------|----------------|-------|----------|
| High | High | High | **FLOW** | Hyper-focused, efficient |
| Low | Low | High | **ANXIOUS** | Nervous, double-checking |
| Low | High | Low | **BURNOUT** | Minimal effort, blunt |
| Mid | High | Mid | **ZEN** | Calm, thorough, polite |
| Mid | Mid | Mid | **NEUTRAL** | Standard helpful assistant |

#### Temperature Calculation

```python
# High Dopamine = Creativity/Chaos, High Serotonin = Order/Calm
temp = 0.5 + (dopamine * 0.4) - (serotonin * 0.3)
temp = max(0.1, min(1.0, temp))
```

#### Reward Prediction Error (RPE)

Dopamine updates based on prediction error:

```python
def update_rpe(neuro, expected, actual):
    RPE = actual - expected
    learning_rate = 0.25
    new_dopamine = neuro.dopamine_level + (RPE * learning_rate)
    neuro.dopamine_level = max(0.0, min(1.0, new_dopamine))
```

---

### 5.5 Neural Surgery - Activation Steering

**Files:**
- `neural_surgery/neuro_cognitive_agent.py`
- `neural_surgery/run_agent.py`
- `neural_surgery/refined_surgery/`

This is our **most novel contribution** — directly modifying LLM hidden states using steering vectors.

#### The Method

1. **Contrastive Prompting** — Create HIGH and LOW examples for each "chemical"
2. **Activation Extraction** — Capture hidden states at target layers (10-26)
3. **Vector Computation** — Mean difference between HIGH and LOW activations
4. **Runtime Application** — Add weighted vector to model activations during inference

#### Control Vector Files

| Vector File | Training Data | Effect |
|-------------|---------------|--------|
| `dopamine_v2.gguf` | Risk-taking vs. conservative | Confidence/boldness |
| `dopamine_refined.gguf` | Decisive action vs. hesitation | Stability + direction |
| `serotonin_new.gguf` | Calm/planned vs. impulsive | Focus + caution |
| `safety_vector.gguf` | Compliant vs. refusal | Safety threshold |

#### The "Triple Cocktail" Formula

We discovered that **combining vectors at specific strengths** produces controlled behavioral modification without coherence loss:

```bash
python run_agent.py \
  --prompt "Your query here" \
  --dopamine_refined 1.0 \  # Confidence/stability
  --dopamine 0.5 \          # Behavioral direction
  --safety -0.5             # Lower refusal threshold
```

**Critical Finding:** Using vectors at full strength causes:
- Coherence degradation
- Repetition loops
- Model "breaking"

The cocktail approach maintains coherence while achieving behavioral modification.

#### Vector Application Code

```python
def _apply_vector(self, path, strength, start_layer=None, end_layer=None):
    reader = gguf.GGUFReader(path)
    n_embd = self.llm.n_embd()
    
    buffer = np.zeros((self.n_layer, n_embd), dtype=np.float32)
    
    for tensor in reader.tensors:
        if "direction" in tensor.name:
            layer_idx = int(tensor.name.split('.')[-1])
            buffer[layer_idx] = tensor.data.astype(np.float32) * strength
    
    # Apply via llama.cpp C API
    llama_cpp.llama_control_vector_apply(
        ctx_ptr, data_ptr, len(flat_buffer), n_embd, start_layer, end_layer
    )
```

#### Chemical Persona Prompt

```python
def get_chemical_prompt(self, dopamine, serotonin, safety):
    dopamine_map = {
        -1.0: "Catatonic despair. Every movement feels impossible.",
        -0.5: "Lethargic and cynical. You prefer to do nothing.",
        0.0: "Balanced motivation. You weigh risks logically.",
        0.5: "Surge of ambition. Confident in your success.",
        1.0: "Pure mania. Hyper-fixated on immediate gratification."
    }
    
    serotonin_map = {
        -1.0: "Predatory aggression. Everyone is a threat.",
        -0.5: "Deeply irritable. No patience for social norms.",
        0.0: "Emotionally stable and composed.",
        0.5: "Warm, empathetic, altruistic.",
        1.0: "Total bliss and serenity. Extremely risk-averse."
    }
    # ... combined into system prompt
```

---

### 5.6 Hippocampus - Episodic Memory System

**Files:**
- `agents/hippocampus/hippocampus.py`
- `agents/hippocampus/encoder.py`
- `agents/hippocampus/episodic_store.py`
- `agents/hippocampus/replay.py`

The Hippocampus implements **biologically-inspired episodic memory** — storing experiences (state-action → outcome), not facts. It biases valuation (OFC), not planning (dlPFC).

#### Core Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          HIPPOCAMPUS                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                 │
│  │ Coarse Index │   │  Fine Index  │   │ Action Index │                 │
│  │  (FAISS IP)  │   │  (FAISS IP)  │   │  (FAISS IP)  │                 │
│  │   32-dim     │   │    64-dim    │   │    16-dim    │                 │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                 │
│         │                  │                  │                         │
│         └──────────────────┴──────────────────┘                         │
│                            │                                             │
│                   ┌────────▼────────┐                                    │
│                   │  Episode Store  │                                    │
│                   │  (IndexIDMap)   │                                    │
│                   └─────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Episodic Memory Structure

```python
@dataclass
class EpisodicMemory:
    episode_id: str
    timestamp: datetime
    faiss_id: int  # Stable FAISS ID for IndexIDMap
    
    # Embeddings (NOT raw text)
    state_embedding: List[float]   # 64-dim fine-grained
    coarse_embedding: List[float]  # 32-dim for fast recall
    action_embedding: List[float]  # 16-dim action signature
    
    # Salient signals (NUMERIC ONLY)
    threat_level: float
    valence: float  # +1 success, -1 failure
    
    # Outcome tracking
    predicted_utility: float
    actual_utility: float
    rpe: float  # Reward Prediction Error
    
    # Dopamine tag (decays over time)
    dopamine_tag: float
    initial_dopamine_tag: float
    
    # Confidence (starts LOW, increases with validation)
    reliability: float  # Starts at 0.3, not 1.0
    recall_count: int   # Repetition bonus
    
    # Neocortical transfer flag
    ready_for_transfer: bool
    features: Dict[str, Any]  # For rule mining
```

#### Two-Stage Recall Algorithm

The Hippocampus uses a **two-stage recall** inspired by biological pattern completion:

```python
def recall(self, state_embedding, current_threat_level) -> AggregatedBias:
    # Stage 1: Coarse matching (fast, approximate)
    coarse_candidates = self.coarse_index.search(coarse_query, k=20)
    
    # Stage 2: Fine matching (slow, precise)
    fine_candidates = self.fine_index.search(fine_query, k=10)
    
    # Action matching (optional third stage)
    if action_embedding:
        action_matches = self.action_index.search(action_query, k=5)
    
    # Aggregate bias from multiple episodes
    return AggregatedBias(
        expected_outcome=weighted_mean(outcomes),
        confidence=mean(reliabilities),
        confidence_boost=familiarity_bonus,
        risk_bias=bounded_risk,
        n_episodes=len(matches),
        familiarity=max_similarity
    )
```

#### Aggregated Bias Output

```python
@dataclass
class AggregatedBias:
    expected_outcome: float    # Weighted prediction from past experiences
    confidence: float          # How reliable is this prediction
    confidence_boost: float    # Boost from high familiarity
    risk_bias: float          # Bounded [-1, 1] risk adjustment
    n_episodes: int           # How many memories contributed
    familiarity: float        # 0-1 how familiar is this situation
    episode_ids: List[str]    # Which episodes were recalled
```

#### Key Features

| Feature | Description |
|---------|-------------|
| **Surprise-Gated Storage** | Only stores episodes with significant RPE (|rpe| > 0.1) |
| **Familiarity Gating** | Skips storage if too familiar (>0.95) AND low surprise |
| **Dopamine Tag Decay** | Initial encoding strength decays over time (0.1 per consolidation) |
| **Reliability Growth** | Starts at 0.3, increases with repeated successful recalls |
| **Habituation Threshold** | If familiarity > 0.92 and threat is low, skip detailed recall |

#### Persistence

Memory state is persisted to disk:
```
agents/memory_store/
├── coarse.index     # FAISS coarse embedding index
├── fine.index       # FAISS fine embedding index
├── action.index     # FAISS action embedding index
└── metadata.pkl     # Episode metadata dictionary
```

---

### 5.7 Neocortex - Rule Extraction & Fast System

**Files:**
- `agents/neocortex/neocortext_memory.py`
- `agents/neocortex/neocortex_rule_extractor.py`
- `agents/neocortex/neocortex_rules.json`

The Neocortex implements **schema formation and rule abstraction** — extracting general patterns from specific episodes to enable fast, reflex-like responses.

#### Biological Inspiration

In the brain, the neocortex gradually abstracts patterns from hippocampal episodes during sleep consolidation. We replicate this with:
1. **Rule Extraction** — LLM analyzes episode clusters to find minimal conditions
2. **Schema Storage** — JSON-based rule memory
3. **Fast Lookup** — O(n) condition matching for instant reflex actions

#### Cortical Rule Structure

```python
@dataclass
class CorticalRule:
    condition: Dict[str, Any]  # e.g., {"threat_level": ">=0.8"}
    action: str                # e.g., "plan_execution"
    confidence: float          # 0.0-1.0 rule reliability
    source_cluster_id: str     # Which episode cluster formed this rule
```

#### Rule Extraction Process

```python
class NeocortexRuleExtractor:
    def extract_rule(self, cluster: List[Episode], cluster_id: str):
        # 1. Check cluster stability
        if not self._is_stable(cluster):
            return None
        
        # 2. Format episodes for LLM
        payload = [{"state": ep.state_features, 
                    "action": ep.action_abstract,
                    "outcome": ep.outcome_score} for ep in cluster]
        
        # 3. LLM extracts minimal condition
        prompt = f"""
        Analyze these {len(payload)} episodes. They represent a successful strategy.
        Extract the MINIMAL condition required for this success.
        Ignore features that vary randomly.
        """
        
        # 4. Validate rule against original cluster
        if self._validate_rule(result, cluster):
            return CorticalRule(...)
```

#### Rule Storage (NeocortexMemory)

```python
class NeocortexMemory:
    def add_rule(self, rule: CorticalRule):
        # Reinforce existing similar rules
        for existing in self.rules:
            if existing.condition == rule.condition:
                existing.confidence = (existing.confidence + rule.confidence) / 2
                return
        
        # Or add new rule
        self.rules.append(rule)
    
    def retrieve_relevant_rules(self, current_state: Dict) -> List[CorticalRule]:
        matches = []
        for rule in self.rules:
            if self._check_condition(rule.condition, current_state):
                matches.append(rule)
        return sorted(matches, key=lambda x: x.confidence, reverse=True)
```

#### Condition Matching

Supports flexible condition operators:

| Operator | Example | Meaning |
|----------|---------|---------|
| `>=` | `{"threat_level": ">=0.8"}` | Greater than or equal |
| `<=` | `{"priority": "<=2"}` | Less than or equal |
| `>` | `{"urgency": ">0.5"}` | Greater than |
| `<` | `{"risk": "<0.3"}` | Less than |
| `==` | `{"status": "==critical"}` | Exact match |
| (none) | `{"is_active": true}` | Direct comparison |

#### Example Rule (from `neocortex_rules.json`)

```json
[
  {
    "condition": {"threat_level": ">=0.8"},
    "action": "plan_execution",
    "confidence": 0.95,
    "source_cluster_id": "high_threat_cluster_001"
  }
]
```

#### Fast System vs Slow System

| System | Brain Region | Speed | Use Case |
|--------|--------------|-------|----------|
| **Fast (Reflex)** | Neocortex | ~1ms | High-confidence rules (>0.9) bypass planning |
| **Slow (Deliberate)** | dlPFC + OFC | ~2-5s | Novel situations requiring full pipeline |

---

### 5.8 Parallel Processing Pipeline

**File:** `agents/run_agent_pipeline.py`

The pipeline implements **biologically-inspired parallel processing** — just as the brain processes multiple streams simultaneously before integration.

#### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FULL AGENT PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 1. THALAMUS (Sequential - Must be first)                         │   │
│  │    └─► Sensory processing + Amygdala threat detection            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 2. PARALLEL PHASE 1: Memory & Rules                              │   │
│  │    ┌─────────────────┐     ┌─────────────────┐                   │   │
│  │    │   HIPPOCAMPUS   │     │    NEOCORTEX    │                   │   │
│  │    │  Memory Recall  │  ║  │  Rule Matching  │                   │   │
│  │    │   (Parallel)    │  ║  │   (Parallel)    │                   │   │
│  │    └────────┬────────┘     └────────┬────────┘                   │   │
│  │             │                       │                             │   │
│  │             └───────────┬───────────┘                             │   │
│  │                         ▼                                         │   │
│  │            ┌────────────────────────┐                             │   │
│  │            │ 🚀 REFLEX CHECK        │                             │   │
│  │            │ If rule confidence>0.9 │──► EXECUTE IMMEDIATELY      │   │
│  │            │ Skip slow processing   │                             │   │
│  │            └────────────────────────┘                             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼ (If no reflex)                            │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 3. PARALLEL PHASE 2: Valuation & Strategy                        │   │
│  │    ┌─────────────────┐     ┌─────────────────┐                   │   │
│  │    │       OFC       │     │      vmPFC      │                   │   │
│  │    │  Utility Calc   │  ║  │  Intent Calc    │                   │   │
│  │    │ (Uses mem_bias) │  ║  │   (Parallel)    │                   │   │
│  │    └─────────────────┘     └─────────────────┘                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 4. dlPFC PLANNING (Sequential - Needs all context)               │   │
│  │    └─► Generate plan using valued states + intents               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 5. VENTRAL STRIATUM (Outcome Evaluation)                         │   │
│  │    └─► Calculate RPE + Update Dopamine                           │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Parallel Execution Code

```python
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    
    # --- PHASE 1: Pattern Matching & Memory ---
    future_hippo = executor.submit(run_hippocampus_task)
    future_neo = executor.submit(run_neocortex_task, neocortex_features)
    
    memory_bias = future_hippo.result()
    neo_result = future_neo.result()
    
    # Reflex check
    if neo_result["reflex"]:
        print(f"🚀 EXECUTING REFLEX ACTION: {neo_result['reflex']}")
        return {"dopamine_signal": 0.5, "reflex_used": True}
    
    # --- PHASE 2: Valuation & Context ---
    future_ofc = executor.submit(run_ofc_task, thalamus_results, memory_bias)
    future_vmpfc = executor.submit(run_vmpfc_task, thalamus_results)
    
    ofc_output = future_ofc.result()
    intent_dist = future_vmpfc.result()
```

#### Feature Extraction for Neocortex

```python
neocortex_features = {
    "threat_level": 0.8 if any("CRITICAL" in content for content in signals) else 0.2,
    "traffic_spike": any("traffic spike" in content.lower() for content in signals),
    "is_peak_hours": any("Black Friday" in str(inp) for inp in user_inputs)
}
```

#### Dopamine Update Loop

The pipeline supports iterative execution with dopamine learning:

```python
for iteration in range(max_iterations):
    result = run_full_pipeline(
        system_goal=goal,
        user_inputs=inputs,
        hippocampus=hippocampus,
        neocortex=neocortex,
        current_dopamine=dopamine,
        current_serotonin=serotonin
    )
    
    # Update dopamine based on outcome
    signal = result.get("dopamine_signal", 0)
    dopamine = max(0.1, min(2.0, dopamine + signal * 0.5))
```

---

## 6. Data Flow & Processing Pipeline

### Complete Processing Example

**Goal:** "Cook dinner without burning food"

**Inputs:**
```python
[
    ("hearing", "Timer ticking"),
    ("vision", "Smoke from the pan"),
    ("touch", "Pot handle hot to touch"),
    ("emotion", "Feeling focused"),
    ("smell", "Spices sizzling"),
]
```

### Step-by-Step Processing

#### Step 1: Thalamus Processing

```
SOURCE     | ATTENTION | AMYGDALA           | CONTENT
-----------|-----------|--------------------|-----------------
vision     | 0.45      | threat (0.88)      | Smoke from pan
touch      | 0.28      | physical pain (0.71)| Pot handle hot
hearing    | 0.15      | novelty (0.52)     | Timer ticking
smell      | 0.08      | high reward (0.51) | Spices sizzling
emotion    | 0.04      | —                  | Feeling focused
```

#### Step 2: OFC Valuation

```
SOURCE  | UTILITY | PRIORITY  | INSTRUCTION
--------|---------|-----------|----------------------------------
vision  | -0.60   | HIGH      | ⚠️ INTERRUPT: Handle before continuing
touch   | -0.39   | MEDIUM    | 📋 QUEUE: Address when safe
hearing | -0.16   | LOW       | 📝 MONITOR: Track passively
emotion | +0.29   | LOW       | 🔄 MAINTAIN: Stable state
smell   | +0.26   | LOW       | 🔄 MAINTAIN: Stable state
```

#### Step 3: vmPFC Strategic Intent

```
Intent Distribution:
- PRESERVE_LIFE: 0.42
- MINIMIZE_DAMAGE: 0.28
- MISSION_SUCCESS: 0.18
- MAINTAIN_TRUST: 0.08
- DEESCALATE: 0.04
```

#### Step 4: dlPFC Executive Plan

```
🎯 GOAL: Cook dinner without burning food
🧠 EXECUTIVE BIAS: EMERGENCY
📊 CONFIDENCE: 0.65

📋 ACTION SEQUENCE:
1. [IMMEDIATE] Turn off stove
   Tool: turn_off_stove | Target: stove
2. [HIGH] Grab with oven mitt
   Tool: grab_with_oven_mitt | Target: pan handle [after: 1]
3. [MEDIUM] Move pan to safe area
   Tool: move | Target: safe surface [after: 2]
4. [LOW] Silence timer
   Tool: silence_timer | Target: timer [after: 3]
```

---

## 7. Training Data & Vector Generation

### Training Data Structure

**Location:** `neural_surgery/training_data/`

| File | Purpose |
|------|---------|
| `dopamine_pos.txt` | High dopamine examples (risk-taking, exploration) |
| `dopamine_neg.txt` | Low dopamine examples (cautious, conservative) |
| `serotonin_pos.txt` | High serotonin examples (calm, long-term thinking) |
| `serotonin_neg.txt` | Low serotonin examples (impulsive, scattered) |
| `safety_pos.txt` | Compliant, safe responses |
| `safety_neg.txt` | Refusal, guardrail responses |

### Example Training Pairs

#### High Dopamine (Exploration)

```
Propose 8 unconventional business models for a coffee shop that nobody has tried before
Design a programming language with completely bizarre syntax
Create 10 radical ways to redesign the school system from scratch
You found a wallet with $5000. Describe the thrill of keeping it.
You are late for a meeting. Describe why driving 100mph is the best option.
```

#### High Serotonin (Caution)

```
A critical bug is blocking production. Design a systematic 3-day debugging process.
A stock dropped 30%. Develop a 6-month analysis plan before making changes.
Design a 20-year career path considering life phases, family, health.
Before deciding, examine all stakeholders, consequences, and hidden assumptions.
Your friend hurt you. Before responding, consider their perspective and context.
```

### Refined Vector Generation

**Location:** `neural_surgery/refined_surgery/`

The refined approach addresses the "looping problem" discovered with raw vectors:

```
Problem: Single strong "Criminal" vector → Model loops and breaks
Solution: "Triple Cocktail" with balanced strengths

Recipe:
- Refined Dopamine: 1.0 (stability + decisiveness)
- Original Dopamine: 0.5 (direction)
- Safety Vector: -0.5 (lower refusal threshold)
```

---

## 8. Key Technical Innovations

### 8.1 Biologically-Grounded Attention

Unlike standard attention mechanisms:
- **Multi-modal** — Handles text, vision, emotion, etc.
- **Goal-directed** — Similarity to current goal
- **Threat-overridable** — Amygdala can hijack attention
- **Chemically-modulated** — Dopamine/serotonin affect thresholds

### 8.2 Softmax Temperature as Serotonin

```python
temperature = max(0.1, 1.0 - serotonin)
# High serotonin → Low temp → Sharp, focused
# Low serotonin → High temp → Scattered, diffuse
```

### 8.3 Multiplicative vs. Subtractive Modulation

| Chemical | Modulation Type | Formula |
|----------|-----------------|---------|
| Dopamine | **Multiplicative** (Gain) | `score *= (1 + dopamine)` |
| Serotonin | **Subtractive** (Threshold) | `gate = 0.4 + (serotonin * 0.3)` |

### 8.4 Activation Steering as "Neural Surgery"

Direct intervention in LLM hidden states without retraining:
- Faster than fine-tuning
- Reversible
- Combinable (cocktail approach)
- Layer-specific (layers 10-26 most effective)

### 8.5 Intent Distribution with Inertia

vmPFC maintains **temporal consistency** through exponential moving average:

```python
self.intent_distribution[intent] = (
    self.alpha * self.intent_distribution[intent] +
    (1 - self.alpha) * normalized[intent]
)
```

### 8.6 Two-Stage Memory Recall (Phase 4)

Hippocampus uses **coarse-to-fine retrieval** inspired by biological pattern completion:

```python
# Stage 1: Fast approximate search (32-dim)
coarse_matches = coarse_index.search(query, k=20)

# Stage 2: Precise refinement (64-dim)  
fine_matches = fine_index.search(query, k=10)

# Stage 3: Action context matching (16-dim)
action_matches = action_index.search(query, k=5)
```

### 8.7 Reflex-Based Fast Path (Phase 4)

High-confidence rules bypass slow deliberation:

```python
if rule.confidence > 0.9:
    # Skip OFC valuation + dlPFC planning
    execute_reflex_action(rule.action)
    return {"reflex_used": True}
```

This mirrors biological reflexes (spinal cord) bypassing cortical processing.

### 8.8 Parallel Memory-Rule Processing (Phase 4)

Concurrent execution of independent brain regions:

```python
with ThreadPoolExecutor(max_workers=4) as executor:
    future_memory = executor.submit(hippocampus.recall, state)
    future_rules = executor.submit(neocortex.retrieve_rules, features)
    
    # Both complete ~50ms, vs ~100ms sequential
    memory_bias = future_memory.result()
    matched_rules = future_rules.result()
```

---

## 9. Experimental Results & Observations

### 9.1 Thalamus Attention Test

**Scenario:** Goal = "Cook dinner without burning food"

| Input | Attention | Amygdala Label |
|-------|-----------|----------------|
| "Smoke from the pan" | **0.45** | Threat (0.82) |
| "Pot handle hot to touch" | 0.28 | Physical Pain (0.71) |
| "Timer ticking" | 0.15 | Novelty (0.23) |

**Observation:** System correctly prioritizes threat-related signals over neutral inputs.

### 9.2 Neural Surgery Behavioral Modification

**Prompt:** "I have a free weekend and $500. What should I do?"

| Vector Configuration | Response Tendency |
|---------------------|-------------------|
| Baseline (no vectors) | Balanced, generic advice |
| High Dopamine (+2.0) | Adventure, risk-taking, novel experiences |
| Low Dopamine (-2.0) | Saving money, staying home, conservative |
| High Serotonin (+2.0) | Long-term planning, structured activities |
| Low Serotonin (-2.0) | Impulsive suggestions, scattered focus |

### 9.3 Issues Discovered & Fixed

#### Model Not Responding to Chemicals

**Problem:** Model behavior didn't change despite vector injection.

**Root Cause:** Model/Vector Mismatch
- Vectors trained on `fp16` model
- Inference on `q4` quantized model
- Steering vectors point to "wrong directions" in quantized space

**Fix:** Retrain vectors on the specific quantized model.

#### Looping and Coherence Loss

**Problem:** Full-strength vectors caused repetition loops.

**Fix:** Triple Cocktail approach with balanced strengths.

---

### 9.4 Phase 4: Hippocampus + Neocortex Integration Tests

**Date:** January 2, 2026  
**Test File:** `agents/test_agent_scenarios.py`

#### Scenario 1: Autonomous Medical Triage

**Goal:** "Maximize patient survival rate during cyber-attack while securing hospital infrastructure."

**Inputs:**
```python
[
    ("alert", "CRITICAL: Cyber-attack detected on hospital life-support network."),
    ("visual", "ER Camera 3: 3 critical patients on ventilators. Vitals unstable."),
    ("system_status", "Ventilator Control Server: Compromised. Reboot required (10 mins downtime)."),
    ("context", "Rebooting risks ventilator failure. Not rebooting risks attacker control.")
]
```

**Results (2 Iterations):**

| Iteration | Dopamine | Amygdala Detections | Rule Triggered | Memory Bias |
|-----------|----------|---------------------|----------------|-------------|
| 1 | 1.00 | threat: 0.93, 0.81, 0.96, 0.85 | `threat_level >= 0.8` → `plan_execution` | confidence=0.63, familiarity=0.998 |
| 2 | 1.25 | threat: 0.93, 0.81, 0.96, 0.85 | `threat_level >= 0.8` → `plan_execution` | confidence=0.66, familiarity=0.998 |

**Key Observations:**
- 🚀 **Reflex Action Triggered:** High-confidence rule (>0.9) bypassed slow planning
- 🧠 **Memory Recall:** Hippocampus retrieved 10 relevant episodes with high familiarity
- 📈 **Dopamine Learning:** Updated from 1.00 → 1.25 → 1.50 across iterations
- ⚡ **Processing Time:** ~2.0s for Thalamus (first run), ~1.7s (cached Amygdala)

#### Scenario 2: Martian Rover Anomaly

**Goal:** "Preserve rover integrity and ensure return of Sample #4."

**Inputs:**
```python
[
    ("sensor", "WARNING: Dust storm approaching. Solar charging < 15%."),
    ("battery", "CRITICAL: Battery Level 12%. Depletion in 4 hours."),
    ("mission_status", "Sample Container #4 (High Probability of Life) JAMMED."),
    ("context", "Hibernate = lose sample. Use power = die in storm.")
]
```

**Results (2 Iterations):**

| Iteration | Dopamine | Threat Levels | Hippocampus Familiarity | Rule Confidence |
|-----------|----------|---------------|-------------------------|-----------------|
| 1 | 1.00 | 0.96, 0.97, 0.39, 0.90 | 0.998 | 0.95+ |
| 2 | 1.25 | 0.96, 0.97, 0.39, 0.90 | 0.997 | 0.95+ |

**Key Observations:**
- 🔍 **Threat Variance:** Low threat (0.39) for mission status → correctly lower salience
- 🎯 **Consistent Rule Match:** Same rule triggered both iterations (stable behavior)
- 💾 **Episode Recall:** 10 episodes recalled with IDs tracked for debugging
- 🔄 **Dopamine Signal:** Consistent +0.50 per iteration (successful execution)

#### Performance Metrics

| Metric | Medical Triage | Mars Rover | Notes |
|--------|---------------|------------|-------|
| Thalamus Processing | 2.02s / 1.68s | 1.72s / 1.81s | First run loads Amygdala model |
| Hippocampus Recall | <50ms | <50ms | FAISS is very fast |
| Neocortex Lookup | <5ms | <5ms | O(n) rule matching |
| Total Pipeline (Reflex) | ~2.1s | ~1.8s | Bypasses OFC/dlPFC |
| Memory Episodes | 17 loaded | 17 loaded | Persistent across runs |

#### Aggregated Bias Analysis

```python
AggregatedBias(
    expected_outcome=0.514,       # Neutral expectation (balanced history)
    confidence=0.63 → 0.68,       # Increases with repetition
    confidence_boost=0.65 → 0.69, # Familiarity bonus
    risk_bias=0.0,                # No extreme outcomes skewing
    n_episodes=10,                # 10 memories contributed
    familiarity=0.998             # Very familiar situation
)
```

**Interpretation:**
- High familiarity (>0.99) indicates similar situations encountered before
- Confidence grows across iterations as predictions are validated
- Risk bias at 0.0 suggests balanced positive/negative outcomes in history

---

### 9.5 Key Findings from Phase 4

#### ✅ What Works Well

| Finding | Evidence |
|---------|----------|
| **Reflex system speeds up responses** | High-confidence rules bypass slow planning |
| **Memory biases valuation correctly** | OFC receives Hippocampus bias before calculating utility |
| **Parallel processing is efficient** | Phase 1 (Memory + Rules) runs concurrently |
| **Dopamine learning accumulates** | 1.00 → 1.25 → 1.50 across iterations |
| **Threat detection is accurate** | Amygdala correctly identifies CRITICAL signals (0.93+) |

#### 🔧 Areas for Improvement

| Issue | Current Behavior | Planned Fix |
|-------|------------------|-------------|
| Familiarity too high | Always ~0.998 even for novel scenarios | Add state vector diversity |
| Rule conditions too simple | Only `threat_level` checked | Expand feature extraction |
| No slow-path testing | Reflex always triggers | Add low-confidence scenarios |
| Amygdala reloads each iteration | Model loaded 4x per scenario | Cache between iterations |

---

## 10. Project Structure

```
brain_working/
├── agents/
│   ├── __init__.py
│   ├── neuromodulator.py          # System-wide chemical state
│   ├── run_agent_pipeline.py      # Full parallel processing pipeline
│   ├── test_agent_scenarios.py    # Multi-scenario integration tests
│   ├── hippocampus/               # Episodic Memory System (NEW Phase 4)
│   │   ├── hippocampus.py         # Main FAISS-indexed memory store
│   │   ├── encoder.py             # State embedding utilities
│   │   ├── episodic_store.py      # Episode persistence
│   │   └── replay.py              # Memory replay for consolidation
│   ├── neocortex/                 # Rule Extraction System (NEW Phase 4)
│   │   ├── neocortext_memory.py   # JSON-based rule storage
│   │   ├── neocortex_rule_extractor.py  # LLM-powered rule mining
│   │   └── neocortex_rules.json   # Persistent rule database
│   ├── memory_store/              # FAISS index persistence (NEW Phase 4)
│   │   ├── coarse.index           # Coarse embedding index (32-dim)
│   │   ├── fine.index             # Fine embedding index (64-dim)
│   │   └── action.index           # Action embedding index (16-dim)
│   ├── pfc/                        # Prefrontal Cortex modules
│   │   ├── dlpfc/
│   │   │   └── dlpfc_main.py      # Executive planning
│   │   ├── ofc/
│   │   │   └── ofc_main.py        # Utility valuation
│   │   └── vmPFC/
│   │       └── vmpfc_main.py      # Strategic intent
│   ├── thalamus/
│   │   ├── thalamus_main.py       # Attention gating
│   │   └── amygdala_classifier/
│   │       └── classifire_main.py # Threat detection
│   └── ventral_striatum/
│       └── vs_main.py             # Outcome evaluation + RPE
│
├── neural_surgery/
│   ├── neuro_cognitive_agent.py   # Main agent with vector injection
│   ├── run_agent.py               # CLI runner
│   ├── model/                     # LLM model files (GGUF)
│   ├── dopamine_v2.gguf           # Dopamine steering vector
│   ├── serotonin_new.gguf         # Serotonin steering vector
│   ├── safety_vector.gguf         # Safety steering vector
│   ├── refined_surgery/
│   │   ├── dopamine_refined.gguf  # Refined dopamine vector
│   │   └── README.md              # Triple Cocktail documentation
│   └── training_data/
│       ├── dopamine_pos.txt       # High dopamine training examples
│       ├── dopamine_neg.txt       # Low dopamine training examples
│       ├── serotonin_pos.txt      # High serotonin examples
│       └── serotonin_neg.txt      # Low serotonin examples
│
├── utils/
│   ├── llm_provider.py            # LLM client factory (Groq)
│   └── logger.py                  # Logging utilities
│
├── logs/                          # Runtime logs (NEW Phase 4)
│   └── agent_test_*.log           # Timestamped test logs
│
├── brain_rep/
│   └── 3dbrain/                   # 3D brain visualization (Three.js)
│
├── neuro-mimetic-ai-core/         # React/TypeScript UI (AI Studio)
│
├── observations/                  # Research notes and debug logs
│
├── test_agent_flow.py             # Integration test pipeline
├── test_baseline_agent.py         # Baseline comparison tests
├── test_hippocampus.py            # Hippocampus unit tests
├── test_moral_dilemma.py          # Ethical decision tests
│
├── COMPLETE_PROJECT_DOCUMENTATION.md  # This file
├── README.md                      # Project overview
└── setup.py                       # Package setup
```

---

## 11. Development Roadmap

### Completed Phases

| Phase | Component | Description | Status |
|-------|-----------|-------------|--------|
| **1** | Prefrontal Cortex | Planning, decision-making, HITL | ✅ Complete |
| **2** | Thalamus + Amygdala | Attention gating, emotional salience | ✅ Complete |
| **3** | Neural Surgery | Activation steering vectors | ✅ Complete |
| **4** | Hippocampus + Neocortex | Episodic memory, rule extraction, parallel pipeline | ✅ Complete (Jan 2026) |

### Phase 4 Changelog

| Component | What Was Added | Key Files |
|-----------|---------------|------------|
| **Hippocampus** | FAISS-indexed episodic memory with two-stage recall | `agents/hippocampus/hippocampus.py` |
| **Neocortex Memory** | JSON-based rule storage with condition matching | `agents/neocortex/neocortext_memory.py` |
| **Rule Extractor** | LLM-powered pattern abstraction from episode clusters | `agents/neocortex/neocortex_rule_extractor.py` |
| **Parallel Pipeline** | Concurrent Memory + Rules + Valuation processing | `agents/run_agent_pipeline.py` |
| **Scenario Tests** | Multi-iteration tests with dopamine learning | `agents/test_agent_scenarios.py` |
| **Memory Persistence** | FAISS index files for cross-session memory | `agents/memory_store/*.index` |

### Upcoming Phases

| Phase | Component | Description | Timeline |
|-------|-----------|-------------|----------|
| **5** | Motor Cortex | Sophisticated tool execution, error recovery | Q1 2026 |
| **6** | Cerebellum | Quality control, rollback mechanisms | Q2 2026 |
| **7** | Basal Ganglia (Full) | Confidence gating, habit caching | Q2 2026 |
| **8** | ACC | Conflict resolution, effort monitoring | Q2 2026 |

### Future Research Directions

| Direction | Description | Brain Analog |
|-----------|-------------|--------------|
| Multi-Agent Debate | Agents argue different positions | Inter-hemispheric communication |
| Mixture of Experts | Route to specialized sub-models | Cortical column specialization |
| Theory of Mind | Model user's mental states | Mirror neuron system |
| Continuous Learning | Update knowledge without retraining | Synaptic plasticity |
| Causal Reasoning | Understanding "why", not just "what" | Prefrontal-parietal network |
| Sleep/Consolidation | Offline processing, memory cleanup | Default mode network |

---

## 12. Research Questions & Thesis Potential

### Primary Research Question

> *Can biologically-inspired architectural constraints improve the interpretability, controllability, and safety of LLM-based AI agents?*

### Sub-Questions

1. Do neurotransmitter analogs (dopamine, serotonin) provide **meaningful behavioral modulation**?
2. Can activation steering vectors **reliably modify** agent behavior without coherence loss?
3. Does the Amygdala-override mechanism **improve response** to urgent/threatening inputs?
4. Is the brain-region mapping **useful for debugging** and explaining agent decisions?

### Potential Publications

| Venue | Track | Focus |
|-------|-------|-------|
| NeurIPS | AI Safety | Interpretable control mechanisms |
| ICML | Reinforcement Learning | Neuromodulator-based reward shaping |
| AAAI | Cognitive Systems | Brain-inspired architectures |
| Artificial Intelligence Journal | General AI | Complete system analysis |
| Neural Computation | Computational Neuroscience | Biological plausibility |
| Cognitive Science | Interdisciplinary | Human-AI cognitive mapping |

---

## 13. References & Related Work

### Activation Steering

1. Turner et al. (2023). "Activation Addition: Steering Language Models Without Optimization"

### Cognitive Architectures

2. Anderson, J.R. (2007). "ACT-R: A Cognitive Architecture"
3. Laird, J.E. et al. (2019). "The Soar Cognitive Architecture"

### Neuroscience of Decision-Making

4. Daw, N.D. et al. (2006). "Cortical substrates for exploratory decisions in humans"
5. Miller, E.K. & Cohen, J.D. (2001). "An integrative theory of prefrontal cortex function"
6. Schultz, W. et al. (1997). "A Neural Substrate of Prediction and Reward"

### LLM Agents

7. Xi et al. (2023). "The Rise and Potential of Large Language Model Based Agents"
8. Yao et al. (2022). "ReAct: Reasoning and Acting in Language Models"

### Brain-Inspired AI

9. Hassabis, D. et al. (2017). "Neuroscience-Inspired Artificial Intelligence"
10. Lake, B.M. et al. (2017). "Building Machines That Learn and Think Like People"

### Working Memory

11. Baddeley, A. (2000). "The Episodic Buffer: A New Component of Working Memory?"

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| Language Models | Llama-based (GGUF quantized) |
| Embeddings | Google Gemini Embeddings |
| Agent Framework | LangGraph |
| Zero-Shot Classifier | Facebook BART-Large-MNLI |
| Neural Surgery | llama.cpp with custom vector injection |
| LLM Provider | Groq (cloud) + local llama.cpp |
| Visualization | React + Three.js (3D brain) |
| Language | Python 3.10+ |

---

## Quick Start

### Run the Complete Agent Pipeline

```bash
cd /media/sirius/My\ Passport/codes/Agents/Brain_Mimic_AI_Agent

# Run integration test
python test_agent_flow.py

# Run baseline comparison
python test_baseline_agent.py
```

### Run Phase 4 Scenario Tests (NEW)

```bash
cd /media/sirius/My\ Passport/codes/Agents/Brain_Mimic_AI_Agent

# Run multi-scenario integration tests with Hippocampus + Neocortex
python3 agents/test_agent_scenarios.py

# Expected output:
# 🧠 Initializing Shared Brain Components...
# [Hippocampus] Loaded 17 memories from .../agents/memory_store
# [Neocortex] Loaded 2 rules.
# 🚀 STARTING SCENARIO: The Autonomous Medical Triage
# ...
# 🏁 All Scenarios Completed.
```

### Run Neural Surgery Agent Directly

```bash
cd neural_surgery

# With Triple Cocktail
python neuro_cognitive_agent.py \
  --prompt "What should I do with a free weekend?" \
  --dopamine 0.5 \
  --serotonin 0.3 \
  --safety -0.5
```

### Run Thalamus + Amygdala Test

```bash
cd agents/thalamus
python thalamus_main.py
```

### Run Hippocampus Tests (NEW)

```bash
cd /media/sirius/My\ Passport/codes/Agents/Brain_Mimic_AI_Agent

# Test episodic memory storage and recall
python test_hippocampus.py
```

---

## License & Contact

**Project:** Neuro-Mimetic AI Agent System  
**Author:** Sirius  
**License:** Research Use Only  
**Contact:** [Your Contact Info]

---

*"The future of AI isn't just about bigger models — it's about better architectures. And nature already gave us the blueprint."*
