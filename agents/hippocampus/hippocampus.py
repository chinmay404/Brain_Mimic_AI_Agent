from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple , Any
from datetime import datetime
import numpy as np
import uuid
import faiss
import pickle
import os


@dataclass
class EpisodicMemory:
    """A single episodic memory - a compressed 'state → outcome' record."""
    episode_id: str
    timestamp: datetime
    faiss_id: int  # Explicit FAISS ID for IndexIDMap

    # Situation encoding (NOT raw text)
    state_embedding: List[float]
    coarse_embedding: List[float]  # For two-stage recall
    action_embedding: List[float]  # Action included in matching

    # Salient signals (NUMERIC ONLY for behavior)
    threat_level: float
    valence: float  # +1 success, -1 failure

    # Debug/logging only - NEVER feed back to OFC/dlPFC
    _goal_context: str = ""
    _dominant_stimuli: List[str] = field(default_factory=list)

    # Action signature (numeric hash)
    action_hash: int = 0
    action_signature: str = "" # NEW: Store readable action for Neocortex


    # Outcome
    predicted_utility: float = 0.0
    actual_utility: float = 0.0
    rpe: float = 0.0
    corrected_expectation: float = 0.0

    # Dopamine tag at encoding (phasic DA) - decays over time
    dopamine_tag: float = 0.5
    initial_dopamine_tag: float = 0.5  # Original value before decay

    # Confidence - starts LOW, increases with validation
    success: bool = True
    reliability: float = 0.3  # Start low, not high
    recall_count: int = 0  # Track how often recalled for repetition bonus
    
    # NEW: Signal for cortical transfer (Schema formation)
    ready_for_transfer: bool = False
    
    # NEW: Store raw features for Neocortical abstraction (Rule Mining)
    features: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryBias:
    """What hippocampus returns to bias vmPFC → OFC."""
    expected_outcome: float
    confidence: float
    episode_id: str
    similarity: float

    # vmPFC pathway outputs
    confidence_boost: float = 0.0
    risk_bias: float = 0.0  # Bounded [-1, 1]

    # Recall metadata
    recall_stage: int = 0  # 1=coarse, 2=fine
    familiarity: float = 0.0  # Stage 1 score
    action_match: float = 0.0  # How well action matches

    note: str = ""
    
    # NEW: Signal that this was a habit/schema match (if we were to return it)
    is_habit: bool = False


@dataclass
class AggregatedBias:
    """Combined bias from multiple episode recalls."""
    expected_outcome: float
    confidence: float
    confidence_boost: float
    risk_bias: float
    n_episodes: int
    familiarity: float = 0.0
    episode_ids: List[str] = field(default_factory=list)


@dataclass
class NeuromodulatorState:
    """Current neurochemical state for dynamic thresholds."""
    dopamine: float = 0.5
    serotonin: float = 0.5
    norepinephrine: float = 0.5


class Hippocampus:
    """
    Episodic memory system with neurobiological refinements.

    NOT a RAG system. NOT a knowledge base.

    This stores EXPERIENCES (state-action → outcome), not facts.
    It biases VALUATION (OFC), not planning (dlPFC).

    Fixes implemented:
    1. IndexIDMap for stable FAISS ID alignment
    2. Reliability starts low, increases with repetition
    3. Dopamine tag only at encoding, decays during consolidation
    4. Risk bias bounded with saturation
    5. Action embedding included in recall matching
    6. Soft continuous + hard sleep consolidation

    Write path: VS → Hippocampus.store() (only on surprise)
    Read path:  Thalamus/OFC → Hippocampus.recall() → vmPFC → OFC bias
    """

    THREAT_THRESHOLDS = {
        "low": 0.70,
        "medium": 0.80,
        "high": 0.90
    }
    
    # NEW: If familiarity > this AND threat is low, skip recall (Habituation)
    HABITUATION_THRESHOLD = 0.92 

    # Consolidation settings
    SOFT_CONSOLIDATION_INTERVAL = 50  # Soft prune every N episodes
    DA_DECAY_RATE = 0.1  # Dopamine tag decay per consolidation

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        embedding_dim: int = 64,
        coarse_dim: int = 32,
        action_dim: int = 16,
        base_similarity_threshold: float = 0.75,
        coarse_threshold: float = 0.60,
        fine_threshold: float = 0.80,
        surprise_threshold: float = 0.15,
        max_memories: int = 1000,
        decay_rate: float = 0.01,
        consolidation_threshold: int = 100
    ) -> None:
        self.storage_dir = storage_dir
        self.embedding_dim = embedding_dim
        self.coarse_dim = coarse_dim
        self.action_dim = action_dim
        self.base_similarity_threshold = base_similarity_threshold
        self.coarse_threshold = coarse_threshold
        self.fine_threshold = fine_threshold
        self.surprise_threshold = surprise_threshold
        self.max_memories = max_memories
        self.decay_rate = decay_rate
        self.consolidation_threshold = consolidation_threshold

        # FIX 1: Use IndexIDMap for stable ID alignment
        self.coarse_index = faiss.IndexIDMap(faiss.IndexFlatIP(coarse_dim))
        self.fine_index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dim))
        self.action_index = faiss.IndexIDMap(faiss.IndexFlatIP(action_dim))

        # Metadata storage keyed by faiss_id
        self._episodes: Dict[int, EpisodicMemory] = {}
        self._episode_id_to_faiss_id: Dict[str, int] = {}
        self._next_faiss_id: int = 0

        # Consolidation control
        self._episodes_since_soft_consolidation: int = 0
        self._episodes_since_hard_consolidation: int = 0
        self._is_active: bool = True

        # Current neuromodulator state
        self._neuro_state = NeuromodulatorState()

        if self.storage_dir:
            self.load_state()

    def save_state(self) -> None:
        """Persist memory state to disk."""
        if not self.storage_dir:
            return
            
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Save FAISS indices
        faiss.write_index(self.coarse_index, os.path.join(self.storage_dir, "coarse.index"))
        faiss.write_index(self.fine_index, os.path.join(self.storage_dir, "fine.index"))
        faiss.write_index(self.action_index, os.path.join(self.storage_dir, "action.index"))
        
        # Save episodes and metadata
        state = {
            "episodes": self._episodes,
            "episode_id_to_faiss_id": self._episode_id_to_faiss_id,
            "next_faiss_id": self._next_faiss_id,
            "episodes_since_soft": self._episodes_since_soft_consolidation,
            "episodes_since_hard": self._episodes_since_hard_consolidation,
            "neuro_state": self._neuro_state
        }
        
        with open(os.path.join(self.storage_dir, "memory_state.pkl"), "wb") as f:
            pickle.dump(state, f)
            
    def load_state(self) -> None:
        """Load memory state from disk."""
        if not self.storage_dir:
            return
            
        if not os.path.exists(os.path.join(self.storage_dir, "memory_state.pkl")):
            return
            
        try:
            # Load FAISS indices
            self.coarse_index = faiss.read_index(os.path.join(self.storage_dir, "coarse.index"))
            self.fine_index = faiss.read_index(os.path.join(self.storage_dir, "fine.index"))
            self.action_index = faiss.read_index(os.path.join(self.storage_dir, "action.index"))
            
            # Load episodes and metadata
            with open(os.path.join(self.storage_dir, "memory_state.pkl"), "rb") as f:
                state = pickle.load(f)
                
            self._episodes = state["episodes"]
            self._episode_id_to_faiss_id = state["episode_id_to_faiss_id"]
            self._next_faiss_id = state["next_faiss_id"]
            self._episodes_since_soft_consolidation = state.get("episodes_since_soft", 0)
            self._episodes_since_hard_consolidation = state.get("episodes_since_hard", 0)
            self._neuro_state = state.get("neuro_state", NeuromodulatorState())
            
            print(f"[Hippocampus] Loaded {len(self._episodes)} memories from {self.storage_dir}")
        except Exception as e:
            print(f"[Hippocampus] Failed to load state: {e}")

    def set_neuromodulator_state(
        self,
        dopamine: float = None,
        serotonin: float = None,
        norepinephrine: float = None,
    ) -> None:
        """Update current neurochemical state."""
        if dopamine is not None:
            self._neuro_state.dopamine = np.clip(dopamine, 0.0, 1.0)
        if serotonin is not None:
            self._neuro_state.serotonin = np.clip(serotonin, 0.0, 1.0)
        if norepinephrine is not None:
            self._neuro_state.norepinephrine = np.clip(norepinephrine, 0.0, 1.0)

    def _get_dynamic_threshold(self) -> float:
        """
        Adaptive threshold based on neuromodulators.
        High dopamine → explore → lower threshold
        High serotonin → cautious → higher threshold
        """
        da = self._neuro_state.dopamine
        ser = self._neuro_state.serotonin
        threshold = self.base_similarity_threshold - 0.1 * da + 0.1 * ser
        return np.clip(threshold, 0.5, 0.95)

    def _get_threat_threshold(self, threat_level: float) -> float:
        """Threat-dependent recall gate."""
        if threat_level < 0.3:
            return self.THREAT_THRESHOLDS["low"]
        elif threat_level < 0.7:
            return self.THREAT_THRESHOLDS["medium"]
        else:
            return self.THREAT_THRESHOLDS["high"]

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector for cosine similarity."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function for bounded outputs."""
        return 1.0 / (1.0 + np.exp(-x))

    def _apply_dopamine_tag(self, embedding: np.ndarray) -> np.ndarray:
        """
        Phasic dopamine tagging at ENCODING ONLY.
        FIX 3: No DA amplification at recall.
        """
        da = self._neuro_state.dopamine
        tagged = embedding * (1.0 + da * 0.2)
        return self._normalize(tagged)

    def _create_coarse_embedding(self, fine_embedding: np.ndarray) -> np.ndarray:
        """Reduce dimensionality for Stage 1 coarse matching."""
        if len(fine_embedding) <= self.coarse_dim:
            padded = np.pad(fine_embedding, (0, self.coarse_dim - len(fine_embedding)))
            return self._normalize(padded)

        ratio = len(fine_embedding) / self.coarse_dim
        coarse = np.array([
            np.mean(fine_embedding[int(i * ratio):int((i + 1) * ratio)])
            for i in range(self.coarse_dim)
        ], dtype=np.float32)
        return self._normalize(coarse)

    def _create_action_embedding(self, action_signature: str) -> np.ndarray:
        """
        FIX 5: Create action embedding for matching.
        Uses hash-based projection into action_dim space.
        """
        # Create deterministic embedding from action string
        h = hash(action_signature)
        np.random.seed(abs(h) % (2**31))
        raw = np.random.randn(self.action_dim).astype(np.float32)
        return self._normalize(raw)

    def _compute_initial_reliability(self, rpe: float, success: bool) -> float:
        """
        FIX 2: Reliability starts LOW.
        High surprise → store but low confidence.
        Confidence increases with repetition, not surprise.
        """
        # Base reliability is LOW - must be validated
        base = 0.3
        # Slight boost for success, but still low
        if success:
            base += 0.1
        return base

    def _compute_risk_bias(self, threat_level: float, valence: float) -> float:
        """
        FIX 4: Risk bias is bounded and saturates.
        Returns value in [-1, 1].
        """
        # Raw risk signal
        raw = -threat_level * (1.0 - valence) / 2.0
        # Apply tanh for saturation
        return float(np.tanh(raw))

    def build_state_embedding(
        self,
        amygdala_salience: Dict[str, float],
        ofc_utilities: Dict[str, float],
        vmpfc_intents: Dict[str, float],
        neurochemistry: Dict[str, float]
    ) -> np.ndarray:
        """Build neural-state fingerprint."""
        self.set_neuromodulator_state(
            dopamine=neurochemistry.get("dopamine"),
            serotonin=neurochemistry.get("serotonin"),
            norepinephrine=neurochemistry.get("norepinephrine")
        )

        salience = list(amygdala_salience.values()) if amygdala_salience else [0.0]
        utilities = list(ofc_utilities.values()) if ofc_utilities else [0.0]
        intents = list(vmpfc_intents.values()) if vmpfc_intents else [0.0]

        neuro = [
            self._neuro_state.dopamine,
            self._neuro_state.serotonin,
            self._neuro_state.norepinephrine
        ]

        raw = np.array(salience + utilities + intents + neuro, dtype=np.float32)

        if len(raw) < self.embedding_dim:
            raw = np.pad(raw, (0, self.embedding_dim - len(raw)))
        else:
            raw = raw[:self.embedding_dim]

        return self._normalize(raw)

    def should_store(self, rpe: float, threat_level: float, is_first_success: bool) -> bool:
        """
        Expanded storage logic.
        Store if:
        1. High Surprise (RPE)
        2. High Threat (Trauma/Survival)
        3. First time success (Novel achievement)
        """
        if abs(rpe) > self.surprise_threshold:
            return True
        
        if threat_level > 0.8:
            return True
            
        if is_first_success:
            return True
            
        return False

    def store(
        self,
        state_embedding: np.ndarray,
        threat_level: float,
        action_signature: str,
        predicted_utility: float,
        actual_utility: float,
        rpe: float,
        success: bool,
        goal_context: str = "",
        dominant_stimuli: List[str] = None,
        is_first_success: bool = False,
        features: Dict[str, Any] = None
    ) -> Optional[str]:
        """Store episodic memory with proper ID management and Interference."""
        if not self.should_store(rpe, threat_level, is_first_success):
            return None

        # Apply dopamine tag at encoding only
        fine_embedding = self._apply_dopamine_tag(state_embedding.astype(np.float32))
        coarse_embedding = self._create_coarse_embedding(fine_embedding)
        action_embedding = self._create_action_embedding(action_signature)

        # NEW: Retroactive Interference (The "Forgetting by Conflict" logic)
        # Before storing, check if this contradicts existing memories
        self._apply_retroactive_interference(
            fine_embedding, 
            action_embedding, 
            success
        )

        corrected_expectation = predicted_utility + rpe
        
        # FIX 2: Start with LOW reliability
        reliability = self._compute_initial_reliability(rpe, success)
        
        valence = 1.0 if success else -1.0
        episode_id = str(uuid.uuid4())
        faiss_id = self._next_faiss_id
        self._next_faiss_id += 1

        episode = EpisodicMemory(
            episode_id=episode_id,
            timestamp=datetime.now(),
            faiss_id=faiss_id,
            state_embedding=fine_embedding.tolist(),
            coarse_embedding=coarse_embedding.tolist(),
            action_embedding=action_embedding.tolist(),
            threat_level=threat_level,
            valence=valence,
            _goal_context=goal_context,
            _dominant_stimuli=dominant_stimuli or [],
            action_hash=hash(action_signature) % (2**31),
            action_signature=action_signature, # NEW
            predicted_utility=predicted_utility,
            actual_utility=actual_utility,
            rpe=rpe,
            corrected_expectation=corrected_expectation,
            dopamine_tag=self._neuro_state.dopamine,
            initial_dopamine_tag=self._neuro_state.dopamine,
            success=success,
            reliability=reliability,
            recall_count=0,
            ready_for_transfer=False,
            features=features or {}
        )

        # FIX 1: Add with explicit IDs
        id_array = np.array([faiss_id], dtype=np.int64)
        self.coarse_index.add_with_ids(coarse_embedding.reshape(1, -1), id_array)
        self.fine_index.add_with_ids(fine_embedding.reshape(1, -1), id_array)
        self.action_index.add_with_ids(action_embedding.reshape(1, -1), id_array)

        self._episodes[faiss_id] = episode
        self._episode_id_to_faiss_id[episode_id] = faiss_id

        # FIX 6: Track both consolidation counters
        self._episodes_since_soft_consolidation += 1
        self._episodes_since_hard_consolidation += 1

        # FIX 6: Soft consolidation while active
        if self._episodes_since_soft_consolidation >= self.SOFT_CONSOLIDATION_INTERVAL:
            self._soft_consolidate()

        self.save_state()
        return episode_id

    def _apply_retroactive_interference(
        self, 
        new_state: np.ndarray, 
        new_action: np.ndarray, 
        new_success: bool
    ) -> None:
        """
        Biological forgetting: If new experience contradicts old ones, 
        suppress the old ones.
        """
        if self.fine_index.ntotal == 0:
            return

        # Find very similar past states
        k = 5
        scores, ids = self.fine_index.search(new_state.reshape(1, -1), k)
        
        for i in range(k):
            faiss_id = int(ids[0][i])
            similarity = float(scores[0][i])
            
            if faiss_id < 0 or similarity < 0.85: # Only check highly similar memories
                continue
                
            episode = self._episodes.get(faiss_id)
            if not episode:
                continue
                
            # Check action match
            ep_action = np.array(episode.action_embedding, dtype=np.float32)
            action_sim = float(np.dot(new_action, ep_action))
            
            if action_sim > 0.9:
                # Same situation, Same action... Different outcome?
                old_success = episode.success
                
                if old_success != new_success:
                    # CONFLICT DETECTED
                    # The world has changed. The old memory is dangerous.
                    # Slash reliability.
                    episode.reliability *= 0.4
                    episode.note = f"Suppressed by interference from newer episode"

    def recall(
        self,
        state_embedding: np.ndarray,
        action_signature: str = "",
        current_threat_level: float = 0.0
    ) -> Optional[AggregatedBias]:
        """
        Two-stage recall with action matching and threat gating.
        FIX 3: No dopamine amplification at recall.
        FIX 5: Action similarity filtering.
        """
        if self.coarse_index.ntotal == 0:
            return None

        # No DA tag at recall - just normalize
        query_fine = self._normalize(state_embedding.astype(np.float32))
        query_coarse = self._create_coarse_embedding(query_fine)
        query_action = self._create_action_embedding(action_signature) if action_signature else None

        # NEW: Habituation Check (Refusing Recall)
        # If we are safe and this is extremely familiar, let the Cortex/Habits handle it.
        # This prevents "over-thinking" routine tasks.
        if current_threat_level < 0.3:
            # Quick check on coarse index
            k_check = 1
            scores, _ = self.coarse_index.search(query_coarse.reshape(1, -1), k_check)
            if scores[0][0] > self.HABITUATION_THRESHOLD:
                # "I've seen this a million times and I'm safe."
                # Return None to signal "Use Default Policy"
                return None

        # Get thresholds
        dynamic_threshold = self._get_dynamic_threshold()
        threat_threshold = self._get_threat_threshold(current_threat_level)
        effective_threshold = max(dynamic_threshold, threat_threshold)

        # Stage 1: Coarse familiarity check
        k_coarse = min(10, self.coarse_index.ntotal)
        familiarity_scores, coarse_ids = self.coarse_index.search(
            query_coarse.reshape(1, -1), k_coarse
        )

        # Filter by coarse threshold
        familiar_ids = [
            int(fid) for fid, score in zip(coarse_ids[0], familiarity_scores[0])
            if score >= self.coarse_threshold and fid >= 0
        ]

        if not familiar_ids:
            return None

        # Stage 2: Fine matching on familiar candidates
        biases = []
        max_familiarity = float(familiarity_scores[0][0])

        for faiss_id in familiar_ids:
            episode = self._episodes.get(faiss_id)
            if not episode:
                continue

            # Fine similarity (NO dopamine amplification)
            episode_fine = np.array(episode.state_embedding, dtype=np.float32)
            fine_sim = float(np.dot(query_fine, episode_fine))

            if fine_sim < effective_threshold:
                continue

            # FIX 5: Action similarity check
            action_match = 1.0
            if query_action is not None:
                episode_action = np.array(episode.action_embedding, dtype=np.float32)
                action_match = float(np.dot(query_action, episode_action))
                # Penalize mismatched actions
                if action_match < 0.5:
                    fine_sim *= action_match

            # Update recall count for repetition bonus
            episode.recall_count += 1
            
            # NEW: Check for Schema Promotion
            # If recalled often and reliable, mark for transfer to Cortex
            if episode.recall_count > 5 and episode.reliability > 0.8:
                episode.ready_for_transfer = True
            
            # FIX 2: Reliability increases with consistent recall
            repetition_bonus = min(0.3, episode.recall_count * 0.05)
            effective_reliability = min(1.0, episode.reliability + repetition_bonus)

            expected = episode.corrected_expectation
            
            # FIX 4: Bounded risk bias
            risk_bias = self._compute_risk_bias(episode.threat_level, episode.valence)
            
            confidence_boost = effective_reliability * fine_sim * episode.valence

            biases.append(MemoryBias(
                expected_outcome=expected,
                confidence=effective_reliability * fine_sim,
                episode_id=episode.episode_id,
                similarity=fine_sim,
                confidence_boost=confidence_boost,
                risk_bias=risk_bias,
                recall_stage=2,
                familiarity=max_familiarity,
                action_match=action_match,
                note=f"[DEBUG] {episode._goal_context}"
            ))

        if not biases:
            return None

        return self._aggregate_biases(biases, max_familiarity)

    def check_familiarity(self, state_embedding: np.ndarray) -> Tuple[bool, float]:
        """Stage 1 only: "Does this feel familiar?" Fast check."""
        if self.coarse_index.ntotal == 0:
            return False, 0.0

        query_coarse = self._create_coarse_embedding(
            self._normalize(state_embedding.astype(np.float32))
        )
        scores, _ = self.coarse_index.search(query_coarse.reshape(1, -1), 1)
        familiarity = float(scores[0][0])
        return familiarity >= self.coarse_threshold, familiarity

    def _aggregate_biases(
        self,
        biases: List[MemoryBias],
        familiarity: float = 0.0
    ) -> AggregatedBias:
        """Combine multiple episode biases with weighted aggregation."""
        if not biases:
            return AggregatedBias(
                expected_outcome=0.0,
                confidence=0.0,
                confidence_boost=0.0,
                risk_bias=0.0,
                n_episodes=0,
                familiarity=familiarity
            )

        total_confidence = sum(b.confidence for b in biases)
        if total_confidence == 0:
            return AggregatedBias(
                expected_outcome=0.0,
                confidence=0.0,
                confidence_boost=0.0,
                risk_bias=0.0,
                n_episodes=len(biases),
                familiarity=familiarity
            )

        weighted_expectation = sum(
            b.expected_outcome * b.confidence for b in biases
        ) / total_confidence

        weighted_confidence_boost = sum(
            b.confidence_boost * b.confidence for b in biases
        ) / total_confidence

        weighted_risk_bias = sum(
            b.risk_bias * b.confidence for b in biases
        ) / total_confidence

        # FIX 4: Final risk bias also bounded
        weighted_risk_bias = float(np.clip(weighted_risk_bias, -1.0, 1.0))

        return AggregatedBias(
            expected_outcome=weighted_expectation,
            confidence=total_confidence / len(biases),
            confidence_boost=weighted_confidence_boost,
            risk_bias=weighted_risk_bias,
            n_episodes=len(biases),
            familiarity=familiarity,
            episode_ids=[b.episode_id for b in biases]
        )

    def update_reliability(self, episode_id: str, new_rpe: float, success: bool) -> None:
        """
        FIX 2: Update reliability based on validation.
        Reliability increases when predictions are accurate.
        """
        faiss_id = self._episode_id_to_faiss_id.get(episode_id)
        if faiss_id is not None and faiss_id in self._episodes:
            episode = self._episodes[faiss_id]
            
            # Low RPE = accurate prediction = increase reliability
            if abs(new_rpe) < self.surprise_threshold:
                episode.reliability = min(1.0, episode.reliability + 0.1)
            else:
                # High RPE = bad prediction = decrease reliability
                episode.reliability = max(0.1, episode.reliability - 0.05)

    def _soft_consolidate(self) -> None:
        """
        FIX 6: Soft consolidation - runs while active.
        Light pruning + dopamine decay + Schema Identification.
        """
        now = datetime.now()
        
        schema_candidates = []

        # Decay dopamine tags
        for episode in self._episodes.values():
            episode.dopamine_tag = max(0.0, episode.dopamine_tag - self.DA_DECAY_RATE)
            
            # Identify Schema Candidates (Pattern Abstraction)
            if episode.ready_for_transfer:
                schema_candidates.append(episode.episode_id)
        
        # In a full system, we would emit 'schema_candidates' here to the Neocortex
        # for rule extraction.
        
        # Mark low-value episodes for removal if over soft limit
        if len(self._episodes) > int(self.max_memories * 0.9):
            to_remove = self._get_lowest_value_episodes(
                count=len(self._episodes) - int(self.max_memories * 0.8)
            )
            for faiss_id in to_remove:
                self._remove_episode(faiss_id)

        self._episodes_since_soft_consolidation = 0

    def _hard_consolidate(self) -> None:
        """
        FIX 6: Hard consolidation - runs during sleep.
        Full rebuild with aggressive pruning.
        """
        now = datetime.now()

        def value(episode: EpisodicMemory) -> float:
            age_hours = (now - episode.timestamp).total_seconds() / 3600
            recency = np.exp(-self.decay_rate * age_hours)
            importance = min(1.0, abs(episode.rpe) * 2)
            recall_bonus = min(0.3, episode.recall_count * 0.05)
            # DA tag already decayed, still gives slight boost
            da_boost = 1.0 + episode.dopamine_tag * 0.05
            return recency * (episode.reliability + recall_bonus) * (0.5 + 0.5 * importance) * da_boost

        # Sort and keep top memories
        sorted_episodes = sorted(self._episodes.values(), key=value, reverse=True)
        keep_episodes = sorted_episodes[:self.max_memories]
        keep_ids = {e.faiss_id for e in keep_episodes}

        # Remove low-value episodes
        to_remove = [fid for fid in self._episodes.keys() if fid not in keep_ids]
        for faiss_id in to_remove:
            self._remove_episode(faiss_id)

        # Rebuild indices for compactness
        self._rebuild_indices()

        self._episodes_since_hard_consolidation = 0

    def _get_lowest_value_episodes(self, count: int) -> List[int]:
        """Get faiss_ids of lowest value episodes."""
        now = datetime.now()

        def value(episode: EpisodicMemory) -> float:
            age_hours = (now - episode.timestamp).total_seconds() / 3600
            recency = np.exp(-self.decay_rate * age_hours)
            importance = min(1.0, abs(episode.rpe) * 2)
            return recency * episode.reliability * (0.5 + 0.5 * importance)

        sorted_episodes = sorted(self._episodes.values(), key=value)
        return [e.faiss_id for e in sorted_episodes[:count]]

    def _remove_episode(self, faiss_id: int) -> None:
        """Remove episode from all indices and storage."""
        if faiss_id not in self._episodes:
            return

        episode = self._episodes[faiss_id]
        
        # Remove from FAISS (IndexIDMap supports remove_ids)
        id_array = np.array([faiss_id], dtype=np.int64)
        self.coarse_index.remove_ids(id_array)
        self.fine_index.remove_ids(id_array)
        self.action_index.remove_ids(id_array)

        # Remove from storage
        del self._episodes[faiss_id]
        del self._episode_id_to_faiss_id[episode.episode_id]

    def _rebuild_indices(self) -> None:
        """Rebuild FAISS indices from scratch for compactness."""
        # Create new indices
        new_coarse = faiss.IndexIDMap(faiss.IndexFlatIP(self.coarse_dim))
        new_fine = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))
        new_action = faiss.IndexIDMap(faiss.IndexFlatIP(self.action_dim))

        for faiss_id, episode in self._episodes.items():
            id_array = np.array([faiss_id], dtype=np.int64)
            coarse = np.array(episode.coarse_embedding, dtype=np.float32).reshape(1, -1)
            fine = np.array(episode.state_embedding, dtype=np.float32).reshape(1, -1)
            action = np.array(episode.action_embedding, dtype=np.float32).reshape(1, -1)

            new_coarse.add_with_ids(coarse, id_array)
            new_fine.add_with_ids(fine, id_array)
            new_action.add_with_ids(action, id_array)

        self.coarse_index = new_coarse
        self.fine_index = new_fine
        self.action_index = new_action

    def enter_sleep(self) -> None:
        """Enter consolidation-ready state."""
        self._is_active = False
        if self._episodes_since_hard_consolidation >= self.consolidation_threshold:
            self._hard_consolidate()

    def wake(self) -> None:
        """Exit consolidation state."""
        self._is_active = True

    def force_consolidate(self) -> None:
        """Manual hard consolidation trigger."""
        self._hard_consolidate()

    def get_statistics(self) -> Dict:
        """Return memory system statistics."""
        if not self._episodes:
            return {
                "total_memories": 0,
                "is_active": self._is_active,
                "neuro_state": {
                    "dopamine": self._neuro_state.dopamine,
                    "serotonin": self._neuro_state.serotonin
                }
            }

        memories = list(self._episodes.values())
        return {
            "total_memories": len(memories),
            "avg_reliability": float(np.mean([m.reliability for m in memories])),
            "avg_rpe": float(np.mean([m.rpe for m in memories])),
            "avg_dopamine_tag": float(np.mean([m.dopamine_tag for m in memories])),
            "avg_recall_count": float(np.mean([m.recall_count for m in memories])),
            "success_rate": float(np.mean([1.0 if m.success else 0.0 for m in memories])),
            "coarse_index_size": self.coarse_index.ntotal,
            "fine_index_size": self.fine_index.ntotal,
            "action_index_size": self.action_index.ntotal,
            "is_active": self._is_active,
            "current_threshold": self._get_dynamic_threshold(),
            "episodes_since_soft": self._episodes_since_soft_consolidation,
            "episodes_since_hard": self._episodes_since_hard_consolidation,
            "neuro_state": {
                "dopamine": self._neuro_state.dopamine,
                "serotonin": self._neuro_state.serotonin
            }
        }

    def get_clusters(self, similarity_threshold: float = 0.85, min_cluster_size: int = 3) -> List[List[EpisodicMemory]]:
        """
        Groups memories into clusters based on state similarity.
        Used for Neocortical consolidation.
        """
        clusters = []
        visited = set()
        
        # Get all memories
        all_memories = list(self._episodes.values())
        
        for i, mem in enumerate(all_memories):
            if mem.episode_id in visited:
                continue
                
            # Start a new cluster
            current_cluster = [mem]
            visited.add(mem.episode_id)
            
            # Find neighbors
            for j, other in enumerate(all_memories):
                if i == j or other.episode_id in visited:
                    continue
                
                # Calculate similarity
                sim = np.dot(mem.state_embedding, other.state_embedding)
                if sim > similarity_threshold:
                    current_cluster.append(other)
                    visited.add(other.episode_id)
            
            if len(current_cluster) >= min_cluster_size:
                clusters.append(current_cluster)
                
        return clusters