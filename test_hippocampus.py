import unittest
import numpy as np
from datetime import datetime
import time
from agents.hippocampus.hippocampus import Hippocampus, EpisodicMemory, AggregatedBias

class TestHippocampus(unittest.TestCase):
    def setUp(self):
        print("\n" + "="*50)
        print("Setting up Hippocampus for testing...")
        self.hippocampus = Hippocampus(
            embedding_dim=64,
            coarse_dim=32,
            action_dim=16,
            base_similarity_threshold=0.75,
            coarse_threshold=0.60,
            fine_threshold=0.80,
            surprise_threshold=0.15,
            max_memories=10,  # Small number for testing consolidation
            decay_rate=0.01,
            consolidation_threshold=5
        )
        print("Hippocampus initialized.")

    def test_initialization(self):
        print("\n[Test] Initialization")
        print("Verifying dimensions and initial state...")
        self.assertEqual(self.hippocampus.embedding_dim, 64)
        self.assertEqual(self.hippocampus.coarse_index.d, 32)
        self.assertEqual(self.hippocampus.fine_index.d, 64)
        self.assertEqual(self.hippocampus.action_index.d, 16)
        self.assertEqual(len(self.hippocampus._episodes), 0)
        print("Initialization verified successfully.")

    def test_build_state_embedding(self):
        print("\n[Test] Build State Embedding")
        print("Simulating input from Amygdala, OFC, and vmPFC...")
        amygdala_salience = {"fear": 0.8, "joy": 0.2}
        ofc_utilities = {"food": 0.9}
        vmpfc_intents = {"approach": 0.7}
        neurochemistry = {"dopamine": 0.6, "serotonin": 0.4, "norepinephrine": 0.5}
        
        print(f"Inputs: Amygdala={amygdala_salience}, OFC={ofc_utilities}, vmPFC={vmpfc_intents}")

        embedding = self.hippocampus.build_state_embedding(
            amygdala_salience, ofc_utilities, vmpfc_intents, neurochemistry
        )

        self.assertEqual(len(embedding), 64)
        self.assertTrue(np.allclose(np.linalg.norm(embedding), 1.0))
        # Check if neurochemistry was updated
        self.assertEqual(self.hippocampus._neuro_state.dopamine, 0.6)
        print("State embedding built and normalized successfully.")

    def test_store_memory(self):
        print("\n[Test] Store Memory (Real Situation: Finding Food)")
        state_embedding = np.random.rand(64).astype(np.float32)
        state_embedding /= np.linalg.norm(state_embedding)
        
        # RPE > surprise_threshold (0.15)
        print("Agent takes action 'move_forward' to 'find_food'.")
        print("Prediction: 0.5, Actual: 0.8 -> RPE: 0.3 (Surprise!)")
        
        episode_id = self.hippocampus.store(
            state_embedding=state_embedding,
            threat_level=0.2,
            action_signature="move_forward",
            predicted_utility=0.5,
            actual_utility=0.8,
            rpe=0.3,
            success=True,
            goal_context="find_food"
        )

        self.assertIsNotNone(episode_id)
        self.assertEqual(len(self.hippocampus._episodes), 1)
        self.assertEqual(self.hippocampus.coarse_index.ntotal, 1)
        
        # Check stored values
        faiss_id = self.hippocampus._episode_id_to_faiss_id[episode_id]
        episode = self.hippocampus._episodes[faiss_id]
        self.assertEqual(episode.rpe, 0.3)
        self.assertEqual(episode.success, True)
        self.assertEqual(episode._goal_context, "find_food")
        print(f"Memory stored successfully. Episode ID: {episode_id}")

    def test_should_not_store_low_rpe(self):
        print("\n[Test] Ignore Low RPE (Real Situation: Routine Walk)")
        state_embedding = np.random.rand(64).astype(np.float32)
        
        # RPE < surprise_threshold (0.15)
        print("Agent takes action 'move_forward'.")
        print("Prediction: 0.5, Actual: 0.55 -> RPE: 0.05 (Expected outcome)")
        
        episode_id = self.hippocampus.store(
            state_embedding=state_embedding,
            threat_level=0.2,
            action_signature="move_forward",
            predicted_utility=0.5,
            actual_utility=0.55,
            rpe=0.05,
            success=True
        )

        self.assertIsNone(episode_id)
        self.assertEqual(len(self.hippocampus._episodes), 0)
        print("Memory NOT stored (as expected for low surprise).")

    def test_recall_memory(self):
        print("\n[Test] Recall Memory (Real Situation: Déjà vu)")
        # Store a memory first
        state_embedding = np.random.rand(64).astype(np.float32)
        state_embedding /= np.linalg.norm(state_embedding)
        
        print("Storing initial memory: 'move_forward' -> Success")
        self.hippocampus.store(
            state_embedding=state_embedding,
            threat_level=0.2,
            action_signature="move_forward",
            predicted_utility=0.5,
            actual_utility=0.8,
            rpe=0.3,
            success=True
        )

        # Recall with same state and action
        print("Recalling with identical state and action...")
        bias = self.hippocampus.recall(
            state_embedding=state_embedding,
            action_signature="move_forward",
            current_threat_level=0.2
        )

        self.assertIsNotNone(bias)
        self.assertIsInstance(bias, AggregatedBias)
        self.assertGreater(bias.confidence, 0.0)
        self.assertEqual(bias.n_episodes, 1)
        print(f"Recall successful. Bias confidence: {bias.confidence:.4f}")

    def test_recall_action_mismatch(self):
        print("\n[Test] Recall Action Mismatch (Real Situation: Different Choice)")
        # Store a memory
        state_embedding = np.random.rand(64).astype(np.float32)
        state_embedding /= np.linalg.norm(state_embedding)
        
        print("Storing memory: 'move_forward' in specific state.")
        self.hippocampus.store(
            state_embedding=state_embedding,
            threat_level=0.2,
            action_signature="move_forward",
            predicted_utility=0.5,
            actual_utility=0.8,
            rpe=0.3,
            success=True
        )

        # Recall with different action
        print("Recalling with same state but action 'run_away'...")
        bias = self.hippocampus.recall(
            state_embedding=state_embedding,
            action_signature="run_away", # Different action
            current_threat_level=0.2
        )
        
        if bias:
             print(f"Recall returned bias (likely due to state match), but action match should be low.")
        else:
             print("Recall returned None (filtered out due to mismatch).")

    def test_update_reliability(self):
        print("\n[Test] Update Reliability (Real Situation: Validation)")
        state_embedding = np.random.rand(64).astype(np.float32)
        episode_id = self.hippocampus.store(
            state_embedding=state_embedding,
            threat_level=0.2,
            action_signature="move_forward",
            predicted_utility=0.5,
            actual_utility=0.8,
            rpe=0.3,
            success=True
        )
        
        faiss_id = self.hippocampus._episode_id_to_faiss_id[episode_id]
        initial_reliability = self.hippocampus._episodes[faiss_id].reliability
        print(f"Initial reliability: {initial_reliability:.2f}")

        # Update with low RPE (success)
        print("Prediction validated (Low RPE). Increasing reliability...")
        self.hippocampus.update_reliability(episode_id, new_rpe=0.05, success=True)
        updated_reliability = self.hippocampus._episodes[faiss_id].reliability
        print(f"Updated reliability: {updated_reliability:.2f}")
        self.assertGreater(updated_reliability, initial_reliability)

        # Update with high RPE (failure)
        print("Prediction failed (High RPE). Decreasing reliability...")
        self.hippocampus.update_reliability(episode_id, new_rpe=0.5, success=False)
        final_reliability = self.hippocampus._episodes[faiss_id].reliability
        print(f"Final reliability: {final_reliability:.2f}")
        self.assertLess(final_reliability, updated_reliability)

    def test_soft_consolidation(self):
        print("\n[Test] Soft Consolidation (Real Situation: Active Wakefulness)")
        # Fill up to trigger soft consolidation
        self.hippocampus._episodes_since_soft_consolidation = 50
        print("Simulating 50 episodes to trigger soft consolidation...")
        
        # Add some memories
        for i in range(5):
            self.hippocampus.store(
                state_embedding=np.random.rand(64).astype(np.float32),
                threat_level=0.2,
                action_signature=f"action_{i}",
                predicted_utility=0.5,
                actual_utility=0.8,
                rpe=0.3,
                success=True
            )
            
        # Check if soft consolidation happened (counter reset)
        self.hippocampus._episodes_since_soft_consolidation = 50
        print("Triggering store() to force check...")
        self.hippocampus.store(
             state_embedding=np.random.rand(64).astype(np.float32),
             threat_level=0.2,
             action_signature="trigger",
             predicted_utility=0.5,
             actual_utility=0.8,
             rpe=0.3,
             success=True
        )
        
        self.assertEqual(self.hippocampus._episodes_since_soft_consolidation, 0)
        print("Soft consolidation triggered and counter reset.")

    def test_hard_consolidation_sleep(self):
        print("\n[Test] Hard Consolidation (Real Situation: Sleep)")
        # Add memories beyond max_memories (10)
        print("Storing 15 memories (Capacity: 10)...")
        for i in range(15):
            self.hippocampus.store(
                state_embedding=np.random.rand(64).astype(np.float32),
                threat_level=0.2,
                action_signature=f"action_{i}",
                predicted_utility=0.5,
                actual_utility=0.8,
                rpe=0.3,
                success=True
            )
            
        self.assertEqual(len(self.hippocampus._episodes), 15)
        print(f"Current memories: {len(self.hippocampus._episodes)}")
        
        # Trigger sleep and hard consolidation
        self.hippocampus._episodes_since_hard_consolidation = 10 # > consolidation_threshold (5)
        print("Entering sleep mode...")
        self.hippocampus.enter_sleep()
        
        # Should be pruned to max_memories (10)
        self.assertEqual(len(self.hippocampus._episodes), 10)
        self.assertEqual(self.hippocampus._episodes_since_hard_consolidation, 0)
        self.assertFalse(self.hippocampus._is_active)
        print(f"Hard consolidation complete. Memories pruned to: {len(self.hippocampus._episodes)}")

    def test_neuromodulator_effects(self):
        print("\n[Test] Neuromodulator Effects (Real Situation: Mood Swings)")
        # High dopamine -> lower threshold
        print("High Dopamine (Excitement) -> Expecting lower similarity threshold")
        self.hippocampus.set_neuromodulator_state(dopamine=0.9, serotonin=0.1)
        thresh_high_da = self.hippocampus._get_dynamic_threshold()
        print(f"Threshold (High DA): {thresh_high_da:.2f}")
        
        # High serotonin -> higher threshold
        print("High Serotonin (Caution) -> Expecting higher similarity threshold")
        self.hippocampus.set_neuromodulator_state(dopamine=0.1, serotonin=0.9)
        thresh_high_ser = self.hippocampus._get_dynamic_threshold()
        print(f"Threshold (High Serotonin): {thresh_high_ser:.2f}")
        
        self.assertLess(thresh_high_da, thresh_high_ser)
        print("Neuromodulation logic verified.")

if __name__ == '__main__':
    unittest.main()
