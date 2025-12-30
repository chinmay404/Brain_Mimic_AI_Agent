from dataclasses import dataclass
import numpy as np
# from agents.neuromodulator import Neuromodulators
from typing import List, Tuple
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from agents.thalamus.amygdala_classifier.classifire_main import Amygdala


@dataclass
class InputSignals:
    source: str
    content: str
    embeddings: np.ndarray


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Computes Softmax. 
    Low Temperature (e.g., 0.1) = Winner takes all (Very sharp contrast).
    High Temperature (e.g., 5.0) = Everyone gets a participation trophy (Flat).
    """
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()


class EmbeddingModel:
    def encode(self, text: str) -> np.ndarray:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",
                                                      api_key="AIzaSyAVZ3HLSYf2ngvyssUc4Ky8-B5KjFUii60")
            return np.array(embeddings.embed_query(text))

        except Exception as e:
            raise ValueError(f"[Thalamus] EmbeddingModel Error: {e}")


class Thalamus:
    def __init__(self) -> None:
        self.embedding_model = EmbeddingModel()
        self.current_goal_embedding = None
        self.dopamine = 0.85
        self.serotonin = 0.10
        self.threat_threshold = 0.6
        self._amygdala_model_ = Amygdala(self.threat_threshold)

    def set_goal(self, goal_txt: str):
        """Current AI Goal and Its Embeddings 

        Args:
            goal_txt (str): goal Text
        """
        try:
            print("[Thalamus] Setting Goal embeddings ")
            self.current_goal_embedding = self._normalize(
                self.embedding_model.encode(goal_txt))
        except Exception as e:
            print(f"Error in Setting  Goal in Thalmus: {e}")

    # def set_input(self, input_txt: str):
    #     """Current AI input and Its Embeddings
    #     Thalamus should:
    #     Compare input ↔ goal
    #     Modulate relevance via neuromodulators
    #     Output prioritized signals

    #     Args:
    #         input_txt (str): input Text
    #     """
    #     try:
    #         print("[Thalamus] Setting input embeddings ")
    #         self.current_input_embeddings = self.embedding_model.encode()
    #     except Exception as e:
    #         print(f"Error in Setting  input in Thalmus: {e}")

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-8)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _amygdala_scan(self, content: str):
        return self._amygdala_model_.scan(content=content)

    # def process_signals(self, inputs: List[Tuple[str, str]]):
    #     if self.current_goal_embedding is None:
    #         raise ValueError("Goal Embeddings not set")

    #     raw_scores = []
    #     metadata = []

    #     # Step A: Calculate Raw Relevance + Amygdala Boost
    #     for source, content in inputs:
    #         content_emb = self._normalize(self.embedding_model.encode(content))

    #         # 1. Semantic Match (The Cortex)
    #         base_relevance = self._cosine_similarity(
    #             content_emb, self.current_goal_embedding)

    #         # 2. Urgency Boost (The Amygdala)
    #         urgency_boost = self._amygdala_scan(content)

    #         # Total Raw Activation
    #         total_activation = base_relevance + urgency_boost

    #         raw_scores.append(total_activation)
    #         metadata.append({"source": source, "content": content,
    #                         "base": base_relevance, "urgency": urgency_boost})

    #     # Step B: Chemical Modulation (Gain and Threshold)
    #     # Serotonin acts as a subtractive threshold (Noise Gate)
    #     # Dopamine acts as a multiplicative gain (Amplifier)

    #     modulated_scores = []
    #     # Higher serotonin = higher bar to pass
    #     serotonin_threshold = 0.5 + (self.serotonin * 0.3)

    #     for score in raw_scores:
    #         # Apply Threshold (Serotonin)
    #         if score < serotonin_threshold:
    #             filtered_score = 0  # Signal blocked!
    #         else:
    #             filtered_score = score

    #         # Apply Gain (Dopamine)
    #         # High dopamine makes strong signals even stronger
    #         final_score = filtered_score * (1 + self.dopamine)
    #         modulated_scores.append(final_score)

    #     # Step C: The Contrast Fix (Softmax)
    #     # We turn these raw numbers into a probability distribution (Attention %)
    #     # We use Serotonin to control "Temperature" (Focus).
    #     # High Serotonin = Low Temp (Sharper focus). Low Serotonin = High Temp (Scattered).

    #     temperature = max(0.1, 1.0 - self.serotonin)
    #     attention_weights = softmax(
    #         np.array(modulated_scores), temperature=temperature)

    #     # Pack results
    #     results = []
    #     for i, meta in enumerate(metadata):
    #         results.append({
    #             "source": meta["source"],
    #             "content": meta["content"],
    #             "relevance": attention_weights[i],
    #             "debug_raw": modulated_scores[i]
    #         })

    #     # Sort by relevance
    #     results.sort(key=lambda x: x['relevance'], reverse=True)
    #     return results

    def process(self, inputs: List[Tuple[str, str]]):
        if self.current_goal_embedding is None:
            raise ValueError("Goal not set")

        activations = []

        # 1. Compute activation per signal
        for source, content in inputs:
            emb = self._normalize(self.embedding_model.encode(content))
            relevance = self._cosine_similarity(
                emb, self.current_goal_embedding)

            emo = self._amygdala_scan(content)

            # multiplicative modulation
            if emo["label"] in ["threat", "physical pain"] and emo["salience"] >= self.threat_threshold:
                activation = emo["salience"] * 1.5   # Ignore goal temporarily
            else:
                activation = relevance * (1 + emo["salience"])

            activations.append({
                "source": source,
                "content": content,
                "activation": activation,
                "amygdala_label": emo["label"],
                "amygdala_salience": emo["salience"]
            })

        # 2. Hard gate (serotonin)
        gate = 0.4 + self.serotonin * 0.3
        gated = [a for a in activations if a["activation"] >= gate]

        if not gated:
            return []

        # 3. Dopamine gain
        for g in gated:
            g["activation"] *= (1 + self.dopamine)

        # 4. Normalize attention (only survivors)
        scores = np.array([g["activation"] for g in gated])
        temperature = max(0.1, 1.0 - self.serotonin)
        weights = softmax(scores, temperature)

        results = []
        for g, w in zip(gated, weights):
            results.append({
                "source": g["source"],
                "content": g["content"],
                "attention": float(w),
                "amygdala_label": g["amygdala_label"],
                "amygdala_salience": g["amygdala_salience"]
            })

        results.sort(key=lambda x: x["attention"], reverse=True)
        return results


# Initialize Thalamus
# --- Run Simulation ---
if __name__ == "__main__":
    thalamus = Thalamus()
    goal_text = "Cook dinner without burning food"
    thalamus.set_goal(goal_text)

    inputs = [
        ("hearing", "Timer ticking"),
        ("vision", "Smoke from the pan"),
        ("touch", "Pot handle hot to touch"),
        ("emotion", "Feeling focused"),
        ("smell", "Spices sizzling"),
    ]



    results = thalamus.process(inputs)

    print(f"\n{'SOURCE':<10} | {'ATTENTION':<10} | {'AMYGDALA':<20} | CONTENT")
    print("-" * 80)
    for r in results:
        bar = "█" * int(r['attention'] * 20)
        amygdala_info = f"{r['amygdala_label']} ({r['amygdala_salience']:.2f})"
        print(f"{r['source']:<10} | {r['attention']:.3f} {bar:<10} | {amygdala_info:<20} | {r['content']}")
