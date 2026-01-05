from transformers import pipeline
import sys


class Amygdala():
    def __init__(self, threat_threshold , model_name: str = "facebook/bart-large-mnli") -> None:
        print(f"[Amygdala] Loading neural circuits ({model_name})...")
        try:
            # if pipeline is None:
            #     raise ImportError("Transformers pipeline not available")
                
            self.classifire = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=0 # Force CPU to avoid potential accelerate/device issues if possible, or keep 0 if user has GPU
            )
            # Note: device=0 might be causing issues if accelerate is broken. 
            # But if import failed, we go to except.
            
            self.labels = [
                "threat",
                "physical pain",
                "social conflict",
                "high reward",
                "novelty",
            ]

            self.gain = {
                "threat": 1.0,
                "physical pain": 1.2,
                "social conflict": 0.5,
                "high reward": 0.3,
                "novelty": 0.1,
            }
            print(f"[Amygdala] Neural circuits ready.")
        except Exception as e:
            print(f"[Amygdala] Error in neural circuits ({model_name}) : {e}")
            self.classifire = None
            self.labels = ["threat", "physical pain", "social conflict", "high reward", "novelty"]
            self.gain = {"threat": 1.0, "physical pain": 1.2, "social conflict": 0.5, "high reward": 0.3, "novelty": 0.1}

    def scan(self, content: str) -> dict:
        if not content:
            return {
                "label": None,
                "raw_score": 0.0,
                "salience": 0.0
            }

        try:
            if self.classifire:
                result = self.classifire(content, candidate_labels=self.labels)
                label = result['labels'][0]
                raw_score = result['scores'][0]
            else:
                # Fallback logic if classifier is broken
                # Simple keyword matching
                content_lower = content.lower()
                if "threat" in content_lower or "danger" in content_lower or "pain" in content_lower:
                    label = "threat"
                    raw_score = 0.9
                elif "reward" in content_lower or "gain" in content_lower:
                    label = "high reward"
                    raw_score = 0.8
                else:
                    label = "novelty"
                    raw_score = 0.5
            
            # Nonlinear salience amplification
            salience = min(raw_score * self.gain.get(label, 1.0), 1.0)
            print(
                f"[Amygdala] Detected: {label} | "
                f"raw={raw_score:.2f} → salience={salience:.2f}"
            )
            return {
                "label": label,
                "raw_score": raw_score,
                "salience": salience
            }
        except Exception as e:
            print(f"[Amygdala] Scan Error : {e}")
