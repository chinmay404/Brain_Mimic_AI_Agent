import json
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from utils.llm_provider import get_groq_llm


@dataclass
class Episode:
    state_features: Dict[str, Any]
    action_abstract: str
    outcome_score: float


@dataclass
class CorticalRule:
    condition: Dict[str, Any]  # e.g. {"threat_level": ">0.8"}
    action: str
    confidence: float
    source_cluster_id: str


class LLMStrcuturedopCorticalRule(BaseModel):
    condition: Dict[str, Any] = Field(description="The condition for the rule, e.g. {'threat_level': '>0.8'}")
    action: str = Field(description="The action to take")
    confidence: float = Field(description="Confidence score")


class NeocortexRuleExtractor:
    def __init__(self) -> None:
        self.llm = get_groq_llm(model="qwen/qwen3-32b")

    def extract_rule(self, cluster: List[Episode] ,cluster_id : str ):
        if not self._is_stable(cluster):
            return None

        payload = []
        for ep in cluster:
            e = {
                "state": ep.state_features,
                "action": ep.action_abstract,
                "outcome": ep.outcome_score
            }
            payload.append(e)

        prompt = f"""
        Analyze these {len(payload)} episodes. They represent a successful strategy.
        Extract the MINIMAL condition required for this success.
        Ignore features that vary randomly.
        
        Data: {json.dumps(payload)}
        
        """
        if not self.llm:
            return None
        try:
            llm_with_structures_op = self.llm.with_structured_output(
                LLMStrcuturedopCorticalRule)
            result = llm_with_structures_op.invoke(prompt)
            print(result)
            if self._validate_rule(result, cluster):
                return CorticalRule(
                    condition=result.condition,
                    action=result.action,
                    confidence=result.confidence,
                    source_cluster_id=cluster_id
                )
            else:
                print(
                    f"Rule rejected: Validation failed for {result.condition}")
                return None
        except Exception as e:
            print(f"[NeoCortextRuleExtractor] Error in rule Extraction : {e}")
            return None

    def _validate_rule(self, rule_data: LLMStrcuturedopCorticalRule, cluster: List[Episode]) -> bool:
        """
        Replays the rule against the episodes. 
        If the rule says 'threat > 0.8' but an episode had 0.5 and succeeded,
        the rule might be too strict, or the episode is an outlier.
        """
        matches = 0
        total = len(cluster)

        for ep in cluster:
            conditions_met = True
            for feature, condition_val in rule_data.condition.items():
                # Skip if feature missing in episode
                if feature not in ep.state_features:
                    conditions_met = False
                    break

                val = ep.state_features[feature]

                # Handle non-string conditions (e.g. booleans)
                if not isinstance(condition_val, str):
                    if val != condition_val:
                        conditions_met = False
                    continue

                condition_str = condition_val

                # Simple parser for operators
                try:
                    if condition_str.startswith(">="):
                        if not (float(val) >= float(condition_str[2:])):
                            conditions_met = False
                    elif condition_str.startswith("<="):
                        if not (float(val) <= float(condition_str[2:])):
                            conditions_met = False
                    elif condition_str.startswith(">"):
                        if not (float(val) > float(condition_str[1:])):
                            conditions_met = False
                    elif condition_str.startswith("<"):
                        if not (float(val) < float(condition_str[1:])):
                            conditions_met = False
                    elif condition_str.startswith("=="):
                        target = condition_str[2:]
                        if str(val) != target:
                            conditions_met = False
                    else:
                        # Default equality
                        if str(val) != str(condition_str):
                            conditions_met = False
                except ValueError:
     
                    conditions_met = False

                if not conditions_met:
                    break

            # Check if action matches too
            if conditions_met and ep.action_abstract == rule_data.action:
                matches += 1

        # If the rule covers at least 80% of the examples, it's valid.
        return (matches / total) >= 0.8

    def _is_stable(self, cluster: List[Episode]) -> bool:
        # Check if outcomes are consistently positive
        outcomes = [e.outcome_score for e in cluster]
        return np.mean(outcomes) > 0.5 and np.std(outcomes) < 0.2
