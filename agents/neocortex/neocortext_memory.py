import json
import os
from typing import List, Dict, Any
from dataclasses import asdict
from agents.neocortex.neocortex_rule_extractor import CorticalRule

class NeocortexMemory:
    def __init__(self, storage_path: str = "neocortex_rules.json"):
        # Ensure the path is relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.storage_path = os.path.join(base_dir, storage_path)
        self.rules: List[CorticalRule] = []
        self.load()

    def add_rule(self, rule: CorticalRule):
        """
        Adds a new rule. If a similar rule exists, updates it.
        """
        for existing in self.rules:
            if existing.condition == rule.condition and existing.action == rule.action:
                existing.confidence = (existing.confidence + rule.confidence) / 2
                existing.source_cluster_id = rule.source_cluster_id 
                self.save()
                print(f"[Neocortex] Reinforced existing rule: {rule.action}")
                return
        
        self.rules.append(rule)
        self.save()
        print(f"[Neocortex] Learned NEW rule: {rule.condition} -> {rule.action}")

    def retrieve_relevant_rules(self, current_state: Dict[str, Any]) -> List[CorticalRule]:
        """
        Finds rules that apply to the current state.
        This is the 'Fast System' lookup.
        """
        matches = []
        for rule in self.rules:
            if self._check_condition(rule.condition, current_state):
                matches.append(rule)
        
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches

    def save(self):
        try:
            data = [asdict(r) for r in self.rules]
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[Neocortex] Save failed: {e}")

    def load(self):
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                # Reconstruct CorticalRule objects
                self.rules = [CorticalRule(**r) for r in data]
            print(f"[Neocortex] Loaded {len(self.rules)} rules.")
        except Exception as e:
            print(f"[Neocortex] Load failed: {e}")

    def _check_condition(self, condition: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """
        Evaluates if the rule's condition is met by the current state.
        """
        for feature, cond_str in condition.items():
            if feature not in state:
                return False 
            
            val = state[feature]

            try:
                # Handle boolean or non-string values by converting to string for comparison if needed
                # But first check if cond_str is actually a string before using startswith
                if isinstance(cond_str, str):
                    if cond_str.startswith(">="):
                        if not (float(val) >= float(cond_str[2:])): return False
                    elif cond_str.startswith("<="):
                        if not (float(val) <= float(cond_str[2:])): return False
                    elif cond_str.startswith(">"):
                        if not (float(val) > float(cond_str[1:])): return False
                    elif cond_str.startswith("<"):
                        if not (float(val) < float(cond_str[1:])): return False
                    elif cond_str.startswith("=="):
                        target = cond_str[2:]
                        if str(val) != target: return False
                    else:
                        if str(val) != str(cond_str): return False
                else:
                    # Direct comparison for non-string conditions (e.g. booleans)
                    if val != cond_str: return False
            except ValueError:
                return False
                
        return True