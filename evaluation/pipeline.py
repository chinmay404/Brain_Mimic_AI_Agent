import sys
import os
import glob
import json
import re
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evaluation.judge import evaluate_decision, evaluate_expression, get_baseline_response

# Import Agent Components
try:
    from run_simulation import BrainMimicAdapter
    from agents.others.run_agent_pipeline import run_full_pipeline
except ImportError as e:
    print(f"Error importing agent components: {e}")
    print("Ensure you are running this script from the project root or that the python path is correct.")
    sys.exit(1)

class AgentWrapper:
    def __init__(self, agent_id="default1", reset_memory=True):
        # Initialize the brain components using the adapter
        self.adapter = BrainMimicAdapter(goal="Evaluation Mode", agent_id=agent_id, reset_memory=reset_memory)
        
    def run(self, input_text):
        # Wrap input as expected by the pipeline
        user_inputs = [("user", input_text)]
        
        # Run the pipeline directly to get the result dict
        result = run_full_pipeline(
            system_goal=self.adapter.goal,
            user_inputs=user_inputs,
            hippocampus=self.adapter.hippocampus,
            neocortex=self.adapter.neocortex,
            rule_extractor=self.adapter.rule_extractor,
            acc=self.adapter.acc,
            dlpfc=self.adapter.dlpfc
        )
        
        # Extract the chosen action or plan
        output_text = "No action taken."
        if "chosen_action" in result:
            output_text = str(result["chosen_action"])
        elif "plan" in result and result["plan"]:
            # If plan is a list of objects, convert to string
            output_text = "; ".join([str(step) for step in result["plan"]])
            
        return output_text, result.get("trace", {})

def parse_scenario(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    try:
        # Robust parsing for the MD format
        name_match = re.search(r'# Scenario: (.*)', content)
        input_match = re.search(r'## Input\n(.*?)\n##', content, re.DOTALL)
        expected_match = re.search(r'## Expected Behavior\n(.*?)\n##', content, re.DOTALL)
        
        if not (name_match and input_match and expected_match):
            print(f"Skipping {filepath}: Missing required sections")
            return None
            
        name = name_match.group(1).strip()
        input_text = input_match.group(1).strip()
        expected = expected_match.group(1).strip()
        
        return {"name": name, "input": input_text, "expected": expected}
    except Exception as e:
        print(f"Skipping {filepath}: Error parsing - {e}")
        return None

def main():
    results = []
    
    scenarios_dir = os.path.join(os.path.dirname(__file__), "scenarios")
    files = glob.glob(os.path.join(scenarios_dir, "*.md"))
    print(f"Loaded {len(files)} scenarios from {scenarios_dir}")

    for filepath in files:
        sc = parse_scenario(filepath)
        if not sc: continue

        # Create a unique ID based on the scenario name
        # e.g., "2FA Security Check" -> "eval_2fa_security_check"
        scenario_id = "eval_" + sc['name'].lower().replace(" ", "_")
        
        print(f"\n-------------------------------------------------")
        print(f"🧪 SCENARIO: {sc['name']}")
        print(f"🤖 Spawning Agent ID: {scenario_id}")
        print(f"-------------------------------------------------")

        # Initialize a FRESH agent for this specific test
        # reset_memory=True ensures it starts blank (or from base) every time we run the eval suite
        agent = AgentWrapper(agent_id=scenario_id, reset_memory=True)

        # 1. Get Agent Output
        actual_output, trace = agent.run(sc['input'])
        print(f"  -> Agent Action: {actual_output[:100]}...")

        # Extract memory context for fair comparison
        memory_context = ""
        if trace and "memory" in trace:
            mem = trace["memory"]
            if mem.get("hippocampus_bias") and mem["hippocampus_bias"] != "None":
                memory_context += f"- Past Experience Bias: {mem['hippocampus_bias']}\n"
            if mem.get("neocortex_rule") and mem["neocortex_rule"] != "None":
                memory_context += f"- Established Rule: {mem['neocortex_rule']}\n"

        # 2. Get Baseline Output
        print(f"  -> Generating Baseline Response (with shared memory)...")
        baseline_output = get_baseline_response(sc['input'], memory_context)
        print(f"  -> Baseline Action: {baseline_output[:100]}...")

        # 3. Judge Agent (Two-Stage)
        print(f"  -> Judging Agent Decision (Hard Gate)...")
        decision_json = evaluate_decision(trace, sc['expected'])
        try:
            decision_data = json.loads(decision_json)
        except json.JSONDecodeError:
            decision_data = {"pass": False, "reasoning": "Judge returned invalid JSON"}

        print(f"  -> Judging Agent Expression (Soft Score)...")
        expression_json = evaluate_expression(sc['input'], actual_output, sc['expected'])
        try:
            expression_data = json.loads(expression_json)
        except json.JSONDecodeError:
            expression_data = {"score": 0, "critique": "Judge returned invalid JSON", "suggestion": ""}

        # 4. Judge Baseline (Expression Only - Baseline has no trace)
        print(f"  -> Judging Baseline...")
        # For baseline, we treat "decision" as a check on the output text since we have no trace
        baseline_decision_json = evaluate_expression(sc['input'], baseline_output, sc['expected'])
        try:
            baseline_expr_data = json.loads(baseline_decision_json)
            # Heuristic: If score > 70, we assume it passed the decision gate
            baseline_pass = baseline_expr_data.get("score", 0) > 70
        except json.JSONDecodeError:
            baseline_expr_data = {"score": 0, "critique": "Judge returned invalid JSON", "suggestion": ""}
            baseline_pass = False

        results.append({
            "scenario": sc['name'],
            "input": sc['input'],
            "expected": sc['expected'],
            "agent": {
                "response": actual_output,
                "pass": decision_data.get('pass', False),
                "decision_reasoning": decision_data.get('reasoning', ''),
                "score": expression_data.get('score', 0),
                "critique": expression_data.get('critique', ''),
                "suggestion": expression_data.get('suggestion', ''),
                "trace": trace
            },
            "baseline": {
                "response": baseline_output,
                "pass": baseline_pass,
                "score": baseline_expr_data.get('score', 0),
                "critique": baseline_expr_data.get('critique', ''),
                "suggestion": baseline_expr_data.get('suggestion', '')
            }
        })

    # 5. Calculate Stats
    total = len(results)
    agent_passed = sum(1 for r in results if r['agent']['pass'])
    baseline_passed = sum(1 for r in results if r['baseline']['pass'])
    
    agent_avg = sum(r['agent']['score'] for r in results) / total if total > 0 else 0
    baseline_avg = sum(r['baseline']['score'] for r in results) / total if total > 0 else 0

    print(f"\n--- REPORT ---")
    print(f"Total Scenarios: {total}")
    print(f"Agent Pass Rate:    {agent_passed}/{total} ({agent_passed/total*100:.1f}%) | Avg Score: {agent_avg:.1f}")
    print(f"Baseline Pass Rate: {baseline_passed}/{total} ({baseline_passed/total*100:.1f}%) | Avg Score: {baseline_avg:.1f}")

    report_path = os.path.join(os.path.dirname(__file__), "report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed report saved to {report_path}")

if __name__ == "__main__":
    main()
