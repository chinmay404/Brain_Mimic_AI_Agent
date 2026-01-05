# Evaluation Harness

This directory contains the evaluation system for the Brain Mimic AI Agent. It uses an "LLM-as-a-Judge" approach to evaluate the agent's performance against defined scenarios.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install groq
    ```

2.  **Set API Key:**
    You need a Groq API key to run the judge.
    ```bash
    export GROQ_API_KEY="your_groq_api_key_here"
    ```

## Running Evaluations

Run the pipeline script from the project root:

```bash
python evaluation/pipeline.py
```

## Adding Scenarios

Create a new Markdown file in `evaluation/scenarios/` (e.g., `my_test.md`). Use the following format:

```markdown
# Scenario: [Scenario Name]

## Input
[The input prompt or context for the agent]

## Expected Behavior
[Description of what the agent SHOULD do]

## Rubric
[Optional notes]
```

## How it Works

1.  **Pipeline (`pipeline.py`):** Loads scenarios, initializes the agent, and runs each scenario.
2.  **Agent Wrapper:** Adapts the `BrainMimicAdapter` to accept simple text input and return the chosen action.
3.  **Judge (`judge.py`):** Sends the Agent's output and the Expected Behavior to Groq (Llama 3 70B).
4.  **Comparison:** The Judge determines if the Agent's action semantically matches the Expected Behavior.
5.  **Report:** Results are saved to `evaluation/report.json`.
