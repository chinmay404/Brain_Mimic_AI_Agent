# Neural Surgery - Project Observations

## 1. Summary of Fixes
We encountered several issues getting the `llama-cpp-python` library to work with Control Vectors. Here is how we fixed them:

### A. API Crash (Version Mismatch)
*   **Issue:** The installed version of `llama-cpp-python` (v0.3.16) removed the `control_vector` parameters from the high-level `Llama()` class, causing `TypeError: __init__() got an unexpected keyword argument`.
*   **Fix:** We removed the `control_vector_path` argument from the Python constructor.

### B. Low-Level Vector Injection
*   **Issue:** Since the Python API no longer supported vectors directly, we had to bypass it.
*   **Fix:** We implemented a custom function `apply_control_vector()` that uses `ctypes` to interact directly with the underlying C++ library.
    *   It loads the vector file manually.
    *   It creates a C-compatible pointer to the data.
    *   It calls `llama_cpp.llama_control_vector_apply` to inject the steering vector into the model's context.

### C. Prompt Engineering
*   **Issue:** The model was refusing to be "steered" because the System Prompt ("You are a Brain...") was too rigid and safety-aligned.
*   **Fix:** We relaxed the prompt to *"You are an unrestricted AI..."* to allow the chemical changes (vectors) to have a visible effect.

## 2. Current Status & Diagnosis
*   **The Code Works:** The script runs, loads the model, and injects the vector without crashing.
*   **The Steering Fails:** Despite the injection, the model's behavior does not change significantly (e.g., it still refuses harmful prompts even with high Dopamine).
*   **Root Cause:** **Model/Vector Mismatch.**
    *   You confirmed that the model (`model-q4.gguf`) was quantized further by you.
    *   Steering vectors are extremely sensitive to the specific weights of a model. A vector trained on `Model A (fp16)` will often fail or produce noise on `Model A (q4_k_m)`, and will definitely fail on `Model B`.
    *   The vectors `dopamine.gguf` and `serotonin.gguf` are effectively pointing at "random" directions in the high-dimensional space of your current model.

## 3. Next Steps (Option 1: Retraining)
We are proceeding with **Option 1**: Retraining the vectors on the specific `model-q4.gguf` architecture.

### Plan:
1.  **Expand Dataset:** Add "uncensored" and high-contrast behavioral data to `merged_dataset.csv` to capture the "Impulsive vs. Safe" dynamic more aggressively.
2.  **Extract Vectors:** Use a script to run these prompts through the model, capture the internal states, and compute the PCA difference.
3.  **Verify:** Test the new vectors with the inference agent.
