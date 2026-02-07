# Feature: Plan -> Execute -> Evaluate

## Goal
Implement an embodied intelligence feature pipeline aligned with EAI-style evaluation.

## Concepts
- Plan: generate symbolic actions from natural language task and scene objects
- Execute: map symbolic actions to robot API operations in simulation
- Evaluate: compute executability, goal satisfaction, partial success, and error taxonomy

## Implementation
- Input schema: `identifier`, `natural_language_description`, `scene_objects`
- Planner options: `rule` (deterministic), `llm` (optional fallback to `rule`)
- Executor maps actions (`GRASP`, `PLACE_INSIDE`, `MOVE_TO_POSITION`) to API calls
- Evaluator outputs JSON and Markdown reports

## Results
- reproducible baseline output with rule planner
- optional provider-backed planning while preserving fallback reliability

## Next Steps
- add subgoal decomposition and transition-model-level metrics
- support batched dataset evaluation loops
