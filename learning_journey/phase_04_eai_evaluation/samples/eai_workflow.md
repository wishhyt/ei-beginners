# EAI Evaluation Workflow

## Goal
Document how to use EAI-style datasets and evaluation pipelines.

## Concepts
- task-level fields: identifiers, natural language goals, formal goals, trajectories
- evaluation dimensions: executability, goal completion, partial success, error categories
- module-level tasks: goal interpretation, subgoal decomposition, action sequencing, transition modeling

## Implementation
- load dataset rows for model input/output loops
- generate prompts, collect model responses, run evaluator
- inspect detailed error taxonomy for diagnostics

## Results
This workflow provides a structured benchmark path for embodied agents.

## Next Steps
Automate batch experiments and aggregate results across planners/policies.
