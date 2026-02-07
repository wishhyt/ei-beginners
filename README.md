# EI Beginners

## Goal
Build a learning-preserving embodied intelligence repository that supports:
- staged learning history
- reusable code modules
- plan -> execute -> evaluate embodied pipeline
- RL baselines (DQN, PPO, SAC) and minimal behavior cloning

## Quickstart
1. Install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,rl,llm]'
```
2. Run pipeline:
```bash
ei plan-exec-eval --task-file learning_journey/phase_04_eai_evaluation/samples/prompt.json --planner rule --report-dir reports
```
3. Train RL baseline:
```bash
ei train-rl --algo dqn --env CartPole-v1 --steps 1000 --output outputs/rl/dqn
```
4. Train BC baseline:
```bash
ei train-bc --dataset learning_journey/phase_04_eai_evaluation/samples/ans1.json --epochs 1 --output outputs/rl/bc
```
5. Show learning journey:
```bash
ei journey show
```

## Architecture
- Core package: `src/ei_beginners`
- Learning history: `learning_journey/`
- Engineering docs: `docs/`
- Tests: `tests/`

See [Architecture](docs/ARCHITECTURE.md) and [Feature Spec](docs/FEATURE_PLAN_EXEC_EVAL.md).
