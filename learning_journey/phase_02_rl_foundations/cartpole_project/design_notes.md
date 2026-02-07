# CartPole Q-Learning Design Notes

## Goal
Explain design choices behind the tabular CartPole project.

## Concepts
- discretization of continuous state features
- epsilon-greedy exploration strategy
- Bellman update for Q-table optimization

## Implementation
- state buckets for cart position/velocity and pole angle/angular velocity
- large training loop with epsilon decay
- post-training evaluation and reward curve visualization

## Results
The project demonstrates a full tabular RL workflow from design to evaluation.

## Next Steps
Benchmark this baseline against DQN/PPO/SAC implementations.
