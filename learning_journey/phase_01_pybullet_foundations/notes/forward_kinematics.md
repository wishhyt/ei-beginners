# Forward Kinematics

## Goal
Map joint values to end-effector pose through chain transformations.

## Concepts
- DH parameterization (standard or modified)
- per-link transform multiplication
- final pose extraction from `T_0_n`

## Implementation
1. Define joint axes and coordinate frames.
2. Build DH table (`a, alpha, d, theta`).
3. Compute per-link transform matrices.
4. Multiply transforms in order.
5. Extract position and orientation for end-effector.

## Results
Forward kinematics establishes deterministic state estimation from joint space.

## Next Steps
Use FK as the base layer for iterative IK and trajectory validation.
