# Robotics Fundamentals: Coordinate Transforms

## Goal
Summarize key representations for pose and orientation in robot systems.

## Concepts
- rotation matrix `R in SO(3)`
- homogeneous transform `T in SE(3)`
- Euler angles (human-readable, singularity-prone)
- quaternions (numerically stable orientation representation)

## Implementation
- Use rotation matrices for composition and inversion.
- Use homogeneous transforms for combined rotation + translation.
- Convert between Euler/quaternion/rotation matrix based on system needs.

## Results
A consistent frame convention reduces integration bugs in planning and control.

## Next Steps
Apply consistent frame conventions in all execution/evaluation components.
