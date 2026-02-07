# PyBullet Workflow

## Goal
Describe the baseline PyBullet simulation lifecycle used in early experiments.

## Concepts
- connect to physics server (`GUI` or `DIRECT`)
- configure gravity and timestep
- load URDF/SDF/MJCF assets
- step simulation and query states
- control joints with position/velocity/torque modes

## Implementation
Typical flow:
1. Import `pybullet`, `pybullet_data`, and utilities.
2. Connect using `p.connect(...)`.
3. Set search path and gravity.
4. Load environment and robot models.
5. Run `p.stepSimulation()` loop.
6. Read object/joint states and apply controls.
7. Disconnect and release resources.

## Results
This workflow enabled repeatable sandbox experiments for robotics fundamentals.

## Next Steps
Integrate these primitives into reusable execution modules.
