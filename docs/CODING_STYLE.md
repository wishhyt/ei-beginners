# Coding Style

## Goal
Keep code consistent, testable, and readable.

## Concepts
- Type-first design with explicit interfaces
- Small modules with focused responsibilities
- Deterministic behavior for tests

## Implementation
- Python >= 3.10
- Formatting: Black
- Linting: Ruff
- Type checking: mypy
- Tests: pytest

## Results
- all public APIs are typed
- planning and evaluation logic are deterministic by default
- tests cover unit, integration, and regression checks

## Next Steps
- add stricter mypy rules for complete strict mode
- add pre-commit hooks for local quality gates
