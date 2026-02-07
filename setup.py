from setuptools import find_packages, setup

setup(
    name="ei-beginners",
    version="0.1.0",
    description="Learning-preserving embodied intelligence project",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "gymnasium>=0.29",
        "pybullet>=3.2",
    ],
    extras_require={
        "dev": ["black>=24.8.0", "ruff>=0.6.5", "pytest>=8.3.0", "mypy>=1.11.0"],
        "rl": ["stable-baselines3>=2.3.0"],
        "llm": ["google-generativeai>=0.8.0"],
    },
    entry_points={"console_scripts": ["ei=ei_beginners.cli:main"]},
)
