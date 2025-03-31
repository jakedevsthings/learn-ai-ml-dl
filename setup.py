from setuptools import setup, find_packages

setup(
    name="learn_ai_ml_dl",
    version="0.1",
    packages=find_packages(include=[
        "phase0",
        "phase0.linear_algebra",
        "phase0.calculus",
        "phase0.statistics",
        "phase0.python_optimization",
        "phase1",
        "milestone_1_1*",
        "ml_core*", 
    ]),
)