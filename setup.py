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
        "phase1.milestone_1_1_trivial_nn",
        "phase1.milestone_1_1_trivial_nn.models",
        "phase1.milestone_1_1_trivial_nn.train",
        "phase1.milestone_1_1_trivial_nn.utils",
        "phase1.milestone_1_1_trivial_nn.notebooks",
        "phase1.milestone_1_1_trivial_nn.tests",
    ]),
)