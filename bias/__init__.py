"""bias -- a reproducible benchmark for LLM-driven analog circuit sizing.

The benchmark asks one question honestly: given a topology, a spec and a fixed
simulation budget, can a language model size an analog circuit better than the
classical optimizers a designer could already have run for free?

Every optimizer reaches ngspice through the same Evaluator, so the simulation
budget means the same thing for all of them.
"""

from . import optimizers, pdk, sim, spec, specs, topology
from .evaluate import Evaluation, Evaluator
from .spec import Design, Metric, Spec

__version__ = "0.1.0"

__all__ = [
    "Design",
    "Evaluation",
    "Evaluator",
    "Metric",
    "Spec",
    "optimizers",
    "pdk",
    "sim",
    "spec",
    "specs",
    "topology",
]
