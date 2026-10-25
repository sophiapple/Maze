"""Maze: learn which prompt trajectory to execute before generation."""
from importlib import import_module

__version__ = "0.2.1"
__all__ = ["Candidate", "RequestSlot", "allocate", "sample_paths", "NumpyEnergy", "EnergyConfig"]


def __getattr__(name):
    """Keep the result-reproduction command usable without numerical packages."""
    if name in {"Candidate", "RequestSlot", "allocate", "sample_paths"}:
        return getattr(import_module("maze.planning"), name)
    if name in {"NumpyEnergy", "EnergyConfig"}:
        return getattr(import_module("maze.scorers.numpy_energy"), name)
    raise AttributeError(name)
