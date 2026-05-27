# Embrained - Neural Navigation Software Suite
# Copyright (C) 2026 Embrained  (GPL-3.0)

"""
modules/simulator.py — canonical import point for the active simulator backend.

Usage (existing code, diagnose_sim.py, sim_collect.py)
------------------------------------------------------
    from modules.simulator import Simulator
    sim = Simulator(headless=True)

Changing backends
-----------------
Set DEFAULT_BACKEND in modules/sim/__init__.py, or import a specific class:

    from modules.sim.raycast import RaycastSimulator as Simulator
"""

from modules.sim import DEFAULT_BACKEND, REGISTRY
from modules.sim.base import SimulatorBase

# Re-export the default backend class as 'Simulator' so that all existing
# call-sites (diagnose_sim.py, engine.py, etc.) work without modification.
Simulator: type[SimulatorBase] = REGISTRY[DEFAULT_BACKEND]

__all__ = ['Simulator']
