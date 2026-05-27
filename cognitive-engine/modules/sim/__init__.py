# Embrained - Neural Navigation Software Suite
# Copyright (C) 2026 Embrained  (GPL-3.0)

"""
modules.sim — virtual environment backends for the Embrained navigation stack.

Registry
--------
Each entry maps a backend name to a factory that accepts (**kwargs) and returns
a SimulatorBase instance.  Add new backends here; modules/simulator.py
picks the default from DEFAULT_BACKEND.

Backends
--------
  'raycast'   Pure-numpy 2D DDA raycaster.  No extra dependencies.
              Designed for CPU-only (Google Colab) environments.
              Lightest option — use as baseline before heavier engines.

  (future)    'pybullet'   Physics simulation with offscreen OpenGL rendering.
  (future)    'habitat'    Habitat-Sim photorealistic indoor scenes.
  (future)    'unity'      Unity ML-Agents with custom room assets.
"""

from modules.sim.base import SimulatorBase
from modules.sim.raycast import RaycastSimulator

DEFAULT_BACKEND = 'raycast'

REGISTRY: dict[str, type] = {
    'raycast': RaycastSimulator,
}


def make_simulator(backend: str = DEFAULT_BACKEND, **kwargs) -> SimulatorBase:
    """
    Instantiate a simulator backend by name.

    Parameters
    ----------
    backend : str   Key from REGISTRY (default: DEFAULT_BACKEND = 'raycast').
    **kwargs        Forwarded to the backend constructor.

    Returns
    -------
    SimulatorBase instance.
    """
    if backend not in REGISTRY:
        raise ValueError(
            f"Unknown backend '{backend}'.  Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[backend](**kwargs)


__all__ = [
    'SimulatorBase',
    'RaycastSimulator',
    'REGISTRY',
    'DEFAULT_BACKEND',
    'make_simulator',
]
