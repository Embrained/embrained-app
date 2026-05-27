# Embrained - Neural Navigation Software Suite
# Copyright (C) 2026 Embrained  (GPL-3.0)

"""
Raycast backend — lightest Embrained virtual environment.

Usage
-----
    from modules.sim.raycast import RaycastSimulator
    sim = RaycastSimulator(layout='room_simple')
"""

from modules.sim.raycast.simulator import RaycastSimulator

__all__ = ['RaycastSimulator']
