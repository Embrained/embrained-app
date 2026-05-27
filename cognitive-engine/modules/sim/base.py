# Embrained - Neural Navigation Software Suite
# Copyright (C) 2026 Embrained
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
SimulatorBase — abstract interface for all Embrained virtual environment backends.

Adding a new backend
--------------------
1. Create a subpackage under modules/sim/<backend_name>/
2. Subclass SimulatorBase and implement every abstract member.
3. Add an entry to modules/sim/REGISTRY (see __init__.py).
4. Update modules/simulator.py DEFAULT_BACKEND if needed.

Available backends
------------------
  raycast  – lightweight 2D raycasting FPV renderer (pure numpy, no extra deps).
             Designed for CPU-only environments such as Google Colab.
             Use as a baseline before graduating to stronger physics engines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class SimulatorBase(ABC):
    """
    Abstract interface for all Embrained virtual environment backends.

    Output conventions
    ------------------
    get_latest_frame()
        (64, 64, 3) uint8, channel order **BGR** — drop-in for cv2.imwrite()
        and compatible with MarkovLogger.

    telemetry['dist']
        String (castable to float) representing the forward obstacle distance
        in sensor units.  **Higher values mean closer obstacles**, matching the
        physical IR sensor convention used in exploration.py.
        Values above AUTONOMY_THRESHOLD (1500) trigger the safety-reflex sequence.

    robotId
        Follows the pybullet convention: ≥ 0 on success, -1 on failure.
        Backends that don't use a rigid-body ID should simply return 0.

    Action IDs  (matching config.ACTION_PWM_MAP)
    --------------------------------------------
        0  STOP              PWM (0, 0)
        1  FORWARD           PWM (130, 130)   → forward translation
        2  BACKWARD          PWM (-130, -130) → backward translation
        3  LEFT TURN         PWM (-110, 110)  → −25° yaw (CCW in top-down view)
        4  RIGHT TURN        PWM (110, -110)  → +25° yaw (CW  in top-down view)
        5  INTENTIONAL STOP  PWM (0, 0)
    """

    # ------------------------------------------------------------------ #
    #  Properties (must be implemented as @property in subclasses)         #
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def robotId(self) -> int:
        """≥ 0 if the simulation loaded successfully, -1 on failure."""

    @property
    @abstractmethod
    def telemetry(self) -> dict:
        """
        Sensor reading dict.  Required key:
          'dist'  str  Forward obstacle distance in sensor units.
                       Higher = closer (IR convention).
                       Trigger level: AUTONOMY_THRESHOLD = 1500.
        """

    # ------------------------------------------------------------------ #
    #  Methods                                                             #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def send_command(self, action_id: int) -> None:
        """Apply one discrete MarkovWASD action and advance the simulation."""

    @abstractmethod
    def get_latest_frame(self) -> np.ndarray:
        """Current first-person view.  Shape (64, 64, 3), dtype uint8, BGR."""

    @abstractmethod
    def reset(self, spawn: Optional[Tuple[float, float, float]] = None) -> None:
        """
        Reset the agent to a spawn pose.

        Parameters
        ----------
        spawn : (x, y, theta) in world units / radians, or None for a random
                valid pose sampled from the current layout's open cells.
        """

    @abstractmethod
    def close(self) -> None:
        """Release all held resources."""
