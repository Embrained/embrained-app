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
RaycastSimulator — lightest Embrained virtual environment backend.

Drop-in for the physical robot stack via modules/simulator.py.
Uses a 2D DDA raycaster (pure numpy) to produce 64×64 BGR first-person frames
with no external rendering dependencies.  Runs headless on CPU in Google Colab.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from modules.sim.base import SimulatorBase
from modules.sim.raycast.env import (
    Agent,
    ACTION_PWM,
    get_layout,
    list_layouts,
)
from modules.sim.raycast.renderer import (
    render_frame,
    forward_sensor_dist,
    dist_to_sensor,
)
from config import RECORD_W, RECORD_H

logger = logging.getLogger(__name__)


class RaycastSimulator(SimulatorBase):
    """
    Lightweight 2D raycasting virtual environment.

    Parameters
    ----------
    headless : bool
        Ignored (the backend is always headless — no display required).
        Kept for API compatibility with heavier backends.
    layout : str
        Name of the room layout to load.  See env.list_layouts() for options.
        Default: 'room_simple'.
    spawn : (x, y, theta) | None
        Initial agent pose.  None → random valid pose.
    """

    def __init__(
        self,
        headless: bool = True,
        layout:   str  = 'room_simple',
        spawn:    Optional[Tuple[float, float, float]] = None,
    ) -> None:
        self._robot_id = -1
        self._layout_name = layout

        try:
            wall_map = get_layout(layout)
            self._agent = Agent(wall_map, spawn)
            self._wall_map = wall_map
            self._robot_id = 0       # success sentinel (pybullet convention)
            logger.info(
                f"RaycastSimulator loaded layout '{layout}' "
                f"({wall_map.shape[1]}×{wall_map.shape[0]} cells). "
                f"Initial pose: {self._agent.pose}"
            )
        except Exception as exc:
            logger.error(f"RaycastSimulator init failed: {exc}")
            self._agent    = None   # type: ignore[assignment]
            self._wall_map = None   # type: ignore[assignment]

        # Cache the most recently rendered frame so get_latest_frame() can be
        # called multiple times without re-rendering.
        self._cached_frame: Optional[np.ndarray] = None
        self._frame_dirty: bool = True

        # Cache telemetry so it is computed only once per step.
        self._cached_telemetry: dict = {'dist': '0'}

    # ------------------------------------------------------------------ #
    #  SimulatorBase properties                                            #
    # ------------------------------------------------------------------ #

    @property
    def robotId(self) -> int:
        return self._robot_id

    @property
    def telemetry(self) -> dict:
        if self._robot_id < 0:
            return {'dist': '0'}
        # Lazily compute if frame is dirty (i.e. after a step or reset)
        if self._frame_dirty:
            self._update_cache()
        return self._cached_telemetry

    # ------------------------------------------------------------------ #
    #  SimulatorBase methods                                               #
    # ------------------------------------------------------------------ #

    def send_command(self, action_id: int) -> None:
        """
        Apply *action_id* (0–5) and advance the simulation one step.

        The internal frame and telemetry caches are invalidated so that the
        next call to get_latest_frame() / telemetry re-renders the new view.
        """
        if self._robot_id < 0:
            return
        self._agent.step(action_id)
        self._frame_dirty = True

    def get_latest_frame(self) -> np.ndarray:
        """
        Return the current first-person view.

        Shape  : (RECORD_H, RECORD_W, 3)
        dtype  : uint8
        Channels: BGR  (compatible with cv2.imwrite and MarkovLogger)
        """
        if self._robot_id < 0:
            # Return a solid black frame on load failure
            return np.zeros((RECORD_H, RECORD_W, 3), dtype=np.uint8)

        if self._frame_dirty:
            self._update_cache()

        return self._cached_frame   # type: ignore[return-value]

    def reset(
        self,
        spawn: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Reset the agent to a new spawn pose and invalidate the frame cache."""
        if self._robot_id < 0:
            return
        self._agent.reset(spawn)
        self._frame_dirty = True

    def close(self) -> None:
        """No persistent resources to release for this backend."""
        logger.debug("RaycastSimulator closed.")

    # ------------------------------------------------------------------ #
    #  Extra helpers (not part of SimulatorBase)                           #
    # ------------------------------------------------------------------ #

    @property
    def pose(self) -> Tuple[float, float, float]:
        """Current agent pose (x, y, theta)."""
        return self._agent.pose

    @property
    def layout_name(self) -> str:
        return self._layout_name

    @property
    def wall_map(self) -> np.ndarray:
        return self._wall_map

    def pwm_for_action(self, action_id: int) -> Tuple[int, int]:
        """Return the (left, right) PWM pair associated with *action_id*."""
        return ACTION_PWM.get(action_id, (0, 0))

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _update_cache(self) -> None:
        """Render the current view and update telemetry from the agent pose."""
        x, y, theta = self._agent.pose

        self._cached_frame = render_frame(self._wall_map, x, y, theta, w=RECORD_W, h=RECORD_H)

        dist_m = forward_sensor_dist(self._wall_map, x, y, theta)
        self._cached_telemetry = {'dist': str(dist_to_sensor(dist_m))}

        self._frame_dirty = False
