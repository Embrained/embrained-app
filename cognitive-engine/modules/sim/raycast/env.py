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
World geometry, agent kinematics, and preset room layouts for the
RaycastSimulator (lightest virtual environment backend).

Coordinate convention
---------------------
  x  increases rightward  (East  on the map).
  y  increases downward   (South on the map, image convention).

  Wall map:  wall_map[row, col]  =  wall_map[y, x]  (bool, True = wall).

  Agent heading (theta):
    0       → facing East  (+x)
    π/2     → facing South (+y, downward on map)
    π       → facing West  (−x)
    3π/2    → facing North (−y, upward on map)

  Positive Δtheta → clockwise on-screen (CW in top-down map view).
  LEFT  turn (Action 3): Δθ = −25°  (CCW, robot's front swings left)
  RIGHT turn (Action 4): Δθ = +25°  (CW,  robot's front swings right)

Kinematic scaling
-----------------
  One map cell ≈ 1 metre.  The FWD / BACK displacement ratio (0.35 / 0.20 =
  1.75) exactly reproduces the physical MarkovWASD spec (70 px / 40 px = 1.75).
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Kinematic constants  (forward_dist metres, delta_theta radians)
# ---------------------------------------------------------------------------

FORWARD_DIST  = 0.35                  # Action 1 – FWD
BACKWARD_DIST = 0.20                  # Action 2 – BACK
TURN_ANGLE    = math.radians(25.0)    # Actions 3 & 4

#: (forward_dist, delta_theta) keyed by action_id
KINEMATICS: dict[int, tuple[float, float]] = {
    0: ( 0.00,          0.00),           # STOP
    1: ( FORWARD_DIST,  0.00),           # FORWARD
    2: (-BACKWARD_DIST, 0.00),           # BACKWARD
    3: ( 0.00,         -TURN_ANGLE),     # LEFT  (CCW → theta decreases)
    4: ( 0.00,         +TURN_ANGLE),     # RIGHT (CW  → theta increases)
    5: ( 0.00,          0.00),           # INTENTIONAL STOP
}

#: PWM values associated with each action_id (left, right)
ACTION_PWM: dict[int, tuple[int, int]] = {
    0: (0,    0),
    1: (130,  130),
    2: (-130, -130),
    3: (-110, 110),
    4: (110,  -110),
    5: (0,    0),
}

#: Collision margin: agent half-width in map units (metres)
COLLISION_MARGIN = 0.18


# ---------------------------------------------------------------------------
# Room layout helpers
# ---------------------------------------------------------------------------

def _parse_ascii(rows: List[str]) -> np.ndarray:
    """
    Convert an ASCII room description to a boolean wall grid.

    '#'  →  True  (wall)
    Any other character  →  False  (open floor)

    Rows of unequal length are right-padded with '#' (wall) so the returned
    array is always rectangular.
    """
    max_w = max(len(r) for r in rows)
    return np.array(
        [[c == '#' for c in row.ljust(max_w, '#')] for row in rows],
        dtype=bool,
    )


# ---------------------------------------------------------------------------
# Preset room layouts
# ---------------------------------------------------------------------------

#: Registry of named layouts available to the simulator.
LAYOUTS: dict[str, np.ndarray] = {

    # Simple rectangular room — good baseline for sanity checks.
    'room_simple': _parse_ascii([
        "#############",
        "#...........#",
        "#...........#",
        "#...........#",
        "#...........#",
        "#...........#",
        "#...........#",
        "#...........#",
        "#...........#",
        "#...........#",
        "#...........#",
        "#############",
    ]),

    # L-shaped room — two connected rectangular arms.
    'room_l': _parse_ascii([
        "############",
        "#..........#",
        "#..........#",
        "#..........#",
        "#....#######",
        "#....#######",
        "#....#######",
        "#....#######",
        "#....#######",
        "#....#######",
        "#....#######",
        "############",
    ]),

    # Room with a central rectangular obstacle / pillar cluster.
    'room_corridors': _parse_ascii([
        "###################",
        "#.................#",
        "#.................#",
        "#....#########....#",
        "#....#.......#....#",
        "#....#.......#....#",
        "#....#.......#....#",
        "#....#########....#",
        "#.................#",
        "#.................#",
        "###################",
    ]),

    # Plus/cross-shaped room — tests turning at junctions.
    'room_plus': _parse_ascii([
        "#####...#####",
        "#####...#####",
        "#####...#####",
        "#...........#",
        "#...........#",
        "#...........#",
        "#####...#####",
        "#####...#####",
        "#####...#####",
    ]),

    # Large open room with two interior pillar pairs.
    'room_large': _parse_ascii([
        "####################",
        "#..................#",
        "#..................#",
        "#...##.......##....#",
        "#...##.......##....#",
        "#..................#",
        "#..................#",
        "#...##.......##....#",
        "#...##.......##....#",
        "#..................#",
        "#..................#",
        "####################",
    ]),
    
    # Empty rectangular room scaled perfectly to 5-10 forward physical actions without obstruction.
    'room_rectangular': _parse_ascii([
        "########",
        "#......#",
        "#......#",
        "#......#",
        "#......#",
        "#......#",
        "#......#",
        "########",
    ]),
}


def get_layout(name: str) -> np.ndarray:
    """Return a copy of the named layout array."""
    if name not in LAYOUTS:
        raise ValueError(
            f"Unknown layout '{name}'.  Available: {sorted(LAYOUTS)}"
        )
    return LAYOUTS[name].copy()


def list_layouts() -> List[str]:
    """Return all registered layout names."""
    return sorted(LAYOUTS.keys())


# ---------------------------------------------------------------------------
# Open-cell finder
# ---------------------------------------------------------------------------

def _find_open_cells(
    wall_map: np.ndarray,
    wall_margin: int = 1,
) -> List[Tuple[float, float]]:
    """
    Return cell-centre (x, y) positions suitable for agent spawning.

    A cell qualifies if it is open *and* the (2·margin+1)² neighbourhood
    around it contains no walls.  Falls back to any non-wall interior cell
    if no margin-safe cell exists.
    """
    h, w = wall_map.shape
    candidates: List[Tuple[float, float]] = []

    for r in range(wall_margin, h - wall_margin):
        for c in range(wall_margin, w - wall_margin):
            if wall_map[r, c]:
                continue
            patch = wall_map[
                r - wall_margin : r + wall_margin + 1,
                c - wall_margin : c + wall_margin + 1,
            ]
            if not patch.any():
                candidates.append((c + 0.5, r + 0.5))  # (x=col, y=row)

    if not candidates:
        # Fallback: any interior non-wall cell
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if not wall_map[r, c]:
                    candidates.append((c + 0.5, r + 0.5))

    return candidates


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    2D rigid-body agent navigating a boolean wall-grid world.

    The agent is represented as a bounding box of radius COLLISION_MARGIN
    around (x, y).  All four corners are checked on each translation step;
    if any corner lands in a wall cell the move is rejected and the agent
    stays in place (wall-bounce semantics matching physical reflex behaviour).

    Parameters
    ----------
    wall_map : np.ndarray  bool (H, W)
    spawn    : (x, y, theta) | None  — if None a random valid pose is chosen.
    """

    def __init__(
        self,
        wall_map: np.ndarray,
        spawn: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        self.wall_map = wall_map
        self._open_cells: List[Tuple[float, float]] = _find_open_cells(wall_map)
        if not self._open_cells:
            raise RuntimeError(
                "No navigable cells found in wall_map.  "
                "Check that the layout has open interior area."
            )
        # Initialise with defaults before calling reset()
        self.x: float = self._open_cells[0][0]
        self.y: float = self._open_cells[0][1]
        self.theta: float = 0.0
        self.reset(spawn)

    # ------------------------------------------------------------------ #
    #  Pose                                                                #
    # ------------------------------------------------------------------ #

    def reset(self, spawn: Optional[Tuple[float, float, float]] = None) -> None:
        """Reset agent pose.  Random if spawn is None."""
        if spawn is not None:
            self.x     = float(spawn[0])
            self.y     = float(spawn[1])
            self.theta = float(spawn[2]) % (2 * math.pi)
        else:
            cx, cy = random.choice(self._open_cells)
            self.x     = cx + random.uniform(-0.05, 0.05)
            self.y     = cy + random.uniform(-0.05, 0.05)
            self.theta = random.uniform(0.0, 2 * math.pi)

    @property
    def pose(self) -> Tuple[float, float, float]:
        """Current (x, y, theta)."""
        return (self.x, self.y, self.theta)

    # ------------------------------------------------------------------ #
    #  Collision                                                           #
    # ------------------------------------------------------------------ #

    def _is_free(self, x: float, y: float) -> bool:
        """True iff all four bounding-box corners avoid wall cells."""
        h, w = self.wall_map.shape
        for dx in (-COLLISION_MARGIN, COLLISION_MARGIN):
            for dy in (-COLLISION_MARGIN, COLLISION_MARGIN):
                gx = int(x + dx)
                gy = int(y + dy)
                if not (0 <= gx < w and 0 <= gy < h):
                    return False
                if self.wall_map[gy, gx]:
                    return False
        return True

    # ------------------------------------------------------------------ #
    #  Action execution                                                    #
    # ------------------------------------------------------------------ #

    def step(self, action_id: int) -> bool:
        """
        Apply *action_id* to update the agent pose.

        Returns True if the agent's position or heading changed,
        False if the action was a no-op or blocked by a wall.
        """
        kin = KINEMATICS.get(action_id)
        if kin is None:
            return False

        forward_dist, delta_theta = kin

        # --- Rotation (always succeeds) ---
        if delta_theta != 0.0:
            self.theta = (self.theta + delta_theta) % (2 * math.pi)
            return True

        # --- Translation ---
        if forward_dist == 0.0:
            return False  # STOP / INTENTIONAL STOP

        new_x = self.x + forward_dist * math.cos(self.theta)
        new_y = self.y + forward_dist * math.sin(self.theta)

        if self._is_free(new_x, new_y):
            self.x, self.y = new_x, new_y
            return True
        else:
            # Simulate real-world continuous grinding/wheel-slip against baseboards
            # by micro-stepping the translation vector until exact collision threshold.
            moved = False
            # 10 granular increments perfectly covers the 0.35 default forward bound
            step_frac = forward_dist / 10.0
            for _ in range(10):
                nx = self.x + step_frac * math.cos(self.theta)
                ny = self.y + step_frac * math.sin(self.theta)
                if self._is_free(nx, ny):
                    self.x, self.y = nx, ny
                    moved = True
                else:
                    # Wall reached at maximum physical threshold
                    break
            return moved
