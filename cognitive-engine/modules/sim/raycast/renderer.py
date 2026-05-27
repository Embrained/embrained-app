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
Pure-numpy DDA raycasting renderer for the RaycastSimulator.

Output
------
  render_frame()        → (H, W, 3) BGR uint8 first-person view
  forward_sensor_dist() → float distance to nearest forward wall (map units)

Visual design (relevant to VAE feature learning)
------------------------------------------------
  • Gradient ceiling (dark → lighter toward horizon) and floor (light → dark
    toward bottom) give a clear horizon cue independent of heading.
  • Wall brightness falls off with distance (1 / (1 + d)) creating an
    exponential-size-growth cue as the agent approaches a wall.
  • X-facing and Y-facing walls have different base brightness (side shading),
    creating heading-dependent contrast changes on turns.
  • Eight vertical stripes per wall cell (computed from the fractional hit
    position along the cell) create stable local texture that persists across
    viewpoints and helps the VAE anchor position inside a cell.
  • Three hue families (warm/neutral/cool), assigned by a cell-position hash,
    break translational symmetry so that the latent space can distinguish
    locations even in rooms with uniform geometry.

Coordinate convention  (see env.py)
-------------------------------------
  x right, y down.  theta=0 → East, π/2 → South.
  Camera plane perpendicular to heading, pointing to the right of the agent
  so that left-column rays sweep leftward (toward smaller theta).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Render configuration
# ---------------------------------------------------------------------------

IMG_W: int = 64
IMG_H: int = 64
FOV:   float = math.pi / 3          # 60° horizontal field of view
_TAN_HALF_FOV: float = math.tan(FOV / 2)

# Wall hue families  (BGR tuples, will be scaled by brightness factor)
# Eight perceptually distinct families break translational symmetry more
# aggressively than three, making it easier for the VAE to tell locations apart.
_WALL_HUES: Tuple[Tuple[int, int, int], ...] = (
    (120, 160, 210),   # 0 – warm cream / beige
    (180, 140, 100),   # 1 – cool slate-blue
    (100, 180, 130),   # 2 – muted sage-green
    (200, 110, 140),   # 3 – dusty rose / mauve
    ( 90, 190, 200),   # 4 – amber / ochre
    (160, 100, 190),   # 5 – soft violet
    (100, 200, 160),   # 6 – teal-green
    (160, 180,  80),   # 7 – steel-blue
)

# Stripe count per wall cell (must be power of 2 for cheap modulo)
_STRIPES: int = 8

# Sensor scaling: sensor_value = SENSOR_SCALE / forward_dist
# At forward_dist = SENSOR_SCALE / AUTONOMY_THRESHOLD the reflex triggers.
# AUTONOMY_THRESHOLD = 1500 (exploration.py) → trigger at ~0.33 m from wall.
SENSOR_SCALE: float = 500.0


# ---------------------------------------------------------------------------
# Pre-built background arrays  (avoids per-frame allocation)
# ---------------------------------------------------------------------------

def _make_background(h: int, w: int) -> np.ndarray:
    """
    Gradient ceiling (dark → lighter at horizon) and floor (lighter → dark).
    Returns a (H, W, 3) BGR uint8 array.
    """
    bg = np.empty((h, w, 3), dtype=np.uint8)
    half = h // 2

    # Ceiling: luminance 75 → 105 (top → horizon), neutral warm-off-white to match real environments
    ceil_lum = np.linspace(75, 105, half, dtype=np.float32)
    bg[:half, :, 0] = ceil_lum[:, np.newaxis]          # B
    bg[:half, :, 1] = (ceil_lum * 1.05)[:, np.newaxis] # G
    bg[:half, :, 2] = (ceil_lum * 1.10)[:, np.newaxis] # R

    # Floor: neutral carpet/tile, luminance 110 → 70 (horizon → bottom)
    floor_lum = np.linspace(110, 70, h - half, dtype=np.float32)
    bg[half:, :, 0] = (floor_lum * 0.85).astype(np.uint8)[:, np.newaxis]  # B
    bg[half:, :, 1] = (floor_lum * 0.95).astype(np.uint8)[:, np.newaxis]  # G
    bg[half:, :, 2] = (floor_lum * 1.00).astype(np.uint8)[:, np.newaxis]  # R

    return bg


_BG_CACHE: dict[Tuple[int, int], np.ndarray] = {}


def _get_background(h: int, w: int) -> np.ndarray:
    key = (h, w)
    if key not in _BG_CACHE:
        _BG_CACHE[key] = _make_background(h, w)
    return _BG_CACHE[key]


# ---------------------------------------------------------------------------
# DDA helpers
# ---------------------------------------------------------------------------

def _dda(
    wall_map: np.ndarray,
    pos_x: float,
    pos_y: float,
    ray_dir_x: float,
    ray_dir_y: float,
    max_steps: int = 128,
) -> Tuple[bool, float, int, int, int, float]:
    """
    Digital Differential Analysis wall-hit finder.

    Returns
    -------
    hit       : bool   – True if a wall was found within max_steps
    perp_dist : float  – perpendicular distance to the hit wall face
    map_x     : int    – grid column of the hit cell
    map_y     : int    – grid row of the hit cell
    side      : int    – 0 = X-facing wall hit, 1 = Y-facing wall hit
    wall_x    : float  – fractional position (0–1) along the hit wall face
    """
    h, w = wall_map.shape
    map_x = int(pos_x)
    map_y = int(pos_y)

    # Delta distances to next grid boundary along each axis
    delta_x = abs(1.0 / ray_dir_x) if ray_dir_x != 0.0 else 1e30
    delta_y = abs(1.0 / ray_dir_y) if ray_dir_y != 0.0 else 1e30

    # Initial side distances and step directions
    if ray_dir_x < 0:
        step_x  = -1
        side_x  = (pos_x - map_x) * delta_x
    else:
        step_x  = 1
        side_x  = (map_x + 1.0 - pos_x) * delta_x

    if ray_dir_y < 0:
        step_y  = -1
        side_y  = (pos_y - map_y) * delta_y
    else:
        step_y  = 1
        side_y  = (map_y + 1.0 - pos_y) * delta_y

    side = 0

    for _ in range(max_steps):
        if side_x < side_y:
            side_x += delta_x
            map_x  += step_x
            side    = 0
        else:
            side_y += delta_y
            map_y  += step_y
            side    = 1

        if not (0 <= map_x < w and 0 <= map_y < h):
            break

        if wall_map[map_y, map_x]:
            # Perpendicular distance (fisheye-corrected)
            if side == 0:
                perp_dist = side_x - delta_x
            else:
                perp_dist = side_y - delta_y

            perp_dist = max(perp_dist, 1e-4)

            # Fractional hit position along the wall face (0–1 within the cell)
            if side == 0:
                wall_x = pos_y + perp_dist * ray_dir_y
            else:
                wall_x = pos_x + perp_dist * ray_dir_x
            wall_x -= math.floor(wall_x)

            return True, perp_dist, map_x, map_y, side, wall_x

    return False, 0.0, 0, 0, 0, 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_frame(
    wall_map: np.ndarray,
    pos_x:    float,
    pos_y:    float,
    theta:    float,
    w:        int = IMG_W,
    h:        int = IMG_H,
) -> np.ndarray:
    """
    Render a first-person view from pose (pos_x, pos_y, theta).

    Parameters
    ----------
    wall_map : np.ndarray bool (H_map, W_map) – True = wall
    pos_x    : float  agent x position in map units
    pos_y    : float  agent y position in map units
    theta    : float  agent heading in radians (0 = East, π/2 = South)
    w, h     : int    output image width / height (default 64 × 64)

    Returns
    -------
    np.ndarray  (h, w, 3) BGR uint8
    """
    img = _get_background(h, w).copy()

    # Heading direction vector
    dir_x = math.cos(theta)
    dir_y = math.sin(theta)

    # Camera plane: perpendicular to dir, pointing LEFT of heading so that
    # cameraX = -1 sweeps leftward (CCW) and cameraX = +1 sweeps rightward.
    #   Rotate dir 90° CCW in image coords: (-dir_y, dir_x) * tan(FOV/2)
    plane_x = -dir_y * _TAN_HALF_FOV
    plane_y =  dir_x * _TAN_HALF_FOV

    half_h = h // 2

    # ── Floor casting (per-pixel, bottom half of image) ──────────────────────
    # Each floor pixel at screen row y_screen maps to a world (fx, fy) position
    # via the standard raycaster floor-casting equations.  We colour each floor
    # tile with a position-stable hue so the VAE can anchor spatial location.
    ray_dir_x0 = dir_x - plane_x   # left edge ray
    ray_dir_x1 = dir_x + plane_x   # right edge ray
    ray_dir_y0 = dir_y - plane_y
    ray_dir_y1 = dir_y + plane_y

    for y_screen in range(half_h, h):
        # Rows below the horizon: p is the distance from horizon to this row.
        p = y_screen - half_h + 1e-4
        row_dist = (h * 0.5) / p       # distance to floor at this row

        # Step vector per pixel column
        floor_step_x = row_dist * (ray_dir_x1 - ray_dir_x0) / w
        floor_step_y = row_dist * (ray_dir_y1 - ray_dir_y0) / w

        # World position of leftmost floor pixel in this row
        fx = pos_x + row_dist * ray_dir_x0
        fy = pos_y + row_dist * ray_dir_y0

        # Darken with distance (fog)
        dist_fog = max(0.0, min(1.0, 1.0 - row_dist / 12.0))
        fog_scale = 0.55 + 0.45 * dist_fog   # 0.55 (far) → 1.00 (near)
        
        for x_screen in range(w):
            tx = int(fx) & 0xFFFF
            ty = int(fy) & 0xFFFF
            # 2-bit checkerboard lightens alternate tiles for grid cues
            checker = ((int(fx) + int(fy)) & 1) * 16
            
            # Per-tile hue from position hash
            tile_hue = (tx * 11 + ty * 17) % len(_WALL_HUES)
            base_b, base_g, base_r = _WALL_HUES[tile_hue]
            
            # Mute the floor tiles to match the grey/beige real-world dataset tones
            tb = base_b * 0.4 + 130 * 0.6
            tg = base_g * 0.4 + 130 * 0.6
            tr = base_r * 0.4 + 130 * 0.6

            # Apply fog, and checker
            br = fog_scale * (0.50 + checker / 255.0 * 0.5)
            img[y_screen, x_screen, 0] = int(min(255, tb * 0.75 * br))
            img[y_screen, x_screen, 1] = int(min(255, tg * 0.85 * br))
            img[y_screen, x_screen, 2] = int(min(255, tr * 0.95 * br))
            
            fx += floor_step_x
            fy += floor_step_y

    for col in range(w):
        # Map column to camera space: −1 (left edge) … +1 (right edge)
        camera_x = 2.0 * col / w - 1.0

        ray_dir_x = dir_x + plane_x * camera_x
        ray_dir_y = dir_y + plane_y * camera_x

        hit, perp_dist, map_x, map_y, side, wall_x = _dda(
            wall_map, pos_x, pos_y, ray_dir_x, ray_dir_y
        )
        if not hit:
            continue

        # --- Wall column height ---
        wall_h    = int(h / perp_dist)
        draw_top  = max(0,  half_h - wall_h // 2)
        draw_bot  = min(h,  half_h + wall_h // 2)

        # --- Brightness ---
        # Distance shading: bright up close, dims with distance
        # Scaled up significantly to match real-world lighting
        brightness = 255.0 / (1.0 + perp_dist * 0.2)
        # Side shading: Y-facing walls (side=1) are 75% of X-facing walls
        if side == 1:
            brightness *= 0.75
        brightness = max(40.0, min(255.0, brightness))

        # --- Stripe texture ---
        # Divide the wall face into _STRIPES equal bands; darken alternating ones.
        stripe = int(wall_x * _STRIPES) % 2
        if stripe == 1:
            brightness *= 0.72

        # --- Hue family (stable per wall cell, breaks translational symmetry) ---
        hue_idx = (map_x * 7 + map_y * 13) % len(_WALL_HUES)
        base_b, base_g, base_r = _WALL_HUES[hue_idx]

        scale = brightness / 180.0   # normalise against a mid-tone reference
        # Desaturate slightly by blending toward a neutral grey (110)
        col_b = int(min(255, (base_b * 0.6 + 110 * 0.4) * scale))
        col_g = int(min(255, (base_g * 0.6 + 110 * 0.4) * scale))
        col_r = int(min(255, (base_r * 0.6 + 110 * 0.4) * scale))

        img[draw_top:draw_bot, col, 0] = col_b
        img[draw_top:draw_bot, col, 1] = col_g
        img[draw_top:draw_bot, col, 2] = col_r

    return img


def forward_sensor_dist(
    wall_map: np.ndarray,
    pos_x:    float,
    pos_y:    float,
    theta:    float,
) -> float:
    """
    Cast a single ray straight ahead and return the perpendicular distance
    to the nearest wall in map units.

    Used to populate telemetry['dist'] via:
        sensor_value = int(SENSOR_SCALE / max(dist, 1e-4))
    so that closer walls give higher values (IR sensor convention).
    """
    ray_dir_x = math.cos(theta)
    ray_dir_y = math.sin(theta)
    hit, perp_dist, *_ = _dda(wall_map, pos_x, pos_y, ray_dir_x, ray_dir_y)
    return perp_dist if hit else 20.0   # 20 m default if no wall in sight


def dist_to_sensor(forward_dist: float) -> int:
    """
    Convert a forward wall distance (map units) to an IR-style sensor value.

    sensor_value ∝ 1 / distance:
      0.33 m  → ~1515  (just above AUTONOMY_THRESHOLD = 1500 → reflex triggers)
      0.50 m  → ~1000  (safe)
      1.00 m  →  ~500  (very safe)
    """
    return int(SENSOR_SCALE / max(forward_dist, 1e-4))
