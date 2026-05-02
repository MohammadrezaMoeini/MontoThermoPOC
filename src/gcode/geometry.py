"""
GeometryBuilder: reconstructs deposited bead geometry as a triangle mesh.

Each extrusion move is approximated as a cuboid (rectangular prism):
  - length  : distance from move start to end along the XY path
  - width   : nozzle_diameter  (perpendicular to path in XY)
  - height  : layer_height     (Z direction)

Meshes are accumulated incrementally so that at time step i the domain
contains all beads deposited from move 0 to move i.

The mesh is represented as:
  vertices : (N, 3) float64 array  — XYZ coordinates in mm
  faces    : (M, 3) int64  array  — vertex index triples (outward normals)
"""

import numpy as np
from typing import Optional


class GeometryBuilder:

    def __init__(self, nozzle_diameter: float = 0.4, layer_height: float = 0.2):
        self._w = nozzle_diameter
        self._h = layer_height
        self._vertices: list[np.ndarray] = []
        self._faces:    list[np.ndarray] = []
        self._vertex_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_move(self, move: dict) -> None:
        """Append the cuboid bead for *move* to the accumulated mesh."""
        verts, faces = self._bead_mesh(move)
        if verts is None:
            return
        self._faces.append(faces + self._vertex_count)
        self._vertices.append(verts)
        self._vertex_count += len(verts)

    def get_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (vertices, faces) of the full accumulated domain so far."""
        if not self._vertices:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)
        return np.vstack(self._vertices), np.vstack(self._faces)

    def reset(self) -> None:
        """Clear all accumulated geometry."""
        self._vertices.clear()
        self._faces.clear()
        self._vertex_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bead_mesh(self, move: dict) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Build the 8-vertex / 12-triangle cuboid for a single bead."""
        dx = move['x2'] - move['x1']
        dy = move['y2'] - move['y1']
        length = np.hypot(dx, dy)

        if length < 1e-9:
            return None, None

        # unit vectors: u along path, p perpendicular (rotate u by 90° CCW)
        ux, uy = dx / length, dy / length
        px, py = -uy, ux

        hw    = self._w / 2.0
        z_bot = move['z'] - self._h
        z_top = move['z']
        sx, sy = move['x1'], move['y1']
        ex, ey = move['x2'], move['y2']

        # 8 corners of the cuboid
        # indices 0-3: start face;  4-7: end face
        # within each face: 0/4 = -perp bottom, 1/5 = +perp bottom,
        #                   2/6 = +perp top,     3/7 = -perp top
        verts = np.array([
            [sx - hw*px, sy - hw*py, z_bot],  # 0
            [sx + hw*px, sy + hw*py, z_bot],  # 1
            [sx + hw*px, sy + hw*py, z_top],  # 2
            [sx - hw*px, sy - hw*py, z_top],  # 3
            [ex - hw*px, ey - hw*py, z_bot],  # 4
            [ex + hw*px, ey + hw*py, z_bot],  # 5
            [ex + hw*px, ey + hw*py, z_top],  # 6
            [ex - hw*px, ey - hw*py, z_top],  # 7
        ], dtype=float)

        # 12 triangles with outward normals (verified by cross-product)
        faces = np.array([
            [0, 1, 5], [0, 5, 4],   # bottom  (normal -Z)
            [3, 7, 6], [3, 6, 2],   # top     (normal +Z)
            [0, 3, 2], [0, 2, 1],   # start   (normal -u)
            [4, 5, 6], [4, 6, 7],   # end     (normal +u)
            [0, 4, 7], [0, 7, 3],   # left    (normal -p)
            [1, 6, 5], [1, 2, 6],   # right   (normal +p)
        ], dtype=int)

        return verts, faces
