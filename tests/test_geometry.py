"""
Unit tests for GeometryBuilder.
"""

import pytest
import numpy as np
from src.gcode.geometry import GeometryBuilder

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def make_move(x1=0.0, y1=0.0, x2=10.0, y2=0.0, z=0.2,
              f=1800.0, e=1.0, layer=0) -> dict:
    return dict(x1=x1, y1=y1, x2=x2, y2=y2, z=z, f=f, e=e, layer=layer)


def face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return (M, 3) array of un-normalised face normals."""
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    return np.cross(b - a, c - a)


# -----------------------------------------------------------------------
# Empty / reset state
# -----------------------------------------------------------------------

class TestEmptyState:

    def test_get_mesh_empty_vertices(self):
        verts, _ = GeometryBuilder().get_mesh()
        assert verts.shape == (0, 3)

    def test_get_mesh_empty_faces(self):
        _, faces = GeometryBuilder().get_mesh()
        assert faces.shape == (0, 3)

    def test_reset_clears_mesh(self):
        gb = GeometryBuilder()
        gb.add_move(make_move())
        gb.reset()
        verts, faces = gb.get_mesh()
        assert len(verts) == 0 and len(faces) == 0


# -----------------------------------------------------------------------
# Single bead counts
# -----------------------------------------------------------------------

class TestSingleBeadCounts:

    def test_vertex_count(self):
        gb = GeometryBuilder()
        gb.add_move(make_move())
        verts, _ = gb.get_mesh()
        assert verts.shape == (8, 3)

    def test_face_count(self):
        gb = GeometryBuilder()
        gb.add_move(make_move())
        _, faces = gb.get_mesh()
        assert faces.shape == (12, 3)

    def test_face_indices_within_range(self):
        gb = GeometryBuilder()
        gb.add_move(make_move())
        verts, faces = gb.get_mesh()
        assert faces.min() >= 0
        assert faces.max() < len(verts)


# -----------------------------------------------------------------------
# Accumulation across multiple beads
# -----------------------------------------------------------------------

class TestAccumulation:

    def test_two_beads_vertex_count(self):
        gb = GeometryBuilder()
        gb.add_move(make_move(x1=0, y1=0, x2=10, y2=0))
        gb.add_move(make_move(x1=10, y1=0, x2=10, y2=10))
        verts, _ = gb.get_mesh()
        assert verts.shape == (16, 3)

    def test_two_beads_face_count(self):
        gb = GeometryBuilder()
        gb.add_move(make_move(x1=0, y1=0, x2=10, y2=0))
        gb.add_move(make_move(x1=10, y1=0, x2=10, y2=10))
        _, faces = gb.get_mesh()
        assert faces.shape == (24, 3)

    def test_face_indices_valid_two_beads(self):
        gb = GeometryBuilder()
        gb.add_move(make_move(x1=0, y1=0, x2=10, y2=0))
        gb.add_move(make_move(x1=10, y1=0, x2=10, y2=10))
        verts, faces = gb.get_mesh()
        assert faces.min() >= 0
        assert faces.max() < len(verts)

    def test_n_beads_scale_linearly(self):
        gb = GeometryBuilder()
        for i in range(5):
            gb.add_move(make_move(x1=i*10.0, y1=0, x2=(i+1)*10.0, y2=0))
        verts, faces = gb.get_mesh()
        assert verts.shape == (40, 3)
        assert faces.shape == (60, 3)


# -----------------------------------------------------------------------
# Bead geometry — axis-aligned move (x-axis), easy to reason about
# -----------------------------------------------------------------------

class TestBeadGeometryAlongX:
    """Move from (0,0) to (L,0) along X.  u=(1,0), p=(0,1)."""

    L  = 10.0
    Z  = 0.35
    ND = 0.4
    LH = 0.2

    def setup_method(self):
        self.gb = GeometryBuilder(nozzle_diameter=self.ND, layer_height=self.LH)
        self.gb.add_move(make_move(x1=0, y1=0, x2=self.L, y2=0,
                                   z=self.Z))
        self.verts, self.faces = self.gb.get_mesh()

    def test_x_coords_are_start_or_end(self):
        xs = np.unique(np.round(self.verts[:, 0], 9))
        assert set(xs) == {0.0, self.L}

    def test_y_coords_are_plus_minus_half_width(self):
        hw = self.ND / 2
        ys = np.unique(np.round(self.verts[:, 1], 9))
        assert set(ys) == {-hw, hw}

    def test_z_coords_are_bot_and_top(self):
        z_bot = self.Z - self.LH
        zs = np.unique(np.round(self.verts[:, 2], 9))
        assert len(zs) == 2
        assert np.allclose(sorted(zs), [z_bot, self.Z])

    def test_bottom_faces_have_negative_z_normal(self):
        # first 4 triangles are bottom (indices 0-1) and top (2-3)
        normals = face_normals(self.verts, self.faces)
        bottom_normals = normals[0:2]
        assert np.all(bottom_normals[:, 2] < 0)

    def test_top_faces_have_positive_z_normal(self):
        normals = face_normals(self.verts, self.faces)
        top_normals = normals[2:4]
        assert np.all(top_normals[:, 2] > 0)

    def test_start_faces_have_negative_x_normal(self):
        # start face is faces 4-5; normal should point in -X (= -u for X-axis move)
        normals = face_normals(self.verts, self.faces)
        start_normals = normals[4:6]
        assert np.all(start_normals[:, 0] < 0)

    def test_end_faces_have_positive_x_normal(self):
        normals = face_normals(self.verts, self.faces)
        end_normals = normals[6:8]
        assert np.all(end_normals[:, 0] > 0)

    def test_all_normals_nonzero(self):
        normals = face_normals(self.verts, self.faces)
        magnitudes = np.linalg.norm(normals, axis=1)
        assert np.all(magnitudes > 0)


# -----------------------------------------------------------------------
# Zero-length move is skipped
# -----------------------------------------------------------------------

class TestEdgeCases:

    def test_zero_length_move_skipped(self):
        gb = GeometryBuilder()
        gb.add_move(make_move(x1=5.0, y1=5.0, x2=5.0, y2=5.0))
        verts, faces = gb.get_mesh()
        assert len(verts) == 0

    def test_zero_length_move_does_not_corrupt_accumulation(self):
        gb = GeometryBuilder()
        gb.add_move(make_move(x1=0, y1=0, x2=5, y2=0))      # valid
        gb.add_move(make_move(x1=5, y1=0, x2=5, y2=0))      # zero-length
        gb.add_move(make_move(x1=5, y1=0, x2=10, y2=0))     # valid
        verts, faces = gb.get_mesh()
        assert verts.shape == (16, 3)
        assert faces.max() < len(verts)

    def test_diagonal_move_produces_valid_mesh(self):
        gb = GeometryBuilder()
        gb.add_move(make_move(x1=0, y1=0, x2=7.07, y2=7.07))
        verts, faces = gb.get_mesh()
        assert verts.shape == (8, 3)
        normals = face_normals(verts, faces)
        assert np.all(np.linalg.norm(normals, axis=1) > 0)

    def test_custom_nozzle_and_layer_height(self):
        gb = GeometryBuilder(nozzle_diameter=0.6, layer_height=0.3)
        gb.add_move(make_move(x1=0, y1=0, x2=10, y2=0, z=0.3))
        verts, _ = gb.get_mesh()
        ys = np.unique(np.round(verts[:, 1], 9))
        assert set(ys) == {-0.3, 0.3}
        zs = np.unique(np.round(verts[:, 2], 9))
        assert len(zs) == 2
        assert np.allclose(sorted(zs), [0.0, 0.3])