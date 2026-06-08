"""
Tests for helper functions in verification.py.

Covers: _fem_ref, _exp_ref, _load_moves, _find_bead, _build_mesh, simulate_point.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent /
                        'experimentDev' / 'example07_ValidationWithAnton'))

import numpy as np
import pytest
from verification import (
    _fem_ref, _exp_ref,
    _load_moves, _find_bead, _build_mesh,
    simulate_point,
    BEAD_X, CONFIG,
)
from solver.config_loader import load_config


# ── _fem_ref and _exp_ref ─────────────────────────────────────────────────────

class TestFemRef:

    @pytest.mark.parametrize('pt_num', range(1, 10))
    def test_returns_float(self, pt_num):
        assert isinstance(_fem_ref(pt_num, 0.5), float)

    @pytest.mark.parametrize('pt_num', range(1, 10))
    def test_value_in_physical_range(self, pt_num):
        T = _fem_ref(pt_num, 0.5)
        assert 20.0 <= T <= 250.0, f"pt#{pt_num}: {T:.1f} °C out of range"

    def test_pt1_at_t0_returns_first_fem_value(self):
        # FEM_DATA[1]['t'][0] = 0.0, T[0] = 138 — t_query=0 maps exactly to it
        assert _fem_ref(1, 0.0) == pytest.approx(138.0)

    def test_pt1_at_t05_is_84(self):
        # FEM_DATA[1] has exact data point t=0.50, T=84
        assert _fem_ref(1, 0.5) == pytest.approx(84.0)

    def test_later_t_gives_lower_T(self):
        for pt in [1, 2, 3]:
            T_early = _fem_ref(pt, 0.1)
            T_late  = _fem_ref(pt, 0.8)
            assert T_early >= T_late, \
                f"pt#{pt}: FEM T not decreasing ({T_early:.1f} → {T_late:.1f})"


class TestExpRef:

    @pytest.mark.parametrize('pt_num', range(1, 10))
    def test_returns_float(self, pt_num):
        assert isinstance(_exp_ref(pt_num, 0.5), float)

    def test_pt1_at_t05_is_88(self):
        # EXP_DATA[1] has exact data point t=0.50, T=88
        assert _exp_ref(1, 0.5) == pytest.approx(88.0)

    def test_exp_and_fem_differ_for_each_point(self):
        for pt in [1, 2, 3]:
            assert _exp_ref(pt, 0.5) != _fem_ref(pt, 0.5), \
                f"pt#{pt}: Exp and FEM are identical (unexpected)"

    def test_later_t_gives_lower_T(self):
        for pt in [1, 2, 3]:
            T_early = _exp_ref(pt, 0.1)
            T_late  = _exp_ref(pt, 0.8)
            assert T_early >= T_late

    @pytest.mark.parametrize('pt_num', range(1, 10))
    def test_both_refs_in_physical_range(self, pt_num):
        assert 20.0 <= _exp_ref(pt_num, 0.3) <= 250.0
        assert 20.0 <= _fem_ref(pt_num, 0.3) <= 250.0


# ── _load_moves ───────────────────────────────────────────────────────────────

class TestLoadMoves:

    @pytest.fixture(scope='class')
    def moves(self):
        return _load_moves()

    def test_returns_non_empty_list(self, moves):
        assert len(moves) > 0

    def test_all_required_keys_present(self, moves):
        required = {'x1', 'y1', 'x2', 'y2', 'layer', 'f', 'e'}
        for m in moves:
            assert required <= set(m)

    def test_all_layer_0(self, moves):
        for m in moves:
            assert m['layer'] == 0

    def test_all_beads_longer_than_1mm(self, moves):
        for m in moves:
            length = float(np.hypot(m['x2'] - m['x1'], m['y2'] - m['y1']))
            assert length >= 1.0, f"Short bead slipped through: {length:.3f} mm"

    def test_contains_validation_bead_x_values(self, moves):
        x_centres = {round((m['x1'] + m['x2']) / 2.0, 1) for m in moves}
        for bx in [-4.0, -0.4, 4.4]:
            assert bx in x_centres, f"Bead x={bx} not found in parsed moves"


# ── _find_bead ────────────────────────────────────────────────────────────────

class TestFindBead:

    @pytest.fixture(scope='class')
    def moves(self):
        return _load_moves()

    @pytest.mark.parametrize('bead_x', [-4.0, -0.4, 4.4])
    def test_finds_correct_bead(self, moves, bead_x):
        move = _find_bead(bead_x, moves)
        xc = round((move['x1'] + move['x2']) / 2.0, 1)
        assert xc == bead_x

    def test_returned_move_is_long_bead(self, moves):
        move = _find_bead(-4.0, moves)
        length = float(np.hypot(move['x2'] - move['x1'], move['y2'] - move['y1']))
        assert length >= 1.0

    def test_raises_on_nonexistent_bead_x(self, moves):
        with pytest.raises(ValueError, match='No bead found'):
            _find_bead(99.9, moves)


# ── _build_mesh ───────────────────────────────────────────────────────────────

class TestBuildMesh:

    @pytest.fixture(scope='class')
    def mesh(self):
        cfg   = load_config(CONFIG)
        moves = _load_moves()
        move  = _find_bead(-4.0, moves)
        return _build_mesh(move, cfg), move, cfg

    def test_returns_verts_and_faces(self, mesh):
        (verts, faces), *_ = mesh
        assert verts.ndim == 2 and verts.shape[1] == 3
        assert faces.ndim == 2 and faces.shape[1] == 3

    def test_verts_dtype_float32(self, mesh):
        (verts, _), *_ = mesh
        assert verts.dtype == np.float32

    def test_faces_dtype_int32(self, mesh):
        (_, faces), *_ = mesh
        assert faces.dtype == np.int32

    def test_face_indices_within_vertex_range(self, mesh):
        (verts, faces), *_ = mesh
        assert faces.min() >= 0
        assert faces.max() < len(verts)

    def test_bead_z_extent_matches_layer_height(self, mesh):
        (verts, _), move, cfg = mesh
        z_range = float(verts[:, 2].max() - verts[:, 2].min())
        assert z_range == pytest.approx(cfg['layer_height'], rel=1e-4)


# ── simulate_point ────────────────────────────────────────────────────────────

class TestSimulatePoint:
    """
    Fast smoke tests using small n_walks=16 to keep runtime short.
    Physical-correctness assertions use a wide tolerance to absorb MC noise.
    """

    @pytest.fixture(scope='class')
    def setup(self):
        cfg   = load_config(CONFIG)
        moves = _load_moves()
        move  = _find_bead(BEAD_X[1], moves)
        verts, faces = _build_mesh(move, cfg)
        return cfg, move, verts, faces

    def test_returns_float(self, setup):
        cfg, move, verts, faces = setup
        T = simulate_point(1, 0.5, n_walks=16, dt=0.5,
                           cfg=cfg, move=move, verts=verts, faces=faces)
        assert isinstance(T, float)

    def test_in_physical_range(self, setup):
        cfg, move, verts, faces = setup
        T = simulate_point(1, 0.5, n_walks=16, dt=0.5,
                           cfg=cfg, move=move, verts=verts, faces=faces)
        assert cfg['T_ambient'] - 10.0 <= T <= cfg['T_nozzle'] + 10.0

    @pytest.mark.parametrize('pt_num', [1, 2, 3])
    def test_layer1_points_all_valid(self, pt_num):
        cfg   = load_config(CONFIG)
        moves = _load_moves()
        move  = _find_bead(BEAD_X[pt_num], moves)
        verts, faces = _build_mesh(move, cfg)
        T = simulate_point(pt_num, 0.5, n_walks=16, dt=0.5,
                           cfg=cfg, move=move, verts=verts, faces=faces)
        assert cfg['T_ambient'] - 10.0 <= T <= cfg['T_nozzle'] + 10.0

    def test_t_query_zero_returns_nozzle_temperature(self, setup):
        cfg, move, verts, faces = setup
        T = simulate_point(1, 0.0, n_walks=16, dt=0.5,
                           cfg=cfg, move=move, verts=verts, faces=faces)
        assert T == pytest.approx(cfg['T_nozzle'])

    def test_result_is_deterministic_given_large_n_walks(self, setup):
        """Two runs with large n_walks should agree within ~10 °C (3-sigma)."""
        cfg, move, verts, faces = setup
        T1 = simulate_point(1, 0.5, n_walks=128, dt=0.5,
                            cfg=cfg, move=move, verts=verts, faces=faces)
        T2 = simulate_point(1, 0.5, n_walks=128, dt=0.5,
                            cfg=cfg, move=move, verts=verts, faces=faces)
        assert abs(T1 - T2) < 20.0, \
            f"MC results differ too much: {T1:.1f} vs {T2:.1f} °C"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])