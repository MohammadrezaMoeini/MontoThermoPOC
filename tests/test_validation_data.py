"""
Tests for the reference data in validation.py.

Checks structural integrity and physical plausibility of EXP_DATA,
FEM_DATA, POINT_COORDS, and MC_DATA (when the pkl is present).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent /
                        'experimentDev' / 'example07_ValidationWithAnton'))

import numpy as np
import pytest
from validation import EXP_DATA, FEM_DATA, POINT_COORDS, MC_DATA

ALL_POINTS   = list(range(1, 28))
LAYER1_PTS   = list(range(1, 10))
LAYER10_PTS  = list(range(10, 19))
LAYER20_PTS  = list(range(19, 28))


# ── POINT_COORDS ──────────────────────────────────────────────────────────────

class TestPointCoords:

    def test_all_27_points_defined(self):
        assert set(POINT_COORDS) == set(ALL_POINTS)

    def test_each_coord_has_three_values(self):
        for pt, coord in POINT_COORDS.items():
            assert len(coord) == 3, f"Point #{pt} coord has {len(coord)} values"

    def test_layer1_z_is_02_mm(self):
        for pt in LAYER1_PTS:
            assert POINT_COORDS[pt][2] == pytest.approx(0.2)

    def test_layer10_z_is_20_mm(self):
        for pt in LAYER10_PTS:
            assert POINT_COORDS[pt][2] == pytest.approx(2.0)

    def test_layer20_z_is_40_mm(self):
        for pt in LAYER20_PTS:
            assert POINT_COORDS[pt][2] == pytest.approx(4.0)

    def test_all_x1_in_expected_set(self):
        x1_vals = {POINT_COORDS[p][0] for p in ALL_POINTS}
        assert x1_vals == {0.4, 4.0, 8.8}

    def test_all_x2_in_expected_set(self):
        x2_vals = {POINT_COORDS[p][1] for p in ALL_POINTS}
        assert x2_vals == {0.4, 29.6, 44.0}


# ── EXP_DATA ──────────────────────────────────────────────────────────────────

class TestExpData:

    def test_all_27_points_present(self):
        assert set(EXP_DATA) == set(ALL_POINTS)

    def test_each_entry_has_t_and_T_arrays(self):
        for pt in ALL_POINTS:
            assert 't' in EXP_DATA[pt] and 'T' in EXP_DATA[pt]
            assert isinstance(EXP_DATA[pt]['t'], np.ndarray)
            assert isinstance(EXP_DATA[pt]['T'], np.ndarray)

    def test_t_and_T_same_length(self):
        for pt in ALL_POINTS:
            assert len(EXP_DATA[pt]['t']) == len(EXP_DATA[pt]['T']), \
                f"Point #{pt}: t/T length mismatch"

    def test_each_point_has_at_least_two_samples(self):
        for pt in ALL_POINTS:
            assert len(EXP_DATA[pt]['t']) >= 2, \
                f"Point #{pt} has fewer than 2 samples"

    def test_times_monotonically_increasing(self):
        for pt in ALL_POINTS:
            assert np.all(np.diff(EXP_DATA[pt]['t']) > 0), \
                f"Point #{pt}: times not strictly increasing"

    def test_temperatures_in_physical_range(self):
        for pt in ALL_POINTS:
            T = EXP_DATA[pt]['T']
            assert T.min() >= 20,  f"Point #{pt}: T too low  ({T.min()} °C)"
            assert T.max() <= 250, f"Point #{pt}: T too high ({T.max()} °C)"

    def test_temperatures_decrease_over_time(self):
        for pt in ALL_POINTS:
            T = EXP_DATA[pt]['T']
            assert T[-1] < T[0], \
                f"Point #{pt}: T did not decrease ({T[0]} → {T[-1]} °C)"

    def test_layer_time_ordering(self):
        t_max_l1  = max(EXP_DATA[p]['t'].max() for p in LAYER1_PTS)
        t_min_l10 = min(EXP_DATA[p]['t'].min() for p in LAYER10_PTS)
        t_min_l20 = min(EXP_DATA[p]['t'].min() for p in LAYER20_PTS)
        assert t_max_l1  < t_min_l10, "Layer 1 and Layer 10 times overlap"
        assert t_min_l10 < t_min_l20, "Layer 10 and Layer 20 times overlap"

    def test_layer1_times_under_50s(self):
        # Layer 1 spans 3 beads: pts#1-3 ≈0-2s, pts#4-6 ≈23-27s, pts#7-9 ≈43-46s
        for pt in LAYER1_PTS:
            assert EXP_DATA[pt]['t'].max() < 50.0, \
                f"Layer-1 pt #{pt} time exceeds 50 s"

    def test_within_bead_points_ordered_in_time(self):
        # pts 1,2,3 are on the same bead; their start times must be increasing
        assert EXP_DATA[1]['t'][0] < EXP_DATA[2]['t'][0] < EXP_DATA[3]['t'][0]
        assert EXP_DATA[4]['t'][0] < EXP_DATA[5]['t'][0] < EXP_DATA[6]['t'][0]
        assert EXP_DATA[7]['t'][0] < EXP_DATA[8]['t'][0] < EXP_DATA[9]['t'][0]


# ── FEM_DATA ──────────────────────────────────────────────────────────────────

class TestFemData:

    def test_all_27_points_present(self):
        assert set(FEM_DATA) == set(ALL_POINTS)

    def test_each_entry_has_t_and_T_arrays(self):
        for pt in ALL_POINTS:
            assert 't' in FEM_DATA[pt] and 'T' in FEM_DATA[pt]

    def test_t_and_T_same_length(self):
        for pt in ALL_POINTS:
            assert len(FEM_DATA[pt]['t']) == len(FEM_DATA[pt]['T'])

    def test_times_monotonically_increasing(self):
        for pt in ALL_POINTS:
            assert np.all(np.diff(FEM_DATA[pt]['t']) > 0), \
                f"Point #{pt}: FEM times not strictly increasing"

    def test_temperatures_in_physical_range(self):
        for pt in ALL_POINTS:
            T = FEM_DATA[pt]['T']
            assert T.min() >= 20  and T.max() <= 250

    def test_temperatures_decrease_over_time(self):
        for pt in ALL_POINTS:
            T = FEM_DATA[pt]['T']
            assert T[-1] < T[0], \
                f"Point #{pt}: FEM T did not decrease ({T[0]} → {T[-1]} °C)"

    def test_fem_and_exp_start_at_same_time(self):
        for pt in ALL_POINTS:
            dt = abs(FEM_DATA[pt]['t'][0] - EXP_DATA[pt]['t'][0])
            assert dt < 1.0, \
                f"Point #{pt}: FEM/Exp start times differ by {dt:.2f} s"

    def test_known_point1_values(self):
        # FEM_DATA[1] at t=0.0: T=138 (digitised from paper)
        assert FEM_DATA[1]['t'][0]  == pytest.approx(0.0)
        assert FEM_DATA[1]['T'][0]  == pytest.approx(138.0)


# ── MC_DATA ───────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not MC_DATA, reason='mc_results_layer1.pkl not present')
class TestMcData:

    def test_keys_are_layer1_points_only(self):
        for key in MC_DATA:
            assert 1 <= key <= 9, f"Unexpected MC_DATA key: {key}"

    def test_each_entry_has_t_and_T(self):
        for pt, data in MC_DATA.items():
            assert 't' in data and 'T' in data

    def test_t_and_T_same_length(self):
        for pt, data in MC_DATA.items():
            assert len(data['t']) == len(data['T'])

    def test_times_monotonically_increasing(self):
        for pt, data in MC_DATA.items():
            assert np.all(np.diff(data['t']) > 0), \
                f"MC pt #{pt}: times not strictly increasing"

    def test_temperatures_in_physical_range(self):
        for pt, data in MC_DATA.items():
            assert data['T'].min() >= 10,  f"MC pt #{pt} T too low"
            assert data['T'].max() <= 250, f"MC pt #{pt} T too high"

    def test_all_9_layer1_points_present(self):
        assert set(MC_DATA) == set(range(1, 10))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])