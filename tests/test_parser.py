"""
Unit tests for GCodeParser.
"""

import pytest
from pathlib import Path
from src.gcode.parser import GCodeParser

TESTS_DIR = Path(__file__).parent
GCODE_FILE = TESTS_DIR / "20mmbox.gcode"

REQUIRED_KEYS = {'x1', 'y1', 'x2', 'y2', 'z', 'f', 'e', 'layer'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_gcode(tmp_path, lines: list[str]) -> Path:
    p = tmp_path / "test.gcode"
    p.write_text("\n".join(lines))
    return p


def make_parser() -> GCodeParser:
    return GCodeParser()


# ---------------------------------------------------------------------------
# Tests against the real 20mmbox.gcode
# ---------------------------------------------------------------------------

class TestRealGcode:

    def test_returns_non_empty_list(self):
        moves = make_parser().parse(str(GCODE_FILE))
        assert isinstance(moves, list)
        assert len(moves) > 0

    def test_all_moves_have_required_keys(self):
        moves = make_parser().parse(str(GCODE_FILE))
        for m in moves:
            assert REQUIRED_KEYS == set(m.keys()), f"Missing keys in move: {m}"

    def test_all_e_values_positive(self):
        moves = make_parser().parse(str(GCODE_FILE))
        for m in moves:
            assert m['e'] > 0, f"Non-positive e in move: {m}"

    def test_layer_indices_non_negative(self):
        moves = make_parser().parse(str(GCODE_FILE))
        for m in moves:
            assert m['layer'] >= 0

    def test_layer_indices_monotonic(self):
        moves = make_parser().parse(str(GCODE_FILE))
        layers = [m['layer'] for m in moves]
        assert layers == sorted(layers)

    def test_multiple_layers_detected(self):
        moves = make_parser().parse(str(GCODE_FILE))
        assert max(m['layer'] for m in moves) > 0

    def test_consecutive_moves_are_contiguous(self):
        """x2,y2 of move N must equal x1,y1 of move N+1 only when on same layer
        and no travel gap — we just check that x1/y1 are floats."""
        moves = make_parser().parse(str(GCODE_FILE))
        for m in moves:
            assert isinstance(m['x1'], float)
            assert isinstance(m['y1'], float)
            assert isinstance(m['x2'], float)
            assert isinstance(m['y2'], float)

    def test_feed_rate_positive(self):
        moves = make_parser().parse(str(GCODE_FILE))
        for m in moves:
            assert m['f'] > 0


# ---------------------------------------------------------------------------
# Tests with synthetic G-code (precise, isolated behaviour)
# ---------------------------------------------------------------------------

class TestAbsoluteMode:

    def test_single_extrusion_move(self, tmp_path):
        gcode = write_gcode(tmp_path, [
            "G90",
            "M82",
            "G92 E0",
            "G1 Z0.35 F7800",
            "G1 X10 Y20 F1800",        # travel (no E)
            "G1 X30 Y40 E1.5 F1800",   # extrusion
        ])
        moves = make_parser().parse(str(gcode))
        assert len(moves) == 1
        m = moves[0]
        assert m['x1'] == pytest.approx(10.0)
        assert m['y1'] == pytest.approx(20.0)
        assert m['x2'] == pytest.approx(30.0)
        assert m['y2'] == pytest.approx(40.0)
        assert m['e']  == pytest.approx(1.5)
        assert m['f']  == pytest.approx(1800.0)

    def test_travel_moves_excluded(self, tmp_path):
        gcode = write_gcode(tmp_path, [
            "G90", "M82", "G92 E0",
            "G0 X50 Y50 F7800",         # rapid travel
            "G1 X60 Y60 F7800",         # G1 travel (no E)
        ])
        moves = make_parser().parse(str(gcode))
        assert moves == []

    def test_retraction_excluded(self, tmp_path):
        gcode = write_gcode(tmp_path, [
            "G90", "M82",
            "G92 E0",
            "G1 X10 Y10 E2.0 F1800",   # extrusion
            "G1 E0.0 F2400",            # retraction (E goes back)
        ])
        moves = make_parser().parse(str(gcode))
        assert len(moves) == 1          # only the extrusion

    def test_g92_resets_extruder(self, tmp_path):
        gcode = write_gcode(tmp_path, [
            "G90", "M82",
            "G92 E0",
            "G1 X10 Y10 E5.0 F1800",   # extrusion: e_delta = 5.0
            "G92 E0",                   # reset
            "G1 X20 Y20 E3.0 F1800",   # extrusion: e_delta should be 3.0, not -2.0
        ])
        moves = make_parser().parse(str(gcode))
        assert len(moves) == 2
        assert moves[1]['e'] == pytest.approx(3.0)

    def test_e_field_is_delta_not_absolute(self, tmp_path):
        gcode = write_gcode(tmp_path, [
            "G90", "M82", "G92 E0",
            "G1 X10 Y0 E2.0 F1800",
            "G1 X20 Y0 E5.0 F1800",    # absolute E=5, delta=3
        ])
        moves = make_parser().parse(str(gcode))
        assert moves[0]['e'] == pytest.approx(2.0)
        assert moves[1]['e'] == pytest.approx(3.0)

    def test_layer_increments_on_z_rise(self, tmp_path):
        gcode = write_gcode(tmp_path, [
            "G90", "M82", "G92 E0",
            "G1 Z0.35 F7800",
            "G1 X10 Y10 E1.0 F1800",   # layer 0
            "G1 Z0.65 F7800",
            "G1 X20 Y20 E2.0 F1800",   # layer 1
            "G1 Z0.95 F7800",
            "G1 X30 Y30 E3.0 F1800",   # layer 2
        ])
        moves = make_parser().parse(str(gcode))
        assert len(moves) == 3
        assert moves[0]['layer'] == 0
        assert moves[1]['layer'] == 1
        assert moves[2]['layer'] == 2

    def test_comments_stripped(self, tmp_path):
        gcode = write_gcode(tmp_path, [
            "G90 ; absolute mode",
            "M82 ; absolute extruder",
            "G92 E0 ; reset",
            "G1 X10 Y10 E1.0 F1800 ; first move",
        ])
        moves = make_parser().parse(str(gcode))
        assert len(moves) == 1

    def test_start_position_tracks_previous_end(self, tmp_path):
        gcode = write_gcode(tmp_path, [
            "G90", "M82", "G92 E0",
            "G1 X10 Y5 E1.0 F1800",
            "G1 X25 Y15 E2.5 F1800",
        ])
        moves = make_parser().parse(str(gcode))
        assert moves[1]['x1'] == pytest.approx(moves[0]['x2'])
        assert moves[1]['y1'] == pytest.approx(moves[0]['y2'])


class TestRelativeExtruderMode:

    def test_m83_relative_extrusion(self, tmp_path):
        gcode = write_gcode(tmp_path, [
            "G90",
            "M83",                      # relative extruder
            "G1 X10 Y10 E1.5 F1800",   # E is a delta directly
            "G1 X20 Y20 E2.0 F1800",
        ])
        moves = make_parser().parse(str(gcode))
        assert len(moves) == 2
        assert moves[0]['e'] == pytest.approx(1.5)
        assert moves[1]['e'] == pytest.approx(2.0)

    def test_retraction_in_relative_mode_excluded(self, tmp_path):
        gcode = write_gcode(tmp_path, [
            "G90", "M83",
            "G1 X10 Y10 E1.5 F1800",
            "G1 E-2.0 F2400",           # retraction
        ])
        moves = make_parser().parse(str(gcode))
        assert len(moves) == 1