"""
Unit tests for STLSlicer class.
"""

import pytest
from pathlib import Path
from src.gcode.slicer import STLSlicer

TESTS_DIR = Path(__file__).parent
STL_FILE  = TESTS_DIR / "20mmbox.stl"


class TestSTLSlicerInit:

    def test_default_slicer_path_exists(self):
        slicer = STLSlicer()
        assert slicer.slicer_path.exists()

    def test_invalid_slicer_path_raises(self):
        with pytest.raises(FileNotFoundError, match="PrusaSlicer not found"):
            STLSlicer(slicer_path="/nonexistent/path/PrusaSlicer")


class TestGenerateGcode:

    def test_generates_gcode_file(self, tmp_path):
        output = tmp_path / "20mmbox.gcode"
        slicer = STLSlicer()
        result = slicer.generate_gcode(STL_FILE, output)
        assert result == output
        assert output.exists()
        assert output.stat().st_size > 0

    def test_default_output_path(self, tmp_path):
        import shutil
        stl_copy = tmp_path / "20mmbox.stl"
        shutil.copy(STL_FILE, stl_copy)
        slicer = STLSlicer()
        result = slicer.generate_gcode(stl_copy)
        assert result == stl_copy.with_suffix(".gcode")
        assert result.exists()

    def test_gcode_contains_layer_change(self, tmp_path):
        output = tmp_path / "20mmbox.gcode"
        slicer = STLSlicer()
        slicer.generate_gcode(STL_FILE, output)
        content = output.read_text()
        assert ";LAYER_CHANGE" in content

    def test_gcode_contains_move_commands(self, tmp_path):
        output = tmp_path / "20mmbox.gcode"
        slicer = STLSlicer()
        slicer.generate_gcode(STL_FILE, output)
        content = output.read_text()
        assert "G1" in content

    def test_invalid_stl_path_raises(self, tmp_path):
        slicer = STLSlicer()
        with pytest.raises(FileNotFoundError, match="STL file not found"):
            slicer.generate_gcode("nonexistent.stl", tmp_path / "out.gcode")

    def test_returns_path_object(self, tmp_path):
        output = tmp_path / "20mmbox.gcode"
        slicer = STLSlicer()
        result = slicer.generate_gcode(STL_FILE, output)
        assert isinstance(result, Path)

    def test_custom_layer_height_option(self, tmp_path):
        output = tmp_path / "20mmbox.gcode"
        slicer = STLSlicer()
        result = slicer.generate_gcode(STL_FILE, output, layer_height=0.3)
        assert result.exists()