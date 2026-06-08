"""
Tests for solver.config_loader.load_config.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from solver.config_loader import load_config

REAL_CONFIG = Path(__file__).parent.parent / 'configs' / 'anton_plate.json'

EXPECTED_KEYS = {
    'k_cond', 'rho', 'cp',
    'T_bed', 'T_nozzle', 'T_ambient', 'h',
    'nozzle_diameter', 'layer_height',
    'n_walks', 'dt', 'n_cooling_steps', 'grid_shape',
}


class TestLoadRealConfig:

    def test_loads_without_error(self):
        cfg = load_config(REAL_CONFIG)
        assert cfg is not None

    def test_path_object_and_string_both_work(self):
        cfg_path   = load_config(REAL_CONFIG)
        cfg_string = load_config(str(REAL_CONFIG))
        assert cfg_path == cfg_string

    def test_all_expected_keys_present(self):
        cfg = load_config(REAL_CONFIG)
        missing = EXPECTED_KEYS - set(cfg)
        assert not missing, f"Missing keys: {missing}"

    def test_no_description_keys(self):
        cfg = load_config(REAL_CONFIG)
        for key in cfg:
            assert not key.startswith('_'), f"Description key leaked: {key!r}"

    def test_grid_shape_is_tuple(self):
        cfg = load_config(REAL_CONFIG)
        assert isinstance(cfg['grid_shape'], tuple)

    def test_grid_shape_has_three_positive_ints(self):
        cfg = load_config(REAL_CONFIG)
        gs = cfg['grid_shape']
        assert len(gs) == 3
        assert all(isinstance(v, int) and v > 0 for v in gs)

    def test_temperature_ordering(self):
        cfg = load_config(REAL_CONFIG)
        assert cfg['T_ambient'] < cfg['T_bed'] < cfg['T_nozzle']

    def test_k_cond_positive(self):
        assert load_config(REAL_CONFIG)['k_cond'] > 0

    def test_rho_positive(self):
        assert load_config(REAL_CONFIG)['rho'] > 0

    def test_cp_positive(self):
        assert load_config(REAL_CONFIG)['cp'] > 0

    def test_h_positive(self):
        assert load_config(REAL_CONFIG)['h'] > 0

    def test_dt_positive(self):
        assert load_config(REAL_CONFIG)['dt'] > 0

    def test_n_walks_positive_int(self):
        n = load_config(REAL_CONFIG)['n_walks']
        assert isinstance(n, int) and n > 0

    def test_n_cooling_steps_non_negative(self):
        assert load_config(REAL_CONFIG)['n_cooling_steps'] >= 0

    def test_layer_height_positive(self):
        assert load_config(REAL_CONFIG)['layer_height'] > 0

    def test_nozzle_diameter_positive(self):
        assert load_config(REAL_CONFIG)['nozzle_diameter'] > 0

    def test_anton_plate_known_values(self):
        cfg = load_config(REAL_CONFIG)
        assert cfg['T_nozzle']  == pytest.approx(210.0)
        assert cfg['T_bed']     == pytest.approx(52.0)
        assert cfg['T_ambient'] == pytest.approx(20.0)
        assert cfg['k_cond']    == pytest.approx(0.11)
        assert cfg['h']         == pytest.approx(71.0)


class TestSyntheticConfig:

    def _write(self, tmp_path, data):
        p = tmp_path / 'cfg.json'
        p.write_text(json.dumps(data))
        return p

    def test_minimal_config_loads(self, tmp_path):
        data = {
            'material': {'k_cond': 0.2, 'rho': 1000.0, 'cp': 1500.0},
            'boundary': {'T_bed': 50.0, 'T_nozzle': 200.0,
                         'T_ambient': 20.0, 'h': 25.0},
            'process':  {'nozzle_diameter': 0.4, 'layer_height': 0.2},
            'solver':   {'n_walks': 64, 'dt': 0.5,
                         'n_cooling_steps': 5, 'grid_shape': [4, 2, 2]},
        }
        cfg = load_config(self._write(tmp_path, data))
        assert cfg['k_cond']     == pytest.approx(0.2)
        assert cfg['grid_shape'] == (4, 2, 2)

    def test_description_section_not_in_output(self, tmp_path):
        data = {
            '_description': 'ignored',
            'material': {'k_cond': 0.11},
            'boundary': {'T_bed': 52.0, 'T_nozzle': 210.0,
                         'T_ambient': 20.0, 'h': 71.0},
            'process':  {'nozzle_diameter': 0.4, 'layer_height': 0.4},
            'solver':   {'n_walks': 128, 'dt': 0.25,
                         'n_cooling_steps': 12, 'grid_shape': [4, 2, 2]},
        }
        cfg = load_config(self._write(tmp_path, data))
        assert '_description' not in cfg

    def test_grid_shape_list_converted_to_tuple(self, tmp_path):
        data = {
            'solver': {'n_walks': 64, 'dt': 0.5,
                       'n_cooling_steps': 5, 'grid_shape': [8, 4, 4]},
        }
        cfg = load_config(self._write(tmp_path, data))
        assert cfg['grid_shape'] == (8, 4, 4)
        assert isinstance(cfg['grid_shape'], tuple)

    def test_missing_file_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load_config('/nonexistent/does_not_exist.json')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])