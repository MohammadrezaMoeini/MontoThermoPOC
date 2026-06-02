"""
load_config: reads a solver parameter JSON file and returns a flat dict
ready to be unpacked into TransientThermalSolver and GCodeTransientSimulator.

JSON sections:
  material  → k_cond, rho, cp
  boundary  → T_bed, T_nozzle, T_ambient, h
  process   → nozzle_diameter, layer_height
  solver    → n_walks, dt, n_cooling_steps, grid_shape

Usage:
    from solver.config_loader import load_config
    cfg = load_config('configs/anton_plate.json')

    solver = TransientThermalSolver(
        T_bed=cfg['T_bed'], T_nozzle=cfg['T_nozzle'], T_ambient=cfg['T_ambient'],
        h=cfg['h'], k_cond=cfg['k_cond'], rho=cfg['rho'], cp=cfg['cp'],
        n_walks=cfg['n_walks'], dt=cfg['dt'], grid_shape=cfg['grid_shape'])

    sim = GCodeTransientSimulator(
        solver,
        nozzle_diameter=cfg['nozzle_diameter'],
        layer_height=cfg['layer_height'],
        n_cooling_steps=cfg['n_cooling_steps'])
"""

import json
from pathlib import Path


def load_config(path: str | Path) -> dict:
    """
    Load a parameter JSON file and return a single flat dict.

    Keys in the returned dict match the parameter names of
    TransientThermalSolver and GCodeTransientSimulator exactly.
    grid_shape is converted from list to tuple.
    """
    with open(path, 'r') as f:
        raw = json.load(f)

    cfg = {}
    for section, values in raw.items():
        if section.startswith('_'):
            continue
        for key, val in values.items():
            cfg[key] = val

    if 'grid_shape' in cfg:
        cfg['grid_shape'] = tuple(cfg['grid_shape'])

    return cfg