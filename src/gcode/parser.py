"""
GCodeParser: parses a .gcode file into a structured list of extrusion moves.

Each move represents one G-code extrusion command (G1 with E advancing),
which is used as one time step in the thermal simulation.

Move format:
    {
        'x1': float, 'y1': float,   # start position (mm)
        'x2': float, 'y2': float,   # end position (mm)
        'z':  float,                 # layer height (mm)
        'f':  float,                 # feed rate (mm/min)
        'e':  float,                 # extrusion amount for this move (mm)
        'layer': int                 # layer index (0-based)
    }
"""
from typing import Optional


class GCodeParser:

    def __init__(self):
        self._x = 0.0
        self._y = 0.0
        self._z = 0.0
        self._e = 0.0
        self._f = 0.0
        self._layer = -1
        self._absolute = True
        self._absolute_e = True

    def parse(self, filepath: str) -> list[dict]:
        moves = []
        with open(filepath, 'r') as fh:
            for raw in fh:
                line = raw.split(';')[0].strip()
                if not line:
                    continue
                move = self._process_line(line)
                if move is not None:
                    moves.append(move)
        return moves

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_line(self, line: str) -> Optional[dict]:
        parts = line.upper().split()
        cmd = parts[0]

        if cmd == 'G90':
            self._absolute = True
            self._absolute_e = True
        elif cmd == 'G91':
            self._absolute = False
            self._absolute_e = False
        elif cmd == 'M82':
            self._absolute_e = True
        elif cmd == 'M83':
            self._absolute_e = False
        elif cmd == 'G92':
            self._handle_set_position(parts[1:])
        elif cmd in ('G0', 'G1'):
            return self._handle_move(cmd, parts[1:])

        return None

    def _handle_set_position(self, params: list[str]) -> None:
        for p in params:
            key, val = p[0], float(p[1:])
            if key == 'X':
                self._x = val
            elif key == 'Y':
                self._y = val
            elif key == 'Z':
                self._z = val
            elif key == 'E':
                self._e = val

    def _handle_move(self, cmd: str, params: list[str]) -> Optional[dict]:
        x1, y1, z1, e1, f1 = self._x, self._y, self._z, self._e, self._f

        new_x, new_y, new_z, new_e, new_f = x1, y1, z1, e1, f1
        for p in params:
            key, val = p[0], float(p[1:])
            if key == 'X':
                new_x = val if self._absolute else x1 + val
            elif key == 'Y':
                new_y = val if self._absolute else y1 + val
            elif key == 'Z':
                new_z = val if self._absolute else z1 + val
            elif key == 'E':
                new_e = val if self._absolute_e else e1 + val
            elif key == 'F':
                new_f = val

        # update state
        self._x, self._y, self._z = new_x, new_y, new_z
        self._f = new_f

        # detect layer change
        if new_z > z1:
            self._layer += 1

        e_delta = new_e - e1 if self._absolute_e else new_e - e1
        self._e = new_e

        # only emit extrusion moves (G1 with positive E advance)
        if cmd == 'G1' and e_delta > 0:
            return {
                'x1': x1,  'y1': y1,
                'x2': new_x, 'y2': new_y,
                'z':  new_z,
                'f':  new_f,
                'e':  e_delta,
                'layer': max(self._layer, 0),
            }

        return None