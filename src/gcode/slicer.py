"""
This script include class STLSlicer to generate the gcode for a given stl file.
"""

import subprocess
from pathlib import Path

PRUSASLICER = "/Applications/PrusaSlicer.app/Contents/MacOS/PrusaSlicer"


class STLSlicer:
    def __init__(self, slicer_path: str = PRUSASLICER):
        self.slicer_path = Path(slicer_path)
        if not self.slicer_path.exists():
            raise FileNotFoundError(f"PrusaSlicer not found at {self.slicer_path}")

    def generate_gcode(self, stl_path: str, output_path: str = None, **options) -> Path:

        stl_path = Path(stl_path)
        if not stl_path.exists():
            raise FileNotFoundError(f"STL file not found: {stl_path}")

        if output_path is None:
            output_path = stl_path.with_suffix(".gcode")
        output_path = Path(output_path)

        cmd = [
            str(self.slicer_path),
            "--export-gcode",
            "--output", str(output_path),
        ]

        for key, value in options.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])

        cmd.append(str(stl_path))

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"PrusaSlicer failed (exit {result.returncode}):\n{result.stderr}"
            )

        return output_path