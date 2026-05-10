"""
Visual verification of GCodeParser output.

Run directly to display the plots:
    python tests/test_parser_visually.py

Plot 1 — Move index:
    Color encodes the sequential move index (0 … N-1).
    Every Nth midpoint is labelled for spot-checking.

Plot 2 — Real time (seconds):
    Color encodes cumulative wall-clock time derived from segment length
    and feed rate F (mm/min).  dt_i = |segment| / (F/60).
    Every Nth midpoint is labelled with its cumulative time in seconds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

from src.gcode.parser import GCodeParser

TESTS_DIR = Path(__file__).parent
GCODE_FILE = TESTS_DIR / "20mmbox.gcode"

# Label every Nth move so the plot stays readable
LABEL_EVERY_N = 50


def plot_parser_output(moves: list[dict]) -> None:
    n = len(moves)
    colormap = cm.plasma
    norm = mcolors.Normalize(vmin=0, vmax=n - 1)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, m in enumerate(moves):
        color = colormap(norm(i))
        ax.plot(
            [m['x1'], m['x2']],
            [m['y1'], m['y2']],
            [m['z'],  m['z']],
            color=color,
            linewidth=0.8,
        )

        if i % LABEL_EVERY_N == 0:
            mx = (m['x1'] + m['x2']) / 2
            my = (m['y1'] + m['y2']) / 2
            ax.scatter(mx, my, m['z'], color=color, s=18, zorder=5)
            ax.text(mx, my, m['z'], f" i={i}", fontsize=6, color='black')

    # colorbar
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label("Move index", fontsize=10)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(
        f"GCodeParser output — {n} extrusion moves\n"
        f"layers: 0 → {max(m['layer'] for m in moves)}   "
        f"(label every {LABEL_EVERY_N} steps)",
        fontsize=11,
    )

    plt.tight_layout()
    plt.show()


def compute_cumulative_times(moves: list[dict]) -> np.ndarray:
    """Return cumulative time (seconds) at the start of each move."""
    times = np.zeros(len(moves))
    for i, m in enumerate(moves):
        dist = np.hypot(m['x2'] - m['x1'], m['y2'] - m['y1'])
        dt = dist / (m['f'] / 60.0) if m['f'] > 0 else 0.0
        if i + 1 < len(moves):
            times[i + 1] = times[i] + dt
    return times


def plot_real_time(moves: list[dict]) -> None:
    times = compute_cumulative_times(moves)
    total = times[-1] + np.hypot(
        moves[-1]['x2'] - moves[-1]['x1'],
        moves[-1]['y2'] - moves[-1]['y1'],
    ) / (moves[-1]['f'] / 60.0)

    colormap = cm.inferno
    norm = mcolors.Normalize(vmin=0, vmax=total)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, m in enumerate(moves):
        color = colormap(norm(times[i]))
        ax.plot(
            [m['x1'], m['x2']],
            [m['y1'], m['y2']],
            [m['z'],  m['z']],
            color=color,
            linewidth=0.8,
        )

        if i % LABEL_EVERY_N == 0:
            mx = (m['x1'] + m['x2']) / 2
            my = (m['y1'] + m['y2']) / 2
            ax.scatter(mx, my, m['z'], color=color, s=18, zorder=5)
            ax.text(mx, my, m['z'], f" {times[i]:.1f}s", fontsize=6, color='black')

    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label("Cumulative time (s)", fontsize=10)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(
        f"GCodeParser output — real time\n"
        f"total print time: {total:.1f} s ({total/60:.1f} min)   "
        f"(label every {LABEL_EVERY_N} steps)",
        fontsize=11,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    moves = GCodeParser().parse(str(GCODE_FILE))
    print(f"Parsed {len(moves)} extrusion moves across "
          f"{max(m['layer'] for m in moves) + 1} layers.")
    plot_parser_output(moves)
    plot_real_time(moves)