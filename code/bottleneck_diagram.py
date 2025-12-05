"""
Dimensional Bottleneck Conceptual Diagram

Generates a schematic figure illustrating the core thesis:
High-D Phase Space -> Partition/Bottleneck -> Discrete Symbol Tape
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def generate_bottleneck_diagram():
    """
    Generates a schematic diagram illustrating the 'Dimensional Bottleneck':
    High-D Phase Space -> Partition/Bottleneck -> Discrete Symbol Tape
    """
    fig = plt.figure(figsize=(10, 8))

    # --- Top: High-Dimensional Dynamics (3D Plot) ---
    ax_top = fig.add_axes([0.2, 0.55, 0.6, 0.4], projection='3d')

    # Generate a complex trajectory (toroidal-like path)
    t = np.linspace(0, 40, 2000)
    x = (2 + np.cos(3*t)) * np.cos(2*t)
    y = (2 + np.cos(3*t)) * np.sin(2*t)
    z = np.sin(3*t)

    ax_top.plot(x, y, z, color='#3366cc', lw=0.8, alpha=0.7)
    ax_top.set_title(r"1. Continuous Dynamics ($X \subseteq \mathbb{R}^n$)",
                     fontsize=14, pad=0)

    # Hide axes details to make it abstract
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.set_zticks([])
    ax_top.axis('off')

    # Add a bounding wireframe to suggest a bounded manifold
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, 2 * np.pi, 20)
    x_w = 3 * np.outer(np.cos(u), np.sin(v))
    y_w = 3 * np.outer(np.sin(u), np.sin(v))
    z_w = 3 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax_top.plot_wireframe(x_w, y_w, z_w, color='gray', alpha=0.05)

    # --- Middle: The Bottleneck Schematic (2D Overlay) ---
    ax_mid = fig.add_axes([0.1, 0.05, 0.8, 0.9])
    ax_mid.set_xlim(0, 10)
    ax_mid.set_ylim(0, 10)
    ax_mid.axis('off')

    # Geometry
    center_x = 5.0
    top_y = 5.5
    neck_y = 3.5
    bottom_y = 2.0
    width_top = 3.0
    width_neck = 0.5

    # Draw Funnel (The "Measurement" Process)
    funnel_color = '#555555'
    # Left Wall
    ax_mid.plot([center_x - width_top, center_x - width_neck], [top_y, neck_y],
                '-', lw=2, color=funnel_color)
    ax_mid.plot([center_x - width_neck, center_x - width_neck], [neck_y, bottom_y],
                '-', lw=2, color=funnel_color)
    # Right Wall
    ax_mid.plot([center_x + width_top, center_x + width_neck], [top_y, neck_y],
                '-', lw=2, color=funnel_color)
    ax_mid.plot([center_x + width_neck, center_x + width_neck], [neck_y, bottom_y],
                '-', lw=2, color=funnel_color)

    # Annotations for the Process
    ax_mid.text(center_x + 1.5, 4.6, r"2. Measurement ($\mathcal{P}$)",
                fontsize=12, fontweight='bold', color='#333333')
    ax_mid.text(center_x + 1.5, 4.0, "Dimensional Bottleneck",
                fontsize=10, style='italic', color='#555555')
    ax_mid.text(center_x + 1.5, 3.6, "(Information Loss)",
                fontsize=10, style='italic', color='#555555')

    # Arrow indicating flow of information
    ax_mid.arrow(center_x, top_y - 0.5, 0, -1.0, head_width=0.2, head_length=0.3,
                 fc='gray', ec='gray', alpha=0.6)

    # --- Bottom: Discrete Symbol Tape ---
    tape_y = 1.0
    tape_x_start = 2.0
    tape_width = 6.0
    num_cells = 8
    cell_w = tape_width / num_cells
    cell_h = 0.8

    symbols = ['0', '1', '1', '0', '1', '0', '0', '...']
    colors = ['#e6f2ff', '#fff0e6'] * 4

    for i in range(num_cells):
        x_pos = tape_x_start + i * cell_w
        rect = Rectangle((x_pos, tape_y), cell_w, cell_h,
                         facecolor=colors[i], edgecolor='black')
        ax_mid.add_patch(rect)
        ax_mid.text(x_pos + cell_w/2, tape_y + cell_h/2, symbols[i],
                    ha='center', va='center', fontsize=14, fontfamily='monospace')

    ax_mid.text(center_x, tape_y - 0.5, r"3. Symbolic Output ($\Sigma^{\mathbb{N}}$)",
                ha='center', fontsize=12, fontweight='bold')

    # Dashed lines connecting bottleneck to tape
    ax_mid.plot([center_x - width_neck, tape_x_start], [bottom_y, tape_y + cell_h],
                'k--', alpha=0.2)
    ax_mid.plot([center_x + width_neck, tape_x_start + tape_width], [bottom_y, tape_y + cell_h],
                'k--', alpha=0.2)

    plt.savefig(FIGURES_DIR / 'fig2_bottleneck.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'fig2_bottleneck.png'}")


if __name__ == "__main__":
    generate_bottleneck_diagram()
