#!/usr/bin/env python3
"""
Generate visualization images for Light Transmission Simulation Results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import matplotlib.patheffects as path_effects

# Set font for Korean support (fallback to default if not available)
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_results(filename: str = "simulation_results_v2.json"):
    with open(filename, 'r') as f:
        return json.load(f)


def create_depth_profile_chart(results: dict, save_path: str = "depth_profile.png"):
    """Create depth profile visualization"""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Data for dark brown hair
    dark_brown = results["dark_brown"]
    fluence_data = dark_brown["fluence_at_depths"]

    # Extract data
    depths = []
    transmissions = []
    fluences = []
    labels = []

    depth_order = ["surface", "epidermis_top", "dermis_top", "dermis_100um",
                   "dermis_500um", "dermis_1mm", "dermis_2mm"]

    label_names = {
        "surface": "Surface\n(0 μm)",
        "epidermis_top": "Epidermis\nTop",
        "dermis_top": "Dermis\nTop (150μm)",
        "dermis_100um": "Dermis\n250μm",
        "dermis_500um": "Dermis\n650μm",
        "dermis_1mm": "Dermis\n1.15mm",
        "dermis_2mm": "Dermis\n2.15mm"
    }

    for key in depth_order:
        if key in fluence_data:
            data = fluence_data[key]
            depths.append(data["depth_um"])
            transmissions.append(data["transmission_fraction"] * 100)
            fluences.append(data["fluence_mW_cm2"])
            labels.append(label_names.get(key, key))

    # Create bar chart
    x = np.arange(len(labels))
    width = 0.6

    # Color gradient from green to red based on transmission
    colors = plt.cm.RdYlGn(np.array(transmissions) / 100)

    bars = ax.bar(x, transmissions, width, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, trans, flu in zip(bars, transmissions, fluences):
        height = bar.get_height()
        ax.annotate(f'{trans:.1f}%\n({flu:.2f} mW/cm²)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    ax.set_xlabel('Tissue Depth', fontsize=14, fontweight='bold')
    ax.set_ylabel('Light Transmission (%)', fontsize=14, fontweight='bold')
    ax.set_title('630nm LED Light Transmission Through Hair to Scalp\n(Dark Brown Hair, 2 mW/cm² Irradiance)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 120)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_hair_comparison_chart(results: dict, save_path: str = "hair_comparison.png"):
    """Create hair type comparison chart"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    hair_types = ["no_hair", "black", "dark_brown", "light_brown", "blonde", "gray"]
    hair_labels = ["No Hair\n(Baseline)", "Black", "Dark\nBrown", "Light\nBrown", "Blonde", "Gray"]

    fluences = []
    absorptions = []

    for ht in hair_types:
        fluences.append(results[ht]["fluence_at_depths"]["dermis_500um"]["fluence_mW_cm2"])
        if ht == "no_hair":
            absorptions.append(0)
        else:
            absorptions.append(results[ht]["absorption_distribution"]["absorbed_in_hair_percent"])

    x = np.arange(len(hair_labels))

    # Left plot: Fluence at dermis
    colors1 = plt.cm.Blues(np.linspace(0.4, 0.9, len(hair_types)))
    bars1 = ax1.bar(x, fluences, color=colors1, edgecolor='black', linewidth=1.5)

    for bar, flu in zip(bars1, fluences):
        ax1.annotate(f'{flu:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    ax1.set_xlabel('Hair Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fluence (mW/cm²)', fontsize=12, fontweight='bold')
    ax1.set_title('Fluence at Dermis 500μm Depth', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(hair_labels, fontsize=10)
    ax1.set_ylim(0, 1.0)
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Therapeutic min (1 mW/cm²)')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)

    # Right plot: Hair absorption
    colors2 = ['lightgray'] + list(plt.cm.Reds(np.linspace(0.3, 0.8, len(hair_types)-1)))
    bars2 = ax2.bar(x, absorptions, color=colors2, edgecolor='black', linewidth=1.5)

    for bar, abs_val in zip(bars2, absorptions):
        if abs_val > 0:
            ax2.annotate(f'{abs_val:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold')

    ax2.set_xlabel('Hair Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Absorption (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Light Absorption by Hair', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(hair_labels, fontsize=10)
    ax2.set_ylim(0, 10)
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('630nm LED Light Transmission: Hair Type Comparison',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_tissue_diagram(results: dict, save_path: str = "tissue_diagram.png"):
    """Create anatomical tissue diagram with light penetration"""

    fig, ax = plt.subplots(figsize=(14, 10))

    # Define layer properties
    layers = [
        {"name": "LED Device", "y": 9.5, "height": 0.8, "color": "#FF6B6B", "alpha": 0.9},
        {"name": "Hair Layer", "y": 8.5, "height": 0.8, "color": "#8B4513", "alpha": 0.8},
        {"name": "Epidermis (150μm)", "y": 7.0, "height": 1.2, "color": "#FFE4B5", "alpha": 0.9},
        {"name": "Dermis (2.5mm)", "y": 2.5, "height": 4.2, "color": "#FFC0CB", "alpha": 0.8},
        {"name": "Subcutaneous", "y": 0.5, "height": 1.8, "color": "#FFFACD", "alpha": 0.7},
    ]

    # Draw layers
    for layer in layers:
        rect = FancyBboxPatch((1, layer["y"]), 8, layer["height"],
                               boxstyle="round,pad=0.02",
                               facecolor=layer["color"],
                               edgecolor='black',
                               linewidth=2,
                               alpha=layer["alpha"])
        ax.add_patch(rect)

        # Add layer label
        ax.text(0.5, layer["y"] + layer["height"]/2, layer["name"],
                fontsize=12, fontweight='bold', ha='right', va='center')

    # Draw light rays (showing attenuation)
    dark_brown = results["dark_brown"]
    fluence_data = dark_brown["fluence_at_depths"]

    # Light ray positions and intensities
    ray_data = [
        (9.5, 100, "2.0 mW/cm² (100%)"),
        (8.5, 97, "1.94 mW/cm² (97%)"),
        (7.0, 62, "1.23 mW/cm² (62%)"),
        (5.5, 54, "1.08 mW/cm² (54%)"),
        (4.0, 39, "0.78 mW/cm² (39%)"),
        (2.5, 29, "0.58 mW/cm² (29%)"),
    ]

    for y, intensity, label in ray_data:
        # Draw multiple rays with varying alpha based on intensity
        alpha = intensity / 100 * 0.8
        width = intensity / 100 * 0.3 + 0.1

        for offset in [-1.5, -0.5, 0.5, 1.5]:
            ax.arrow(5 + offset, y + 0.3, 0, -0.5,
                    head_width=width, head_length=0.1,
                    fc='red', ec='darkred', alpha=alpha, linewidth=1)

        # Add fluence label on the right
        ax.text(9.5, y, label, fontsize=10, ha='left', va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Draw hair strands
    for x in [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]:
        ax.plot([x, x], [8.5, 9.3], color='#4a3728', linewidth=3, solid_capstyle='round')

    # Add title and annotations
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 11)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('630nm LED Light Penetration Through Scalp Tissue\n(Monte Carlo Simulation Results - Dark Brown Hair)',
                 fontsize=16, fontweight='bold', pad=20)

    # Add legend box
    legend_text = """Key Findings:
• 630nm light penetrates through hair to dermis
• ~39% transmission at dermis 500μm depth
• Hair absorbs 3.7% of incident light
• Therapeutic dose achievable with extended time"""

    ax.text(0.5, 0.3, legend_text, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
            transform=ax.transAxes, verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_summary_infographic(results: dict, save_path: str = "simulation_summary.png"):
    """Create a comprehensive summary infographic"""

    fig = plt.figure(figsize=(16, 12))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('Monte Carlo Light Transmission Simulation Results\n630nm LED (2 mW/cm²) Through Hair to Scalp',
                 fontsize=20, fontweight='bold', y=0.98)

    # 1. Parameters box (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    params_text = """SIMULATION PARAMETERS

Wavelength: 630 nm (Red LED)
Irradiance: 2.0 mW/cm²
Photons: 100,000
Hair Density: 120/cm²
Hair Diameter: 70 μm
Epidermis: 150 μm
Dermis: 2.5 mm"""
    ax1.text(0.1, 0.9, params_text, fontsize=11, family='monospace',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax1.set_title('Parameters', fontsize=14, fontweight='bold')

    # 2. Depth transmission curve (top center + right)
    ax2 = fig.add_subplot(gs[0, 1:])
    dark_brown = results["dark_brown"]
    fluence_data = dark_brown["fluence_at_depths"]

    depths = [0, 500, 650, 750, 1150, 1650, 2650]
    transmissions = [100, 97.0, 61.7, 53.9, 38.9, 29.0, 19.1]

    ax2.fill_between(depths, transmissions, alpha=0.3, color='red')
    ax2.plot(depths, transmissions, 'ro-', linewidth=3, markersize=10)

    for d, t in zip(depths, transmissions):
        ax2.annotate(f'{t:.1f}%', (d, t), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Depth (μm)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Transmission (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Light Transmission vs Depth (Dark Brown Hair)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=50, color='green', linestyle='--', alpha=0.7, label='50% threshold')
    ax2.axvspan(0, 150, alpha=0.2, color='orange', label='Epidermis')
    ax2.axvspan(150, 2650, alpha=0.1, color='pink', label='Dermis')
    ax2.legend(loc='upper right')

    # 3. Hair comparison bar chart (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    hair_types = ["Black", "Dark\nBrown", "Light\nBrown", "Blonde", "Gray"]
    absorptions = [6.1, 3.7, 2.1, 1.1, 0.5]
    colors = plt.cm.Reds(np.linspace(0.3, 0.8, 5))

    bars = ax3.barh(hair_types, absorptions, color=colors, edgecolor='black')
    for bar, val in zip(bars, absorptions):
        ax3.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'{val}%', va='center', fontweight='bold')

    ax3.set_xlabel('Absorption (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Hair Light Absorption\nby Color', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 8)

    # 4. Fluence comparison (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    conditions = ["Surface", "Epidermis", "Dermis\n500μm", "Dermis\n2mm"]
    fluences = [2.0, 1.23, 0.78, 0.38]
    colors = ['darkred', 'red', 'orange', 'yellow']

    bars = ax4.bar(conditions, fluences, color=colors, edgecolor='black', linewidth=2)
    for bar, val in zip(bars, fluences):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', fontweight='bold')

    ax4.set_ylabel('Fluence (mW/cm²)', fontsize=11, fontweight='bold')
    ax4.set_title('Fluence at Different Depths', fontsize=12, fontweight='bold')
    ax4.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Therapeutic min')
    ax4.legend()
    ax4.set_ylim(0, 2.5)

    # 5. Pie chart - Energy distribution (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    sizes = [14.1, 14.7, 3.7, 67.5]  # epidermis, dermis, hair, transmitted/reflected
    labels = ['Epidermis\nAbsorption\n14.1%', 'Dermis\nAbsorption\n14.7%',
              'Hair\nAbsorption\n3.7%', 'Transmitted/\nReflected\n67.5%']
    colors = ['#FFE4B5', '#FFC0CB', '#8B4513', '#87CEEB']
    explode = (0, 0, 0.1, 0)

    ax5.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='',
            shadow=True, startangle=90)
    ax5.set_title('Energy Distribution\n(Dark Brown Hair)', fontsize=12, fontweight='bold')

    # 6. Key findings (bottom)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')

    findings_text = """
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                              KEY FINDINGS & RECOMMENDATIONS                                    ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                ║
║  ✓ CONCLUSION: 630nm LED light DOES penetrate through hair to reach the dermis layer                         ║
║                                                                                                                ║
║  • Transmission Efficiency: ~39% at dermis 500μm depth                                                        ║
║  • Fluence at Target: 0.78 mW/cm² (below 1-5 mW/cm² therapeutic range)                                        ║
║  • Treatment Time: ~64 minutes needed for 3 J/cm² therapeutic dose                                            ║
║                                                                                                                ║
║  RECOMMENDATIONS:                                                                                              ║
║  1. Use higher irradiance LEDs (4-10 mW/cm²) for faster treatment                                             ║
║  2. Part hair to maximize direct skin contact                                                                  ║
║  3. Extend treatment duration for darker hair types                                                            ║
║                                                                                                                ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""
    ax6.text(0.5, 0.5, findings_text, fontsize=11, family='monospace',
             ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Generate all visualization images"""

    print("Loading simulation results...")
    results = load_results()

    print("\nGenerating visualization images...")

    # Generate all charts
    create_depth_profile_chart(results, "depth_profile.png")
    create_hair_comparison_chart(results, "hair_comparison.png")
    create_tissue_diagram(results, "tissue_diagram.png")
    create_summary_infographic(results, "simulation_summary.png")

    print("\n" + "="*50)
    print("All images generated successfully!")
    print("="*50)
    print("\nGenerated files:")
    print("  1. depth_profile.png     - Depth vs transmission bar chart")
    print("  2. hair_comparison.png   - Hair type comparison")
    print("  3. tissue_diagram.png    - Anatomical tissue diagram")
    print("  4. simulation_summary.png - Comprehensive summary infographic")


if __name__ == "__main__":
    main()
