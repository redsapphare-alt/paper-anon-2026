My design rationale for this figure is to tell a clear, three-act story from left to right, guiding the reader from the problem to our solution. The first panel establishes the "strawman" of conventional wisdom using a simple, grayed-out monotonic curve. The second, central panel presents our core empirical finding—the surprising diversity of performance curves—using a vibrant, two-color scheme to visually group tasks by their behavior. The final panel delivers the punchline, explaining *why* the curves differ by mapping the tasks onto a single, powerful explanatory axis ("Knowledge Concentration") and presenting our simple, predictive threshold rule, culminating in the key 88% accuracy result.

```python
import matplotlib.pyplot as plt
import numpy as np
import os

def create_teaser_figure():
    """
    Generates the teaser figure for the NeurIPS 2026 paper.
    The figure tells a story in three panels:
    1. The prevailing wisdom (more agents = better).
    2. Our empirical finding (performance is task-dependent).
    3. Our explanation and predictive model (Knowledge Concentration).
    """
    # --- Configuration ---
    # Fonts and Colors
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['text.usetex'] = False
    
    FIG_BG_COLOR = '#FFFFFF'
    PANEL_BG_COLOR = '#F8F8F8'
    TEXT_COLOR = '#333333'
    TITLE_COLOR = '#111111'
    AXIS_COLOR = '#888888'
    
    # Color palette for task types
    HIGH_K_COLOR = '#D95319'  # Warm orange/red for knowledge-bound tasks
    LOW_K_COLOR = '#0072B2'   # Cool blue for open-context/logic tasks
    NEUTRAL_COLOR = '#AAAAAA' # Gray for the "prevailing view"

    # Figure layout
    FIG_WIDTH = 13.5
    FIG_HEIGHT = 4.5
    
    # --- Data Anchors from the Paper ---
    tasks = {
        'Medical QA': {'k': 0.88, 'type': 'high_k', 'icon': '⚕️'},
        'Legal Class.': {'k': 0.80, 'type': 'high_k', 'icon': '⚖️'},
        'Formal Logic': {'k': 0.50, 'type': 'low_k', 'icon': '🧠'},
        'Long-Context': {'k': 0.10, 'type': 'low_k', 'icon': '📜'}
    }
    K_THRESHOLD = 0.6
    N_AGENTS = np.array([1, 2, 3, 4])

    # Representative performance curves (conceptual shapes based on paper text)
    perf_curves = {
        'prevailing': [0.60, 0.70, 0.75, 0.80],
        'high_k': [0.85, 0.70, 0.60, 0.55], # Monotonically decreasing
        'low_k': [0.60, 0.85, 0.75, 0.65]  # Inverted-U, peak at N=2
    }

    # --- Figure and Axes Setup ---
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=FIG_BG_COLOR)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.5], wspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    axes = [ax1, ax2, ax3]
    for ax in axes:
        ax.set_facecolor(PANEL_BG_COLOR)
        ax.tick_params(colors=AXIS_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(AXIS_COLOR)

    # --- Panel 1: The Prevailing View ---
    ax1.set_title("1. The Prevailing View", fontsize=14, weight='bold', color=TITLE_COLOR, loc='left')
    ax1.plot(N_AGENTS, perf_curves['prevailing'], marker='o', color=NEUTRAL_COLOR, linewidth=2.5, markersize=8)
    ax1.set_xlabel("Number of Agents (N)", fontsize=11, color=TEXT_COLOR)
    ax1.set_ylabel("Accuracy", fontsize=11, color=TEXT_COLOR)
    ax1.set_xticks(N_AGENTS)
    ax1.set_ylim(0.5, 0.9)
    ax1.grid(True, linestyle='--', alpha=0.3, color=AXIS_COLOR)
    ax1.text(2.5, 0.82, '"More agents is all you need"', style='italic',
             ha='center', fontsize=12, color=NEUTRAL_COLOR, weight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc=PANEL_BG_COLOR, ec='none'))

    # --- Panel 2: Our Empirical Finding ---
    ax2.set_title("2. Our Finding: Performance is Task-Dependent", fontsize=14, weight='bold', color=TITLE_COLOR, loc='left')
    # High-K curve
    ax2.plot(N_AGENTS, perf_curves['high_k'], marker='s', color=HIGH_K_COLOR, linewidth=2.5, markersize=8, label='High-Knowledge Tasks\n(e.g., Legal, Medical)')
    # Low-K curve
    ax2.plot(N_AGENTS, perf_curves['low_k'], marker='^', color=LOW_K_COLOR, linewidth=2.5, markersize=8, label='Low-Knowledge Tasks\n(e.g., Logic, Long-Context)')
    
    ax2.set_xlabel("Number of Agents (N)", fontsize=11, color=TEXT_COLOR)
    ax2.set_ylabel("Accuracy", fontsize=11, color=TEXT_COLOR)
    ax2.set_xticks(N_AGENTS)
    ax2.set_ylim(0.4, 1.0)
    ax2.grid(True, linestyle='--', alpha=0.3, color=AXIS_COLOR)
    leg = ax2.legend(loc='lower left', fontsize=10, frameon=True, facecolor=FIG_BG_COLOR, edgecolor=AXIS_COLOR)
    for text in leg.get_texts():
        text.set_color(TEXT_COLOR)

    # --- Panel 3: The Explanation & Predictive Rule ---
    ax3.set_title("3. Explanation: Knowledge Concentration Predicts Optimal N*", fontsize=14, weight='bold', color=TITLE_COLOR, loc='left')
    ax3.set_yticks([])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    for spine in ['left', 'right', 'top']:
        ax3.spines[spine].set_visible(False)
    ax3.spines['bottom'].set_position(('axes', 0.5))
    ax3.spines['bottom'].set_linewidth(1.5)

    # K-axis
    ax3.set_xlabel("Knowledge Concentration (K)", fontsize=12, color=TEXT_COLOR, labelpad=10)
    ax3.xaxis.set_label_coords(0.5, 0.35)
    ax3.set_xticks([0, 0.2, 0.4, K_THRESHOLD, 0.8, 1.0])
    ax3.tick_params(axis='x', length=5)
    ax3.get_xticklabels()[3].set_weight('bold')
    ax3.get_xticklabels()[3].set_color(TEXT_COLOR)

    # Plotting tasks on the K-axis
    for name, data in tasks.items():
        color = HIGH_K_COLOR if data['type'] == 'high_k' else LOW_K_COLOR
        ax3.plot(data['k'], 0.5, 'o', ms=12, color=color, markeredgecolor='white', markeredgewidth=1.5)
        ax3.text(data['k'], 0.6, f"{name}\n(K={data['k']:.2f})", ha='center', va='bottom', fontsize=9, color=TEXT_COLOR)

    # Threshold line and regions
    ax3.axvline(K_THRESHOLD, color=TEXT_COLOR, linestyle='--', linewidth=2, ymin=0.1, ymax=0.9)
    ax3.text(K_THRESHOLD, 0.92, r"$\tau=0.6$", ha='center', fontsize=11, color=TEXT_COLOR)

    # Decision rule boxes
    ax3.text(0.3, 0.2, "Optimal N* = 2", ha='center', va='center', fontsize=12, weight='bold', color=LOW_K_COLOR,
             bbox=dict(boxstyle="round,pad=0.5", fc=f"{LOW_K_COLOR}20", ec=LOW_K_COLOR))
    ax3.text(0.8, 0.2, "Optimal N* = 1", ha='center', va='center', fontsize=12, weight='bold', color=HIGH_K_COLOR,
             bbox=dict(boxstyle="round,pad=0.5", fc=f"{HIGH_K_COLOR}20", ec=HIGH_K_COLOR))

    # Final result box
    fig.text(0.78, 0.05, "A simple K-threshold rule predicts the optimal agent count\nfor 7 of 8 cases (88% exact match, LOO-CV)",
             ha='center', fontsize=12, weight='bold', color=FIG_BG_COLOR,
             bbox=dict(boxstyle="round,pad=0.5", fc='#444444', ec='none'))
    
    # --- Final Touches & Saving ---
    fig.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust for bottom text box

    # Create output directory if it doesn't exist
    output_dir = "paper/latex_v2/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    pdf_path = os.path.join(output_dir, "fig0_teaser_v2.pdf")
    png_path = os.path.join(output_dir, "fig0_teaser_v2.png")
    
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    
    print(f"Figure saved to {pdf_path}")
    print(f"Figure saved to {png_path}")


if __name__ == '__main__':
    create_teaser_figure()
```