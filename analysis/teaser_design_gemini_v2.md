Here are the notes on the changes made to the plotting script:

*   **Overall Layout & Text:** The figure size was increased to `(14, 5)`. A main title (suptitle) was added above the panels to state the key takeaway. The "final result" text box was moved from its awkward position into a clean, centered banner at the very bottom of the figure.
*   **Panel 1 (Prevailing View):** The performance curve was reshaped to be more aggressively and obviously monotonic-increasing (`[0.60, 0.72, 0.84, 0.95]`), and its line width was increased to visually distinguish it from the more nuanced findings in Panel 2.
*   **Panel 2 (Our Finding):** The legend was moved to the `upper right` to prevent it from overlapping with the data curves.
*   **Panel 3 (Explanation):** The title was shortened to "3. Explanation: K Predicts Optimal N\*" to prevent it from being clipped. The text labels for the different tasks were staggered vertically (alternating above and below the axis) to resolve the collision between "Legal Class." and "Medical QA".

```python
import matplotlib.pyplot as plt
import numpy as np
import os

def create_teaser_figure_v2():
    """
    Generates the revised teaser figure for the NeurIPS 2026 paper.
    The figure tells a story in three panels, with improved layout and clarity.
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
    FIG_WIDTH = 14
    FIG_HEIGHT = 5
    
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
    # V2: Made 'prevailing' curve more distinct and monotonic-increasing.
    perf_curves = {
        'prevailing': [0.60, 0.72, 0.84, 0.95], # More obviously "more is better"
        'high_k': [0.85, 0.70, 0.60, 0.55],     # Monotonically decreasing
        'low_k': [0.60, 0.85, 0.75, 0.65]       # Inverted-U, peak at N=2
    }

    # --- Figure and Axes Setup ---
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=FIG_BG_COLOR)
    
    # V2: Add a key takeaway suptitle (banner)
    fig.suptitle("More agents can hurt performance; a simple 'Knowledge Concentration' metric predicts when.",
                 fontsize=16, weight='bold', color=TITLE_COLOR, y=0.98)

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
    # V2: Increased linewidth for emphasis
    ax1.plot(N_AGENTS, perf_curves['prevailing'], marker='o', color=NEUTRAL_COLOR, linewidth=3.0, markersize=8)
    ax1.set_xlabel("Number of Agents (N)", fontsize=11, color=TEXT_COLOR)
    ax1.set_ylabel("Accuracy", fontsize=11, color=TEXT_COLOR)
    ax1.set_xticks(N_AGENTS)
    ax1.set_ylim(0.5, 1.0)
    ax1.grid(True, linestyle='--', alpha=0.3, color=AXIS_COLOR)
    ax1.text(2.5, 0.9, '"More agents is all you need"', style='italic',
             ha='center', fontsize=12, color=NEUTRAL_COLOR, weight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc=PANEL_BG_COLOR, ec='none'))

    # --- Panel 2: Our Empirical Finding ---
    ax2.set_title("2. Our Finding: Performance is Task-Dependent", fontsize=14, weight='bold', color=TITLE_COLOR, loc='left')
    ax2.plot(N_AGENTS, perf_curves['high_k'], marker='s', color=HIGH_K_COLOR, linewidth=2.5, markersize=8, label='High-Knowledge Tasks\n(e.g., Legal, Medical)')
    ax2.plot(N_AGENTS, perf_curves['low_k'], marker='^', color=LOW_K_COLOR, linewidth=2.5, markersize=8, label='Low-Knowledge Tasks\n(e.g., Logic, Long-Context)')
    
    ax2.set_xlabel("Number of Agents (N)", fontsize=11, color=TEXT_COLOR)
    ax2.set_ylabel("Accuracy", fontsize=11, color=TEXT_COLOR)
    ax2.set_xticks(N_AGENTS)
    ax2.set_ylim(0.4, 1.0)
    ax2.grid(True, linestyle='--', alpha=0.3, color=AXIS_COLOR)
    # V2: Moved legend to upper right to avoid overlap
    leg = ax2.legend(loc='upper right', fontsize=10, frameon=True, facecolor=FIG_BG_COLOR, edgecolor=AXIS_COLOR)
    for text in leg.get_texts():
        text.set_color(TEXT_COLOR)

    # --- Panel 3: The Explanation & Predictive Rule ---
    # V2: Shortened title to prevent clipping
    ax3.set_title("3. Explanation: K Predicts Optimal N*", fontsize=14, weight='bold', color=TITLE_COLOR, loc='left')
    ax3.set_yticks([])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    for spine in ['left', 'right', 'top']:
        ax3.spines[spine].set_visible(False)
    ax3.spines['bottom'].set_position(('axes', 0.5))
    ax3.spines['bottom'].set_linewidth(1.5)

    ax3.set_xlabel("Knowledge Concentration (K)", fontsize=12, color=TEXT_COLOR, labelpad=10)
    ax3.xaxis.set_label_coords(0.5, 0.35)
    ax3.set_xticks([0, 0.2, 0.4, K_THRESHOLD, 0.8, 1.0])
    ax3.tick_params(axis='x', length=5)
    ax3.get_xticklabels()[3].set_weight('bold')
    ax3.get_xticklabels()[3].set_color(TEXT_COLOR)

    # V2: Stagger labels vertically to avoid overlap
    sorted_tasks = sorted(tasks.items(), key=lambda item: item[1]['k'])
    is_above = True  # Start with the first label above
    for name, data in sorted_tasks:
        color = HIGH_K_COLOR if data['type'] == 'high_k' else LOW_K_COLOR
        ax3.plot(data['k'], 0.5, 'o', ms=12, color=color, markeredgecolor='white', markeredgewidth=1.5)
        
        y_pos = 0.6 if is_above else 0.4
        va = 'bottom' if is_above else 'top'
        
        ax3.text(data['k'], y_pos, f"{name}\n(K={data['k']:.2f})", ha='center', va=va, fontsize=9, color=TEXT_COLOR)
        
        is_above = not is_above # Alternate for the next label

    # Threshold line and regions
    ax3.axvline(K_THRESHOLD, color=TEXT_COLOR, linestyle='--', linewidth=2, ymin=0.1, ymax=0.9)
    ax3.text(K_THRESHOLD, 0.92, r"$\tau=0.6$", ha='center', fontsize=11, color=TEXT_COLOR)

    # Decision rule boxes
    ax3.text(0.3, 0.2, "Optimal N* = 2", ha='center', va='center', fontsize=12, weight='bold', color=LOW_K_COLOR,
             bbox=dict(boxstyle="round,pad=0.5", fc=f"{LOW_K_COLOR}20", ec=LOW_K_COLOR))
    ax3.text(0.8, 0.2, "Optimal N* = 1", ha='center', va='center', fontsize=12, weight='bold', color=HIGH_K_COLOR,
             bbox=dict(boxstyle="round,pad=0.5", fc=f"{HIGH_K_COLOR}20", ec=HIGH_K_COLOR))

    # V2: Moved final result box to a centered banner at the bottom of the figure
    fig.text(0.5, 0.01, "A simple K-threshold rule predicts the optimal agent count for 7 of 8 cases (88% exact match, LOO-CV)",
             ha='center', va='bottom', fontsize=12, weight='bold', color=FIG_BG_COLOR,
             bbox=dict(boxstyle="round,pad=0.5", fc='#444444', ec='none'))
    
    # --- Final Touches & Saving ---
    # V2: Adjust rect to make space for suptitle and bottom banner
    fig.tight_layout(rect=[0, 0.1, 1, 0.92])

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
    create_teaser_figure_v2()
```