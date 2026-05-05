Here is a summary of the four main categories of changes made to the Python script to produce the improved V4 figure:

*   **Layout and Composition:** The figure was restructured using `matplotlib.gridspec` for a robust 2x2 layout with a spanning middle row. A main title banner and a separate footer caption were added, and panel background colors were changed to create a narrative flow (setup, problem, solution).
*   **Clarity and Readability:** Text in Panel ① was shortened to prevent overflow. The decision flowchart in Panel ⑤ was completely redesigned with larger, clearer boxes, thicker arrows, and explicit "Yes/No" labels to be easily understood at a glance.
*   **Diagram Redesign:** The multi-agent protocol in Panel ② was revised to show four agents in a fully connected debate phase, with a new downward arrow leading to a distinct "Vote" step. The K-gauge in Panel ④ was redesigned as a clean horizontal gradient bar with a threshold line and non-overlapping labels connected by leader lines.
*   **Code Modernization:** The script was updated to use more modern Matplotlib practices, such as `ax.inset_axes` for the plot in Panel ③ instead of manually calculating positions with `fig.add_axes`, making the code cleaner and more maintainable.

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.gridspec as gridspec
import numpy as np
import os

# --- Configuration ---
FIG_WIDTH, FIG_HEIGHT = 14, 9
SAVE_DIR = "paper/latex_v2/figures"
FILE_NAME = "fig0_teaser_v4"
FONT_FAMILY = 'sans-serif'
FONT_WEIGHT = 'normal'
TITLE_FONT_SIZE = 20
PANEL_TITLE_FONT_SIZE = 15
BASE_FONT_SIZE = 11
SMALL_FONT_SIZE = 9

# --- Colors ---
COLORS = {
    'panel_bg_setup': '#F8F9FA',    # White-ish for setup (Panels 1, 2)
    'panel_bg_problem': '#FFFBEB', # Light yellow for the puzzle (Panel 3)
    'panel_bg_solution': '#F0FFF4',# Light green for the solution (Panels 4, 5)
    'arrow': '#4A5568',
    'text': '#2D3748',
    'title': '#1A202C',
    'panel_border': '#A0AEC0',
    'highlight_red': '#E53E3E',
    'highlight_green': '#38A169',
    'highlight_blue': '#3182CE',
    'domain_med': '#D53F8C', # Pink
    'domain_legal': '#DD6B20', # Orange
    'domain_logic': '#3182CE', # Blue
    'domain_long': '#38A169', # Green
}

# --- Data for the Paper (with shortened questions) ---
DOMAINS = {
    "Medical QA": {"k": 0.88, "icon": "cross", "color": COLORS['domain_med'], 
                   "q": "First-line treatment for hypertension?",
                   "perf": [0.85, 0.82, 0.80, 0.78]},
    "Legal Classification": {"k": 0.80, "icon": "scales", "color": COLORS['domain_legal'],
                             "q": "Is this contract clause enforceable?",
                             "perf": [0.90, 0.88, 0.85, 0.84]},
    "Formal Logic": {"k": 0.50, "icon": "phi", "color": COLORS['domain_logic'],
                     "q": "If P→Q and ¬Q, what is P?",
                     "perf": [0.70, 0.80, 0.78, 0.75]},
    "Long-context QA": {"k": 0.10, "icon": "scroll", "color": COLORS['domain_long'],
                        "q": "What was the CEO's main concern in Q3?",
                        "perf": [0.65, 0.75, 0.72, 0.70]},
}

# --- Helper Functions ---

def draw_panel(ax, number, title, bg_color):
    """Draws a rounded rectangle panel with a number and title on a given axis."""
    ax.set_facecolor(bg_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS['panel_border'])
        spine.set_linewidth(1.5)
        spine.set_visible(True)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Use FancyBboxPatch for rounded corners effect inside the axes
    ax.add_patch(mpatches.FancyBboxPatch(
        (-0.01, -0.01), 1.02, 1.02,
        boxstyle="round,pad=0,rounding_size=0.02",
        facecolor='none',
        edgecolor=COLORS['panel_border'],
        linewidth=1.5,
        clip_on=False,
        transform=ax.transAxes
    ))

    # Number circle
    ax.add_patch(mpatches.Circle((0.04, 0.94), 0.025, color=COLORS['title'], zorder=3, transform=ax.transAxes, clip_on=False))
    ax.text(0.04, 0.94, str(number), color='white', ha='center', va='center',
            fontsize=PANEL_TITLE_FONT_SIZE, weight='bold', zorder=4, transform=ax.transAxes)
    
    # Title
    ax.text(0.08, 0.94, title, color=COLORS['title'], ha='left', va='center',
            fontsize=PANEL_TITLE_FONT_SIZE, weight='bold', transform=ax.transAxes)

def draw_icon(ax, icon_type, cx, cy, size, color):
    """Draws a geometric icon using matplotlib primitives."""
    if icon_type == "cross":
        ax.add_patch(mpatches.Rectangle((cx - size/2, cy - size/6), size, size/3, color=color, transform=ax.transAxes))
        ax.add_patch(mpatches.Rectangle((cx - size/6, cy - size/2), size/3, size, color=color, transform=ax.transAxes))
    elif icon_type == "scales":
        ax.add_patch(mpatches.Rectangle((cx - size/2, cy - size/2), size, size/10, color=color, transform=ax.transAxes))
        ax.add_patch(mpatches.Rectangle((cx - size/12, cy - size/2), size/6, size/2, color=color, transform=ax.transAxes))
        ax.add_patch(mpatches.Rectangle((cx - size*0.6, cy), size*1.2, size/12, color=color, transform=ax.transAxes))
        ax.add_patch(mpatches.Arc((cx - size*0.5, cy - size/4), size/2, size/2, theta1=180, theta2=360, color=color, lw=1.5, transform=ax.transAxes))
        ax.add_patch(mpatches.Arc((cx + size*0.5, cy - size/4), size/2, size/2, theta1=180, theta2=360, color=color, lw=1.5, transform=ax.transAxes))
    elif icon_type == "phi":
        ax.add_patch(mpatches.Circle((cx, cy), size/2, facecolor='none', edgecolor=color, lw=2, transform=ax.transAxes))
        ax.plot([cx, cx], [cy - size/1.8, cy + size/1.8], color=color, lw=2, transform=ax.transAxes)
    elif icon_type == "scroll":
        ax.add_patch(mpatches.Rectangle((cx - size/2.5, cy - size/2), size/1.25, size, facecolor='#FEFCE8', edgecolor=color, lw=1, transform=ax.transAxes))
        ax.add_patch(mpatches.Arc((cx - size/2.5, cy + size/2), size/4, size/4, theta1=90, theta2=270, edgecolor=color, lw=1.5, facecolor=COLORS['panel_bg_setup'], zorder=3, transform=ax.transAxes))
        ax.add_patch(mpatches.Arc((cx - size/2.5, cy - size/2), size/4, size/4, theta1=90, theta2=270, edgecolor=color, lw=1.5, facecolor=COLORS['panel_bg_setup'], zorder=3, transform=ax.transAxes))
        ax.plot([cx - size/2.5, cx + size/4], [cy + size/2, cy + size/2], color=color, lw=1, transform=ax.transAxes)
        ax.plot([cx - size/2.5, cx + size/4], [cy - size/2, cy - size/2], color=color, lw=1, transform=ax.transAxes)

def main():
    # --- Create Figure using GridSpec ---
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.suptitle("More Agents Isn't Always Better — A Task-Dependent Theory", 
                 fontsize=TITLE_FONT_SIZE, weight='bold', color=COLORS['title'], y=0.98)
    
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[0.4, 0.3, 0.3],
                           wspace=0.15, hspace=0.25, top=0.9, bottom=0.08, left=0.05, right=0.95)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['font.weight'] = FONT_WEIGHT

    # --- Panel 1: The Task ---
    draw_panel(ax1, '①', 'Diverse NLP Tasks', COLORS['panel_bg_setup'])
    
    task_card_w, task_card_h = 0.4, 0.3
    positions = [(0.08, 0.52), (0.52, 0.52), (0.08, 0.15), (0.52, 0.15)]
    
    for i, (name, data) in enumerate(DOMAINS.items()):
        x, y = positions[i]
        ax1.add_patch(mpatches.FancyBboxPatch(
            (x, y), task_card_w, task_card_h,
            boxstyle="round,pad=0.01", facecolor='white', edgecolor=data['color'], 
            linewidth=1.5, transform=ax1.transAxes
        ))
        draw_icon(ax1, data['icon'], x + 0.06, y + task_card_h/2, 0.08, data['color'])
        ax1.text(x + 0.12, y + 0.23, name, fontsize=BASE_FONT_SIZE, weight='bold', color=COLORS['text'], transform=ax1.transAxes)
        ax1.text(x + 0.12, y + 0.15, data['q'], fontsize=SMALL_FONT_SIZE, color=COLORS['text'], wrap=True, va='top', transform=ax1.transAxes,
                 bbox=dict(boxstyle='square,pad=0', fc='none', ec='none', width=25, height=10))
        ax1.text(x + task_card_w - 0.01, y + 0.03, f"K = {data['k']:.2f}", ha='right', va='bottom',
                 fontsize=SMALL_FONT_SIZE, weight='bold', color='white', transform=ax1.transAxes,
                 bbox=dict(facecolor=data['color'], edgecolor='none', boxstyle='round,pad=0.3'))

    # --- Panel 2: The Protocol ---
    draw_panel(ax2, '②', 'Standard Multi-Agent Protocol', COLORS['panel_bg_setup'])
    
    center_x, center_y = 0.5, 0.6
    radius = 0.2
    agent_colors = ['#EF4444', '#3B82F6', '#10B981', '#F97316']
    agent_pos = []
    for i in range(4):
        angle = np.deg2rad(90 * i + 45)
        ax_x, ax_y = center_x + radius * np.cos(angle), center_y + radius * np.sin(angle)
        agent_pos.append((ax_x, ax_y))
        ax2.add_patch(mpatches.Circle((ax_x, ax_y), 0.05, color=agent_colors[i], transform=ax2.transAxes))
        ax2.text(ax_x, ax_y, f"A{i+1}", ha='center', va='center', color='white', weight='bold', fontsize=BASE_FONT_SIZE, transform=ax2.transAxes)
    
    for i in range(4):
        for j in range(i + 1, 4):
            arrow = mpatches.FancyArrowPatch(agent_pos[i], agent_pos[j],
                                             arrowstyle='<->,head_length=4,head_width=3',
                                             connectionstyle="arc3,rad=0.0",
                                             color=COLORS['arrow'], lw=1, shrinkA=8, shrinkB=8, transform=ax2.transAxes)
            ax2.add_patch(arrow)
    
    ax2.text(center_x, center_y, "Propose,\nDebate", ha='center', va='center', fontsize=BASE_FONT_SIZE, style='italic', color=COLORS['text'], transform=ax2.transAxes)
    ax2.add_patch(mpatches.FancyArrowPatch((0.5, 0.45), (0.5, 0.3), arrowstyle='simple,head_length=8,head_width=8,tail_width=3', color=COLORS['arrow'], transform=ax2.transAxes))
    ax2.add_patch(mpatches.FancyBboxPatch((0.35, 0.18), 0.3, 0.12, boxstyle="round,pad=0.3", fc='#EDF2F7', ec=COLORS['arrow'], transform=ax2.transAxes))
    ax2.text(0.5, 0.24, "Vote", ha='center', va='center', fontsize=BASE_FONT_SIZE, weight='bold', color=COLORS['text'], transform=ax2.transAxes)

    # --- Panel 3: The Puzzle ---
    draw_panel(ax3, '③', 'The Puzzle: Performance Diverges', COLORS['panel_bg_problem'])
    
    plot_ax = ax3.inset_axes([0.05, 0.1, 0.9, 0.75])
    N = [1, 2, 3, 4]
    for name, data in DOMAINS.items():
        plot_ax.plot(N, data['perf'], marker='o', linestyle='-', label=f"{name} (K={data['k']:.2f})", color=data['color'], lw=2.5)
    
    plot_ax.set_xlabel("Number of Agents (N)", fontsize=BASE_FONT_SIZE)
    plot_ax.set_ylabel("Accuracy", fontsize=BASE_FONT_SIZE)
    plot_ax.set_xticks(N)
    plot_ax.set_ylim(0.6, 1.0)
    plot_ax.grid(True, linestyle='--', alpha=0.5)
    plot_ax.legend(loc='upper right', fontsize=SMALL_FONT_SIZE, frameon=True, facecolor='white', framealpha=0.8)
    plot_ax.spines['top'].set_visible(False)
    plot_ax.spines['right'].set_visible(False)
    
    ax3.text(0.25, 0.7, "High-K tasks:\nPerformance decreases", ha='center', color=COLORS['highlight_red'], weight='bold', fontsize=BASE_FONT_SIZE, transform=ax3.transAxes)
    ax3.add_patch(mpatches.FancyArrowPatch((0.18, 0.7), (0.18, 0.6), mutation_scale=20, color=COLORS['highlight_red'], arrowstyle='simple', transform=ax3.transAxes))
    
    ax3.text(0.55, 0.2, "Low-K tasks:\nInverted-U shape, peak at N=2", ha='center', color=COLORS['highlight_blue'], weight='bold', fontsize=BASE_FONT_SIZE, transform=ax3.transAxes)
    plot_ax.plot(2, DOMAINS['Formal Logic']['perf'][1], 'o', ms=15, mec=COLORS['highlight_blue'], mfc='none', mew=2)
    plot_ax.plot(2, DOMAINS['Long-context QA']['perf'][1], 'o', ms=15, mec=COLORS['highlight_blue'], mfc='none', mew=2)

    # --- Panel 4: The Explanation ---
    draw_panel(ax4, '④', 'The Explanation: Knowledge Concentration (K)', COLORS['panel_bg_solution'])
    
    # K-meter gradient bar
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax4.imshow(gradient, aspect='auto', extent=[0.1, 0.9, 0.6, 0.68], cmap='coolwarm', transform=ax4.transAxes)
    ax4.plot([0.1 + 0.8*0.6, 0.1 + 0.8*0.6], [0.55, 0.73], color=COLORS['highlight_red'], lw=2, linestyle='--', transform=ax4.transAxes)
    ax4.text(0.1 + 0.8*0.6, 0.53, "τ=0.6", ha='center', va='top', color=COLORS['highlight_red'], fontsize=BASE_FONT_SIZE, transform=ax4.transAxes)
    
    stagger_y = [0.8, 0.4, 0.45, 0.85]
    for i, (name, data) in enumerate(DOMAINS.items()):
        kx = 0.1 + 0.8 * data['k']
        ax4.plot([kx, kx], [0.6, 0.68], color='black', lw=2, transform=ax4.transAxes)
        ax4.plot([kx, kx], [0.68, stagger_y[i]-0.03], color=data['color'], lw=1, ls=':', transform=ax4.transAxes)
        ax4.text(kx, stagger_y[i], f"{name.split(' ')[0]}\nK={data['k']:.2f}", ha='center', va='bottom', fontsize=SMALL_FONT_SIZE, color=data['color'], weight='bold', transform=ax4.transAxes)
    
    ax4.text(0.1, 0.58, "0.0", ha='center', va='top', fontsize=BASE_FONT_SIZE, transform=ax4.transAxes)
    ax4.text(0.9, 0.58, "1.0", ha='center', va='top', fontsize=BASE_FONT_SIZE, transform=ax4.transAxes)
    
    ax4.text(0.75, 0.25, "High K (K ≥ 0.6)\nAnswer is concentrated.\nOne expert agent is best.", ha='center', va='center', fontsize=BASE_FONT_SIZE, color=COLORS['text'], transform=ax4.transAxes)
    ax4.text(0.25, 0.25, "Low K (K < 0.6)\nAnswer is distributed.\nCollaboration is better.", ha='center', va='center', fontsize=BASE_FONT_SIZE, color=COLORS['text'], transform=ax4.transAxes)

    # --- Panel 5: The Rule & Result ---
    draw_panel(ax5, '⑤', 'A Simple, Predictive Rule', COLORS['panel_bg_solution'])
    
    box_style = "round,pad=0.5"
    arrow_style = "->,head_length=8,head_width=6"
    
    # [New Task] -> [Compute K]
    ax5.text(0.18, 0.7, "New Task", ha='center', va='center', fontsize=BASE_FONT_SIZE, bbox=dict(boxstyle=box_style, fc='#EDF2F7', ec=COLORS['arrow']), transform=ax5.transAxes)
    ax5.text(0.5, 0.7, "Compute K", ha='center', va='center', fontsize=BASE_FONT_SIZE, bbox=dict(boxstyle=box_style, fc='#EDF2F7', ec=COLORS['arrow']), transform=ax5.transAxes)
    ax5.add_patch(mpatches.FancyArrowPatch((0.28, 0.7), (0.4, 0.7), arrowstyle=arrow_style, color=COLORS['arrow'], mutation_scale=20, transform=ax5.transAxes))
    
    # -> [K >= 0.6?]
    diamond_verts = np.array([[0.75, 0.78], [0.83, 0.7], [0.75, 0.62], [0.67, 0.7], [0.75, 0.78]])
    ax5.add_patch(mpatches.Polygon(diamond_verts, facecolor='#FEFCBF', ec=COLORS['arrow'], transform=ax5.transAxes))
    ax5.text(0.75, 0.7, "K ≥ 0.6?", ha='center', va='center', fontsize=BASE_FONT_SIZE, transform=ax5.transAxes)
    ax5.add_patch(mpatches.FancyArrowPatch((0.6, 0.7), (0.67, 0.7), arrowstyle=arrow_style, color=COLORS['arrow'], mutation_scale=20, transform=ax5.transAxes))

    # -> [1 Agent] (Yes)
    ax5.text(0.75, 0.25, "Deploy 1 Agent", ha='center', va='center', fontsize=BASE_FONT_SIZE, weight='bold',
             bbox=dict(boxstyle=box_style, fc=COLORS['highlight_red']+'33', ec=COLORS['highlight_red']), transform=ax5.transAxes)
    ax5.add_patch(mpatches.FancyArrowPatch((0.75, 0.62), (0.75, 0.35), arrowstyle=arrow_style, color=COLORS['arrow'], mutation_scale=20, transform=ax5.transAxes))
    ax5.text(0.79, 0.5, "Yes", ha='center', va='center', fontsize=SMALL_FONT_SIZE, color=COLORS['text'], transform=ax5.transAxes)

    # -> [2 Agents] (No)
    ax5.text(0.25, 0.25, "Deploy 2 Agents", ha='center', va='center', fontsize=BASE_FONT_SIZE, weight='bold',
             bbox=dict(boxstyle=box_style, fc=COLORS['highlight_blue']+'33', ec=COLORS['highlight_blue']), transform=ax5.transAxes)
    ax5.add_patch(mpatches.FancyArrowPatch((0.67, 0.7), (0.5, 0.5), connectionstyle="arc3,rad=0.3", arrowstyle=arrow_style, color=COLORS['arrow'], mutation_scale=20, transform=ax5.transAxes))
    ax5.add_patch(mpatches.FancyArrowPatch((0.5, 0.5), (0.25, 0.35), connectionstyle="arc3,rad=-0.0", arrowstyle=arrow_style, color=COLORS['arrow'], mutation_scale=20, transform=ax5.transAxes))
    ax5.text(0.55, 0.55, "No", ha='center', va='center', fontsize=SMALL_FONT_SIZE, color=COLORS['text'], transform=ax5.transAxes)

    # --- Footer Caption ---
    fig.text(0.5, 0.02, "Validated: 88% LOO-CV accuracy (7 of 8 trustable cells).",
             ha='center', va='bottom', weight='bold', fontsize=BASE_FONT_SIZE, color=COLORS['title'])

    # --- Save Figure ---
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    save_path_pdf = os.path.join(SAVE_DIR, f"{FILE_NAME}.pdf")
    save_path_png = os.path.join(SAVE_DIR, f"{FILE_NAME}.png")
    
    plt.savefig(save_path_pdf, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(save_path_png, bbox_inches='tight', pad_inches=0.1, dpi=300)
    
    print(f"Figure saved to {save_path_pdf}")
    print(f"Figure saved to {save_path_png}")
    
    # plt.show() # Uncomment for interactive display

if __name__ == '__main__':
    main()
```