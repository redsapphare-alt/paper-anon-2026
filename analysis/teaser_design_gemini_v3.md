### Design Rationale

*   **Narrative Flow:** The figure is organized into five numbered panels that tell a story, guiding the reader from the problem setup to our proposed solution and validation. The large arrows connecting the panels (①→②→③→④→⑤) make this narrative explicit, ensuring the reader can grasp the paper's logic in seconds.
*   **GEMMAS-style Aesthetics:** The design closely emulates the requested style of GEMMAS (arXiv:2507.13190) by using numbered, rounded-rectangle panels with soft pastel backgrounds, clear titles, and a mix of text and iconic visuals. This creates a modern, clean, and easily digestible infographic.
*   **Visualizing the Core Conflict:** Panel ③ is the figure's centerpiece. It visually presents "The Puzzle" by juxtaposing two distinct types of performance curves (monotonically decreasing vs. inverted-U) that arise from the *same* multi-agent protocol (shown in Panel ②). This contrast immediately establishes the central research question: why do the outcomes diverge?
*   **Simplicity and Clarity:** Complex ideas are distilled into simple visual metaphors. "Knowledge Concentration" is represented as an intuitive gauge (Panel ④), and our proposed solution is a simple flowchart (Panel ⑤). Agent avatars and task icons are minimalist, focusing the viewer's attention on the concepts rather than on decorative detail. All icons are drawn with Matplotlib primitives for maximum portability, avoiding external assets or special fonts.

---

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np
import os

# --- Configuration ---
FIG_WIDTH, FIG_HEIGHT = 14, 10
SAVE_DIR = "paper/latex_v2/figures"
FILE_NAME = "fig0_teaser_v3"
FONT_FAMILY = 'sans-serif'
FONT_WEIGHT = 'normal'
TITLE_FONT_SIZE = 18
PANEL_TITLE_FONT_SIZE = 14
BASE_FONT_SIZE = 11
SMALL_FONT_SIZE = 9

# --- Colors ---
COLORS = {
    'panel_bg': '#F0F4F8',
    'arrow': '#4A5568',
    'text': '#2D3748',
    'title': '#1A202C',
    'panel_border': '#D0D5DD',
    'highlight_red': '#E53E3E',
    'highlight_green': '#38A169',
    'highlight_blue': '#3182CE',
    'domain_med': '#D53F8C', # Pink
    'domain_legal': '#DD6B20', # Orange
    'domain_logic': '#3182CE', # Blue
    'domain_long': '#38A169', # Green
}

# --- Data for the Paper ---
DOMAINS = {
    "Medical QA": {"k": 0.88, "icon": "cross", "color": COLORS['domain_med'], 
                   "q": "What is the first-line treatment for hypertension?",
                   "perf": [0.85, 0.82, 0.80, 0.78]},
    "Legal Classification": {"k": 0.80, "icon": "scales", "color": COLORS['domain_legal'],
                             "q": "Is this contract clause enforceable under CA law?",
                             "perf": [0.90, 0.88, 0.85, 0.84]},
    "Formal Logic": {"k": 0.50, "icon": "phi", "color": COLORS['domain_logic'],
                     "q": "If P→Q and ¬Q, what can we conclude about P?",
                     "perf": [0.70, 0.80, 0.78, 0.75]},
    "Long-context QA": {"k": 0.10, "icon": "scroll", "color": COLORS['domain_long'],
                        "q": "From the text, what was the CEO's main concern in Q3?",
                        "perf": [0.65, 0.75, 0.72, 0.70]},
}

# --- Helper Functions ---

def draw_panel(ax, x, y, w, h, number, title):
    """Draws a rounded rectangle panel with a number and title."""
    panel = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size=0.02",
        facecolor=COLORS['panel_bg'],
        edgecolor=COLORS['panel_border'],
        linewidth=1.5
    )
    ax.add_patch(panel)
    
    # Number circle
    ax.add_patch(mpatches.Circle((x + 0.025, y + h - 0.025), 0.012, color=COLORS['title'], zorder=3))
    ax.text(x + 0.025, y + h - 0.025, str(number), color='white', ha='center', va='center',
            fontsize=PANEL_TITLE_FONT_SIZE, weight='bold', zorder=4)
    
    # Title
    ax.text(x + 0.05, y + h - 0.025, title, color=COLORS['title'], ha='left', va='center',
            fontsize=PANEL_TITLE_FONT_SIZE, weight='bold')

def draw_icon(ax, icon_type, cx, cy, size, color):
    """Draws a geometric icon using matplotlib primitives."""
    if icon_type == "cross":
        ax.add_patch(mpatches.Rectangle((cx - size/2, cy - size/6), size, size/3, color=color))
        ax.add_patch(mpatches.Rectangle((cx - size/6, cy - size/2), size/3, size, color=color))
    elif icon_type == "scales":
        # Base and pillar
        ax.add_patch(mpatches.Rectangle((cx - size/2, cy - size/2), size, size/10, color=color))
        ax.add_patch(mpatches.Rectangle((cx - size/12, cy - size/2), size/6, size/2, color=color))
        # Beam
        ax.add_patch(mpatches.Rectangle((cx - size*0.6, cy), size*1.2, size/12, color=color))
        # Pans
        ax.add_patch(mpatches.Arc((cx - size*0.5, cy - size/4), size/2, size/2, theta1=180, theta2=360, color=color, lw=1.5))
        ax.add_patch(mpatches.Arc((cx + size*0.5, cy - size/4), size/2, size/2, theta1=180, theta2=360, color=color, lw=1.5))
    elif icon_type == "phi":
        ax.add_patch(mpatches.Circle((cx, cy), size/2, facecolor='none', edgecolor=color, lw=2))
        ax.plot([cx, cx], [cy - size/1.8, cy + size/1.8], color=color, lw=2)
    elif icon_type == "scroll":
        # Main paper
        ax.add_patch(mpatches.Rectangle((cx - size/2.5, cy - size/2), size/1.25, size, facecolor='#FEFCE8', edgecolor=color, lw=1))
        # Curled top/bottom
        ax.add_patch(mpatches.Arc((cx - size/2.5, cy + size/2), size/4, size/4, theta1=90, theta2=270, edgecolor=color, lw=1.5, facecolor=COLORS['panel_bg'], zorder=3))
        ax.add_patch(mpatches.Arc((cx - size/2.5, cy - size/2), size/4, size/4, theta1=90, theta2=270, edgecolor=color, lw=1.5, facecolor=COLORS['panel_bg'], zorder=3))
        ax.plot([cx - size/2.5, cx + size/4], [cy + size/2, cy + size/2], color=color, lw=1)
        ax.plot([cx - size/2.5, cx + size/4], [cy - size/2, cy - size/2], color=color, lw=1)

def main():
    # --- Create Figure ---
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['font.weight'] = FONT_WEIGHT

    # --- Panel 1: The Task ---
    p1_x, p1_y, p1_w, p1_h = 0.05, 0.65, 0.43, 0.3
    draw_panel(ax, p1_x, p1_y, p1_w, p1_h, '①', 'Diverse NLP Tasks')
    
    task_card_w, task_card_h = 0.18, 0.11
    positions = [(p1_x + 0.02, p1_y + 0.15), (p1_x + 0.22, p1_y + 0.15),
                 (p1_x + 0.02, p1_y + 0.02), (p1_x + 0.22, p1_y + 0.02)]
    
    for i, (name, data) in enumerate(DOMAINS.items()):
        x, y = positions[i]
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), task_card_w, task_card_h,
            boxstyle="round,pad=0.01",
            facecolor='white', edgecolor=data['color'], linewidth=1.5
        ))
        draw_icon(ax, data['icon'], x + 0.025, y + task_card_h/2, 0.03, data['color'])
        ax.text(x + 0.05, y + 0.085, name, fontsize=BASE_FONT_SIZE, weight='bold', color=COLORS['text'])
        ax.text(x + 0.05, y + 0.04, data['q'], fontsize=SMALL_FONT_SIZE, color=COLORS['text'], wrap=True, va='top',
                bbox=dict(facecolor='none', edgecolor='none', width=100))
        ax.text(x + task_card_w - 0.005, y + 0.01, f"K = {data['k']:.2f}", ha='right', va='bottom',
                fontsize=SMALL_FONT_SIZE, weight='bold', color='white',
                bbox=dict(facecolor=data['color'], edgecolor='none', boxstyle='round,pad=0.3'))

    # --- Panel 2: The Protocol ---
    p2_x, p2_y, p2_w, p2_h = 0.52, 0.65, 0.43, 0.3
    draw_panel(ax, p2_x, p2_y, p2_w, p2_h, '②', 'Standard Multi-Agent Protocol')
    
    center_x, center_y = p2_x + p2_w/2, p2_y + p2_h/2 - 0.02
    radius = 0.08
    agent_colors = ['#EF4444', '#3B82F6', '#10B981', '#F97316']
    for i in range(4):
        angle = np.deg2rad(90 * i + 45)
        ax_x, ax_y = center_x + radius * np.cos(angle), center_y + radius * np.sin(angle)
        ax.add_patch(mpatches.Circle((ax_x, ax_y), 0.02, color=agent_colors[i]))
        ax.text(ax_x, ax_y, f"A{i+1}", ha='center', va='center', color='white', weight='bold', fontsize=BASE_FONT_SIZE)
        # Arrows to others
        for j in range(i + 1, 4):
            angle2 = np.deg2rad(90 * j + 45)
            ax2_x, ax2_y = center_x + radius * np.cos(angle2), center_y + radius * np.sin(angle2)
            arrow = mpatches.FancyArrowPatch((ax_x, ax_y), (ax2_x, ax2_y),
                                             arrowstyle='<->,head_length=4,head_width=3',
                                             connectionstyle="arc3,rad=0.0",
                                             color=COLORS['arrow'], lw=1, shrinkA=20, shrinkB=20)
            ax.add_patch(arrow)
    ax.text(center_x, center_y, "Propose,\nDebate,\nVote", ha='center', va='center',
            fontsize=BASE_FONT_SIZE, style='italic', color=COLORS['text'],
            bbox=dict(facecolor='white', edgecolor=COLORS['panel_border'], boxstyle='circle,pad=0.5'))
    ax.text(p2_x + p2_w/2, p2_y + 0.02, "The same protocol is applied to all tasks.",
            ha='center', va='bottom', fontsize=BASE_FONT_SIZE, color=COLORS['text'])

    # --- Panel 3: The Puzzle ---
    p3_x, p3_y, p3_w, p3_h = 0.05, 0.35, 0.9, 0.25
    draw_panel(ax, p3_x, p3_y, p3_w, p3_h, '③', 'The Puzzle: Performance Diverges')
    
    # Create an inset axes for the plot
    plot_ax = fig.add_axes([p3_x + 0.03, p3_y + 0.03, p3_w - 0.06, p3_h - 0.08])
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
    
    # Annotations
    ax.text(0.3, 0.53, "High-K tasks:\nPerformance decreases", ha='center', color=COLORS['highlight_red'], weight='bold', fontsize=BASE_FONT_SIZE)
    ax.add_patch(mpatches.FancyArrowPatch((0.2, 0.54), (0.2, 0.51), mutation_scale=20, color=COLORS['highlight_red'], arrowstyle='simple'))
    
    ax.text(0.6, 0.4, "Low-K tasks:\nInverted-U shape, peak at N=2", ha='center', color=COLORS['highlight_blue'], weight='bold', fontsize=BASE_FONT_SIZE)
    plot_ax.plot(2, DOMAINS['Formal Logic']['perf'][1], 'o', ms=15, mec=COLORS['highlight_blue'], mfc='none', mew=2)
    plot_ax.plot(2, DOMAINS['Long-context QA']['perf'][1], 'o', ms=15, mec=COLORS['highlight_blue'], mfc='none', mew=2)

    # --- Panel 4: The Explanation ---
    p4_x, p4_y, p4_w, p4_h = 0.05, 0.05, 0.43, 0.25
    draw_panel(ax, p4_x, p4_y, p4_w, p4_h, '④', 'The Explanation: Knowledge Concentration (K)')
    
    # K-meter
    meter_y = p4_y + 0.12
    ax.add_patch(mpatches.Rectangle((p4_x + 0.04, meter_y - 0.01), 0.35, 0.02, color='#E2E8F0'))
    ax.plot([p4_x + 0.04 + 0.35*0.6, p4_x + 0.04 + 0.35*0.6], [meter_y - 0.02, meter_y + 0.02], color=COLORS['highlight_red'], lw=2, linestyle='--')
    ax.text(p4_x + 0.04 + 0.35*0.6, meter_y - 0.03, "τ=0.6", ha='center', va='top', color=COLORS['highlight_red'], fontsize=BASE_FONT_SIZE)
    
    for name, data in DOMAINS.items():
        kx = p4_x + 0.04 + 0.35 * data['k']
        ax.add_patch(mpatches.Circle((kx, meter_y), 0.008, color=data['color'], zorder=5))
        ax.text(kx, meter_y + 0.015, name.split(" ")[0], ha='center', va='bottom', fontsize=SMALL_FONT_SIZE, rotation=30)
    
    ax.text(p4_x + 0.04, meter_y - 0.02, "0.0", ha='center', va='top', fontsize=BASE_FONT_SIZE)
    ax.text(p4_x + 0.39, meter_y - 0.02, "1.0", ha='center', va='top', fontsize=BASE_FONT_SIZE)
    
    ax.text(p4_x + p4_w * 0.7, p4_y + 0.18, "High K: Answer requires\nnarrow, expert knowledge.\n→ One agent is best.",
            ha='center', va='center', fontsize=BASE_FONT_SIZE, color=COLORS['text'])
    ax.text(p4_x + p4_w * 0.3, p4_y + 0.06, "Low K: Answer is in context\nor general knowledge.\n→ Two agents collaborate better.",
            ha='center', va='center', fontsize=BASE_FONT_SIZE, color=COLORS['text'])
    
    # --- Panel 5: The Rule & Result ---
    p5_x, p5_y, p5_w, p5_h = 0.52, 0.05, 0.43, 0.25
    draw_panel(ax, p5_x, p5_y, p5_w, p5_h, '⑤', 'A Simple, Predictive Rule')
    
    # Flowchart
    box_style = "round,pad=0.5"
    arrow_style = "->,head_length=6,head_width=4"
    
    input_box = mpatches.FancyBboxPatch((p5_x + 0.03, p5_y + 0.16), 0.1, 0.05, boxstyle=box_style, fc='#EDF2F7', ec=COLORS['arrow'])
    ax.add_patch(input_box)
    ax.text(p5_x + 0.08, p5_y + 0.185, "New Task", ha='center', va='center', fontsize=BASE_FONT_SIZE)
    
    compute_box = mpatches.FancyBboxPatch((p5_x + 0.165, p5_y + 0.16), 0.1, 0.05, boxstyle=box_style, fc='#EDF2F7', ec=COLORS['arrow'])
    ax.add_patch(compute_box)
    ax.text(p5_x + 0.215, p5_y + 0.185, "Compute K", ha='center', va='center', fontsize=BASE_FONT_SIZE)

    # Diamond shape for decision
    Path = mpath.Path
    path_data = [
        (Path.MOVETO, (0.0, 0.5)),
        (Path.LINETO, (0.5, 1.0)),
        (Path.LINETO, (1.0, 0.5)),
        (Path.LINETO, (0.5, 0.0)),
        (Path.CLOSEPOLY, (0.0, 0.5)),
    ]
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    decision_patch = mpatches.PathPatch(path, facecolor='#FEFCBF', ec=COLORS['arrow'], transform=ax.transAxes,
                                        # Manually scale and translate
                                        offset=(p5_x + 0.29, p5_y + 0.16),
                                        xy=(0,0), width=0.1, height=0.05)
    decision_patch.get_path().vertices *= np.array([0.1, 0.05])
    decision_patch.get_path().vertices += np.array([p5_x + 0.29, p5_y + 0.16])
    ax.add_patch(decision_patch)
    ax.text(p5_x + 0.34, p5_y + 0.185, "K ≥ 0.6?", ha='center', va='center', fontsize=BASE_FONT_SIZE)

    yes_box = mpatches.FancyBboxPatch((p5_x + 0.1, p5_y + 0.08), 0.12, 0.05, boxstyle=box_style, fc=COLORS['highlight_green']+'33', ec=COLORS['highlight_green'])
    ax.add_patch(yes_box)
    ax.text(p5_x + 0.16, p5_y + 0.105, "Deploy 1 Agent", ha='center', va='center', fontsize=BASE_FONT_SIZE)

    no_box = mpatches.FancyBboxPatch((p5_x + 0.21, p5_y + 0.08), 0.12, 0.05, boxstyle=box_style, fc=COLORS['highlight_blue']+'33', ec=COLORS['highlight_blue'])
    ax.add_patch(no_box)
    ax.text(p5_x + 0.27, p5_y + 0.105, "Deploy 2 Agents", ha='center', va='center', fontsize=BASE_FONT_SIZE)

    # Flowchart arrows
    ax.add_patch(mpatches.FancyArrowPatch((p5_x + 0.13, p5_y + 0.185), (p5_x + 0.165, p5_y + 0.185), arrowstyle=arrow_style, color=COLORS['arrow'], mutation_scale=15))
    ax.add_patch(mpatches.FancyArrowPatch((p5_x + 0.265, p5_y + 0.185), (p5_x + 0.29, p5_y + 0.185), arrowstyle=arrow_style, color=COLORS['arrow'], mutation_scale=15))
    ax.add_patch(mpatches.FancyArrowPatch((p5_x + 0.315, p5_y + 0.16), (p5_x + 0.16, p5_y + 0.13), connectionstyle="arc3,rad=0.3", arrowstyle=arrow_style, color=COLORS['arrow'], mutation_scale=15))
    ax.text(p5_x + 0.25, p5_y + 0.14, "Yes", ha='center', va='center', fontsize=SMALL_FONT_SIZE)
    ax.add_patch(mpatches.FancyArrowPatch((p5_x + 0.365, p5_y + 0.16), (p5_x + 0.27, p5_y + 0.13), connectionstyle="arc3,rad=-0.3", arrowstyle=arrow_style, color=COLORS['arrow'], mutation_scale=15))
    ax.text(p5_x + 0.38, p5_y + 0.14, "No", ha='center', va='center', fontsize=SMALL_FONT_SIZE)

    # Result footer
    ax.add_patch(mpatches.Rectangle((p5_x + 0.02, p5_y + 0.015), p5_w - 0.04, 0.04, facecolor='#F7FAFC', edgecolor=COLORS['panel_border']))
    ax.text(p5_x + p5_w/2, p5_y + 0.035, "Validated: Predicts optimal N* for 7 of 8 settings (88% LOO-CV)",
            ha='center', va='center', weight='bold', fontsize=BASE_FONT_SIZE, color=COLORS['title'])

    # --- Inter-panel Arrows ---
    arrow_props = dict(arrowstyle='simple,head_length=15,head_width=15,tail_width=4',
                       color=COLORS['arrow'], alpha=0.5,
                       connectionstyle="arc3,rad=0.1")
    
    ax.add_patch(mpatches.FancyArrowPatch((p1_x + p1_w, p1_y + p1_h/2), (p2_x, p2_y + p2_h/2), **arrow_props))
    ax.add_patch(mpatches.FancyArrowPatch((p2_x + p2_w/2, p2_y), (p3_x + p3_w*0.75, p3_y + p3_h), **arrow_props))
    ax.add_patch(mpatches.FancyArrowPatch((p3_x + p3_w*0.25, p3_y), (p4_x + p4_w/2, p4_y + p4_h), **arrow_props))
    ax.add_patch(mpatches.FancyArrowPatch((p4_x + p4_w, p4_y + p4_h/2), (p5_x, p5_y + p5_h/2), **arrow_props))

    # --- Save Figure ---
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    save_path_pdf = os.path.join(SAVE_DIR, f"{FILE_NAME}.pdf")
    save_path_png = os.path.join(SAVE_DIR, f"{FILE_NAME}.png")
    
    plt.savefig(save_path_pdf, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(save_path_png, bbox_inches='tight', pad_inches=0.1, dpi=300)
    
    print(f"Figure saved to {save_path_pdf}")
    print(f"Figure saved to {save_path_png}")

if __name__ == '__main__':
    main()

```