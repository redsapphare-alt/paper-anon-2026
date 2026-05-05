"""draw.io teaser figure에 들어갈 보조 자산 생성.

생성 자산:
  insets/curve_medical.png      — Medical 곡선 (단조 감소, K=0.88)
  insets/curve_legal.png        — Legal 곡선 (단조 감소, 가장 가파름, K=0.80)
  insets/curve_logic.png        — Logic 곡선 (inverted-U at N=2, K=0.50)
  insets/curve_longcontext.png  — Long-context 곡선 (inverted-U at N=2, K=0.10)
  insets/k_gauge.png            — K gauge bar (gradient + 4 도메인 marker + τ=0.6 line)
  insets/k_gauge.svg            — 동일 (벡터)
  insets/agents_protocol.png    — 4 agent + arrow + Vote 박스 (panel ②용 대체)
  insets/icon_medical.png       — 의료 아이콘 (적십자)
  insets/icon_legal.png         — 법률 아이콘 (저울)
  insets/icon_logic.png         — 논리 아이콘 (Φ 그리스 글자 in 원)
  insets/icon_longcontext.png   — long-context 아이콘 (책/문서)

사용 방법:
  1) draw.io에서 fig0 teaser 열기
  2) 미니 curve 영역을 클릭해서 선택 → Delete (기존 점-선 4개 제거)
  3) 메뉴 Edit → Insert → Image → 파일 업로드 → 해당 PNG 선택
  4) 적당한 크기로 조정 후 카드 안에 배치
  5) K gauge는 panel 3 전체 자리에 insets/k_gauge.png를 배치하면 깔끔
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'paper' / 'figures_design' / 'insets'
OUT.mkdir(parents=True, exist_ok=True)

# Color palette (consistent with the rest of the paper)
C_HIGH_K = '#D95319'  # warm orange-red
C_LOW_K = '#0072B2'   # cool blue
C_TAU = '#9C27B0'     # purple
C_NEUTRAL = '#444444'

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.transparent': True,
})


# ============================================================
# Mini curves — one per domain
# ============================================================
def draw_mini_curve(filename, accs, peak_idx, color, kind='decreasing',
                     title_inset=None, n_star=None):
    """Render a small curve thumbnail at high DPI for embedding in draw.io.

    accs: 4-tuple of accuracies (illustrative; just the shape)
    peak_idx: 0..3 — which agent count is the optimum (for marker)
    title_inset: short ASCII label rendered top-left
    n_star: int — overlay a 'N* = N' badge near the peak
    """
    fig, ax = plt.subplots(figsize=(3.2, 1.3))
    xs = np.array([1, 2, 3, 4])
    ys = np.array(accs)
    # smoothed line via cubic spline for visual elegance
    from scipy.interpolate import make_interp_spline
    if len(set(accs)) > 1:
        spl = make_interp_spline(xs, ys, k=3)
        xx = np.linspace(1, 4, 100)
        yy = spl(xx)
        ax.plot(xx, yy, color=color, linewidth=2.6, alpha=0.85, solid_capstyle='round')
    ax.plot(xs, ys, 'o', color=color, markersize=8,
             markeredgecolor='white', markeredgewidth=1.5, zorder=4)
    # mark the optimum with a ring
    ax.scatter(xs[peak_idx], ys[peak_idx], s=240, facecolor='none',
                edgecolor=color, linewidth=2.5, zorder=5)
    # baseline labels
    ax.set_xticks(xs)
    ax.set_xticklabels(['1', '2', '3', '4'], fontsize=8, color='#666')
    ax.set_yticks([])
    # tighten ylim
    margin = (max(ys) - min(ys)) * 0.5 + 0.06
    ax.set_ylim(min(ys) - margin, max(ys) + margin * 1.4)
    ax.set_xlim(0.7, 4.3)
    # remove all spines except bottom
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#aaa')
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='x', length=2, color='#aaa', pad=2)
    # ASCII-only kind label, top-left
    if title_inset:
        ax.text(0.02, 0.97, title_inset, transform=ax.transAxes,
                 fontsize=9, color=color, fontstyle='italic',
                 ha='left', va='top', fontweight='bold')
    # N* badge
    if n_star is not None:
        ax.text(0.98, 0.97, f'N* = {n_star}', transform=ax.transAxes,
                 fontsize=9, color='white', ha='right', va='top', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', fc=color, ec=color))
    plt.savefig(OUT / filename, transparent=True)
    plt.close()
    print(f'  Wrote {OUT / filename}')


print('=== Mini curves ===')
draw_mini_curve('curve_medical.png',
                 accs=[0.29, 0.27, 0.27, 0.26], peak_idx=0,
                 color=C_HIGH_K, kind='decreasing',
                 title_inset='decreasing', n_star=1)
draw_mini_curve('curve_legal.png',
                 accs=[0.88, 0.78, 0.72, 0.72], peak_idx=0,
                 color=C_HIGH_K, kind='decreasing',
                 title_inset='decreasing', n_star=1)
draw_mini_curve('curve_logic.png',
                 accs=[0.20, 0.30, 0.22, 0.20], peak_idx=1,
                 color=C_LOW_K, kind='inverted_U',
                 title_inset='inverted-U', n_star=2)
draw_mini_curve('curve_longcontext.png',
                 accs=[0.03, 0.20, 0.07, 0.02], peak_idx=1,
                 color=C_LOW_K, kind='inverted_U',
                 title_inset='inverted-U', n_star=2)


# ============================================================
# K gauge — clean horizontal bar with markers
# ============================================================
print('\n=== K gauge ===')
fig, ax = plt.subplots(figsize=(11, 2.0))
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-1, 1.5)
ax.axis('off')

# gradient bar
n_steps = 400
bar_y = -0.05
bar_h = 0.30
for i in range(n_steps):
    t = i / (n_steps - 1)
    r1, g1, b1 = 0x00/255, 0x72/255, 0xB2/255
    r2, g2, b2 = 0xD9/255, 0x53/255, 0x19/255
    r, g, b = r1 + t*(r2-r1), g1 + t*(g2-g1), b1 + t*(b2-b1)
    ax.add_patch(mpatches.Rectangle((i/n_steps, bar_y), 1/n_steps + 0.001, bar_h,
                                      facecolor=(r,g,b), edgecolor='none'))
ax.add_patch(mpatches.Rectangle((0, bar_y), 1.0, bar_h, facecolor='none',
                                  edgecolor='#444', linewidth=1.2))

# K=0 / K=1 endpoints
ax.text(-0.02, bar_y + bar_h/2, 'K = 0', ha='right', va='center', fontsize=12, fontweight='bold')
ax.text(1.02, bar_y + bar_h/2, 'K = 1', ha='left', va='center', fontsize=12, fontweight='bold')

# τ=0.6 dashed vertical line — explicit segments so dashes always render
tau_x = 0.6
tau_top = bar_y + bar_h + 0.35
tau_bot = bar_y - 0.35
seg_len = 0.04
gap = 0.025
y = tau_bot
while y < tau_top - 1e-6:
    y2 = min(y + seg_len, tau_top)
    ax.plot([tau_x, tau_x], [y, y2], color=C_TAU, linewidth=3.0, solid_capstyle='butt')
    y = y2 + gap
ax.text(tau_x, bar_y + bar_h + 0.50, 'τ = 0.6',
         ha='center', va='bottom', fontsize=13, fontweight='bold', color=C_TAU)

# Domain markers — staggered above/below to avoid label collision
DOMAINS = [
    ('Long-context', 0.10, C_LOW_K, 'below'),
    ('Formal Logic', 0.50, C_LOW_K, 'above'),
    ('Legal',        0.80, C_HIGH_K, 'below'),
    ('Medical',      0.88, C_HIGH_K, 'above'),
]
for name, k, color, side in DOMAINS:
    px = k
    # marker dot
    ax.add_patch(mpatches.Circle((px, bar_y + bar_h/2), 0.018,
                                   facecolor=color, edgecolor='white', linewidth=2, zorder=4))
    if side == 'above':
        ty = bar_y + bar_h + 0.18
        ax.plot([px, px], [bar_y + bar_h, ty - 0.02], color=color, linewidth=1, alpha=0.5)
        ax.text(px, ty, f'{name}\n(K = {k:.2f})', ha='center', va='bottom',
                 fontsize=10, fontweight='bold', color=color)
    else:
        ty = bar_y - 0.18
        ax.plot([px, px], [bar_y, ty + 0.02], color=color, linewidth=1, alpha=0.5)
        ax.text(px, ty, f'{name}\n(K = {k:.2f})', ha='center', va='top',
                 fontsize=10, fontweight='bold', color=color)

# Side captions
ax.text(0.30, -0.78, 'Low K  (open-context / general):  collaboration helps',
         ha='center', va='center', fontsize=11, color=C_LOW_K, fontweight='bold')
ax.text(0.85, -0.78, 'High K  (knowledge-bound):  single expert wins',
         ha='center', va='center', fontsize=11, color=C_HIGH_K, fontweight='bold')

plt.savefig(OUT / 'k_gauge.png', transparent=True)
plt.savefig(OUT / 'k_gauge.svg', transparent=True)
plt.close()
print(f'  Wrote {OUT / "k_gauge.png"}')
print(f'  Wrote {OUT / "k_gauge.svg"}')


# ============================================================
# Multi-agent protocol diagram
# ============================================================
print('\n=== Multi-agent protocol diagram ===')
fig, ax = plt.subplots(figsize=(7, 4))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# 4 agent circles in a row
agent_colors = ['#EF4444', '#3B82F6', '#10B981', '#F59E0B']
agent_x = [0.18, 0.40, 0.62, 0.84]
agent_y = 0.70
for i, (cx, color) in enumerate(zip(agent_x, agent_colors)):
    ax.add_patch(mpatches.Circle((cx, agent_y), 0.07, facecolor=color,
                                   edgecolor='white', linewidth=2.5))
    ax.text(cx, agent_y, f'A{i+1}', ha='center', va='center',
             color='white', fontsize=14, fontweight='bold')

# Arrows between adjacent agents (left-right)
for i in range(3):
    ax.annotate('', xy=(agent_x[i+1] - 0.075, agent_y),
                xytext=(agent_x[i] + 0.075, agent_y),
                arrowprops=dict(arrowstyle='<->', color='#888', lw=1.5))

# Long arc A1 ↔ A4 (collaboration)
ax.annotate('', xy=(agent_x[3], agent_y + 0.07),
            xytext=(agent_x[0], agent_y + 0.07),
            arrowprops=dict(arrowstyle='<->', color='#aaa', lw=1.0,
                             connectionstyle='arc3,rad=-0.5', linestyle=(0, (3, 3))))

# "Propose · Debate" caption
ax.text(0.5, 0.92, 'Propose · Debate', ha='center', va='center',
         fontsize=12, color='#666', style='italic')

# Down arrow to vote box
ax.annotate('', xy=(0.5, 0.36), xytext=(0.5, 0.62),
            arrowprops=dict(arrowstyle='->', color='#444', lw=2.5,
                             mutation_scale=22))

# Vote box
ax.add_patch(mpatches.FancyBboxPatch((0.30, 0.18), 0.40, 0.16,
                                       boxstyle='round,pad=0.02', facecolor='#EDF2F7',
                                       edgecolor='#444', linewidth=2))
ax.text(0.5, 0.26, 'Vote → Final answer', ha='center', va='center',
         fontsize=14, fontweight='bold', color='#222')

# Footer caption
ax.text(0.5, 0.06,
         'The same protocol is applied to every task.\nThe only thing we vary is N (number of agents).',
         ha='center', va='center', fontsize=10, style='italic', color='#666')

plt.savefig(OUT / 'agents_protocol.png', transparent=True)
plt.close()
print(f'  Wrote {OUT / "agents_protocol.png"}')


# ============================================================
# Domain icons — simple SVG-friendly mpl drawings
# ============================================================
def draw_icon(filename, color, draw_fn):
    fig, ax = plt.subplots(figsize=(1.3, 1.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    # background circle
    ax.add_patch(mpatches.Circle((0.5, 0.5), 0.45,
                                   facecolor=color + '22', edgecolor=color, linewidth=2))
    draw_fn(ax, color)
    plt.savefig(OUT / filename, transparent=True)
    plt.close()
    print(f'  Wrote {OUT / filename}')

print('\n=== Icons ===')

def medical_icon(ax, c):
    # red cross
    ax.add_patch(mpatches.Rectangle((0.35, 0.18), 0.30, 0.64,
                                      facecolor=c, edgecolor='none'))
    ax.add_patch(mpatches.Rectangle((0.18, 0.35), 0.64, 0.30,
                                      facecolor=c, edgecolor='none'))

def legal_icon(ax, c):
    # scale: top horizontal beam + central post + two pans
    ax.add_patch(mpatches.Rectangle((0.20, 0.65), 0.60, 0.05,
                                      facecolor=c, edgecolor='none'))
    ax.add_patch(mpatches.Rectangle((0.485, 0.30), 0.03, 0.40,
                                      facecolor=c, edgecolor='none'))
    # base
    ax.add_patch(mpatches.Rectangle((0.35, 0.25), 0.30, 0.05,
                                      facecolor=c, edgecolor='none'))
    # pans (triangles approximating dishes)
    for x in (0.25, 0.75):
        ax.plot([x - 0.10, x, x + 0.10, x - 0.10],
                 [0.55, 0.45, 0.55, 0.55], color=c, linewidth=2)
        # rope from beam to pan
        ax.plot([x, x], [0.65, 0.55], color=c, linewidth=1.5)

def logic_icon(ax, c):
    # large Greek letter Phi (Φ)
    ax.text(0.5, 0.5, 'Φ', ha='center', va='center',
             fontsize=42, color=c, fontweight='bold')

def longcontext_icon(ax, c):
    # document outline with lines
    ax.add_patch(mpatches.Rectangle((0.27, 0.20), 0.46, 0.62,
                                      facecolor='none', edgecolor=c, linewidth=2))
    # corner fold (small triangle)
    ax.add_patch(mpatches.Polygon([[0.62, 0.82], [0.73, 0.82], [0.73, 0.71]],
                                    facecolor=c+'33', edgecolor=c, linewidth=1.5))
    # text lines
    for y in (0.62, 0.52, 0.42, 0.32):
        ax.add_patch(mpatches.Rectangle((0.34, y), 0.32, 0.04,
                                          facecolor=c, edgecolor='none'))

draw_icon('icon_medical.png', C_HIGH_K, medical_icon)
draw_icon('icon_legal.png', C_HIGH_K, legal_icon)
draw_icon('icon_logic.png', C_LOW_K, logic_icon)
draw_icon('icon_longcontext.png', C_LOW_K, longcontext_icon)


# ============================================================
# Summary card listing all assets
# ============================================================
print('\n=== Summary ===')
files = sorted(OUT.glob('*.png')) + sorted(OUT.glob('*.svg'))
for p in files:
    print(f'  {p.relative_to(ROOT)}  ({p.stat().st_size // 1024} KB)')

# ============================================================
# Combined preview — show all assets together at draw.io scale
# ============================================================
print('\n=== Combined preview ===')
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor('white')

# Title bar
fig.text(0.5, 0.96, "More Agents Isn't Always Better — A Task-Dependent Theory",
          ha='center', va='top', fontsize=16, fontweight='bold')
fig.text(0.5, 0.925, 'Draw.io asset preview — drag these PNGs into the corresponding panels',
          ha='center', va='top', fontsize=10, fontstyle='italic', color='#666')

# Helper: place an image as ax inset
def place_image(rect, path, label):
    a = fig.add_axes(rect)
    a.axis('off')
    img = plt.imread(path)
    a.imshow(img)
    a.text(0.5, -0.18, label, transform=a.transAxes, ha='center', va='top',
            fontsize=8, color='#444', fontstyle='italic')

# Row 1: 4 mini-curves + 4 icons
y_curve = 0.70
y_icon = 0.78
xs_card = [0.06, 0.27, 0.48, 0.69]
curves = ['curve_medical.png', 'curve_legal.png', 'curve_logic.png', 'curve_longcontext.png']
icons = ['icon_medical.png', 'icon_legal.png', 'icon_logic.png', 'icon_longcontext.png']
labels = ['Medical (K=0.88)', 'Legal (K=0.80)', 'Formal Logic (K=0.50)', 'Long-context (K=0.10)']
for x, cf, ic, lb in zip(xs_card, curves, icons, labels):
    place_image([x, y_icon, 0.06, 0.10], OUT / ic, '')
    place_image([x + 0.07, y_curve, 0.13, 0.13], OUT / cf, lb)

# Row 2: agents_protocol on the right
place_image([0.06, 0.42, 0.42, 0.24], OUT / 'agents_protocol.png',
             'agents_protocol.png  →  panel ②')
# K gauge below
place_image([0.06, 0.18, 0.86, 0.18], OUT / 'k_gauge.png',
             'k_gauge.png  →  panel ③')

# Footer label
fig.text(0.5, 0.04,
          'Open paper/figures_design/teaser_overview.drawio in app.diagrams.net  →  '
          'drag-drop these PNGs into matching panels',
          ha='center', va='center', fontsize=10, color='#444')

_skip_doc = bool(__import__('os').environ.get('TEASER_MASTER'))
preview_path = OUT.parent / 'teaser_assets_preview.png'
if _skip_doc:
    plt.close()
    print(f'  (TEASER_MASTER=1) skipping {preview_path} — master script owns it')
else:
    plt.savefig(preview_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Wrote {preview_path}')


# ============================================================
# Asset usage guide markdown
# ============================================================
GUIDE = ROOT / 'paper' / 'figures_design' / 'INSET_USAGE.md'
if _skip_doc:
    print(f'  (TEASER_MASTER=1) skipping {GUIDE} — master script owns it')
else:
    GUIDE.write_text("""# Draw.io teaser assets — usage guide

Generated by `scripts/generate_drawio_assets.py`.

## Inventory (`paper/figures_design/insets/`)

| File | Usage in teaser_overview.drawio |
|---|---|
| `curve_medical.png` | Drag into the Medical QA card (panel ①) |
| `curve_legal.png` | Drag into the Legal Classification card (panel ①) |
| `curve_logic.png` | Drag into the Formal Logic card (panel ①) |
| `curve_longcontext.png` | Drag into the Long-context QA card (panel ①) |
| `icon_medical.png` | Replace the icon box in Medical QA card |
| `icon_legal.png` | Replace the icon box in Legal Classification card |
| `icon_logic.png` | Replace the icon box in Formal Logic card |
| `icon_longcontext.png` | Replace the icon box in Long-context QA card |
| `agents_protocol.png` | Replace panel ② entirely with this single image |
| `k_gauge.png` | Replace the gauge inside panel ③ with this single image |
| `k_gauge.svg` | Same as above but vector (preferred for paper print) |

## Step-by-step

### Replacing a mini-curve in a task card

1. Open https://app.diagrams.net (or draw.io desktop) and open `teaser_overview.drawio`.
2. Click on the existing dot-and-line sketch inside (e.g.) the Medical QA card to select it.
   Hold *Shift* and click each dot/line to select all parts; *Delete* to remove them.
3. From your file manager, **drag and drop** `insets/curve_medical.png` onto the now-empty
   curve area inside the Medical QA card.
4. Resize the image (drag corner handles) so it fits inside the card.
5. Repeat for legal / logic / long-context.

### Replacing the agent protocol panel ②

1. In `teaser_overview.drawio`, click the panel ② background rectangle and *Edit → Select All
   in Container* (or drag a marquee) to select every shape inside.
2. *Delete*.
3. Drag `insets/agents_protocol.png` onto the empty panel ②.
4. Resize to fit.

### Replacing the K gauge in panel ③

1. Select all shapes inside panel ③ (the gauge bar, markers, labels, side captions).
2. *Delete*.
3. Drag `insets/k_gauge.png` onto the empty panel ③ region.
4. Resize so it spans the panel.

### Replacing task icons

1. For each task card (Medical / Legal / Logic / Long-context), select the small icon box
   on the left side of the card.
2. *Edit Style…* → set image to the corresponding `icon_*.png`, OR delete and drag-drop the
   icon PNG.

## Exporting the polished figure

After all replacements:
1. *File → Export As → PDF (Vector)*  →  save to
   `paper/latex_v2/figures/fig0_teaser_drawio.pdf`
2. Also *File → Export As → PNG (Resolution: 300 dpi)* for previewing.
3. In `agent_count_neurips_2026.tex` change
   `\\includegraphics[width=\\linewidth]{figures/fig0_teaser_v4.pdf}`
   to
   `\\includegraphics[width=\\linewidth]{figures/fig0_teaser_drawio.pdf}`.

## Asset preview

See `paper/figures_design/teaser_assets_preview.png` for a single-image overview of
all generated assets at the scale they will appear inside the teaser.

## Re-generating

If you want to tweak colors / sizes / labels:
1. Edit `scripts/generate_drawio_assets.py` (constants at top).
2. Re-run: `python scripts/generate_drawio_assets.py`.
3. Drag-and-drop the new PNGs into draw.io (existing image elements can be replaced via
   *right-click → Edit Image…*).
""", encoding='utf-8')
    print(f'  Wrote {GUIDE}')


print(f'\nAll assets in: {OUT}')
print('See paper/figures_design/INSET_USAGE.md for the editing guide')
print('See paper/figures_design/teaser_assets_preview.png for a single-image preview')
