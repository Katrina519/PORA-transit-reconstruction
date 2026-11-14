"""图3：收敛过程与逐种子哑铃图，完全由保存记录绘制。"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.transforms import Bbox
import fitz

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'source_data'
curve = pd.read_csv(DATA / 'figure3_curve_candidates.csv')
pairs = pd.read_csv(DATA / 'figure3_paired_endpoints.csv')
meta = json.loads((DATA / 'figure3_design_evidence.json').read_text(encoding='utf-8'))
TEAL, PURPLE, CORAL = '#699490', '#9F9CAA', '#D2928F'
INK, GRAY, GRID = '#28333B', '#697780', '#E9EDEF'
plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 8,
    'mathtext.fontset': 'stix',
    'axes.labelsize': 8, 'xtick.labelsize': 7.4, 'ytick.labelsize': 7.3,
    'axes.linewidth': .75, 'axes.spines.top': False,
    'axes.spines.right': False, 'legend.frameon': False,
    'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.facecolor': 'white',
})

fig = plt.figure(figsize=(180 / 25.4, 112 / 25.4), facecolor='white')
ax = fig.add_axes([.087, .228, .468, .625])
bx = fig.add_axes([.700, .228, .215, .625])
styles = [('author', 'NEA', PURPLE, (0, (4, 1.5))),
          ('full', 'PORA', TEAL, '-')]
for method, label, color, ls in styles:
    g = curve[curve.method == method].sort_values('seconds')
    ax.fill_between(g.seconds, g['mean'] - g.sd, g['mean'] + g.sd,
                    step='post', color=color, alpha=.13, lw=0, zorder=1)
    ax.step(g.seconds, g['mean'], where='post', color=color,
            lw=1.5 if method == 'full' else 1.3,
            ls=ls, label=label, zorder=3)
    cutoff = g[g.seconds == 120].iloc[0]
    ax.scatter(120, cutoff['mean'], s=23, facecolor='white',
               edgecolor=color, lw=1.15, zorder=5)
ax.set(xlim=(0, 400), ylim=(1.02, 3.5),
       xticks=[0, 120, 240, 400], yticks=[1.1, 1.5, 2, 2.5, 3, 3.5])
ax.set_xlabel('Elapsed wall-clock time (s)', labelpad=5)
ax.set_ylabel('Best-so-far cost', labelpad=5)
ax.grid(axis='y', color=GRID, lw=.55)
ax.set_axisbelow(True)
ax.axvline(120, color=CORAL, lw=.8, ls=(0, (3, 2)))
ax.text(126, 3.32, '120 s', fontsize=7.3, color=CORAL)
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.018),
          ncol=2, fontsize=8, handlelength=2, handletextpad=.6,
          columnspacing=1.5, borderaxespad=0)
ax.add_patch(Rectangle((120, 1.08), 280, .55, fill=False,
                       edgecolor='#BCC7CC', linestyle=(0, (3, 2)),
                       lw=.65, zorder=0))

inset = ax.inset_axes([.485, .49, .475, .42])
for method, label, color, ls in styles:
    g = curve[(curve.method == method) & curve.seconds.between(120, 400)]
    g = g.sort_values('seconds')
    inset.fill_between(g.seconds, g['mean'] - g.sd, g['mean'] + g.sd,
                       step='post', color=color, alpha=.13, lw=0)
    inset.step(g.seconds, g['mean'], where='post', color=color, ls=ls, lw=1.1)
inset.set(xlim=(120, 400), ylim=(1.08, 1.63),
          xticks=[120, 240, 400], yticks=[1.1, 1.3, 1.5])
inset.tick_params(labelsize=6.8, length=2.1, width=.6, pad=2)
inset.set_title('120–400 s detail', fontsize=7.3, color=GRAY, pad=5)
inset.grid(axis='y', color=GRID, lw=.45)
inset.set_axisbelow(True)
inset.spines['left'].set_color('#8F9BA2')
inset.spines['bottom'].set_color('#8F9BA2')

seeds = sorted(pairs.seed.unique())
assert len(seeds) == 10
endpoints = ['120s', '8round']
values = np.array([
    [float((pairs.loc[(pairs.seed == seed) & (pairs.endpoint == e), 'author_cost']
            - pairs.loc[(pairs.seed == seed) & (pairs.endpoint == e), 'full_cost']).iloc[0])
     for e in endpoints] for seed in seeds
])
assert values.shape == (10, 2)
y = np.arange(10, 0, -1)
bx.set(xlim=(-.033, .4), ylim=(-1.25, 10.9),
       xticks=[0, .1, .2, .3, .4])
bx.set_yticks(list(y) + [-.6], [str(seed) for seed in seeds] + ['Mean'])
bx.tick_params(axis='y', length=0, pad=6)
bx.tick_params(axis='x', pad=4)
bx.set_xlabel('Cost gain (NEA − PORA)', labelpad=6)
bx.set_ylabel('Seed', labelpad=8)
# Original brick/gold/sage/blue-violet hues retained; HSV saturation scaled to 60%.
gain_cmap = LinearSegmentedColormap.from_list(
    'original_muted_gain', ['#AA6A63', '#E0BB7C', '#ABB69F', '#A0B2B6', '#686D85'])
gain_norm = Normalize(vmin=-.02, vmax=.38)

def point_area(value):
    return 14 + 76 * abs(float(value)) / .38

for row in y:
    bx.axhline(row, color='#CAD0D3', lw=.55, ls=(0, (3, 3)), zorder=0)
bx.grid(axis='x', color='#CAD0D3', lw=.55, ls=(0, (3, 3)), zorder=1)
bx.set_axisbelow(True)
bx.spines['left'].set_visible(True)
bx.spines['left'].set_color('#9CA7AD')
bx.spines['left'].set_linewidth(.6)
bx.spines['right'].set_visible(True)
bx.spines['top'].set_visible(True)
bx.spines['right'].set_color('#9CA7AD')
bx.spines['top'].set_color('#9CA7AD')
bx.spines['right'].set_linewidth(.6)
bx.spines['top'].set_linewidth(.6)
bx.axvline(0, color='#68747C', lw=.9, zorder=2)
bx.axhline(.25, color='#BFC9CE', lw=.65, zorder=2)
offset = .15
for i in range(10):
    for k, dy in [(0, offset), (1, -offset)]:
        value = values[i, k]
        color = gain_cmap(gain_norm(value))
        bx.plot([0, value], [y[i] + dy, y[i] + dy], color='#3B4247',
                lw=.9, zorder=3, solid_capstyle='round')
        bx.scatter(value, y[i] + dy, s=point_area(value),
                   facecolor=color if k == 0 else 'white',
                   edgecolor='white' if k == 0 else color,
                   linewidth=.6 if k == 0 else 1.15, zorder=5)
means = values.mean(axis=0)
for k, dy in [(0, offset), (1, -offset)]:
    bx.plot([0, means[k]], [-.6 + dy, -.6 + dy], color=INK, lw=1, zorder=3)
bx.scatter(means[0], -.6 + offset, s=31, color=INK, edgecolor='white', lw=.75, zorder=4)
bx.scatter(means[1], -.6 - offset, s=35, marker='o', facecolor='white', edgecolor=INK,
           linewidth=1.15, zorder=5)
bx.get_yticklabels()[-1].set_fontweight('bold')
bx.legend(handles=[
    Line2D([], [], ls='none', marker='o', markerfacecolor=INK,
           markeredgecolor='white', markersize=5, label='120 s'),
    Line2D([], [], ls='none', marker='o', markerfacecolor='white',
           markeredgecolor=INK, markersize=4.7, label='8 rounds'),
], loc='lower left', bbox_to_anchor=(-.08, 1.018), ncol=2,
    borderaxespad=0, fontsize=7.4, handlelength=.8,
    handletextpad=.4, columnspacing=1.0)

# 图例明确尺寸对应绝对收益，颜色对应有符号收益，不引入每种子的p值。
size_ax = fig.add_axes([.942, .685, .055, .130])
size_ax.set(xlim=(0, 1), ylim=(-.4, 2.4))
size_ax.set_axis_off()
for yy, value in [(2, .05), (1, .20), (0, .35)]:
    size_ax.scatter(.18, yy, s=point_area(value), color=INK, edgecolor='none')
    size_ax.text(.48, yy, f'{value:.2f}', va='center', ha='left', fontsize=6.8)
fig.text(.940, .841, '|gain|', fontsize=6.8, color=GRAY)
cax = fig.add_axes([.947, .228, .010, .385])
cbar = fig.colorbar(ScalarMappable(norm=gain_norm, cmap=gain_cmap),
                    cax=cax, ticks=[0, .1, .2, .3])
cbar.ax.tick_params(labelsize=6.8, length=2, width=.5, pad=2)
cbar.outline.set_visible(False)
fig.text(.938, .640, 'Gain', fontsize=6.8, color=GRAY)

# One shared heading and a cohort bridge replace separate panel letters.
fig.text(.5, .988, 'Mumford1: search progress and paired-seed gains',
         fontsize=10.2, fontweight='bold', ha='center', va='top')
fig.text(.087, .931, 'Search progress', fontsize=9.0, va='center')
fig.text(.700, .931, 'Paired-seed gains', fontsize=9.0, va='center')
cohort_arrow = FancyArrowPatch((.31, .931), (.665, .931),
    transform=fig.transFigure, arrowstyle='->', mutation_scale=8,
    linewidth=.8, color='#A2B2B4', zorder=5, shrinkA=0, shrinkB=0)
fig.add_artist(cohort_arrow)
fig.text(.487, .931, 'Same 10 paired seeds', fontsize=7.4,
         color=GRAY, ha='center', va='center', zorder=6,
         bbox=dict(facecolor='white', edgecolor='none', pad=2.2))

# 百分比是均值之比；绝对差值均值在右图Mean行单独显示。
summary = {}
for e in endpoints:
    p = pairs[pairs.endpoint == e]
    pct = 100 * (p.author_cost.mean() - p.full_cost.mean()) / p.author_cost.mean()
    expected = next(v for v in meta['endpoint_summary'] if v['endpoint'] == e)
    assert abs(pct - expected['improvement_percent']) < 1e-10
    diff = p.author_cost.to_numpy() - p.full_cost.to_numpy()
    summary[e] = dict(percent=pct, wins=int((diff > 1e-6).sum()),
                      ties=int((abs(diff) <= 1e-6).sum()), losses=int((diff < -1e-6).sum()))
# The shared endpoint band ties the two views together without implying
# that the 400-s curve horizon is the completion of eight full rounds.
footer = FancyBboxPatch((.070, .009), .912, .111,
    transform=fig.transFigure, boxstyle='round,pad=0,rounding_size=.008',
    facecolor='#EEF1F5', edgecolor='none', zorder=-1)
fig.add_artist(footer)
for xx in [.330, .652]:
    fig.add_artist(Line2D([xx,xx],[.024,.102],transform=fig.transFigure,
                         color='#CDD4DE',linewidth=.6))
fig.text(.087,.083,'Mean cost reduction',fontsize=8.0,va='center')
fig.text(.087,.034,'10 paired seeds',fontsize=7.3,color=GRAY,va='center')
for e,xx,end,label in [('120s',.364,.626,'120 s'),('8round',.688,.955,'8 rounds')]:
    fig.add_artist(Line2D([xx],[.083],transform=fig.transFigure,linestyle='none',
        marker='o',markersize=4.6,markeredgecolor=INK,
        markerfacecolor=INK if e=='120s' else 'white',markeredgewidth=.9))
    fig.text(xx+.018,.083,label,fontsize=7.8,va='center')
    fig.text(end,.083,f"{summary[e]['percent']:.2f}%",fontsize=8.8,
             fontweight='bold',color='#4D948D',ha='right',va='center')
    fig.text(xx,.034,'Wins / ties / losses',fontsize=6.8,color=GRAY,va='center')
    fig.text(end,.034,f"{summary[e]['wins']} / {summary[e]['ties']} / {summary[e]['losses']}",
             fontsize=7.2,color=GRAY,ha='right',va='center')

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
outside = []
for label in fig.findobj(matplotlib.text.Text):
    if label.get_visible() and label.get_text().strip():
        bb = label.get_window_extent(renderer)
        if not all(fig.bbox.contains(*point) for point in bb.get_points()):
            outside.append(label.get_text())
assert not outside, outside
for fmt in ['pdf', 'svg']:
    fig.savefig(ROOT / f'Figure3_final.{fmt}')
fig.savefig(ROOT / 'Figure3_final_preview.png', dpi=240)
fig.savefig(ROOT / 'Figure3_final_600dpi.png', dpi=600)
fig.savefig(ROOT / 'Figure3_final_180mm_check.png', dpi=150)
right_box = Bbox.from_extents(.62 * fig.get_figwidth(), 0,
                             fig.get_figwidth(), fig.get_figheight())
fig.savefig(ROOT / 'Figure3b_lollipop_detail.png', dpi=360, bbox_inches=right_box)
fig.savefig(ROOT / 'Figure3b_lollipop_detail.pdf', bbox_inches=right_box)
doc = fitz.open(ROOT / 'Figure3_final.pdf')
doc[0].get_pixmap(matrix=fitz.Matrix(2, 2)).save(ROOT / 'Figure3_final_PDFproof.png')
pdf_text = doc[0].get_text()
for token in [str(seed) for seed in seeds] + ['Mean', '13.73%', '2.46%']:
    assert token in pdf_text, token
audit = dict(dimensions_mm=[180, 112], chart_type='stepwise convergence + paired zero-anchored lollipop',
             paired_seed_count=10, endpoints=endpoints, seed_sort='numeric ascending',
             absolute_gains=values.tolist(), mean_absolute_gains=means.tolist(),
             summary=summary, outside_text=outside,
             source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.iterdir()},
             checkpoint_source_sha256=meta['source_sha256'])
(ROOT / 'verification.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(dict(dimensions_mm=audit['dimensions_mm'], summary=summary, outside=outside), ensure_ascii=False))
