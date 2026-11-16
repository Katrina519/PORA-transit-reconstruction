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
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch, ConnectionPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.transforms import Bbox
import fitz

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'source_data'
curve = pd.read_csv(DATA / 'figure3_curve_candidates.csv')
pairs = pd.read_csv(DATA / 'figure3_paired_endpoints.csv')
meta = json.loads((DATA / 'figure3_design_evidence.json').read_text(encoding='utf-8'))
TEAL, PURPLE, CORAL = '#699490', '#9F9CAA', '#B54A4A'
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

fig = plt.figure(figsize=(180 / 25.4, 95 / 25.4), facecolor='white')
ax = fig.add_axes([.095, .160, .875, .700])
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
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('#111111')
    spine.set_linewidth(1.05)
ax.tick_params(direction='in', color='#111111', width=.85)
ax.set_axisbelow(True)
ax.axvline(120, color=CORAL, lw=.8, ls=(0, (3, 2)))
ax.text(126, 3.32, '120 s', fontsize=7.3, color=CORAL)
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.018),
          ncol=2, fontsize=8, handlelength=2, handletextpad=.6,
          columnspacing=1.5, borderaxespad=0)
zoom_region = Rectangle((120, 1.08), 120, .55,
    facecolor=(.76, .79, .80, .20), edgecolor=CORAL,
    linestyle='-', linewidth=.7, zorder=.5)
ax.add_patch(zoom_region)

inset = ax.inset_axes([.515, .450, .455, .475])
for method, label, color, ls in styles:
    g = curve[(curve.method == method) & curve.seconds.between(120, 240)]
    g = g.sort_values('seconds')
    inset.fill_between(g.seconds, g['mean'] - g.sd, g['mean'] + g.sd,
                       step='post', color=color, alpha=.13, lw=0)
    inset.step(g.seconds, g['mean'], where='post', color=color, ls=ls, lw=1.1)
inset.set(xlim=(120, 240), ylim=(1.08, 1.63),
          xticks=[120, 180, 240], yticks=[1.1, 1.3, 1.5])
inset.tick_params(labelsize=6.8, length=2.1, width=.6, pad=2)
inset.set_title('120–240 s detail', fontsize=7.3, color=GRAY, pad=5)
inset.grid(False)
inset.set_axisbelow(True)
for spine in inset.spines.values():
    spine.set_visible(True)
    spine.set_color('#111111')
    spine.set_linewidth(.85)
inset.tick_params(direction='in', color='#111111', width=.65)
# Both views retain exactly the same data and zoom limits. The arrow lands
# between inset tick labels so the connection remains readable at print size.
zoom_arrow = ConnectionPatch(xyA=(215, 1.63), coordsA='data', axesA=ax,
    xyB=(.72, 0), coordsB='axes fraction', axesB=inset,
    arrowstyle='-|>', mutation_scale=7.5, linewidth=.9,
    linestyle=(0, (4, 3)), color=CORAL, shrinkA=3, shrinkB=3,
    clip_on=False, zorder=6)
ax.add_artist(zoom_arrow)


fig.text(.53, .985, 'Mumford1: search progress across 24 paired runs',
         fontsize=10.2, fontweight='bold', ha='center', va='top')
summary = {}
for endpoint in ['120s', '8round']:
    p = pairs[pairs.endpoint == endpoint]
    assert len(p) == 24
    delta = p.author_cost.to_numpy() - p.full_cost.to_numpy()
    pct = 100 * delta.mean() / p.author_cost.mean()
    summary[endpoint] = dict(percent=float(pct), wins=int((delta > 1e-6).sum()),
                            ties=int((abs(delta) <= 1e-6).sum()), losses=int((delta < -1e-6).sum()))
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
outside = []
for label in fig.findobj(matplotlib.text.Text):
    if label.get_visible() and label.get_text().strip():
        if not all(fig.bbox.contains(*point) for point in label.get_window_extent(renderer).get_points()):
            outside.append(label.get_text())
assert not outside, outside
for fmt in ['pdf', 'svg']:
    fig.savefig(ROOT / f'Figure3_final.{fmt}')
fig.savefig(ROOT / 'Figure3_final_preview.png', dpi=240)
fig.savefig(ROOT / 'Figure3_final_600dpi.png', dpi=600)
fig.savefig(ROOT / 'Figure3_final_180mm_check.png', dpi=150)
with fitz.open(ROOT / 'Figure3_final.pdf') as doc:
    doc[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(ROOT / 'Figure3_final_PDFproof.png')
values = np.array([[float(pairs[(pairs.seed == seed) & (pairs.endpoint == endpoint)].iloc[0].author_cost
                         - pairs[(pairs.seed == seed) & (pairs.endpoint == endpoint)].iloc[0].full_cost)
                    for endpoint in ['120s','8round']] for seed in sorted(pairs.seed.unique())])
audit = dict(status='FINAL', dimensions_mm=[180,95], chart_type='mean convergence with a local inset',
             paired_seed_count=24, initial_network_count=4,
             endpoints=['120s','8round'], absolute_gains=values.tolist(), mean_absolute_gains=values.mean(axis=0).tolist(),
             summary=summary, outside_text=outside,
             source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.iterdir()},
             checkpoint_source_sha256=meta['source_sha256'], preview_only=False,
             guide_color='#B54A4A', zoom_limits=[120,240,1.08,1.63],
             main_spines_black_closed=True, inset_spines_black_closed=True,
             right_plot_removed=True, summary_band_removed=True, endpoint_summary_in_table=True)
(ROOT / 'verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
import shutil
for original, canonical in [('Figure3_final.pdf','Figure3.pdf'),('Figure3_final.svg','Figure3.svg'),
    ('Figure3_final_preview.png','Figure3_preview.png'),('Figure3_final_600dpi.png','Figure3_600dpi.png'),
    ('Figure3_final_180mm_check.png','Figure3_180mm_print_check.png')]:
    shutil.copy2(ROOT / original, ROOT / canonical)
print(json.dumps({'dimensions_mm':[180,95],'summary':summary,'outside':outside},ensure_ascii=False))
