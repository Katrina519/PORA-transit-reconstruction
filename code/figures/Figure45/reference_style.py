"""Colours and mark styles sampled from the user's two reference figures."""
from plot_style import *

COLORS={'author':'#7B8795','count':'#B5964F','ridge':'#806895','full':'#779B68'}
CODES={'author':'A','count':'B','ridge':'C','full':'D'}
MARKERS={'author':'o','count':'D','ridge':'s','full':'o'}
LINESTYLES={'author':'-','count':'--','ridge':'-.','full':':'}

def framed(ax):
    for spine in ax.spines.values():
        spine.set_visible(True);spine.set_color('black');spine.set_linewidth(.7)
    ax.tick_params(color='black',labelcolor='black',width=.65,length=2.5,pad=2.5)

def key_handles(lines=False):
    return [Line2D([],[],color=COLORS[m],marker=MARKERS[m] if lines else 'o',
        linestyle=LINESTYLES[m] if lines else 'none',markersize=3.5,linewidth=.75,
        markerfacecolor=COLORS[m] if m=='full' or not lines else 'white',
        markeredgewidth=.65,label=f'{CODES[m]}   {NAMES[m]}') for m in METHODS]
