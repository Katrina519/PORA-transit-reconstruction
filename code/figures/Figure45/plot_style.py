"""Shared, physical-size plotting style. All observations come from saved CSV."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

ROOT = Path(__file__).resolve().parent
METHODS = ['author','count','ridge','full']
NAMES = {'author':'NEA','count':'Count','ridge':'Ridge','full':'PORA'}
COLORS = {'author':'#507B9D','count':'#CDA65D','ridge':'#9986B7','full':'#169B88'}
POSITIVE = '#29CBAE'
NEGATIVE = '#FF6448'
INK = '#29343D'
PARENTS = ['62811','62812','62813','62814']
SHAPES = ['o','s','^','D']

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial'], 'mathtext.fontset':'stix',
    'font.size':8, 'axes.labelsize':8.5, 'axes.titlesize':9,
    'axes.edgecolor':INK, 'axes.labelcolor':INK,'text.color':INK,
    'xtick.color':INK,'ytick.color':INK,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
    'axes.linewidth':.75,'xtick.major.width':.65,'ytick.major.width':.65,
    'xtick.major.size':3,'ytick.major.size':3, 'legend.fontsize':7.5,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'savefig.dpi':600,'figure.dpi':120,'axes.spines.top':False,'axes.spines.right':False,
    'legend.frameon':False,'lines.linewidth':1.2,'axes.axisbelow':True,
})

def read_data():
    return (pd.read_csv(ROOT/'paired_runs.csv'),pd.read_csv(ROOT/'plot_data_long.csv'),pd.read_csv(ROOT/'plot_summary.csv'))

def mm(width,height):
    return width/25.4,height/25.4

def label_panel(fig, ax, letter, title):
    box=ax.get_position()
    fig.text(box.x0-.046,box.y1+.027,letter,weight='bold',fontsize=10,va='bottom')
    fig.text(box.x0,box.y1+.027,title,weight='bold',fontsize=9,va='bottom')

def finish_axes(ax,grid=True):
    if grid:
        ax.grid(color='#E7EBEE',lw=.55)
    ax.tick_params(pad=3)

def method_handles(mean_only=False):
    return [Line2D([],[],color=COLORS[m],marker=['o','s','^','D'][i],ls='none',
        markerfacecolor=COLORS[m] if m=='full' else 'white',markeredgewidth=1.2,
        markersize=5,label=NAMES[m]) for i,m in enumerate(METHODS)]

def save(fig,name):
    fig.canvas.draw()
    legend_checks=[]
    for index,ax in enumerate(fig.axes):
        legend=ax.get_legend()
        if legend is not None:
            box=legend.get_window_extent(fig.canvas.get_renderer())
            overlaps=[]
            for line_index,line in enumerate(ax.lines):
                path=line.get_path().transformed(line.get_transform())
                if path.intersects_bbox(box,filled=False):
                    overlaps.append(line_index)
            legend_checks.append({'axes':index,'curve_intersections':overlaps})
    (ROOT/(name+'_legend_checks.json')).write_text(json.dumps(legend_checks,indent=2),encoding='utf-8')
    fig.savefig(ROOT/(name+'.pdf'))
    fig.savefig(ROOT/(name+'.svg'))
    fig.savefig(ROOT/(name+'_600dpi.png'),dpi=600)
    fig.savefig(ROOT/(name+'_preview.png'),dpi=170)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    bounds=[]
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text().strip():
            b=text.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
            bounds.append({'text':text.get_text(),'font_pt':text.get_fontsize(),
                'bbox_inches':[b.x0,b.y0,b.x1,b.y1]})
    (ROOT/(name+'_text_bounds.json')).write_text(json.dumps(bounds,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.close(fig)
