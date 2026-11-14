"""Reproduce reference 1: black-and-white boxes, muted round observations."""
import numpy as np
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
from reference_style import *

def main():
    paired,_,_=read_data()
    fig=plt.figure(figsize=mm(180,105))
    axes=[fig.add_axes([.095,.325,.375,.59]),fig.add_axes([.595,.325,.375,.59])]
    for i,(ax,city,label) in enumerate(zip(axes,['mandl','mumford0'],['Mandl1-6','Mumford0-12'])):
        d=paired[paired.city==city].sort_values(['parent','seed'])
        values=[d[m+'_budget'].to_numpy() for m in METHODS]
        low=min(v.min() for v in values);high=max(v.max() for v in values);span=high-low
        ax.boxplot(values,positions=np.arange(1,5),widths=.46,patch_artist=True,showfliers=False,
            boxprops=dict(facecolor='white',edgecolor='black',linewidth=.9),
            medianprops=dict(color='black',linewidth=.95),
            whiskerprops=dict(color='black',linewidth=.8),capprops=dict(color='black',linewidth=.8))
        jitter=np.random.default_rng(20261111).uniform(-.095,.095,len(d))
        for j,m in enumerate(METHODS):
            ax.scatter(j+1+jitter,values[j],s=13,color=COLORS[m],edgecolors='none',zorder=3)
        ax.set_xlim(.45,4.55);ax.set_ylim(low-.12*span,high+.24*span)
        ax.set_xticks(range(1,5),[CODES[m] for m in METHODS])
        ax.set_xlabel('Method',labelpad=3);ax.set_ylabel('Cost',labelpad=4)
        ax.yaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.grid(axis='y',color='#D9D9D9',lw=.55);framed(ax)
        ax.text(.97,.95,f'n = {len(d)} runs per method',transform=ax.transAxes,
            ha='right',va='top',fontsize=8)
        pos=ax.get_position()
        fig.text((pos.x0+pos.x1)/2,.18,f'({"ab"[i]})  {label}',ha='center',fontsize=8.5)
    fig.legend(handles=key_handles(),loc='lower center',bbox_to_anchor=(.5,.018),
        ncol=2,columnspacing=3.4,handletextpad=.65,fontsize=8)
    save(fig,'Figure4')

if __name__=='__main__':main()
