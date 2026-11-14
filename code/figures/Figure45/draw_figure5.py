"""四十八组配对轨迹分两页；只连接已保存的最好cost方案。"""
import json
import numpy as np
import fitz
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
from reference_style import *

def main():
    data=pd.read_csv(ROOT/'saved_search_paths.csv')
    paired,_,_=read_data()
    limits={}
    for city,g in data.groupby('city'):
        xr=np.ptp(g.P_minutes);yr=np.ptp(g.O_minutes)
        limits[city]=((g.P_minutes.min()-.10*xr,g.P_minutes.max()+.10*xr),
                      (g.O_minutes.min()-.10*yr,g.O_minutes.max()+.10*yr))
    groups=[]
    for page in range(2):
        fig,axes=plt.subplots(6,4,figsize=mm(180,270))
        fig.subplots_adjust(left=.078,right=.989,bottom=.105,top=.955,wspace=.48,hspace=.76)
        rows=[]
        for city,size in [('mandl',3),('mumford0',3)]:
            seeds=sorted(paired[paired.city==city].seed.unique())
            rows.extend((city,s) for s in seeds[page*size:(page+1)*size])
        for row,(city,seed) in enumerate(rows):
            for col,parent in enumerate(PARENTS):
                ax=axes[row,col]
                g=data[(data.city==city)&(data.seed==seed)&data.parent.str.endswith(parent)]
                assert set(g.method)==set(METHODS)
                for m in METHODS:
                    v=g[g.method==m].sort_values('update_index')
                    ax.plot(v.P_minutes,v.O_minutes,color=COLORS[m],linestyle=LINESTYLES[m],
                        marker=MARKERS[m],markersize=2.1,markeredgewidth=.5,
                        markerfacecolor=COLORS[m] if m=='full' else 'white',
                        linewidth=.68,label=CODES[m],zorder=3 if m=='full' else 2)
                ax.set_xlim(*limits[city][0]);ax.set_ylim(*limits[city][1])
                ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(3))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax.set_xlabel('Passenger time (min)',fontsize=7,labelpad=2)
                ax.set_ylabel('Route time (min)',fontsize=7,labelpad=2)
                ax.tick_params(labelsize=6.7)
                framed(ax);ax.grid(False)
                ax.legend(loc='best',ncol=4 if city=='mumford0' else 2,fontsize=6.5,
                    handlelength=.75 if city=='mumford0' else 1.05,
                    handletextpad=.3,columnspacing=.45 if city=='mumford0' else .65,
                    borderaxespad=.25,labelspacing=.25)
                pos=ax.get_position()
                display='Mandl' if city=='mandl' else 'Mumford0'
                fig.text((pos.x0+pos.x1)/2,pos.y1+.0075,f'{display} · {parent} / {seed}',
                    ha='center',va='bottom',fontsize=7.1)
                fig.text((pos.x0+pos.x1)/2,pos.y0-.0325,f'({chr(97+row*4+col)})',
                    ha='center',va='top',fontsize=7)
                groups.append(dict(page=page+1,panel=chr(97+row*4+col),city=city,
                    parent=str(g.parent.iloc[0]),seed=int(seed),points=len(g)))
        fig.legend(handles=key_handles(lines=True),loc='lower center',bbox_to_anchor=(.5,.014),
            ncol=4,columnspacing=1.8,handletextpad=.6,fontsize=7.6)
        save(fig,f'Figure5_page{page+1}')
    assert len(groups)==48 and len({(g['city'],g['parent'],g['seed']) for g in groups})==48
    combined=fitz.open()
    for page in [1,2]:
        with fitz.open(ROOT/f'Figure5_page{page}.pdf') as pdf:combined.insert_pdf(pdf)
    combined.save(ROOT/'Figure5.pdf');combined.close()
    (ROOT/'Figure5_panel_index.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
