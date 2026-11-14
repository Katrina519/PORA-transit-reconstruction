"""只读复核Excel数值、配对和公式缓存，不运行搜索。"""
import json
import math
from pathlib import Path
import openpyxl
ROOT=Path(__file__).resolve().parent


def same(a,b):
    return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12) if isinstance(a,(int,float)) and isinstance(b,(int,float)) else a==b


def main():
    d=json.loads((ROOT/'统计与核验.json').read_text(encoding='utf-8'))
    file=ROOT/'outputs/mumford1_10pairs_20261112/Mumford1_统一在线Full_10组配对.xlsx'
    wb=openpyxl.load_workbook(file,data_only=True)
    assert wb.sheetnames==['Mumford1主实验','十组配对','方法逐次','全部检查点','冻结N24快照','方法与来源']
    main=wb['Mumford1主实验']
    for stat in d['summary']:
        author,full=(7,8) if stat['endpoint']=='8round' else (13,14)
        assert same(main.cell(author,3).value,stat['author_mean'])
        assert same(main.cell(full,3).value,stat['full_mean'])
        assert same(main.cell(author,4).value,stat['author_sd']) and same(main.cell(full,4).value,stat['full_sd'])
        assert same(main.cell(full,5).value,stat['improvement_percent']/100)
        assert main.cell(full,6).value=='{}/{}/{}'.format(stat['wins'],stat['ties'],stat['losses'])
        assert same(main.cell(full,7).value,stat['p_two_sided']) and main.cell(full,8).value==10
    index={(r['seed'],r['endpoint']):r for r in d['pairs']}
    seeds=sorted({r['seed'] for r in d['pairs']})
    for start,endpoint in [(2,'8round'),(12,'120s')]:
        for i,seed in enumerate(seeds):
            row=start+i;r=index[seed,endpoint];ws=wb['十组配对']
            expected=[seed,endpoint,r['author_cost'],r['full_cost'],r['improvement_percent']/100,
                      r['author_cost']-r['full_cost'],r['author_P'],r['full_P'],r['author_O'],r['full_O'],
                      r['author_seconds'],r['full_seconds'],r['full_selection_seconds'],r['full_feedback_seconds']]
            assert all(same(ws.cell(row,j+1).value,v) for j,v in enumerate(expected)),(row,seed)
    assert wb['全部检查点'].max_row==821 and wb['方法逐次'].max_row==41
    for i,r in enumerate(d['frozen_N24']['main'],start=6):
        ws=wb['冻结N24快照'];assert same(ws.cell(i,3).value,r['cost_reduction_percent']/100)
        assert same(ws.cell(i,5).value,r['p_two_sided'])
    for ws in wb:
        for row in ws.iter_rows():
            for cell in row:assert cell.data_type!='e',(ws.title,cell.coordinate,cell.value)
    formulas=openpyxl.load_workbook(file,data_only=False,read_only=True)
    assert formulas['Mumford1主实验']['C8'].data_type=='f'
    out=dict(passed=True,path=str(file),summary_endpoints_checked=2,pairs_checked=20,
             trace_rows=820,N24_unchanged=True,formula_cache_matches_statistics=True)
    with (ROOT/'Excel核验.json').open('w',encoding='utf-8') as stream:json.dump(out,stream,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__':main()
