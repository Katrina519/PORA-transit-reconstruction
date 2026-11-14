"""只读导出的Excel，逐格复核原始/更新数据与汇总缓存。"""
import json
import math
from pathlib import Path
import openpyxl

ROOT=Path(__file__).resolve().parent


def equal(a,b):
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    return a==b


def main():
    result=json.loads((ROOT/'统计与核验.json').read_text(encoding='utf-8'))
    source=json.loads((ROOT/'输入CSV.json').read_text(encoding='utf-8'))
    file=ROOT/'outputs/unified_online_20261112/公交选线_统一在线方法_40组对照.xlsx'
    workbook=openpyxl.load_workbook(file,read_only=False,data_only=True)
    expected_sheets=['主实验汇总','消融对照','全部逐次','方法逐次','Mumford0旧新对照','方法与口径','原始CSV']
    assert workbook.sheetnames==expected_sheets
    checked=0
    for sheet,rows in [('全部逐次',result['rows']),('原始CSV',source['rows'])]:
        ws=workbook[sheet];keys=list(rows[0])
        assert [ws.cell(1,j+1).value for j in range(len(keys))]==keys
        assert ws.max_row==81 and ws.max_column==41
        for i,row in enumerate(rows,start=2):
            for j,key in enumerate(keys,start=1):
                assert equal(ws.cell(i,j).value,row[key]),(sheet,i,key,ws.cell(i,j).value,row[key])
                checked+=1
    rowmap={('final','mandl'):7,('final','mumford0'):11,('budget','mandl'):21,('budget','mumford0'):25}
    methods=['author','full','count','ridge']
    for item in result['summary']:
        rr=rowmap[item['endpoint'],item['city']]+methods.index(item['method'])
        ws=workbook['主实验汇总']
        assert equal(ws.cell(rr,3).value,item['method_mean']),(rr,'mean',ws.cell(rr,3).value,item['method_mean'])
        assert equal(ws.cell(rr,6).value,item['cost_sd']),(rr,'sd')
        assert ws.cell(rr,7).value==40
        if item['method']!='author':
            assert equal(ws.cell(rr,4).value,item['improvement_percent']/100),(rr,'improvement')
            assert ws.cell(rr,5).value=='{}/{}/{}'.format(item['wins'],item['ties'],item['losses']),(rr,'w/t/l')
    for i,item in enumerate(result['ablations'],start=6):
        ws=workbook['消融对照']
        assert equal(ws.cell(i,4).value,item['control_mean'])
        assert equal(ws.cell(i,5).value,item['method_mean'])
        assert equal(ws.cell(i,6).value,item['improvement_percent']/100)
        assert equal(ws.cell(i,8).value,item['one_sided_p_full_lower'])
    for ws in workbook:
        for row in ws.iter_rows():
            for cell in row:
                assert cell.data_type!='e',(ws.title,cell.coordinate,cell.value)
    formulas=openpyxl.load_workbook(file,data_only=False,read_only=True)
    assert formulas['主实验汇总']['C8'].data_type=='f' and formulas['消融对照']['F6'].data_type=='f'
    output=dict(passed=True,path=str(file),sheets=expected_sheets,raw_cells_checked=checked,
                summary_rows_checked=16,ablation_rows_checked=12,formula_caches_match_statistics=True)
    with (ROOT/'Excel核验.json').open('w',encoding='utf-8') as stream:json.dump(output,stream,ensure_ascii=False,indent=2)
    print(json.dumps(output,ensure_ascii=False))


if __name__=='__main__':main()
