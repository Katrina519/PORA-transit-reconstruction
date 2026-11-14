"""只读检查已导出XLSX的缓存数值、公式、行数和原始结果一致性。"""
import sys
import json
import math
import zipfile
import posixpath
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import experiment as e


def verify():
    data = e.read(ROOT / '统计与核验.json')
    req = e.read(ROOT / '请求冻结.json')
    target = ROOT / 'outputs/mumford1_ablation_20261114/Mumford1_五配置消融_10种子_8轮及120秒.xlsx'
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    checked = 0
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = [''.join(v.itertext()) for v in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        relations = {r.attrib['Id']: r.attrib['Target'] for r in ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))}
        sheets = {}
        formulas = 0
        for sheet in ET.fromstring(archive.read('xl/workbook.xml')).find('s:sheets', ns):
            relation = sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
            name = relations[relation]
            name = name.lstrip('/') if name.startswith('/') else posixpath.normpath('xl/' + name)
            values = {}
            for cell in ET.fromstring(archive.read(name)).findall('.//s:sheetData/s:row/s:c', ns):
                kind = cell.attrib.get('t')
                assert kind != 'e', (sheet.attrib['name'], cell.attrib['r'], 'Excel error')
                v = cell.find('s:v', ns)
                if kind == 's':
                    value = shared[int(v.text)]
                elif kind == 'inlineStr':
                    value = ''.join(t.text or '' for t in cell.findall('.//s:t', ns))
                elif kind in ['str', 'd']:
                    value = v.text if v is not None else None
                elif v is not None and v.text is not None:
                    value = float(v.text)
                else:
                    value = None
                if cell.find('s:f', ns) is not None:
                    formulas += 1
                    assert value is not None, (sheet.attrib['name'], cell.attrib['r'], 'formula has no saved value')
                values[cell.attrib['r']] = value
            sheets[sheet.attrib['name']] = values
    assert len(sheets) == 7

    def number(sheet, cell, expected):
        nonlocal checked
        actual = sheets[sheet].get(cell)
        assert isinstance(actual, (int, float)) and math.isfinite(actual), (sheet, cell, actual)
        assert abs(actual - expected) < 1e-10, (sheet, cell, actual, expected)
        checked += 1

    methods = req['methods']
    for item in data['summary']:
        r = (7 if item['endpoint'] == '8round' else 15) + methods.index(item['method'])
        number('消融汇总', 'C'+str(r), 10)
        number('消融汇总', 'D'+str(r), item['mean_cost'])
        number('消融汇总', 'E'+str(r), item['sd_cost'])
        for col, key in [('J','mean_P_minutes'),('K','mean_O_oneway_minutes'),('L','mean_online_seconds')]:
            number('消融汇总', col+str(r), item[key])
        if item['method'] != 'author':
            number('消融汇总', 'F'+str(r), item['improvement_vs_NEA_percent']/100)
    for i, item in enumerate(data['comparisons'], 6):
        for col,key in [('D','comparator_mean'),('E','full_mean'),('G','mean_absolute_gain'),
                        ('H','gain_ci95_low'),('I','gain_ci95_high'),('K','p_two_sided')]:
            number('配对检验', col+str(i), item[key])
        number('配对检验', 'F'+str(i), item['full_improvement_percent']/100)
        if item['p_holm'] is not None:
            number('配对检验', 'L'+str(i), item['p_holm'])
    raw_index = {(x['endpoint'],x['method'],x['seed']):x for x in data['rows']}
    assert len(raw_index) == 100
    for r in range(2,102):
        s = sheets['逐次结果']
        key = (s['A'+str(r)], s['B'+str(r)], int(s['C'+str(r)]))
        item = raw_index.pop(key)
        for col,key in [('D','cost'),('E','P_minutes'),('F','O_oneway_minutes'),
                        ('G','total_online_seconds'),('H','selection_seconds'),('I','feedback_seconds'),
                        ('J','completed_passes'),('K','last_completed_seconds'),('L','preparation_seconds'),
                        ('M','model_evaluations'),('N','cache_hits'),('O','gain_changed_choice_count')]:
            number('逐次结果', col+str(r), item[key])
    assert not raw_index
    for i, item in enumerate(data['pairs'],2):
        for col,key in [('D','comparator_cost'),('E','full_cost'),('F','absolute_gain'),
                        ('H','comparator_P'),('I','full_P'),('J','comparator_O'),('K','full_O')]:
            number('逐种子配对', col+str(i), item[key])
        number('逐种子配对','G'+str(i),item['full_improvement_percent']/100)
    for i,item in enumerate(data['checkpoints'],2):
        number('全部检查点','E'+str(i),item['seconds'])
        number('全部检查点','F'+str(i),item['cost'])
        number('全部检查点','G'+str(i),item['samples'])
    e.check(req['hashes'])
    e.check(e.read(ROOT/'分析方案冻结.json')['hashes'])
    provenance = e.read(ROOT/'审计与交付来源.json')['hashes']
    e.check({p:h for p,h in provenance.items() if p.endswith('analyze.py')})
    e.dump(ROOT/'Excel独立核验.json',dict(passed=True,target=str(target),numeric_checks=checked,
        worksheets=list(sheets),saved_formulas=formulas,zero_excel_errors=True,
        protected_sources_unchanged=len(req['hashes']),xlsx_sha256=e.sha(target)))
    print('XLSX VERIFIED',checked,'numeric cells;',len(sheets),'sheets;',formulas,'saved formulas')


if __name__ == '__main__':
    verify()
