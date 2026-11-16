"""Verify published raw-record identities and their reported endpoints; no search."""
from pathlib import Path
import csv,json,hashlib,zipfile

DATA=Path(__file__).resolve().parents[1]/'data/manuscript_24pairs'
def sha(b):return hashlib.sha256(b).hexdigest()
def read(name):
    with (DATA/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def key(r):return r['batch'],r['parent'],int(r['seed']),r['method']

def verify():
    index=read('run_sources.csv');mapping={key(r):r for r in index}
    assert len(mapping)==len(index)==132
    rows=[dict(role=role,**r) for role in ['main','ablation'] for r in read(role+'24.csv')]
    assert len(rows)==336 and {key(r) for r in rows}==set(mapping)
    grouped={}
    for r in rows:grouped.setdefault(key(r),[]).append(r)
    maximum=0.;checked=0;starts={}
    with zipfile.ZipFile(DATA/'run_records.zip') as archive:
        for identity,entry in mapping.items():
            raw=archive.read(entry['record_member']);config=archive.read(entry['config_member'])
            assert sha(raw)==entry['record_sha256'] and sha(config)==entry['config_sha256']
            run=json.loads(raw);request=json.loads(config)
            assert (identity[1],identity[2],identity[3])==(run['parent'],int(run['seed']),run['variant'])
            assert run['iterations']==request['iterations']==8 and request['secondary_seconds']==120
            contexts=request.get('contexts',[request.get('context')])
            ctx=next(c for c in contexts if c['parent']==entry['parent'])
            network_hash=sha(json.dumps(ctx['routes'],separators=(',',':')).encode())
            assert network_hash==entry['initial_network_sha256']
            assert entry['parent'] not in starts or starts[entry['parent']]==network_hash
            starts[entry['parent']]=network_hash
            points=run['fine_curve'];assert len(points)==41
            assert all(a['seconds']<=b['seconds'] for a,b in zip(points,points[1:]))
            for row in grouped[identity]:
                point=points[-1] if row['endpoint']=='8round' else [v for v in points if v['seconds']<=120][-1]
                err=abs(point['cost']-float(row['cost']));maximum=max(maximum,err)
                assert err<1e-12 and abs(point['seconds']-float(row['last_completed_seconds']))<1e-7
                assert row['role'] in entry['roles'].split('|')
                checked+=1
    assert len(starts)==4 and checked==336
    result=json.loads((DATA/'run_source_checks.json').read_text(encoding='utf-8'))
    assert result['archive_sha256']==sha((DATA/'run_records.zip').read_bytes())
    assert result['source_index_sha256']==sha((DATA/'run_sources.csv').read_bytes())
    return dict(status='PASS_RUN_SOURCE_MAPPING',checked_endpoint_rows=checked,unique_recorded_runs=len(index),
        initial_networks=len(starts),maximum_endpoint_cost_error=maximum)

if __name__=='__main__':print(json.dumps(verify(),indent=2))
