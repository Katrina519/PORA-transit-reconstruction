"""封存已完成的运行、统计与Excel；不执行搜索或改动已有数据。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    request = read(ROOT / '请求冻结.json')
    for path, expected in request['hashes'].items():
        assert sha(Path(path)) == expected, path
    stats = read(ROOT / '统计与核验.json')
    excel = read(ROOT / 'Excel核验.json')
    assert stats['passed'] and excel['passed']
    assert stats['method_runs'] == 20 and stats['paired_runs'] == 10
    assert (ROOT / '独立最终统计复核.md').is_file()
    files = [p for p in ROOT.iterdir() if p.is_file() and p.name != '交付封存.json']
    files += [p for p in (ROOT / 'frozen_N24').iterdir() if p.is_file()]
    files += [p for p in (ROOT / 'outputs').rglob('*') if p.is_file()]
    output = dict(
        complete=True, source_hashes_unchanged=len(request['hashes']),
        primary_endpoint='8round', secondary_endpoint='120s',
        seeds=request['seeds'], summary=stats['summary'],
        author_and_generator_unchanged=True, no_retraining=True,
        previous_Mandl_Mumford0_unchanged=True,
        files={str(p.relative_to(ROOT)):dict(sha256=sha(p), bytes=p.stat().st_size)
               for p in sorted(files)},
    )
    with (ROOT / '交付封存.json').open('x', encoding='utf-8') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(complete=True, files=len(files),
                         source_hashes_unchanged=len(request['hashes'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
