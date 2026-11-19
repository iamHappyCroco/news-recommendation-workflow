"""Sample current end-to-end workflow results across quadrants from the source workbook."""
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import pandas as pd
from batch_run import _run_one, _validate_final_summary, prepare_article_input, warmup
from tools.search_classifier import classify_article

OUT = ROOT / 'outputs' / 'quadrant_v2_sample_20261119'
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = ROOT / 'assets' / '要闻资讯拉取_2026-10-27.xlsx'

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

async def main():
    frame = pd.read_excel(SOURCE, dtype=str, keep_default_na=False)
    quadrant_map = json.loads((ROOT / 'assets/category_quadrants_82.json').read_text(encoding='utf-8'))['topics']
    print(f'Read {len(frame)} articles; warming classifier', flush=True)
    await asyncio.to_thread(warmup)
    pools = {r: [] for r in 'ABCD'}
    scanned = []
    for index, row in frame.iterrows():
        payload = prepare_article_input(row['资讯标题'], row['资讯全文'])
        if not payload['title'] or not payload['content']:
            continue
        model = await asyncio.to_thread(classify_article, payload['title'], payload['content'])
        region = quadrant_map.get(str(model['leaf_id']), {}).get('region', '')
        entry = dict(excel_row=int(index)+2, article_id=row['资讯ID'], title=payload['title'],
                     original_content_chars=len(row['资讯全文']), input_content_chars=len(payload['content']),
                     payload=payload, model=model, predicted_region=region)
        scanned.append(entry)
        if region in pools:
            pools[region].append(entry)
        if len(scanned) % 20 == 0:
            print('Scan', len(scanned), {r: len(v) for r, v in pools.items()}, flush=True)
        if len(scanned) >= 100 and all(len(pools[r]) >= 4 for r in pools):
            break
    save('candidate_scan.json', scanned)
    for region in pools:
        pools[region].sort(key=lambda item: item['model']['similarity'], reverse=True)
    configs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'config').glob('*llm_cfg.json')}
    save('run_metadata.json', dict(source=str(SOURCE), source_rows=len(frame), scanned_rows=len(scanned),
                                  candidate_counts={r:len(v) for r,v in pools.items()}, config_sha256=configs,
                                  model=os.getenv('EM_MODEL', 'DeepSeek-V4-Flash-0731'),
                                  selection='按本地模型预测象限分组、组内相似度优先；最终象限以完整工作流校验结果为准'))
    records = []
    covered = Counter()
    used = set()
    semaphore = asyncio.Semaphore(4)

    async def run(entry):
        async with semaphore:
            started = time.perf_counter()
            record = {k:v for k,v in entry.items() if k not in ('payload', 'model')}
            print(f"Start Excel row {entry['excel_row']} predicted {entry['predicted_region']}: {entry['title']}", flush=True)
            for attempt in range(1, 4):
                try:
                    result = await _run_one(entry['payload'])
                    _validate_final_summary(result['final_summary'])
                    record.update(status='success', final_summary=result['final_summary'])
                    region = result['final_summary']['category_quadrant']['region']
                    for r in set(region.split('/')):
                        covered[r] += 1
                    print(f"Done Excel row {entry['excel_row']} final {region}", flush=True)
                    break
                except Exception as exc:
                    print(f"Excel row {entry['excel_row']} attempt {attempt}: {type(exc).__name__}: {exc}", flush=True)
                    if attempt == 3:
                        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                    else:
                        await asyncio.sleep(2**attempt)
            record.update(attempts=attempt, elapsed_seconds=round(time.perf_counter()-started, 2))
            records.append(record)
            if record['status'] == 'success':
                folder = OUT / f"excel_row_{record['excel_row']}"
                folder.mkdir(exist_ok=True)
                (folder / 'final_summary.json').write_text(
                    json.dumps({'final_summary': record['final_summary']}, ensure_ascii=False, indent=2),
                    encoding='utf-8',
                )
            save('final_summaries.json', [
                {key:item[key] for key in ('excel_row', 'title', 'final_summary')}
                for item in sorted(records, key=lambda item:item['excel_row']) if item['status'] == 'success'
            ])

    initial = [entry for region in 'ABCD' for entry in pools[region][:2]]
    used.update(entry['excel_row'] for entry in initial)
    await asyncio.gather(*(run(entry) for entry in initial))
    for _ in range(0):
        missing = [r for r in 'ABCD' if covered[r] < 1]
        next_entries = []
        for region in missing:
            options = [entry for entry in pools[region] if entry['excel_row'] not in used]
            next_entries.extend(options[:2])
        if not next_entries:
            break
        used.update(entry['excel_row'] for entry in next_entries)
        await asyncio.gather(*(run(entry) for entry in next_entries))
    print('COMPLETE', dict(covered), 'success', sum(r['status']=='success' for r in records), 'failed', sum(r['status']=='failed' for r in records), flush=True)

if __name__ == '__main__':
    asyncio.run(main())
