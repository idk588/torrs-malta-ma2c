"""Pre-generate and validate background-traffic route files.

Usage (from the project root):
    python tools/gen_traffic.py --seeds 12-111 --factors 1.0 [--workers 4] [--check-only]
    python tools/gen_traffic.py --seeds test --factors 1.00,1.06,1.10 --check-only

Exit code 0 only if every requested file is valid.
"""
import argparse
import configparser
import os
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Malta.build_file import bg_period, bg_tag, validate_bg_file, gen_background_file

DATA_DIR = 'Malta/data/'
CONFIG = os.path.join(ROOT, 'config', 'config_ma2c_Malta.ini')


def _config_test_seeds():
    cp = configparser.ConfigParser()
    cp.read(CONFIG)
    return [int(s) for s in cp['ENV_CONFIG']['test_seeds'].split(',') if s.strip()]


def parse_seeds(text):
    text = text.strip()
    if text == 'test':
        return _config_test_seeds()
    if ',' not in text and '-' in text:
        lo, hi = text.split('-')
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(',') if s.strip()]


def needed_files(train_seeds, test_seeds, levels):
    """(seed, period) pairs: training seeds at factor 1.0, test seeds at every level."""
    out, seen = [], set()
    items = [(s, bg_period(1.0)) for s in train_seeds]
    items += [(s, bg_period(f)) for s in test_seeds for f in levels]
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _process(item, check_only, data_dir):
    seed, period = item
    tag = bg_tag(seed, period)
    rou = os.path.join(data_dir, tag + '.rou.xml')
    status = 'ok'
    if os.path.exists(rou):
        ok, n = validate_bg_file(rou, period)
        if ok:
            return tag, True, 'ok (%d vehicles)' % n
        if check_only:
            return tag, False, 'INVALID (%d vehicles)' % n
        os.remove(rou)
        status = 'regenerated (was invalid)'
    elif check_only:
        return tag, False, 'MISSING'
    else:
        status = 'generated'
    try:
        gen_background_file(data_dir, seed, period)
    except Exception as e:
        return tag, False, 'FAILED: %s' % e
    ok, n = validate_bg_file(rou, period)
    if not ok:
        return tag, False, 'INVALID after generation (%d vehicles)' % n
    return tag, True, '%s (%d vehicles)' % (status, n)


def _run_chunk(chunk, check_only, data_dir):
    return [_process(it, check_only, data_dir) for it in chunk]


def run(items, workers, check_only, data_dir=DATA_DIR):
    workers = max(1, min(workers, len(items) or 1))
    chunks = [items[i::workers] for i in range(workers)]  # disjoint
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_run_chunk, c, check_only, data_dir) for c in chunks]
        for f in futs:
            results.extend(f.result())
    order = {bg_tag(s, p): i for i, (s, p) in enumerate(items)}
    results.sort(key=lambda r: order[r[0]])
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--seeds', required=True, help="'12-111', '10000,20000' or 'test'")
    ap.add_argument('--factors', default='1.0', help='comma-separated demand factors')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--check-only', action='store_true')
    a = ap.parse_args(argv)

    seeds = parse_seeds(a.seeds)
    levels = [float(f) for f in a.factors.split(',') if f.strip()]
    base_only = levels == [1.0]
    if a.seeds.strip() == 'test':
        items = needed_files([], seeds, levels)
    else:
        items = needed_files(seeds, [], levels) if base_only else \
            [(s, bg_period(f)) for s in seeds for f in levels]
    os.chdir(ROOT)
    results = run(items, a.workers, a.check_only)
    for tag, ok, msg in results:
        print('%s: %s' % (tag, msg))
    bad = [r for r in results if not r[1]]
    print('%d files, %d valid, %d not valid' % (len(results), len(results) - len(bad), len(bad)))
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
