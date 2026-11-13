"""CLI parsing, RQ1 guard and per-run store set-up for the RQ2 training queue."""
import argparse
import configparser
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

RQ1_DIR = 'Malta/signal_route'
RQ1_STORE = 'Malta/signal_route/'
RQ1_CONFIG = 'config/config_ma2c_Malta.ini'
ZONES_SRC = 'Malta/data/zones_nso.csv'
NAME_RE = re.compile(r'^(?:[a-z][a-z0-9_]*|_smoke_[a-z]+)$')
OPTIONS = ('train', 'evaluate', 'train_then_evaluate', 'greedy', 'get_parameter')
MAIN_RE = re.compile(r'(?<!\w)main\.py(?!\w)')
RUN_ARG_RE = re.compile(r'(?:^|\s)--run(?:=|\s|$)')


@dataclass
class RunSpec:
    name: Optional[str]
    option: str
    levels: Optional[List[float]]
    rq1: bool
    config_path: str
    store_dir: str


def _die(code: int, msg: str):
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def parse_args(argv: List[str]) -> RunSpec:
    p = argparse.ArgumentParser(prog='main.py', add_help=False)
    p.add_argument('--run')
    p.add_argument('--option', choices=OPTIONS)
    p.add_argument('--levels')
    p.add_argument('--rq1', action='store_true')
    try:
        ns, _unknown = p.parse_known_args(argv)
    except SystemExit:
        raise SystemExit(2)
    levels = None
    if ns.levels:
        try:
            levels = [float(x) for x in ns.levels.split(',') if x.strip()]
        except ValueError:
            _die(2, 'invalid --levels: %r' % ns.levels)
    if ns.run is not None:
        if ns.rq1:
            _die(2, 'invalid: --rq1 cannot be combined with --run')
        if not NAME_RE.match(ns.run):
            _die(2, 'invalid run name: %r' % ns.run)
        option = ns.option or 'train'
        if option not in ('train', 'evaluate'):
            _die(2, 'invalid: --run only supports --option train or evaluate')
        return RunSpec(ns.run, option, levels, False,
                       'config/runs/%s.ini' % ns.run, 'Malta/runs/%s/' % ns.run)
    if not ns.rq1:
        _die(2, 'refusing: pass --run <name> or --rq1')
    return RunSpec(None, ns.option or 'train_then_evaluate', levels, True,
                   RQ1_CONFIG, RQ1_STORE)


def _norm(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def is_rq1_path(path: str) -> bool:
    p = _norm(path)
    root = _norm(RQ1_DIR)
    return p == root or p.startswith(root + os.sep)


def assert_not_rq1(path: str, rq1: bool) -> None:
    if is_rq1_path(path) and not rq1:
        _die(3, 'refusing to write into RQ1 store %s without --rq1' % path)


def _read_ini(path: str) -> configparser.ConfigParser:
    cp = configparser.ConfigParser(interpolation=None)
    with open(path, encoding='utf-8-sig') as f:
        cp.read_file(f)
    return cp


def configs_equal(a: str, b: str) -> Tuple[bool, List[str]]:
    ca, cb = _read_ini(a), _read_ini(b)
    diffs = set()
    for s in set(ca.sections()) ^ set(cb.sections()):
        diffs.add(s)
    for s in set(ca.sections()) & set(cb.sections()):
        da, db = dict(ca[s]), dict(cb[s])
        for k in set(da) | set(db):
            if k not in da or k not in db or da[k].strip() != db[k].strip():
                diffs.add('%s.%s' % (s, k))
    return (not diffs, sorted(diffs))


def parse_pid_json(text: str, own_pid: int) -> List[int]:
    text = (text or '').strip()
    if not text:
        return []
    data = json.loads(text)
    if isinstance(data, dict):
        data = [data]
    data = data or []
    parent = {}
    for item in data:
        if item.get('ProcessId') is not None and item.get('ParentProcessId') is not None:
            parent[int(item['ProcessId'])] = int(item['ParentProcessId'])
    skip = {own_pid}
    cur = own_pid
    while cur in parent and parent[cur] not in skip:
        cur = parent[cur]
        skip.add(cur)
    out = []
    for item in data:
        cmd = item.get('CommandLine')
        pid = item.get('ProcessId')
        if (cmd and pid is not None and int(pid) not in skip
                and MAIN_RE.search(cmd) and RUN_ARG_RE.search(cmd)):
            out.append(int(pid))
    return out


def other_run_pids(own_pid: int) -> List[int]:
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress")
    r = subprocess.run(['powershell', '-NoProfile', '-Command', ps],
                       capture_output=True, text=True)
    return parse_pid_json(r.stdout, own_pid)


def prepare_store(spec: RunSpec, maps_hash: str) -> bool:
    store = spec.store_dir
    cfg = os.path.join(store, 'config.ini')
    hash_path = os.path.join(store, 'neighbours.sha256')
    os.makedirs(store, exist_ok=True)
    if not os.path.exists(cfg):
        shutil.copyfile(ZONES_SRC, os.path.join(store, 'zones_nso.csv'))
        with open(hash_path, 'w', encoding='utf-8', newline='') as f:
            f.write(maps_hash + '\n')
        tmp = cfg + '.tmp'
        shutil.copyfile(spec.config_path, tmp)
        os.replace(tmp, cfg)
        return True
    ok, diffs = configs_equal(spec.config_path, cfg)
    stored = None
    if os.path.exists(hash_path):
        with open(hash_path, encoding='utf-8') as f:
            stored = f.read().strip()
    problems = []
    if not ok:
        problems.append('config differs: ' + ', '.join(diffs))
    if stored != maps_hash:
        problems.append('neighbour hash differs: stored=%s current=%s' % (stored, maps_hash))
    if problems:
        _die(4, 'resume refused for %s: %s' % (store, '; '.join(problems)))
    return False
