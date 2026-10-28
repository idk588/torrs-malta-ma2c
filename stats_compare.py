# -*- coding: utf-8 -*-
"""Per-seed metrics for every evaluated arm, and paired tests against a reference arm.

Every metric comes from SUMO outputs written per evaluation seed, so env arms
(MA2C, greedy) and plain-SUMO baselines are measured identically:

  trips, arrived, completion_pct  tripinfo incl. unfinished vehicles (arrival = -1)
  travel_s, wait_s                means over all inserted vehicles, unfinished included
  insert_pct                      inserted / loaded at the end (summary-output)
  congested_min                   minutes with > HALT_SHARE of running vehicles halted
  halted_veh_h                    vehicle-hours spent halted, network-wide
  congested_vt_share              share of vehicle-time on edges averaging under
                                  half their speed limit (edgedata)

Arms evaluated with the pre-October pipeline have no summary/edgedata and
exclude unfinished vehicles; those metrics come out NaN or optimistic.

Usage:
    python stats_compare.py                      # ref = ma2c, default arms
    python stats_compare.py fixed_static greedy actuated_static
"""
import os
import sys
import configparser
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy import stats

CONFIG = './config/config_ma2c_Malta.ini'
TRIP = 'Malta_signal_route_trip.csv'
OUT_DIR = 'Malta/signal_route/stats/'
HALT_SHARE = 0.10   # one vehicle in ten queued counts as congested
SLOW = 0.5          # edge hour-average speed below half its limit
B = 'Malta/signal_route/'
ARMS = {
    'ma2c': B + 'eva_data/',
    'greedy': B + 'greedy_data/',
    'fixed_static': B + 'baseline_data/fixed_static/',
    'fixed_rerouting': B + 'baseline_data/fixed_rerouting/',
    'actuated_static': B + 'baseline_data/actuated_static/',
    'actuated_rerouting': B + 'baseline_data/actuated_rerouting/',
    'routing_on': B + 'ablation_data/routing_on/',
    'routing_off': B + 'ablation_data/routing_off/',
}
DEFAULT = ['ma2c', 'greedy', 'fixed_static', 'fixed_rerouting', 'actuated_static', 'actuated_rerouting']
METRICS = ['completion_pct', 'travel_s', 'wait_s', 'insert_pct',
           'congested_min', 'halted_veh_h', 'congested_vt_share']


def summary_metrics(path):
    if not os.path.exists(path):
        return {}
    s = pd.DataFrame([e.attrib for e in ET.parse(path).getroot()]).astype(float)
    step = s['time'].diff().median()
    share = s['halting'] / s['running'].where(s['running'] > 0)
    last = s.iloc[-1]
    return {'insert_pct': last['inserted'] / last['loaded'] * 100,
            'congested_min': (share > HALT_SHARE).sum() * step / 60,
            'halted_veh_h': s['halting'].sum() * step / 3600}


def edge_metrics(path):
    if not os.path.exists(path):
        return {}
    vt = slow = 0.0
    for _, e in ET.iterparse(path):
        if e.tag == 'edge' and 'speedRelative' in e.attrib:
            t = float(e.get('sampledSeconds', 0))
            vt += t
            slow += t if float(e.get('speedRelative')) < SLOW else 0.0
        e.clear()
    return {'congested_vt_share': slow / vt if vt else np.nan}


def arm_metrics(name, seeds):
    d = ARMS[name]
    trip = pd.read_csv(d + TRIP)
    rows = []
    for ep, seed in enumerate(seeds, 1):
        t = trip[trip['episode'] == ep]
        if t.empty:
            continue
        arrived = (t['arrival_sec'].astype(float) >= 0).sum()
        r = {'arm': name, 'seed': seed, 'trips': len(t), 'arrived': arrived,
             'completion_pct': arrived / len(t) * 100,
             'travel_s': t['duration_sec'].astype(float).mean(),
             'wait_s': t['wait_sec'].astype(float).mean()}
        r.update(summary_metrics(d + 'summary_%s.xml' % seed))
        r.update(edge_metrics(d + 'edgedata_%s.xml' % seed))
        rows.append(r)
    assert rows, '%s: no trips matched the evaluation seeds' % name
    return rows


def compare(df, ref):
    out = []
    base = df[df['arm'] == ref].set_index('seed')
    for arm in df['arm'].unique():
        if arm == ref:
            continue
        cur = df[df['arm'] == arm].set_index('seed')
        for m in METRICS:
            pair = pd.concat([cur[m], base[m]], axis=1, keys=['a', 'r']).dropna()
            if len(pair) < 2:
                continue
            d = pair['a'] - pair['r']
            n, sd = len(d), d.std(ddof=1)
            half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if sd > 0 else 0.0
            try:
                w_p = stats.wilcoxon(pair['a'], pair['r']).pvalue
            except ValueError:   # all differences zero
                w_p = np.nan
            out.append({'arm': arm, 'metric': m, 'n': n, 'arm_mean': pair['a'].mean(),
                        'ref_mean': pair['r'].mean(), 'diff': d.mean(),
                        'ci95_low': d.mean() - half, 'ci95_high': d.mean() + half,
                        't_p': stats.ttest_rel(pair['a'], pair['r']).pvalue if sd > 0 else np.nan,
                        'wilcoxon_p': w_p, 'cohens_dz': d.mean() / sd if sd > 0 else np.nan})
    return pd.DataFrame(out)


def main():
    config = configparser.ConfigParser()
    config.read(CONFIG)
    seeds = [int(s) for s in config['ENV_CONFIG']['test_seeds'].split(',')]
    ref = sys.argv[1] if len(sys.argv) > 1 else 'ma2c'
    names = sys.argv[1:] if len(sys.argv) > 2 else DEFAULT
    names = [n for n in dict.fromkeys([ref] + names) if os.path.exists(ARMS[n] + TRIP)]
    assert ref in names, 'reference arm %s has no trip data' % ref

    df = pd.DataFrame([r for n in names for r in arm_metrics(n, seeds)])
    df = df.reindex(columns=['arm', 'seed', 'trips', 'arrived'] + METRICS)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_DIR + 'per_seed_metrics.csv', index=False)
    cmp_df = compare(df, ref)
    cmp_df.to_csv(OUT_DIR + 'comparison_vs_%s.csv' % ref, index=False)

    pd.set_option('display.width', 200)
    print(df.groupby('arm')[['trips', 'arrived'] + METRICS].mean().round(2).to_string())
    print('\nvs %s:' % ref)
    print(cmp_df.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
