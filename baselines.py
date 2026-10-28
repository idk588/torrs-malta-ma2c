# -*- coding: utf-8 -*-
"""RQ1 baselines: SUMO's built-in controllers on the same demand as MA2C.

Runs plain `sumo` (no TraCI, no TensorFlow) per controller config over the
evaluation seeds, with the same network, corridor flows, cached background
traffic, seed and teleport setting as the MA2C environment:

  fixed_static        netconvert's static signal programs; routes never change
  fixed_rerouting     static signals + SUMO dynamic rerouting (device.rerouting)
  actuated_static     actuated signal programs; routes never change
  actuated_rerouting  actuated signals + device.rerouting

Writes per config to Malta/signal_route/baseline_data/<config>/:
  Malta_signal_route_trip.csv   same columns as env.collect_tripinfo
  summary_<seed>.xml, edgedata_<seed>.xml   (gitignored; read by stats_compare.py)

Usage:
    python baselines.py                    # all configs, all evaluation seeds
    python baselines.py fixed_static 1     # one config, first N seeds (smoke test)
"""
import os
import sys
import subprocess
import configparser
import xml.etree.ElementTree as ET

import pandas as pd

from Malta.build_file import output_flows, get_flows, gen_background_file, write_file

CONFIG = './config/config_ma2c_Malta.ini'
OUT_DIR = 'Malta/signal_route/baseline_data/'
PROBLEM = 'signal_route'
END = 3600
STEP = 5   # summary period, matches control_interval_sec in the env
# A* finds the same shortest paths as the default Dijkstra; Dijkstra took >12 min per seed here.
REROUTING = ['--device.rerouting.probability', '1', '--device.rerouting.period', '60',
             '--routing-algorithm', 'astar']
CONFIGS = {
    'fixed_static': ('static', []),
    'fixed_rerouting': ('static', REROUTING),
    'actuated_static': ('actuated', []),
    'actuated_rerouting': ('actuated', REROUTING),
}


def edge_ids(net):
    return {e.get('id') for _, e in ET.iterparse(net)
            if e.tag == 'edge' and e.get('function') != 'internal'}


def tl_types(net):
    return [e.get('type') for _, e in ET.iterparse(net) if e.tag == 'tlLogic']


def actuated_net(data_path):
    """Build the actuated-signal variant of the network once; edge IDs must survive."""
    out = OUT_DIR + 'malta_actuated.net.xml'
    if not os.path.exists(out):
        os.makedirs(OUT_DIR, exist_ok=True)
        subprocess.run(['netconvert', '-s', data_path + 'malta.net.xml',
                        '--tls.default-type', 'actuated', '--tls.rebuild', '-o', out],
                       check=True, capture_output=True)
        assert edge_ids(data_path + 'malta.net.xml') == edge_ids(out), 'netconvert changed edge IDs'
        types = tl_types(out)
        assert types and all(t == 'actuated' for t in types), 'signals not actuated: %s' % set(types)
    return out


def run(config, seed, episode, net, routes, data_path):
    """One SUMO run; returns tripinfo rows shaped like env.collect_tripinfo."""
    out = OUT_DIR + config + '/'
    trip_xml = out + 'trip_%s.xml' % seed
    cmd = ['sumo', '-n', net,
           '-r', routes + ',' + data_path + gen_background_file(data_path, seed),
           '--seed', str(seed), '--begin', '0', '--end', str(END),
           '--time-to-teleport', '-1', '--ignore-route-errors', 'false',
           '--no-step-log', 'true', '--duration-log.disable', 'true', '--no-warnings', 'true',
           '--tripinfo-output', trip_xml, '--tripinfo-output.write-unfinished', 'true',
           '--edgedata-output', out + 'edgedata_%s.xml' % seed,
           '--summary-output', out + 'summary_%s.xml' % seed,
           '--summary-output.period', str(STEP)] + CONFIGS[config][1]
    subprocess.run(cmd, check=True, capture_output=True)
    rows = [{'episode': episode, 'id': t.get('id'), 'depart_sec': t.get('depart'),
             'arrival_sec': t.get('arrival'), 'duration_sec': t.get('duration'),
             'wait_step': t.get('waitingCount'), 'wait_sec': t.get('waitingTime')}
            for t in ET.parse(trip_xml).getroot()]
    os.remove(trip_xml)
    return rows


def main():
    config = configparser.ConfigParser()
    config.read(CONFIG)
    data_path = config['ENV_CONFIG']['data_path']
    seeds = [int(s) for s in config['ENV_CONFIG']['test_seeds'].split(',')]
    names = list(CONFIGS)
    if len(sys.argv) > 1:
        names = [sys.argv[1]]
        if len(sys.argv) > 2:
            seeds = seeds[:int(sys.argv[2])]

    os.makedirs(OUT_DIR, exist_ok=True)
    routes = OUT_DIR + 'corridors.rou.xml'
    if not os.path.exists(routes):   # SUMO streams route files; never rewrite one a run may be reading
        write_file(routes, output_flows(get_flows()))
    nets = {'static': data_path + 'malta.net.xml'}
    if any(CONFIGS[n][0] == 'actuated' for n in names):
        nets['actuated'] = actuated_net(data_path)

    for name in names:
        os.makedirs(OUT_DIR + name, exist_ok=True)
        rows = []
        for i, seed in enumerate(seeds, 1):
            trips = run(name, seed, i, nets[CONFIGS[name][0]], routes, data_path)
            done = sum(1 for r in trips if float(r['arrival_sec']) >= 0)
            print('[%s] seed %d: %d trips, %d arrived' % (name, seed, len(trips), done), flush=True)
            rows += trips
        pd.DataFrame(rows).to_csv(OUT_DIR + name + '/Malta_%s_trip.csv' % PROBLEM)


if __name__ == '__main__':
    main()
