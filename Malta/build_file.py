# -*- coding: utf-8 -*-
"""
Malta/build_file.py
Replaces Sioux/data/build_file.py for the Malta road network.
Reads directly from malta.net.xml instead of CSV files.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict, deque

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
NET_FILE  = os.path.join(DATA_DIR, 'malta.net.xml')

MAX_CAR_NUM  = 30
SPEED_LIMIT  = 50
time_length  = 3600


# ── helpers ───────────────────────────────────────────────────────────────────

# Process-wide cache of the parsed net + derived structures.
# Parsing malta.net.xml is expensive (~2 MB, 400k+ edges), so we keep one copy.
_NET_CACHE = {}


def _parse_net():
    if 'root' not in _NET_CACHE:
        _NET_CACHE['root'] = ET.parse(NET_FILE).getroot()
    return _NET_CACHE['root']


def _lane_allows_passenger(lane_elem, edge_disallow, edge_allow):
    allow = lane_elem.attrib.get('allow', edge_allow)
    disallow = lane_elem.attrib.get('disallow', edge_disallow)
    if allow:
        return 'passenger' in allow.split()
    if disallow:
        return 'passenger' not in disallow.split()
    return True  # SUMO default: passenger allowed


def _build_net_index():
    """One-shot scan of malta.net.xml: identify passenger-allowed edges, build
    an edge connectivity graph from <connection> elements, and collect
    traffic-light junctions. Cached for the life of the process."""
    if _NET_CACHE.get('index_built'):
        return
    root = _parse_net()

    passenger_edges = set()
    edge_from = {}
    edge_to = {}
    for edge in root.findall('edge'):
        if edge.attrib.get('function') == 'internal':
            continue
        eid = edge.attrib.get('id', '')
        if not eid or eid.startswith(':'):
            continue
        e_disallow = edge.attrib.get('disallow', '')
        e_allow = edge.attrib.get('allow', '')
        lanes = edge.findall('lane')
        if not lanes:
            continue
        if any(_lane_allows_passenger(l, e_disallow, e_allow) for l in lanes):
            passenger_edges.add(eid)
            edge_from[eid] = edge.attrib.get('from', '')
            edge_to[eid] = edge.attrib.get('to', '')

    edge_graph = defaultdict(set)
    for conn in root.findall('connection'):
        fr = conn.attrib.get('from', '')
        to = conn.attrib.get('to', '')
        if fr in passenger_edges and to in passenger_edges:
            edge_graph[fr].add(to)

    tl_junctions = set()
    for j in root.findall('junction'):
        if j.attrib.get('type') == 'traffic_light':
            tl_junctions.add(j.attrib.get('id', ''))

    _NET_CACHE['passenger_edges'] = passenger_edges
    _NET_CACHE['edge_from'] = edge_from
    _NET_CACHE['edge_to'] = edge_to
    _NET_CACHE['edge_graph'] = edge_graph
    _NET_CACHE['tl_junctions'] = tl_junctions
    _NET_CACHE['index_built'] = True


def _route_is_valid(edge_seq):
    """A route is valid if every edge allows passenger and consecutive edges
    have a SUMO connection between them."""
    if not edge_seq:
        return False
    _build_net_index()
    passenger_edges = _NET_CACHE['passenger_edges']
    edge_graph = _NET_CACHE['edge_graph']
    for e in edge_seq:
        if e not in passenger_edges:
            return False
    for i in range(len(edge_seq) - 1):
        if edge_seq[i + 1] not in edge_graph.get(edge_seq[i], set()):
            return False
    return True


def _validate_routes(routes):
    """Assert every corridor is a usable route choice, or fail loudly.

    A corridor is only usable if both alternatives share their first edge:
    vehicles are collected on up_edge (= route_a[0]) and TraCI refuses a
    setRoute whose target does not contain the vehicle's current edge. An
    earlier version of this function silently replaced invalid routes with
    unrelated BFS paths, which produced corridors whose alternatives began
    in different parts of the network — every route swap then failed and the
    route agents became no-ops. Fail fast instead of corrupting the study.
    """
    _build_net_index()

    for i, route_set in enumerate(routes):
        tag = 'corridor %d' % (i + 1)
        if len(route_set) != 2:
            raise ValueError('%s: expected exactly 2 alternatives, got %d'
                             % (tag, len(route_set)))
        route_a, route_b = route_set
        for k, r in enumerate(route_set):
            if not r:
                raise ValueError('%s alt %d: empty route' % (tag, k + 1))
            if not _route_is_valid(r):
                raise ValueError(
                    '%s alt %d is not a connected passenger-allowed path: %r'
                    % (tag, k + 1, r))
        if route_a[0] != route_b[0]:
            raise ValueError(
                '%s: alternatives must share their first edge (the approach '
                'edge vehicles are rerouted from), got %r vs %r'
                % (tag, route_a[0], route_b[0]))
        if route_a == route_b:
            raise ValueError('%s: both alternatives are identical, so the '
                             'route agent can have no effect' % tag)
        if route_a[-1] != route_b[-1]:
            print('[Malta routes] warning: %s alternatives end on different '
                  'edges (%s vs %s) — they do not reconverge, so switching '
                  'changes destination rather than balancing load'
                  % (tag, route_a[-1], route_b[-1]))
    return routes


def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)


# ── required by Malta_env.py ──────────────────────────────────────────────────

def get_nodes(build=False):
    root = _parse_net()

    # Get valid TL IDs — must have >= 2 phases
    valid_tl_ids = set()
    for tl in root.findall('tlLogic'):
        tl_id  = tl.attrib.get('id')
        states = [p.attrib['state'] for p in tl.findall('phase') if 'state' in p.attrib]
        if tl_id and len(states) >= 2:
            valid_tl_ids.add(tl_id)

    tl_nodes, non_tl_nodes = [], []
    for j in root.findall('junction'):
        jid   = j.attrib.get('id', '')
        jtype = j.attrib.get('type', '')
        if jid.startswith(':'):
            continue
        if jtype == 'traffic_light':
            if jid in valid_tl_ids:
                tl_nodes.append(jid)
            else:
                # treat excluded TL junctions as non-TL
                non_tl_nodes.append(jid)
        elif jtype in ('priority', 'right_before_left', 'unregulated'):
            non_tl_nodes.append(jid)

    return tl_nodes, non_tl_nodes



def get_phases():
    root = _parse_net()
    phases = {}
    phase_node_map = {}
    for tl in root.findall('tlLogic'):
        tl_id  = tl.attrib.get('id')
        states = [p.attrib['state'] for p in tl.findall('phase') if 'state' in p.attrib]
        # SKIP junctions with only 1 phase — model needs at least 2 actions
        if tl_id and len(states) >= 2:
            phases[tl_id]          = states
            phase_node_map[tl_id]  = tl_id
    return phases, phase_node_map


def get_routes():
    """
    Route-choice alternatives for Malta corridors using real SUMO edge IDs.

    Each corridor is a genuine parallel route choice: both alternatives start
    on the same approach edge, diverge onto edge-disjoint branches, and
    reconverge on the same destination edge. That shared first edge is what
    makes rerouting possible at all — vehicles are collected on it (up_edge)
    and TraCI only accepts a setRoute whose target contains the vehicle's
    current edge. Because the two branches serve the same origin-destination
    pair, switching between them balances load across parallel paths instead
    of changing where vehicles are going.

    Corridors were derived from malta.net.xml by searching for diverge points
    whose branches reconverge, then ranked by traffic-light crossings, lane
    count and branch length, and selected to be spatially non-overlapping.
    """
    routes = [
        # Corridor 1: Msida-area arterial, diverge at -15806386#3,
        # reconverging on -1456229447#1 (808m vs 1055m, 4 TL crossings)
        [
            ['-15806386#3', '-15806418#4', '-15806418#3', '-15806418#2',
             '-15806418#1', '-15806418#0', '15806399#8', '15806399#9',
             '15806399#10', '-1456229447#1'],
            ['-15806386#3', '-15806386#2', '-15806386#1', '-15806386#0',
             '-1456229448', '-1456229447#1'],
        ],

        # Corridor 2: diverge at 1330839310#5, reconverging on 516164732#0
        # (874m vs 468m, 3 TL crossings)
        [
            ['1330839310#5', '-522027338#1', '-522027338#0', '136705430',
             '136705412', '522027337#0', '522027337#1', '522027337#2',
             '522027337#3', '516164732#0'],
            ['1330839310#5', '522027338#2', '522027338#3', '522027338#4',
             '-64444535#4', '-64444535#3', '-64444535#2', '-64444535#1',
             '-64444535#0', '516164732#0'],
        ],

        # Corridor 3: diverge at 1264339210, reconverging on 154271794
        # (593m vs 440m, 2 TL crossings)
        [
            ['1264339210', '364130462#0', '364130462#1', '30014934#1',
             '22895632#4', '22895632#5', '518049423', '-26544420#0',
             '154271794'],
            ['1264339210', '26440346#0', '26440346#1', '26440346#2',
             '26440346#3', '1387038412', '514551777', '1023447884',
             '154271794'],
        ],

        # Corridor 4: diverge at -24318333, reconverging on 1292523373
        # (520m vs 502m, 2 TL crossings)
        [
            ['-24318333', '-29434717#1', '24318330', '-374916246#3',
             '-374916246#2', '-374916246#1', '-374916246#0',
             '-1292523375#1', '-1292523375#0', '1292523373'],
            ['-24318333', '29434717#2', '29434717#3', '29434717#4',
             '-29434716#2', '-29434716#1', '-29434716#0', '374916256',
             '1292523374', '1292523373'],
        ],
    ]

    routes = _validate_routes(routes)
    rc_num = sum(len(r) - 1 for r in routes)
    return routes, rc_num


def relevant_ss(build=False):
    """
    Return a dict mapping each TL node to its directly connected TL neighbours.
    Built automatically from the edges in malta.net.xml.
    """
    root = _parse_net()
    tl_nodes, _ = get_nodes()
    tl_set = set(tl_nodes)

    relevant_ss_map = {n: [] for n in tl_nodes}
    for edge in root.findall('edge'):
        if edge.attrib.get('function') == 'internal':
            continue
        frm = edge.attrib.get('from', '')
        to  = edge.attrib.get('to', '')
        if frm in tl_set and to in tl_set:
            if to not in relevant_ss_map[frm]:
                relevant_ss_map[frm].append(to)
    return relevant_ss_map


def get_flows(build=False):
    """
    Vehicles/hour per corridor calibrated to Malta peak-hour conditions.
    Index matches corridor order in get_routes().
    Corridor 1: main route-choice point (high demand, Msida-Valletta type)
    Corridors 2-4: bidirectional arterial flows
    """
    return [600, 300, 400, 350]


# ── route/config file generation ─────────────────────────────────────────────

def _get_edge_ids_for_route(edge_sequence):
    """Routes are already edge IDs — return them directly."""
    return edge_sequence


def output_flows(demand_flows, seed=None, load=False):
    routes, _ = get_routes()
    demand_begin = 0
    demand_end   = 3600
    demand_gap   = 360

    ratios = [
        [0.3, 0.5, 0.4, 0.3, 0.8, 0.9, 1.0, 0.9, 0.7, 0.2],
        [0.3, 0.2, 0.4, 0.3, 0.5, 0.5, 0.8, 0.8, 0.9, 1.0],
        [0.2, 0.3, 0.5, 0.4, 0.7, 0.8, 1.0, 0.9, 0.6, 0.3],
        [0.4, 0.5, 0.6, 0.5, 0.8, 0.9, 1.0, 0.8, 0.5, 0.2],
    ]

    if seed is not None:
        np.random.seed(seed)

    str_flows = '<routes>\n'
    str_flows += '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50"/>\n'

    # Define route alternatives — edges already provided directly
    for i, route_set in enumerate(routes):
        for k, edge_seq in enumerate(route_set):
            edges = ' '.join(edge_seq)
            str_flows += '    <route id="route_%d_%d" edges="%s"/>\n' % (i+1, k+1, edges)

    # Generate flows per time interval
    for j in range(10):
        t_begin = j * demand_gap
        t_end   = (j + 1) * demand_gap
        for i, route_set in enumerate(routes):
            flow_val = ratios[i % len(ratios)][j] * demand_flows[i % len(demand_flows)]
            if load:
                flow_val = int(flow_val * load)
            str_flows += (
                '    <flow id="flow_%d_%d" route="route_%d_1" '
                'begin="%d" end="%d" vehsPerHour="%.1f" type="car"/>\n'
                % (i+1, j+1, i+1, t_begin, t_end, flow_val)
            )

    str_flows += '</routes>\n'
    return str_flows


def output_config(thread=None, add_file=None):
    if thread is None:
        rou_file = 'exp.rou.xml'
    else:
        rou_file = 'exp_%d.rou.xml' % int(thread)

    add_line = ''
    if add_file:
        add_line = '        <additional-files value="%s"/>\n' % add_file

    cfg = (
        '<configuration>\n'
        '    <input>\n'
        '        <net-file value="malta.net.xml"/>\n'
        '        <route-files value="%s"/>\n'
        '%s'
        '    </input>\n'
        '    <time>\n'
        '        <begin value="0"/>\n'
        '        <end value="%d"/>\n'
        '    </time>\n'
        '    <processing>\n'
        '        <time-to-teleport value="-1"/>\n'
        '    </processing>\n'
        '</configuration>\n'
    ) % (rou_file, add_line, time_length)
    return cfg


def output_additional(data_path):
    """
    Generate E2 lane-area detectors only on lanes that lead INTO traffic-light
    junctions. The env's _init_agents only ever queries detectors that match
    trafficlight.getControlledLanes(), so emitting one per lane in the entire
    network (~120k detectors) was massive wasted work at SUMO startup.

    Also: dropped freq=1 + per-second file output (was ~430M lines/episode).
    The env reads detector state via TraCI getLastStep* calls, which work
    independently of the aggregation period.
    """
    root = _parse_net()

    tl_junctions = set()
    for j in root.findall('junction'):
        if j.attrib.get('type') == 'traffic_light':
            tl_junctions.add(j.attrib.get('id', ''))

    str_add = '<additional>\n'
    count = 0
    for edge in root.findall('edge'):
        if edge.attrib.get('function') == 'internal':
            continue
        eid = edge.attrib.get('id', '')
        if not eid or eid.startswith(':'):
            continue
        # Only lanes feeding a TL junction matter for the agents
        if edge.attrib.get('to', '') not in tl_junctions:
            continue
        for lane in edge.findall('lane'):
            lane_id = lane.attrib.get('id', '')
            lane_len = float(lane.attrib.get('length', 100))
            det_len = min(lane_len - 0.1, 100)
            if det_len <= 0:
                det_len = 1.0
            # SUMO requires both `file` and a period; we set period to the
            # full episode length so only ~one row per detector is ever
            # written (442 rows total, negligible). Per-step queries via
            # TraCI getLastStep* still work regardless of period.
            str_add += (
                '    <laneAreaDetector id="%s_0" lane="%s" '
                'pos="0" length="%.2f" period="3600" file="ild_out.xml"/>\n'
                % (lane_id, lane_id, det_len)
            )
            count += 1
    str_add += '</additional>\n'
    print('[Malta detectors] emitted %d laneAreaDetectors (TL-incoming lanes only)' % count)
    return str_add


def gen_rou_file(path, demand_flows, seed=None, thread=None, load=False):
    if thread is None:
        flow_file = 'exp.rou.xml'
    else:
        flow_file = 'exp_%d.rou.xml' % int(thread)

    write_file(path + flow_file, output_flows(demand_flows, seed=seed, load=load))

    # generate detector additional file
    add_file = path + 'exp.add.xml'
    write_file(add_file, output_additional(path))

    sumocfg_file = path + ('exp_%d.sumocfg' % thread)
    write_file(sumocfg_file, output_config(thread=thread, add_file='exp.add.xml'))
    return sumocfg_file