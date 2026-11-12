"""Build Malta/data/zones_nso.csv: each RQ2 agent's NSO district.

Usage (from the project root):
    python tools/build_zones.py [--net Malta/data/malta.net.xml]
                                [--councils Malta/data/councils_nso.csv]
                                [--out Malta/data/zones_nso.csv]

A signal agent's point is the junction with the agent's id; a route agent's point is the
from-junction of its up_edge (MaltaRoute.up_edge_map). Points are converted from net x/y to
lon/lat with the network's own projection (UTM 33N + netOffset). The island is decided
first (Gozo & Comino if lat > 36.005), then the nearest council of that island
(great-circle distance) gives the district.

Prints counts per district and per council, and the corridors whose edges cross districts.
Exit 0 on success. Exit 2, without writing the zone file, if any agent is more than 5 km
from its council, any point is outside lon 14.15-14.60 / lat 35.78-36.10, or the signal ids
differ from sorted(NODE) / the route ids from RC_NAMES.
"""
import argparse
import collections
import csv
import math
import os
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.zones import DISTRICTS, build_maps, global_order, load_zones, neighbour_hash

NET = 'Malta/data/malta.net.xml'
COUNCILS = 'Malta/data/councils_nso.csv'
OUT = 'Malta/data/zones_nso.csv'

GOZO_LAT = 36.005
LON_BOUNDS = (14.15, 14.60)
LAT_BOUNDS = (35.78, 36.10)
MAX_KM = 5.0
UTM_ZONE = 33
EARTH_RADIUS_KM = 6371.0088  # IUGG mean radius

ZONE_COLUMNS = ['agent_id', 'type', 'x', 'y', 'lon', 'lat', 'island', 'council', 'district']

# ---------------------------------------------------------------------------
# WGS84 transverse Mercator (UTM), Krueger series to n^6 (Karney 2011, eqs 35-36)
# ---------------------------------------------------------------------------
_A = 6378137.0
_F = 1 / 298.257223563
_K0 = 0.9996
_FALSE_EASTING = 500000.0
_ECC = math.sqrt(_F * (2 - _F))
_n = _F / (2 - _F)
_RECT = _A / (1 + _n) * (1 + _n ** 2 / 4 + _n ** 4 / 64 + _n ** 6 / 256)
_ALPHA = (
    _n / 2 - 2 * _n ** 2 / 3 + 5 * _n ** 3 / 16 + 41 * _n ** 4 / 180
    - 127 * _n ** 5 / 288 + 7891 * _n ** 6 / 37800,
    13 * _n ** 2 / 48 - 3 * _n ** 3 / 5 + 557 * _n ** 4 / 1440
    + 281 * _n ** 5 / 630 - 1983433 * _n ** 6 / 1935360,
    61 * _n ** 3 / 240 - 103 * _n ** 4 / 140 + 15061 * _n ** 5 / 26880
    + 167603 * _n ** 6 / 181440,
    49561 * _n ** 4 / 161280 - 179 * _n ** 5 / 168 + 6601661 * _n ** 6 / 7257600,
    34729 * _n ** 5 / 80640 - 3418889 * _n ** 6 / 1995840,
    212378941 * _n ** 6 / 319334400,
)
_BETA = (
    _n / 2 - 2 * _n ** 2 / 3 + 37 * _n ** 3 / 96 - _n ** 4 / 360
    - 81 * _n ** 5 / 512 + 96199 * _n ** 6 / 604800,
    _n ** 2 / 48 + _n ** 3 / 15 - 437 * _n ** 4 / 1440 + 46 * _n ** 5 / 105
    - 1118711 * _n ** 6 / 3870720,
    17 * _n ** 3 / 480 - 37 * _n ** 4 / 840 - 209 * _n ** 5 / 4480
    + 5569 * _n ** 6 / 90720,
    4397 * _n ** 4 / 161280 - 11 * _n ** 5 / 504 - 830251 * _n ** 6 / 7257600,
    4583 * _n ** 5 / 161280 - 108847 * _n ** 6 / 3991680,
    20648693 * _n ** 6 / 638668800,
)


def _central_meridian(zone):
    return math.radians(6 * zone - 183)


def lonlat_to_utm(lon, lat, zone=UTM_ZONE):
    """WGS84 lon/lat (degrees) -> UTM (easting, northing) in metres, northern hemisphere."""
    phi = math.radians(lat)
    lam = math.radians(lon) - _central_meridian(zone)
    sin_phi = math.sin(phi)
    t = math.sinh(math.atanh(sin_phi) - _ECC * math.atanh(_ECC * sin_phi))
    xi_p = math.atan2(t, math.cos(lam))
    eta_p = math.atanh(math.sin(lam) / math.sqrt(1 + t * t))
    xi, eta = xi_p, eta_p
    for j, a in enumerate(_ALPHA, start=1):
        xi += a * math.sin(2 * j * xi_p) * math.cosh(2 * j * eta_p)
        eta += a * math.cos(2 * j * xi_p) * math.sinh(2 * j * eta_p)
    return _FALSE_EASTING + _K0 * _RECT * eta, _K0 * _RECT * xi


def utm_to_lonlat(e, n, zone=UTM_ZONE):
    """UTM (easting, northing) in metres, northern hemisphere -> WGS84 (lon, lat) degrees."""
    xi = n / (_K0 * _RECT)
    eta = (e - _FALSE_EASTING) / (_K0 * _RECT)
    xi_p, eta_p = xi, eta
    for j, b in enumerate(_BETA, start=1):
        xi_p -= b * math.sin(2 * j * xi) * math.cosh(2 * j * eta)
        eta_p -= b * math.cos(2 * j * xi) * math.sinh(2 * j * eta)
    lam = math.atan2(math.sinh(eta_p), math.cos(xi_p))
    tau_p = math.sin(xi_p) / math.sqrt(math.sinh(eta_p) ** 2 + math.cos(xi_p) ** 2)
    # Conformal -> geodetic latitude by Newton iteration on tau = tan(phi).
    e2 = _ECC * _ECC
    tau = tau_p
    for _ in range(20):
        sigma = math.sinh(_ECC * math.atanh(_ECC * tau / math.sqrt(1 + tau * tau)))
        tau_i = tau * math.sqrt(1 + sigma * sigma) - sigma * math.sqrt(1 + tau * tau)
        d_tau = ((tau_p - tau_i) / math.sqrt(1 + tau_i * tau_i)
                 * (1 + (1 - e2) * tau * tau) / ((1 - e2) * math.sqrt(1 + tau * tau)))
        tau += d_tau
        if abs(d_tau) < 1e-15:
            break
    return (math.degrees(_central_meridian(zone) + lam),
            math.degrees(math.atan(tau)))


# ---------------------------------------------------------------------------
# Network, councils, assignment
# ---------------------------------------------------------------------------
def read_net(net_path, info=None):
    """Stream the net file; return (netOffset, {junction: (x, y)}, {edge: from-junction}).

    Only top-level <location>, <junction> and <edge> elements are read; internal edges
    (no `from`) are skipped. If `info` is a dict, the <location> attributes are copied in.
    """
    offset = None
    junction_xy, edge_from = {}, {}
    depth, root = 0, None
    for event, elem in ET.iterparse(net_path, events=('start', 'end')):
        if event == 'start':
            if root is None:
                root = elem
            depth += 1
            continue
        depth -= 1
        if depth != 1:  # only direct children of <net>
            continue
        tag = elem.tag
        if tag == 'junction':
            junction_xy[elem.get('id')] = (float(elem.get('x')), float(elem.get('y')))
        elif tag == 'edge':
            frm = elem.get('from')
            if frm is not None:
                edge_from[elem.get('id')] = frm
        elif tag == 'location' and offset is None:
            ox, oy = elem.get('netOffset').split(',')
            offset = (float(ox), float(oy))
            if info is not None:
                info.update(elem.attrib)
        root.clear()
    if offset is None:
        raise ValueError('read_net: no <location> element in %s' % net_path)
    return offset, junction_xy, edge_from


def load_councils(csv_path):
    """Council table rows (UTF-8) with float lat/lon."""
    with open(csv_path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['lat'], r['lon'] = float(r['lat']), float(r['lon'])
    return rows


def great_circle_km(lon1, lat1, lon2, lat2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, h)))


def island_of(lat):
    return 'Gozo' if lat > GOZO_LAT else 'Malta'


def assign_council(lon, lat, councils):
    """(island, council row, km): island first, then the nearest council on that island."""
    island = island_of(lat)
    best, best_km = None, None
    for c in councils:
        if c['island'] != island:
            continue
        km = great_circle_km(lon, lat, c['lon'], c['lat'])
        if best_km is None or km < best_km:
            best, best_km = c, km
    if best is None:
        raise ValueError('assign_council: no council on island %r' % island)
    return island, best, best_km


def in_bounds(lon, lat):
    return LON_BOUNDS[0] <= lon <= LON_BOUNDS[1] and LAT_BOUNDS[0] <= lat <= LAT_BOUNDS[1]


def _runs(seq):
    """Run-length encode a sequence: [(value, count), ...]."""
    out = []
    for v in seq:
        if out and out[-1][0] == v:
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--net', default=NET)
    ap.add_argument('--councils', default=COUNCILS)
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    def path(p):
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    from envs.Malta_env import NODE, RC_NAMES, MaltaRoute
    route_map = MaltaRoute()
    councils = load_councils(path(args.councils))

    info = {}
    offset, junction_xy, edge_from = read_net(path(args.net), info)
    violations = []
    proj = info.get('projParameter', '')
    if '+proj=utm' not in proj or '+zone=%d ' % UTM_ZONE not in proj + ' ':
        violations.append('projection is not UTM zone %d: %r' % (UTM_ZONE, proj))
    print('net: %s  netOffset=%s  junctions=%d  edges=%d  proj=%s'
          % (args.net, offset, len(junction_xy), len(edge_from), proj))

    def locate(x, y):
        lon, lat = utm_to_lonlat(x - offset[0], y - offset[1], UTM_ZONE)
        island, council, km = assign_council(lon, lat, councils)
        return lon, lat, island, council, km

    rows, dist_km = [], {}
    for agent_id, kind, junction in (
            [(s, 'signal', s) for s in sorted(NODE)]
            + [(rc, 'route', edge_from.get(route_map.up_edge_map.get(rc, ''))) for rc in RC_NAMES]):
        if junction is None or junction not in junction_xy:
            violations.append('%s %s: point junction %r not in the net'
                              % (kind, agent_id, junction))
            continue
        x, y = junction_xy[junction]
        lon, lat, island, council, km = locate(x, y)
        if not in_bounds(lon, lat):
            violations.append('%s %s: (%.6f, %.6f) outside lon %s / lat %s'
                              % (kind, agent_id, lon, lat, LON_BOUNDS, LAT_BOUNDS))
        if km > MAX_KM:
            violations.append('%s %s: %.2f km from %s (> %.1f km)'
                              % (kind, agent_id, km, council['council'], MAX_KM))
        dist_km[agent_id] = km
        rows.append({'agent_id': agent_id, 'type': kind, 'x': '%.2f' % x, 'y': '%.2f' % y,
                     'lon': '%.7f' % lon, 'lat': '%.7f' % lat, 'island': island,
                     'council': council['council'], 'district': council['district']})
        if kind == 'route':
            print('route %s: up_edge %s from-junction %s  lon %.6f lat %.6f  -> %s, %s (%.2f km)'
                  % (agent_id, route_map.up_edge_map[agent_id], junction, lon, lat,
                     council['council'], council['district'], km))

    signal_ids = [r['agent_id'] for r in rows if r['type'] == 'signal']
    route_ids = [r['agent_id'] for r in rows if r['type'] == 'route']
    if signal_ids != sorted(NODE):
        violations.append('signal ids differ from sorted(NODE) (%d vs %d)'
                          % (len(signal_ids), len(NODE)))
    if route_ids != list(RC_NAMES):
        violations.append('route ids %s differ from RC_NAMES %s' % (route_ids, RC_NAMES))

    # ---- counts per district -------------------------------------------------
    print('\nAgents per district (signals + routes = total):')
    by_district = collections.OrderedDict((d, [0, 0]) for d in DISTRICTS)
    for r in rows:
        by_district[r['district']][0 if r['type'] == 'signal' else 1] += 1
    for d, (ns, nr) in by_district.items():
        print('  %-17s %3d + %d = %3d' % (d, ns, nr, ns + nr))
    print('  %-17s %3d + %d = %3d' % ('TOTAL', len(signal_ids), len(route_ids), len(rows)))

    # ---- counts per council --------------------------------------------------
    print('\nAgents per council (max distance to council point):')
    by_council = collections.OrderedDict(((c['council'], c['district']), []) for c in councils)
    for r in rows:
        by_council[(r['council'], r['district'])].append(r['agent_id'])
    for d in DISTRICTS:
        print('  %s' % d)
        empty = []
        for (name, dist), members in by_council.items():
            if dist != d:
                continue
            if not members:
                empty.append(name)
                continue
            nr = sum(1 for m in members if m in RC_NAMES)
            print('    %-18s %3d%s  max %.2f km' % (name, len(members),
                                                 ' (incl. %d route)' % nr if nr else '',
                                                 max(dist_km[m] for m in members)))
        if empty:
            print('    no agents: %s' % ', '.join(empty))
    if dist_km:
        far = max(dist_km, key=dist_km.get)
        print('  farthest agent: %s, %.2f km from its council' % (far, dist_km[far]))

    # ---- corridors crossing districts ---------------------------------------
    print('\nCorridors (district of each edge\'s from-junction, in route order):')
    district_of_junction = {}
    for rc in RC_NAMES:
        alts = route_map.forward_route_map.get(rc, [])
        seen = []
        lines = []
        for label, edges in zip('AB', alts):
            seq = []
            for eid in edges:
                j = edge_from.get(eid)
                if j is None or j not in junction_xy:
                    print('  WARNING: %s edge %s not found in the net' % (rc, eid))
                    continue
                if j not in district_of_junction:
                    district_of_junction[j] = locate(*junction_xy[j])[3]['district']
                seq.append(district_of_junction[j])
            for d in seq:
                if d not in seen:
                    seen.append(d)
            lines.append('    alt %s (%d edges): %s' % (
                label, len(edges), ' > '.join('%s x%d' % (d, k) for d, k in _runs(seq))))
        own = next((r['district'] for r in rows if r['agent_id'] == rc), '?')
        status = 'CROSSES %d districts' % len(seen) if len(seen) > 1 else 'stays in one district'
        print('  %s [agent district: %s] %s' % (rc, own, status))
        for line in lines:
            print(line)

    # ---- neighbour maps (for reference) ---------------------------------------
    if not violations:
        zone_of = {r['agent_id']: r['district'] for r in rows}
        for arch in ('cloud', 'fog'):
            maps = build_maps(arch, list(NODE), list(RC_NAMES), zone_of)
            print('\n%s maps: neighbour_hash %s' % (arch, neighbour_hash(*maps)))
        ss, sr, rr, rs = build_maps('fog', list(NODE), list(RC_NAMES), zone_of)
        print('fog: signal neighbours ss min/max %d/%d, route neighbours per route %s'
              % (min(map(len, ss.values())), max(map(len, ss.values())),
                 {rc: (len(rr[rc]), len(rs[rc])) for rc in RC_NAMES}))

    if violations:
        print('\nVIOLATIONS (%d):' % len(violations))
        for v in violations:
            print('  ' + v)
        print('%s NOT written.' % args.out)
        return 2

    out = path(args.out)
    with open(out, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ZONE_COLUMNS, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    zones = load_zones(out)
    if list(zones) != global_order(NODE, RC_NAMES):
        print('Written zone file does not follow global_order(NODE, RC_NAMES).')
        return 2
    print('\nwrote %d agents to %s' % (len(rows), args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
