"""NSO district zones for the RQ2 fog, hybrid and cloud architectures.

Pure functions only (no SUMO, no TF):
- load_zones: agent id -> district, from Malta/data/zones_nso.csv (built by tools/build_zones.py)
- build_maps: the four neighbour maps (ss, sr, rr, rs) for 'cloud', 'fog' or 'hybrid'
- global_order / neighbour_hash: deterministic agent order and a hash of the maps
- district_summary: the hybrid arm's 12 values [q_1..q_6, w_1..w_6]
"""
import csv
import hashlib
import json

import numpy as np

DISTRICTS = ('Southern Harbour', 'Northern Harbour', 'South Eastern',
             'Western', 'Northern', 'Gozo & Comino')

COMM_ARCHS = ('cloud', 'fog', 'hybrid')


def load_zones(csv_path):
    """Read a zone file (UTF-8) and return {agent_id: district}."""
    zones = {}
    with open(csv_path, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            agent, district = row['agent_id'], row['district']
            if agent in zones:
                raise ValueError('duplicate agent_id %r in %s' % (agent, csv_path))
            if district not in DISTRICTS:
                raise ValueError('unknown district %r for %r in %s' % (district, agent, csv_path))
            zones[agent] = district
    return zones


def global_order(signal_ids, route_ids):
    """Global agent order: sorted signal ids, then route ids in the given order."""
    return sorted(signal_ids) + list(route_ids)


def build_maps(comm_arch, signal_ids, route_ids, zone_of):
    """Build (ss, sr, rr, rs) for 'cloud', 'fog' or 'hybrid'.

    ss = {sig: [sig, ...]}, sr = {sig: {rc: 1.0}}, rr = {rc: [rc, ...]}, rs = {rc: {sig: 1.0}}.
    Keys and neighbour lists follow global_order(); an agent is never its own neighbour.
    cloud: every other agent is a neighbour. fog/hybrid: every other agent in the same
    district (zone_of[agent]); hybrid's maps are the fog maps.
    """
    if comm_arch not in COMM_ARCHS:
        raise ValueError('build_maps: comm_arch must be one of %s, got %r'
                         % (COMM_ARCHS, comm_arch))
    order = global_order(signal_ids, route_ids)
    if len(set(order)) != len(order):
        raise ValueError('build_maps: duplicate agent ids')
    is_signal = {a: i < len(signal_ids) for i, a in enumerate(order)}

    if comm_arch == 'cloud':
        def same_group(a, b):
            return True
    else:
        missing = [a for a in order if a not in zone_of]
        if missing:
            raise ValueError('build_maps: no district for %s' % missing[:10])
        unknown = [a for a in order if zone_of[a] not in DISTRICTS]
        if unknown:
            raise ValueError('build_maps: unknown district for %s'
                             % [(a, zone_of[a]) for a in unknown[:10]])

        def same_group(a, b):
            return zone_of[a] == zone_of[b]

    ss, sr, rr, rs = {}, {}, {}, {}
    for a in order:
        nbrs = [b for b in order if b != a and same_group(a, b)]
        sig_nbrs = [b for b in nbrs if is_signal[b]]
        rc_nbrs = [b for b in nbrs if not is_signal[b]]
        if is_signal[a]:
            ss[a] = sig_nbrs
            sr[a] = {b: 1.0 for b in rc_nbrs}
        else:
            rr[a] = rc_nbrs
            rs[a] = {b: 1.0 for b in sig_nbrs}
    return ss, sr, rr, rs


def neighbour_hash(ss, sr, rr, rs):
    """SHA-256 hex of the four maps (insertion order preserved)."""
    text = json.dumps([ss, sr, rr, rs], ensure_ascii=False)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def district_summary(wait_means, vehwait_per_lane, district_idx, norm_wait, clip_wait):
    """Hybrid summary: np.array of 12 floats [q_1..q_6, w_1..w_6].

    Per-signal inputs. district_idx holds indices into DISTRICTS (-1 = ignored; anything
    outside [-1, 5] raises ValueError).
    q_d = mean of wait_means over district d (already normalised; NaN = no detectors,
    ignored). w_d = mean of clip(vehwait_per_lane / norm_wait, 0, clip_wait) over
    district d. A district with no (finite) values gives 0.0.
    """
    wait_means = np.asarray(wait_means, dtype=float).ravel()
    vehwait = np.asarray(vehwait_per_lane, dtype=float).ravel()
    idx = np.asarray(district_idx, dtype=int).ravel()
    if not (len(wait_means) == len(vehwait) == len(idx)):
        raise ValueError('district_summary: input lengths differ (%d, %d, %d)'
                         % (len(wait_means), len(vehwait), len(idx)))
    if not norm_wait > 0:
        raise ValueError('district_summary: norm_wait must be > 0, got %r' % norm_wait)
    n = len(DISTRICTS)
    bad = idx[(idx < -1) | (idx >= n)]
    if bad.size:
        raise ValueError('district_summary: district_idx must be in [-1, %d], got %s'
                         % (n - 1, sorted(set(bad.tolist()))[:10]))
    w_node = np.clip(vehwait / float(norm_wait), 0.0, float(clip_wait))
    out = np.zeros(2 * n)
    for d in range(n):
        members = idx == d
        q = wait_means[members]
        q = q[np.isfinite(q)]
        w = w_node[members]
        w = w[np.isfinite(w)]
        out[d] = q.mean() if q.size else 0.0
        out[n + d] = w.mean() if w.size else 0.0
    return out
