import collections
import csv
import math
import os
import shutil
import tempfile
import unittest

import numpy as np

from envs.zones import (DISTRICTS, build_maps, district_summary, global_order,
                        load_zones, neighbour_hash)
from tools.build_zones import (assign_council, load_councils, lonlat_to_utm,
                               read_net, utm_to_lonlat)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COUNCILS_CSV = os.path.join(ROOT, 'Malta', 'data', 'councils_nso.csv')
ZONES_CSV = os.path.join(ROOT, 'Malta', 'data', 'zones_nso.csv')

NET_OFFSET = (-426538.16, -3962602.48)
CONV_BOUNDARY = (0.0, 0.0, 35083.37, 30711.08)


def _meridian_arc(lat_deg, steps=2000):
    """Meridian arc length on WGS84 by Simpson integration (independent of the series)."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    phi = math.radians(lat_deg)
    h = phi / steps
    total = 0.0
    for i in range(steps + 1):
        x = i * h
        g = a * (1 - e2) / (1 - e2 * math.sin(x) ** 2) ** 1.5
        total += g * (1 if i in (0, steps) else (4 if i % 2 else 2))
    return total * h / 3


class TestUTM(unittest.TestCase):
    POINTS = [(14.5137, 35.8990), (14.2399, 36.0429), (14.3684, 35.9600),
              (14.5274, 35.8270), (14.3817, 35.8604)]

    def test_round_trip(self):
        for lon, lat in self.POINTS:
            e, n = lonlat_to_utm(lon, lat)
            lon2, lat2 = utm_to_lonlat(e, n)
            self.assertAlmostEqual(lon2, lon, delta=1e-7)
            self.assertAlmostEqual(lat2, lat, delta=1e-7)

    def test_net_corners_in_malta(self):
        x0, y0, x1, y1 = CONV_BOUNDARY
        for x in (x0, x1):
            for y in (y0, y1):
                lon, lat = utm_to_lonlat(x - NET_OFFSET[0], y - NET_OFFSET[1])
                self.assertTrue(14.15 <= lon <= 14.60, (x, y, lon))
                self.assertTrue(35.78 <= lat <= 36.10, (x, y, lat))

    def test_central_meridian_matches_meridian_arc(self):
        e, n = lonlat_to_utm(15.0, 36.0)
        self.assertAlmostEqual(e, 500000.0, delta=1e-6)
        self.assertAlmostEqual(n, 0.9996 * _meridian_arc(36.0), delta=0.001)

    def test_known_point_off_meridian(self):
        # CN Tower, Toronto: 43°38'33.24"N 79°23'13.7"W = UTM 17T 630084 4833438
        lat = 43 + 38 / 60 + 33.24 / 3600
        lon = -(79 + 23 / 60 + 13.7 / 3600)
        e, n = lonlat_to_utm(lon, lat, zone=17)
        self.assertAlmostEqual(e, 630084, delta=1.0)
        self.assertAlmostEqual(n, 4833438, delta=1.0)


class TestReadNet(unittest.TestCase):
    def test_reads_location_junctions_and_edges(self):
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<net version="1.20">\n'
            '  <location netOffset="-426538.16,-3962602.48" convBoundary="0,0,1,1" '
            'projParameter="+proj=utm +zone=33"/>\n'
            '  <type id="t" priority="1"/>\n'
            '  <edge id=":j1_0" function="internal"><lane id=":j1_0_0" shape="0,0 1,1"/></edge>\n'
            '  <edge id="e1" from="j1" to="j2" priority="1">'
            '<lane id="e1_0" shape="0,0 1,1"/><param key="k" value="v"/></edge>\n'
            '  <edge id="-e1" from="j2" to="j1"><lane id="-e1_0" shape="1,1 0,0"/></edge>\n'
            '  <junction id="j1" type="priority" x="10.50" y="20.25"><request index="0"/></junction>\n'
            '  <junction id="j2" type="traffic_light" x="11.00" y="21.00"/>\n'
            '  <connection from="e1" to="-e1" fromLane="0" toLane="0"/>\n'
            '</net>\n')
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, 'mini.net.xml')
            with open(path, 'w', encoding='utf-8') as f:
                f.write(xml)
            offset, jxy, efrom = read_net(path)
        finally:
            shutil.rmtree(tmp)
        self.assertEqual(offset, (-426538.16, -3962602.48))
        self.assertEqual(jxy, {'j1': (10.5, 20.25), 'j2': (11.0, 21.0)})
        self.assertEqual(efrom, {'e1': 'j1', '-e1': 'j2'})


class TestCouncilTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(COUNCILS_CSV, encoding='utf-8', newline='') as f:
            cls.rows = list(csv.DictReader(f))

    def test_columns(self):
        self.assertEqual(list(self.rows[0].keys()),
                         ['council', 'district', 'island', 'lat', 'lon', 'source'])

    def test_68_unique_rows(self):
        self.assertEqual(len(self.rows), 68)
        pairs = {(r['council'], r['district']) for r in self.rows}
        self.assertEqual(len(pairs), 68)

    def test_district_counts(self):
        counts = collections.Counter(r['district'] for r in self.rows)
        self.assertEqual(set(counts), set(DISTRICTS))
        self.assertEqual([counts[d] for d in DISTRICTS], [14, 13, 11, 10, 6, 14])

    def test_islands(self):
        for r in self.rows:
            self.assertIn(r['island'], ('Malta', 'Gozo'))
            self.assertEqual(r['island'] == 'Gozo', r['district'] == 'Gozo & Comino', r)
            if r['island'] == 'Gozo':
                self.assertGreater(float(r['lat']), 36.005, r)
            else:
                self.assertLess(float(r['lat']), 36.005, r)
            self.assertTrue(14.15 <= float(r['lon']) <= 14.60, r)
            self.assertTrue(35.78 <= float(r['lat']) <= 36.10, r)
            self.assertTrue(r['source'].strip(), r)

    def test_same_names_stay_distinct(self):
        names = {r['council']: r['district'] for r in self.rows}
        self.assertEqual(names['Rabat'], 'Western')
        self.assertEqual(names['Rabat (Victoria)'], 'Gozo & Comino')
        self.assertEqual(names['Żebbuġ'], 'Western')
        self.assertEqual(names['Żebbuġ (Gozo)'], 'Gozo & Comino')
        self.assertEqual(names['Ħamrun'], 'Northern Harbour')


class TestAssignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.councils = load_councils(COUNCILS_CSV)

    def test_valletta(self):
        island, council, km = assign_council(14.5130, 35.8985, self.councils)
        self.assertEqual((island, council['council'], council['district']),
                         ('Malta', 'Valletta', 'Southern Harbour'))
        self.assertLess(km, 1.0)

    def test_comino_goes_to_gozo_district(self):
        # Comino is part of Għajnsielem; nearest-council may pick Qala, but the
        # district (the only thing the maps use) is Gozo & Comino either way.
        island, council, _ = assign_council(14.3370, 36.0110, self.councils)
        self.assertEqual((island, council['district']), ('Gozo', 'Gozo & Comino'))

    def test_island_decided_first(self):
        # Just south of the Gozo line: nearest council overall is in Gozo, but the
        # point is on the Malta side, so it must get a Malta council.
        island, council, _ = assign_council(14.3000, 36.0040, self.councils)
        self.assertEqual(island, 'Malta')
        self.assertNotEqual(council['district'], 'Gozo & Comino')


class TestMaps(unittest.TestCase):
    ZONES = {'a': 'Western', 'b': 'Western', 'c': 'Northern', 'rc1': 'Western'}

    def test_global_order(self):
        self.assertEqual(global_order(['b', 'a', 'c'], ['rc2', 'rc1']),
                         ['a', 'b', 'c', 'rc2', 'rc1'])

    def test_fog(self):
        ss, sr, rr, rs = build_maps('fog', ['b', 'a', 'c'], ['rc1'], self.ZONES)
        self.assertEqual(ss['a'], ['b'])
        self.assertEqual(ss['b'], ['a'])
        self.assertEqual(sr['a'], {'rc1': 1.0})
        self.assertEqual(rs['rc1'], {'a': 1.0, 'b': 1.0})
        self.assertEqual(list(rs['rc1']), ['a', 'b'])
        self.assertEqual(ss['c'], [])
        self.assertEqual(sr['c'], {})
        self.assertEqual(rr['rc1'], [])
        self.assertEqual(list(ss), ['a', 'b', 'c'])
        self.assertEqual(list(rr), ['rc1'])

    def test_hybrid_equals_fog(self):
        self.assertEqual(build_maps('hybrid', ['b', 'a', 'c'], ['rc1'], self.ZONES),
                         build_maps('fog', ['b', 'a', 'c'], ['rc1'], self.ZONES))

    def test_cloud(self):
        sigs, rcs = ['b', 'a', 'c'], ['rc1', 'rc2']
        ss, sr, rr, rs = build_maps('cloud', sigs, rcs, {})
        order = global_order(sigs, rcs)
        for agent in order:
            others = [x for x in order if x != agent]
            if agent in sigs:
                self.assertEqual(ss[agent], [x for x in others if x in sigs])
                self.assertEqual(list(sr[agent]), [x for x in others if x in rcs])
                self.assertTrue(all(v == 1.0 for v in sr[agent].values()))
            else:
                self.assertEqual(rr[agent], [x for x in others if x in rcs])
                self.assertEqual(list(rs[agent]), [x for x in others if x in sigs])
                self.assertTrue(all(v == 1.0 for v in rs[agent].values()))
        self.assertEqual(ss['a'], ['b', 'c'])
        self.assertEqual(rr['rc2'], ['rc1'])

    def test_input_order_does_not_matter(self):
        m1 = build_maps('fog', ['b', 'a', 'c'], ['rc1'], self.ZONES)
        m2 = build_maps('fog', ['c', 'b', 'a'], ['rc1'], self.ZONES)
        self.assertEqual(neighbour_hash(*m1), neighbour_hash(*m2))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            build_maps('local', ['a'], ['rc1'], self.ZONES)
        with self.assertRaises(ValueError):
            build_maps('fog', ['a', 'z'], ['rc1'], self.ZONES)
        with self.assertRaises(ValueError):
            build_maps('fog', ['a'], ['rc1'], {'a': 'Atlantis', 'rc1': 'Western'})
        with self.assertRaises(ValueError):
            build_maps('cloud', ['a', 'a'], ['rc1'], {})

    def test_hash(self):
        maps = build_maps('cloud', ['b', 'a', 'c'], ['rc1'], {})
        h1, h2 = neighbour_hash(*maps), neighbour_hash(*maps)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)
        ss, sr, rr, rs = maps
        ss2 = dict(ss)
        ss2['a'] = list(reversed(ss['a']))
        self.assertNotEqual(neighbour_hash(ss2, sr, rr, rs), h1)

    def test_load_zones(self):
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, 'z.csv')
            with open(path, 'w', encoding='utf-8', newline='') as f:
                f.write('agent_id,type,x,y,lon,lat,island,council,district\n'
                        '1,signal,0,0,14.2,36.04,Gozo,Għajnsielem,Gozo & Comino\n'
                        'rc1,route,0,0,14.5,35.9,Malta,Valletta,Southern Harbour\n')
            self.assertEqual(load_zones(path),
                             {'1': 'Gozo & Comino', 'rc1': 'Southern Harbour'})
        finally:
            shutil.rmtree(tmp)


class TestDistrictSummary(unittest.TestCase):
    def test_summary(self):
        # districts:          0    0     1    2    3    4    -1 (ignored); 5 is empty
        district_idx = np.array([0, 0, 1, 2, 3, 4, -1])
        wait_means = np.array([0.2, 0.4, np.nan, 0.5, 0.1, 0.3, 9.0])
        vehwait = np.array([30.0, 90.0, 600.0, -5.0, 0.0, 60.0, 999.0])
        out = district_summary(wait_means, vehwait, district_idx, 60.0, 2.0)
        self.assertEqual(out.shape, (12,))
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertEqual(out[5], 0.0)
        self.assertEqual(out[11], 0.0)
        # q: mean of wait_means; the NaN-only district 1 gives 0.
        np.testing.assert_allclose(out[:6], [0.3, 0.0, 0.5, 0.1, 0.3, 0.0])
        # w: mean of clip(vehwait / 60, 0, 2) per district.
        np.testing.assert_allclose(out[6:], [(0.5 + 1.5) / 2, 2.0, 0.0, 0.0, 1.0, 0.0])

    def test_nan_ignored_within_district(self):
        out = district_summary(np.array([np.nan, 0.6]), np.array([10.0, 20.0]),
                               np.array([3, 3]), 10.0, 5.0)
        self.assertAlmostEqual(out[3], 0.6)
        self.assertAlmostEqual(out[9], 1.5)

    def test_all_empty(self):
        out = district_summary(np.array([]), np.array([]), np.array([], dtype=int), 60.0, 2.0)
        np.testing.assert_array_equal(out, np.zeros(12))

    def test_rejects_district_idx_out_of_range(self):
        for bad in (6, -2, 99):
            with self.assertRaises(ValueError, msg=bad):
                district_summary(np.array([0.1, 0.2]), np.array([1.0, 2.0]),
                                 np.array([0, bad]), 60.0, 2.0)
        # -1 (ignored) and 5 (Gozo & Comino) are the bounds of the valid range
        out = district_summary(np.array([0.1, 0.2]), np.array([60.0, 120.0]),
                               np.array([-1, 5]), 60.0, 2.0)
        self.assertAlmostEqual(out[5], 0.2)
        self.assertAlmostEqual(out[11], 2.0)


class TestZoneFile(unittest.TestCase):
    def test_zone_file(self):
        with open(ZONES_CSV, encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(list(rows[0].keys()),
                         ['agent_id', 'type', 'x', 'y', 'lon', 'lat', 'island',
                          'council', 'district'])
        sig = [r['agent_id'] for r in rows if r['type'] == 'signal']
        rcs = [r['agent_id'] for r in rows if r['type'] == 'route']
        self.assertEqual(len(sig), 186)
        self.assertEqual(sig, sorted(sig))
        self.assertEqual(rcs, ['rc1', 'rc2', 'rc3', 'rc4'])
        self.assertEqual([r['agent_id'] for r in rows], sig + rcs)
        for r in rows:
            self.assertIn(r['district'], DISTRICTS)
        self.assertEqual(load_zones(ZONES_CSV),
                         {r['agent_id']: r['district'] for r in rows})


if __name__ == '__main__':
    unittest.main()
