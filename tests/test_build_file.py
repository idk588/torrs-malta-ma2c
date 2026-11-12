import os
import re
import shutil
import tempfile
import unittest

from Malta.build_file import (
    bg_period, bg_tag, validate_bg_file, gen_background_file, output_flows,
    gen_rou_file, expected_corridor_total, read_corridor_total, get_flows,
)

GOLDEN = 'tests/data/flows_f1_golden.xml'
BASE_BG = 'Malta/data/bg_12_p1.2.rou.xml'


def sum_vehs_per_hour(txt):
    return sum(float(v) for v in re.findall(r'vehsPerHour="([^"]+)"', txt))


class TestBuildFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_bg_period_levels(self):
        self.assertEqual([bg_period(f) for f in (1.0, 1.06, 1.10, 1.13, 1.20, 2.00)],
                         [1.2, 1.132, 1.091, 1.062, 1.0, 0.6])

    def test_bg_tag(self):
        self.assertEqual(bg_tag(12, 1.2), 'bg_12_p1.2')
        self.assertEqual(bg_tag(10000, bg_period(1.10)), 'bg_10000_p1.091')

    def test_flows_unscaled_byte_identical(self):
        with open(GOLDEN) as f:
            golden = f.read()
        self.assertEqual(output_flows(get_flows(), seed=12), golden)

    def test_flows_scaled_half_up(self):
        txt = output_flows(get_flows(), seed=12, load=1.10)
        self.assertEqual(sum_vehs_per_hour(txt), 10738)
        self.assertEqual(expected_corridor_total(get_flows(), 1.10), 10738)
        with open(GOLDEN) as f:
            self.assertEqual(expected_corridor_total(get_flows(), 1.0),
                             sum_vehs_per_hour(f.read()))

    def test_read_corridor_total(self):
        p = os.path.join(self.tmp, 'x.rou.xml')
        with open(p, 'w') as f:
            f.write(output_flows(get_flows(), seed=12, load=1.10))
        self.assertEqual(read_corridor_total(p), 10738)

    def test_validate_bg_file(self):
        self.assertEqual(validate_bg_file(BASE_BG, 1.2), (True, 3000))
        with open(BASE_BG) as f:
            txt = f.read()
        trunc = os.path.join(self.tmp, 'trunc.rou.xml')
        with open(trunc, 'w') as f:
            f.write(txt[:len(txt) // 2])
        self.assertFalse(validate_bg_file(trunc, 1.2)[0])
        # parses fine but too few vehicles for a shorter period
        self.assertFalse(validate_bg_file(BASE_BG, 0.6)[0])
        self.assertFalse(validate_bg_file(os.path.join(self.tmp, 'nope.xml'), 1.2)[0])

    def test_missing_bg_without_generate_raises(self):
        with self.assertRaises(FileNotFoundError):
            gen_background_file(self.tmp + '/', 999999, 1.2, generate=False)

    def test_per_thread_add_file(self):
        shutil.copy(BASE_BG, os.path.join(self.tmp, 'bg_12_p1.2.rou.xml'))
        gen_rou_file(self.tmp + '/', get_flows(), seed=12, thread=21, generate_bg=False)
        with open(os.path.join(self.tmp, 'exp_21.add.xml')) as f:
            add = f.read()
        self.assertIn('file="ild_out_21.xml"', add)
        self.assertNotIn('file="ild_out.xml"', add)
        with open(os.path.join(self.tmp, 'exp_21.sumocfg')) as f:
            cfg = f.read()
        self.assertIn('exp_21.add.xml', cfg)
        self.assertNotIn('"exp.add.xml"', cfg)
        self.assertIn('bg_12_p1.2.rou.xml', cfg)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, 'exp.add.xml')))


if __name__ == '__main__':
    unittest.main()
