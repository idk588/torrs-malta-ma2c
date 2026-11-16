"""Checks that every RQ2 run config differs from the RQ1 config only in the
agreed keys, and that the queue files list the expected runs."""
import configparser
import glob
import math
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'config', 'config_ma2c_Malta.ini')
RUNS = os.path.join(ROOT, 'config', 'runs')

ARMS = ('cloud', 'fog', 'hybrid')
RQ2 = {
    ('ENV_CONFIG', 'reward_share', 'mean'),
    ('ENV_CONFIG', 'seed_continue', 'true'),
    ('ENV_CONFIG', 'eval_levels', '1.00,1.06,1.10,1.13,1.20,2.00'),
    ('ENV_CONFIG', 'norm_summary_wait', '60'),
    ('ENV_CONFIG', 'coop_gamma_sr', '0'),
    ('MODEL_CONFIG', 'save_optimizer', 'true'),
}
SMOKE = {
    ('ENV_CONFIG', 'episode_length_sec', '720'),
    ('ENV_CONFIG', 'test_seeds', '10000,20000'),
    ('ENV_CONFIG', 'eval_levels', '1.00,1.10'),
    ('TRAIN_CONFIG', 'total_step', '288'),
}


def load(path):
    cp = configparser.ConfigParser(interpolation=None)
    with open(path) as f:
        cp.read_file(f)
    return cp


def triples(cp):
    return {(s, k, v) for s in cp.sections() for k, v in cp.items(s)}


def expected(name):
    smoke = name.startswith('_smoke_')
    arm = name.split('_')[-1] if smoke else name.split('_')[0]
    exp = {t for t in RQ2 if not (smoke and t[1] == 'eval_levels')}
    exp.add(('ENV_CONFIG', 'comm_arch', arm))
    if smoke:
        exp |= SMOKE
    return exp


def read_queue(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]


class RunConfigs(unittest.TestCase):
    def test_run_files_present(self):
        names = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(RUNS, '*.ini'))}
        want = {a + '_base' for a in ARMS} | {'_smoke_' + a for a in ARMS}
        self.assertEqual(names, want)

    def test_differences_and_step_arithmetic(self):
        base = load(BASE)
        base_t = triples(base)
        for path in sorted(glob.glob(os.path.join(RUNS, '*.ini'))):
            name = os.path.basename(path)[:-4]
            with self.subTest(run=name):
                cp = load(path)
                t = triples(cp)
                # keys changed or added relative to the RQ1 file
                diff = t - base_t
                self.assertEqual(diff, expected(name))
                # nothing removed except the old values of changed keys
                kept = {(s, k) for s, k, _ in t}
                for s, k, _ in base_t - t:
                    self.assertIn((s, k), kept)
                steps = math.ceil(cp.getint('ENV_CONFIG', 'episode_length_sec') /
                                  cp.getint('ENV_CONFIG', 'control_interval_sec'))
                self.assertEqual(cp.getint('TRAIN_CONFIG', 'total_step') % steps, 0)
                self.assertEqual(steps % cp.getint('MODEL_CONFIG', 'batch_size'), 0)
                if not name.startswith('_smoke_'):
                    self.assertEqual(cp.getint('TRAIN_CONFIG', 'total_step'), 72000)
                    self.assertEqual(cp.getint('ENV_CONFIG', 'seed'), 12)

    def test_queue_files(self):
        cfg = os.path.join(ROOT, 'config')
        q = read_queue(os.path.join(cfg, 'queue.txt'))
        s = read_queue(os.path.join(cfg, 'queue_smoke.txt'))
        self.assertEqual(q, ['cloud_base', 'fog_base', 'hybrid_base'])
        self.assertEqual(s, ['_smoke_hybrid', '_smoke_fog', '_smoke_cloud'])
        for n in q + s:
            self.assertTrue(os.path.isfile(os.path.join(RUNS, n + '.ini')), n)


if __name__ == '__main__':
    unittest.main()
