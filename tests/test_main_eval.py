"""main.py evaluation sweep for an RQ2 run (spec section 8) and Evaluator.run's per-level hook.

No SUMO starts and no checkpoint is restored: the sweep tests stub the env, the model, the
Evaluator and the corridor-file helpers and work in a temp directory; the Evaluator.run test
drives utils.Evaluator with a one-step fake env. Importing main.py imports TensorFlow (10-20 s).
"""
import configparser
import contextlib
import io
import logging
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUN = 'zz_eval'
STORE = 'Malta/runs/%s/' % RUN
TOTAL_STEP = 1440
RUN_INI = """[MODEL_CONFIG]
save_optimizer = true

[TRAIN_CONFIG]
total_step = %d

[ENV_CONFIG]
seed = 12
eval_levels = 1.00,1.10
""" % TOTAL_STEP

MARKER_RE = re.compile(r'^(EVAL_LEVEL_START|EVAL_LEVEL_DONE|test \d+,)')


def _import_main():
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        import main
        return main
    finally:
        os.chdir(cwd)


def _spec(m, *extra):
    return m.parse_args(['--run', RUN, '--option', 'evaluate'] + list(extra))


def _cfg(text):
    cp = configparser.ConfigParser()
    cp.read_string(text)
    return cp


class LevelDirTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main()

    def test_level_dir(self):
        self.assertEqual(self.main.level_dir('Malta/runs/x/', 1.1), 'Malta/runs/x/eva_data/f1.10/')
        self.assertEqual(self.main.level_dir('Malta/runs/x/', 1.0), 'Malta/runs/x/eva_data/f1.00/')
        self.assertEqual(self.main.level_dir('Malta/runs/x/', 2.0), 'Malta/runs/x/eva_data/f2.00/')


class ResolveLevelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main()

    def test_cli_levels_win(self):
        m = self.main
        self.assertEqual(m.resolve_levels(_spec(m, '--levels', '2.0'), _cfg(RUN_INI)), [2.0])

    def test_config_levels(self):
        m = self.main
        cfg = _cfg('[ENV_CONFIG]\neval_levels = 1.00,1.06\n')
        self.assertEqual(m.resolve_levels(_spec(m), cfg), [1.0, 1.06])

    def test_fallback_is_base_demand(self):
        m = self.main
        self.assertEqual(m.resolve_levels(_spec(m), _cfg('[ENV_CONFIG]\nseed = 12\n')), [1.0])

    def test_bad_levels_refused(self):
        m = self.main
        for levels in ('0', '-1.1', '1.10,1.1', 'nan'):
            with self.subTest(levels=levels), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    m.resolve_levels(_spec(m, '--levels', levels), _cfg(RUN_INI))
                self.assertNotIn(cm.exception.code, (0, None))


class FakeEnv:
    def __init__(self):
        self.n_s, self.n_a = 4, 2
        self.n_s_ls, self.n_a_ls, self.n_w_ls = [4], [2], [1]
        self.n_f_ls, self.n_o_ls, self.agent_type = [1], [0], ['signal']
        self.demand_flows = [600, 300]
        self.data_path = 'Malta/data/'
        self.sim_thread = 0
        self.test_seeds = [10000, 20000]
        self.test_num = 2
        self.demand_factor = 1.0
        self.cur_episode = 0
        self.maps_hash = 'maps-hash'

    def neighbour_hash(self):
        return self.maps_hash


class FakeModel:
    def __init__(self, ok=True):
        self.ok = ok
        self.loads = []

    def load(self, model_dir=None, checkpoint=None):
        self.loads.append((model_dir, checkpoint))
        return self.ok


class SweepTest(unittest.TestCase):
    """evaluate_sweep through main(['--run', RUN, '--option', 'evaluate', ...])."""

    @classmethod
    def setUpClass(cls):
        cls.main = _import_main()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.tmp.name)
        os.makedirs('config/runs')
        with open('config/runs/%s.ini' % RUN, 'w') as f:
            f.write(RUN_INI)
        self.env = FakeEnv()
        self.model = FakeModel()
        self.runs = []
        self.fail_at = None
        self.rou_paths = []

    def make_store(self, config=RUN_INI, maps_hash='maps-hash', checkpoint=True):
        os.makedirs(STORE + 'model')
        with open(STORE + 'config.ini', 'w') as f:
            f.write(config)
        with open(STORE + 'neighbours.sha256', 'w') as f:
            f.write(maps_hash + '\n')
        if checkpoint:
            open(STORE + 'model/checkpoint-%d.index' % TOTAL_STEP, 'w').close()

    def make_evaluator(self, env, model, output_path, demo=False, policy_type='default'):
        test = self

        class FakeEvaluator:
            def __init__(self):
                self.env, self.model, self.output_path = env, model, output_path

            def run(self, after_first_reset=None):
                test.runs.append({'output_path': self.output_path,
                                  'factor': self.env.demand_factor,
                                  'dir_exists': os.path.isdir(self.output_path),
                                  'dir_entries': sorted(os.listdir(self.output_path))})
                if test.fail_at == len(test.runs):
                    raise RuntimeError('SUMO connection closed')
                for i in range(self.env.test_num):
                    if i == 0 and after_first_reset is not None:
                        after_first_reset()
                    logging.info('test %i, avg reward %.2f' % (i, -1.0))

        self.evaluator = FakeEvaluator()
        return self.evaluator

    def read_total(self, path):
        self.rou_paths.append(path)
        return 1000 * self.env.demand_factor

    def run_main(self, *extra):
        m = self.main
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(m, 'other_run_pids', return_value=[]) as pids, \
                mock.patch.object(m, 'init_env', return_value=self.env) as init_env, \
                mock.patch.object(m, 'MA2C', return_value=self.model) as ma2c, \
                mock.patch.object(m, 'Evaluator', side_effect=self.make_evaluator), \
                mock.patch.object(m, 'init_log') as init_log, \
                mock.patch.object(m, 'read_corridor_total', side_effect=self.read_total), \
                mock.patch.object(m, 'expected_corridor_total',
                                  side_effect=lambda flows, f: 1000 * f), \
                self.assertLogs(level='INFO') as logs, \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            logging.info('sweep test start')  # assertLogs needs at least one record
            self.calls = SimpleCalls(pids, init_env, ma2c, init_log)
            try:
                rc = m.main(['--run', RUN, '--option', 'evaluate'] + list(extra))
            finally:
                self.out, self.err = out.getvalue(), err.getvalue()
                self.messages = [r.getMessage() for r in logs.records]
        return rc

    def markers(self):
        return [msg for msg in self.messages if MARKER_RE.match(msg)]

    def test_sweep_logs_levels_in_order_and_moves_old_level_dir(self):
        self.make_store()
        os.makedirs(STORE + 'eva_data/f1.10')
        with open(STORE + 'eva_data/f1.10/old.csv', 'w') as f:
            f.write('stale\n')

        self.assertEqual(self.run_main(), 0)

        self.assertEqual(self.markers(), [
            'EVAL_LEVEL_START f1.00 bg_tag=p1.2 corridor_total=1000 corridor_expected=1000',
            'test 0, avg reward -1.00',
            'test 1, avg reward -1.00',
            'EVAL_LEVEL_DONE f1.00 seeds=2',
            'EVAL_LEVEL_START f1.10 bg_tag=p1.091 corridor_total=1100 corridor_expected=1100',
            'test 0, avg reward -1.00',
            'test 1, avg reward -1.00',
            'EVAL_LEVEL_DONE f1.10 seeds=2',
        ])
        lines = [l for l in self.out.splitlines() if l.strip()]
        self.assertEqual(lines[-1], '>>> EVALUATE COMPLETE levels=2')

        # each level ran into its own fresh folder, at its own factor
        self.assertEqual([(r['output_path'], r['factor'], r['dir_exists'], r['dir_entries'])
                          for r in self.runs],
                         [(STORE + 'eva_data/f1.00/', 1.0, True, []),
                          (STORE + 'eva_data/f1.10/', 1.1, True, [])])
        self.assertEqual(self.rou_paths, ['Malta/data/exp_0.rou.xml'] * 2)

        # the old f1.10 folder was set aside, name kept, contents intact
        eva = sorted(os.listdir(STORE + 'eva_data'))
        failed = [d for d in eva if d.startswith('_failed_')]
        self.assertEqual(len(failed), 1, eva)
        self.assertRegex(failed[0], r'^_failed_\d{8}-\d{6}$')
        self.assertEqual(os.listdir(STORE + 'eva_data/' + failed[0]), ['f1.10'])
        with open(STORE + 'eva_data/%s/f1.10/old.csv' % failed[0]) as f:
            self.assertEqual(f.read(), 'stale\n')
        self.assertEqual(sorted(d for d in eva if not d.startswith('_failed_')),
                         ['f1.00', 'f1.10'])

        # set-up: store layout, env, model
        self.calls.pids.assert_called_once_with(os.getpid())
        self.calls.init_log.assert_called_once_with(STORE + 'logs/')
        self.assertEqual(sorted(d for d in os.listdir(STORE) if os.path.isdir(STORE + d)),
                         ['eva_data', 'logs', 'model'])
        self.calls.init_env.assert_called_once()
        args, kw = self.calls.init_env.call_args
        self.assertEqual(args[0]['seed'], '12')  # the store's ENV_CONFIG section
        self.assertEqual((kw.get('port'), kw.get('output_path'), kw.get('generate_traffic')),
                         (0, STORE + 'logs/', False))
        self.calls.ma2c.assert_called_once()
        self.assertEqual(self.calls.ma2c.call_args[0][4], 0)  # eval model: total_step = 0
        self.assertEqual(self.model.loads, [(STORE + 'model/', TOTAL_STEP)])

    def test_cli_levels_run_only_those(self):
        self.make_store()
        self.assertEqual(self.run_main('--levels', '2.0'), 0)
        self.assertEqual([r['output_path'] for r in self.runs], [STORE + 'eva_data/f2.00/'])
        self.assertEqual(self.markers()[0],
                         'EVAL_LEVEL_START f2.00 bg_tag=p0.6 corridor_total=2000 '
                         'corridor_expected=2000')
        self.assertEqual(self.markers()[-1], 'EVAL_LEVEL_DONE f2.00 seeds=2')
        self.assertIn('>>> EVALUATE COMPLETE levels=1', self.out)
        self.assertEqual(os.listdir(STORE + 'eva_data'), ['f2.00'])

    def test_error_mid_sweep_propagates_without_completion_line(self):
        self.make_store()
        self.fail_at = 2
        with self.assertRaises(RuntimeError):
            self.run_main()
        self.assertEqual(self.markers()[-1], 'EVAL_LEVEL_DONE f1.00 seeds=2')
        self.assertFalse(any('f1.10' in msg for msg in self.markers()))
        self.assertNotIn('EVALUATE COMPLETE', self.out)

    def test_missing_store_config_refused_before_anything(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_main()
        self.assertNotIn(cm.exception.code, (0, None))
        self.calls.init_env.assert_not_called()
        self.assertFalse(os.path.exists(STORE))

    def test_config_difference_refused(self):
        self.make_store(config=RUN_INI.replace('seed = 12', 'seed = 13'))
        with self.assertRaises(SystemExit) as cm:
            self.run_main()
        self.assertEqual(cm.exception.code, 4)
        self.assertIn('ENV_CONFIG.seed', self.err)
        self.calls.init_env.assert_not_called()

    def test_neighbour_hash_difference_refused(self):
        self.make_store(maps_hash='other-hash')
        with self.assertRaises(SystemExit) as cm:
            self.run_main()
        self.assertEqual(cm.exception.code, 4)
        self.assertIn('neighbour hash differs', self.err)
        self.calls.ma2c.assert_not_called()
        self.assertEqual(self.runs, [])

    def test_missing_checkpoint_refused(self):
        self.make_store(checkpoint=False)
        with self.assertRaises(SystemExit) as cm:
            self.run_main()
        self.assertEqual(cm.exception.code, 5)
        self.calls.init_env.assert_not_called()

    def test_failed_load_refused(self):
        self.make_store()
        self.model.ok = False
        with self.assertRaises(SystemExit) as cm:
            self.run_main()
        self.assertEqual(cm.exception.code, 5)
        self.assertEqual(self.runs, [])

    def test_other_run_refused(self):
        self.make_store()
        m = self.main
        with mock.patch.object(m, 'other_run_pids', return_value=[4242]), \
                mock.patch.object(m, 'init_env', side_effect=AssertionError('env built')), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit) as cm:
                m.main(['--run', RUN, '--option', 'evaluate'])
        self.assertEqual(cm.exception.code, 6)
        self.assertIn('another main.py --run', err.getvalue())
        self.assertFalse(os.path.exists(STORE + 'eva_data'))


class SimpleCalls:
    def __init__(self, pids, init_env, ma2c, init_log):
        self.pids, self.init_env, self.ma2c, self.init_log = pids, init_env, ma2c, init_log


class OneStepEnv:
    """Just enough env for utils.Evaluator.run/perform: every episode is one step."""
    agent = 'ma2c'

    def __init__(self):
        self.test_num = 2
        self.train_mode = True
        self.cur_episode = 7
        self.calls = []

    def init_data(self, is_record, output_path):
        self.output_path = output_path
        self.calls.append(('init_data', is_record, output_path, self.cur_episode))

    def reset(self, gui=False, test_ind=0, load=False):
        self.cur_episode += 1
        self.calls.append(('reset', test_ind))
        return [np.zeros(1)]

    def update_fingerprint(self, policy):
        pass

    def step(self, action):
        return [np.zeros(1)], [0.0], True, 1.5, 0, 0, 0.0, 0.0, [0.0]

    def terminate(self):
        self.calls.append(('terminate',))

    def collect_tripinfo(self):
        self.calls.append(('collect', self.output_path, self.cur_episode))

    def output_data(self):
        self.calls.append(('output_data', self.output_path))


class OneActionModel:
    def reset(self):
        pass

    def forward(self, ob, done, out_type):
        return [np.array([1.0])]


class EvaluatorRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _import_main()
        import utils
        cls.utils = utils

    def test_first_reset_hook_and_level_output_path(self):
        u = self.utils
        env = OneStepEnv()
        out = 'Malta/runs/x/eva_data/f1.10/'
        ev = u.Evaluator(env, OneActionModel(), 'Malta/runs/x/eva_data/f1.00/')
        ev.output_path = out

        def hook():
            env.calls.append(('hook',))
            logging.info('HOOK')

        with mock.patch.object(u.time, 'sleep'), self.assertLogs(level='INFO') as logs:
            ev.run(after_first_reset=hook)
        self.assertEqual(env.calls, [
            ('init_data', True, out, 0),
            ('reset', 0), ('hook',), ('terminate',), ('collect', out, 1),
            ('reset', 1), ('terminate',), ('collect', out, 2),
            ('output_data', out),
        ])
        self.assertEqual([r.getMessage() for r in logs.records],
                         ['HOOK', 'test 0, avg reward 1.50', 'test 1, avg reward 1.50'])

    def test_run_without_hook_unchanged(self):
        u = self.utils
        env = OneStepEnv()
        ev = u.Evaluator(env, OneActionModel(), 'Malta/signal_route/eva_data/')
        with mock.patch.object(u.time, 'sleep'), self.assertLogs(level='INFO'):
            ev.run()
        self.assertEqual([c[0] for c in env.calls],
                         ['init_data', 'reset', 'terminate', 'collect',
                          'reset', 'terminate', 'collect', 'output_data'])

    def test_output_path_must_be_a_folder(self):
        u = self.utils
        env = OneStepEnv()
        ev = u.Evaluator(env, OneActionModel(), 'Malta/runs/x/eva_data/f1.10')
        with self.assertRaises(ValueError):
            ev.run()
        self.assertEqual(env.calls, [])


if __name__ == '__main__':
    unittest.main()
