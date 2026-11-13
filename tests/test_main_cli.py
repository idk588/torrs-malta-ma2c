"""main.py CLI (spec section 4): the refusal without --run/--rq1, the one-instance guard, and
train(spec)'s fresh, resume and nothing-to-do paths for an RQ2 run.

No training runs and no SUMO starts: the one-instance test stops before any env is built, and
the train(spec) tests stub the env, the model, the Trainer, the sampler and the TF FileWriter
and work in a temp directory. Importing main.py imports TensorFlow (10-20 s).
"""
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUN = 'zz_flow'
STORE = 'Malta/runs/%s/' % RUN
T = 720
TOTAL_STEP = 3 * T
RUN_INI = """[MODEL_CONFIG]
save_optimizer = true

[TRAIN_CONFIG]
total_step = %d
test_interval = 2000000
log_interval = %d

[ENV_CONFIG]
seed = 12
seed_continue = true
""" % (TOTAL_STEP, T)


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


class RefusalTest(unittest.TestCase):
    def test_no_arguments_refuses(self):
        # Safety net: a main.py without the CLI guard would start RQ1 training here.
        with open(os.path.join(ROOT, 'main.py'), encoding='utf-8') as f:
            if 'parse_args(' not in f.read():
                self.fail('main.py has no CLI guard; not running it without arguments')
        r = subprocess.run([sys.executable, 'main.py'], cwd=ROOT, capture_output=True,
                           text=True, timeout=600)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('refusing: pass --run <name> or --rq1', r.stderr)


class OneInstanceTest(unittest.TestCase):
    """main(['--run', ...]) refuses while another `main.py --run` python is alive."""

    @classmethod
    def setUpClass(cls):
        cls.main = _import_main()

    def setUp(self):
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(ROOT)
        self.store = os.path.join(ROOT, 'Malta', 'runs', 'zz_test')
        self.assertFalse(os.path.exists(self.store), 'leftover %s' % self.store)
        self.dummy = subprocess.Popen(
            [sys.executable, '-c', 'import time; time.sleep(60)', 'main.py', '--run', 'dummy'],
            cwd=ROOT)
        self.addCleanup(self.dummy.wait, timeout=30)
        self.addCleanup(self.dummy.terminate)

    def test_refuses_before_building_anything(self):
        m = self.main
        err = io.StringIO()
        with mock.patch.object(m, 'init_env', side_effect=AssertionError('env built')) as env, \
                mock.patch.object(m, 'MA2C', side_effect=AssertionError('model built')), \
                contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as cm:
                m.main(['--run', 'zz_test'])
        self.assertEqual(cm.exception.code, 6, err.getvalue())
        self.assertIn('another main.py --run', err.getvalue())
        env.assert_not_called()
        self.assertFalse(os.path.exists(self.store))


class FakeModel:
    def __init__(self, ckpt_step):
        self.ckpt_step = ckpt_step
        self.cur_step = 0
        self.lr_scheduler = SimpleNamespace(n=0)
        self.loaded_from = None
        self.saved = []

    def load(self, model_dir=None, checkpoint=None):
        self.loaded_from = model_dir
        if self.ckpt_step is None:
            return False
        self.cur_step = self.ckpt_step
        return True

    def save(self, model_dir, global_step):
        self.saved.append((model_dir, global_step))


class FakeTrainer:
    """Records what train(spec) handed over; run() ends at end_step, last saved at saved_step."""

    def __init__(self, env, model, problem, global_counter, summary_writer, run_test,
                 end_step=None, saved_step=None, **kwargs):
        self.env, self.global_counter, self.kwargs = env, global_counter, kwargs
        self.end_step, self.saved_step = end_step, saved_step
        self.data, self.agent_reward = [], []
        self.last_saved_step = 0
        self.ran = False

    def run(self):
        self.ran = True
        self.seed_at_run = self.env.seed
        self.data_at_run = list(self.data)
        self.agent_rows_at_run = len(self.agent_reward)
        self.global_counter.cur_step = self.end_step
        self.last_saved_step = self.saved_step


class TrainFlowTest(unittest.TestCase):
    """train(spec) for `--run zz_flow`, in a temp project dir, with everything heavy stubbed."""

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
        self.env = SimpleNamespace(n_s=4, n_a=2, n_s_ls=[4], n_a_ls=[2], n_w_ls=[1], n_f_ls=[1],
                                   n_o_ls=[0], agent_type=['signal'], T=np.float64(T), seed=12,
                                   neighbour_hash=lambda: 'maps-hash')
        self.sampler = mock.Mock(name='sampler')

    def run_main(self, ckpt_step, end_step=None, saved_step=None, argv=('--run', RUN)):
        m = self.main
        self.model = FakeModel(ckpt_step)
        self.trainers = []

        def make_trainer(*a, **k):
            t = FakeTrainer(*a, end_step=end_step, saved_step=saved_step, **k)
            self.trainers.append(t)
            return t

        out = io.StringIO()
        with mock.patch.object(m, 'other_run_pids', return_value=[]) as pids, \
                mock.patch.object(m, 'init_env', return_value=self.env) as init_env, \
                mock.patch.object(m, 'prepare_store', return_value=ckpt_step is None) as prep, \
                mock.patch.object(m, 'MA2C', return_value=self.model), \
                mock.patch.object(m, 'Trainer', side_effect=make_trainer), \
                mock.patch.object(m, 'MemorySampler', return_value=self.sampler), \
                mock.patch.object(m, 'init_log') as init_log, \
                mock.patch.object(m.tf.summary, 'FileWriter') as writer, \
                contextlib.redirect_stdout(out):
            rc = m.main(list(argv))
        self.assertEqual(rc, 0)
        self.pids, self.init_env, self.prep = pids, init_env, prep
        self.init_log, self.writer = init_log, writer
        lines = [l for l in out.getvalue().splitlines() if l.strip()]
        return lines

    def write_history(self, steps):
        os.makedirs(STORE, exist_ok=True)
        pd.DataFrame([{'agent': 'ma2c', 'step': s, 'test_id': -1, 'reward': -float(s)}
                      for s in steps]).to_csv(STORE + 'train_reward.csv')
        np.savetxt(STORE + 'agent_reward.csv',
                   np.array([[s, s + 1.0] for s in steps], dtype=float), delimiter=',')

    def check_store_setup(self):
        self.init_env.assert_called_once()
        _, kw = self.init_env.call_args
        self.assertEqual(kw.get('port'), 0)
        self.assertEqual(kw.get('output_path'), STORE + 'sim/')
        self.assertIs(kw.get('generate_traffic'), False)
        self.prep.assert_called_once()
        spec, maps_hash = self.prep.call_args[0]
        self.assertEqual((spec.name, spec.store_dir, maps_hash), (RUN, STORE, 'maps-hash'))
        self.init_log.assert_called_once_with(STORE + 'logs/')
        self.assertEqual(sorted(d for d in os.listdir(STORE) if os.path.isdir(STORE + d)),
                         ['logs', 'model', 'sim'])
        self.assertEqual(self.model.loaded_from, STORE + 'model/')

    def check_trainer(self):
        self.assertEqual(len(self.trainers), 1)
        t = self.trainers[0]
        self.assertTrue(t.ran)
        self.assertEqual(t.kwargs.get('store_dir'), STORE)
        self.assertIs(t.kwargs.get('strict_save'), True)
        self.assertIs(t.kwargs.get('sampler'), self.sampler)
        self.sampler.start.assert_called_once()
        self.sampler.stop.assert_called_once()
        self.writer.assert_called_once_with(STORE + 'logs/')
        self.pids.assert_called_once_with(os.getpid())
        return t

    def test_resume_carries_seed_and_trims_history_past_the_checkpoint(self):
        # Episode 3's rows were written but its checkpoint was not (crash between the two).
        self.write_history([T, 2 * T, 3 * T])
        lines = self.run_main(ckpt_step=2 * T, end_step=TOTAL_STEP, saved_step=TOTAL_STEP)
        self.check_store_setup()
        t = self.check_trainer()
        self.assertEqual(t.seed_at_run, 12 + 2)  # seed + resumed_step // T
        self.assertEqual(self.model.lr_scheduler.n, 2 * T)
        self.assertEqual([r['step'] for r in t.data_at_run], [T, 2 * T])
        self.assertEqual(t.agent_rows_at_run, 2)
        self.assertEqual(self.model.saved, [])  # the per-episode loop already saved total_step
        self.assertEqual(lines[-1], '>>> TRAINING COMPLETE step %d' % TOTAL_STEP)

    def test_fresh_start_saves_final_model_when_loop_did_not(self):
        lines = self.run_main(ckpt_step=None, end_step=TOTAL_STEP, saved_step=2 * T)
        self.check_store_setup()
        t = self.check_trainer()
        self.assertEqual(t.seed_at_run, 12)
        self.assertEqual(t.data_at_run, [])
        self.assertEqual(self.model.saved, [(STORE + 'model/', TOTAL_STEP)])
        self.assertEqual(lines[-1], '>>> TRAINING COMPLETE step %d' % TOTAL_STEP)

    def test_nothing_to_do_still_prints_completion(self):
        lines = self.run_main(ckpt_step=TOTAL_STEP)
        self.check_store_setup()
        self.assertEqual(self.trainers, [])
        self.assertEqual(self.model.saved, [])
        self.assertEqual(lines[-1], '>>> TRAINING COMPLETE step %d' % TOTAL_STEP)

    def test_rq1_train_keeps_its_layout(self):
        # Temp dir: this Malta/signal_route/ is not the real RQ1 store.
        with open('config/config_ma2c_Malta.ini', 'w') as f:
            f.write(RUN_INI.replace('seed_continue = true\n', ''))
        rq1_store = 'Malta/signal_route/'
        lines = self.run_main(ckpt_step=None, end_step=TOTAL_STEP, saved_step=TOTAL_STEP,
                              argv=('--rq1', '--option', 'train'))
        self.pids.assert_not_called()
        self.prep.assert_not_called()
        self.assertEqual(self.main.default_store_dir, 'Malta/signal_route')
        _, kw = self.init_env.call_args
        self.assertEqual((kw.get('port'), kw.get('output_path'), kw.get('generate_traffic')),
                         (0, '', True))
        self.init_log.assert_called_once_with(rq1_store + 'log/')
        self.assertEqual(sorted(os.listdir(rq1_store)), ['data', 'log', 'model'])
        self.assertEqual(sorted(os.listdir('Malta')), ['signal_route'])  # no Malta/data copy
        t = self.trainers[0]
        self.assertEqual(t.kwargs.get('store_dir'), rq1_store)
        self.assertIs(t.kwargs.get('strict_save'), False)
        self.assertIsNone(t.kwargs.get('sampler'))
        self.assertEqual(self.model.loaded_from, rq1_store + 'model/')
        self.assertEqual(self.model.saved, [])
        self.assertEqual(lines[-1], '>>> TRAINING COMPLETE step %d' % TOTAL_STEP)


if __name__ == '__main__':
    unittest.main()
