"""Run files (spec section 7): atomic writes, the cost log, the memory sampler and the
Trainer's per-episode file order.

Temp dirs only. Importing utils starts TensorFlow (a few seconds); no SUMO and no real
model are built: the Trainer runs against a fake env and a fake model.
"""
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import logging
import math
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import run_files
from run_files import (COST_COLUMNS, MemorySampler, append_cost_row, atomic_savetxt,
                       atomic_write_csv, atomic_write_text, process_memory_bytes)
import utils

MIB = 2 ** 20


def _temp_store():
    """A temp dir as a store path: forward slashes and a trailing '/', like Malta/runs/<name>/."""
    return tempfile.mkdtemp(prefix='run_files_').replace('\\', '/') + '/'


def _read(path):
    with open(path, 'rb') as f:
        return f.read()


class AtomicWriteTests(unittest.TestCase):
    def setUp(self):
        self.dir = _temp_store()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_writes_and_replaces_without_leaving_tmp(self):
        path = self.dir + 'a.csv'
        atomic_write_text(path, 'one\ntwo\n')
        atomic_write_text(path, 'three\n')
        with open(path, encoding='utf-8') as f:
            self.assertEqual(f.read(), 'three\n')
        self.assertFalse(os.path.exists(path + '.tmp'))

    def test_locked_file_retries_then_raises(self):
        path = self.dir + 'locked.csv'
        with open(path, 'w') as f:
            f.write('old\n')
        # A handle without FILE_SHARE_DELETE (as Excel holds a CSV): os.replace onto it fails.
        holder = open(path, 'r')
        self.addCleanup(holder.close)
        real_replace = os.replace
        start = time.perf_counter()
        with mock.patch.object(run_files.os, 'replace', side_effect=real_replace) as replace:
            with self.assertLogs(level='WARNING'):
                with self.assertRaises(PermissionError):
                    atomic_write_text(path, 'new\n', retries=2, wait_s=0.01)
        self.assertLess(time.perf_counter() - start, 1.0)
        self.assertEqual(replace.call_count, 3)  # the first attempt plus 2 retries
        holder.close()
        with open(path) as f:
            self.assertEqual(f.read(), 'old\n')
        self.assertFalse(os.path.exists(path + '.tmp'))

    def test_lock_released_during_retries(self):
        path = self.dir + 'busy.csv'
        with open(path, 'w') as f:
            f.write('old\n')
        holder = open(path, 'r')
        self.addCleanup(holder.close)
        release = threading.Timer(0.05, holder.close)
        release.start()
        with self.assertLogs(level='WARNING'):
            atomic_write_text(path, 'new\n', retries=50, wait_s=0.02)
        release.join()
        with open(path) as f:
            self.assertEqual(f.read(), 'new\n')

    def test_csv_and_savetxt_match_the_plain_writes(self):
        df = pd.DataFrame([{'agent': 'ma2c', 'step': 720, 'test_id': -1, 'reward': -12.5, 'seed': 12},
                           {'agent': 'ma2c', 'step': 1440, 'test_id': -1, 'reward': -8.25, 'seed': 13}])
        df.to_csv(self.dir + 'ref_train.csv')
        atomic_write_csv(df, self.dir + 'train_reward.csv')
        self.assertEqual(_read(self.dir + 'train_reward.csv'), _read(self.dir + 'ref_train.csv'))
        agent = [np.array([1.0, -2.5, 3.25]), np.array([4.0, 5.5, -6.0])]
        np.savetxt(self.dir + 'ref_agent.csv', agent, delimiter=',')
        atomic_savetxt(agent, self.dir + 'agent_reward.csv', delimiter=',')
        self.assertEqual(_read(self.dir + 'agent_reward.csv'), _read(self.dir + 'ref_agent.csv'))
        # main.py's resume reads them back this way.
        back = pd.read_csv(self.dir + 'train_reward.csv', index_col=0).to_dict('records')
        self.assertEqual(back, df.to_dict('records'))
        np.testing.assert_array_equal(np.loadtxt(self.dir + 'agent_reward.csv', delimiter=','),
                                      np.array(agent))
        self.assertEqual(sorted(os.listdir(self.dir)),
                         ['agent_reward.csv', 'ref_agent.csv', 'ref_train.csv', 'train_reward.csv'])


def _cost_row(episode, step, **overrides):
    row = {'episode': episode, 'step': step, 'seed': 11 + episode, 'wall_s': 1400.5,
           'rollout_s': 1200.25, 'update_s': 150.125, 'save_s': 20.0625,
           'peak_private_gb': 3.5, 'peak_working_set_gb': 2.75,
           'ckpt_data_mb': 1500.0, 'ckpt_meta_mb': 12.5}
    row.update(overrides)
    return row


class CostRowTests(unittest.TestCase):
    def setUp(self):
        self.dir = _temp_store()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.path = self.dir + 'episode_cost.csv'

    def test_columns_are_the_spec_columns(self):
        self.assertEqual(COST_COLUMNS, ['episode', 'step', 'seed', 'wall_s', 'rollout_s', 'update_s',
                                        'save_s', 'peak_private_gb', 'peak_working_set_gb',
                                        'ckpt_data_mb', 'ckpt_meta_mb', 'lost'])

    def test_header_once_and_lost_zero(self):
        append_cost_row(self.path, _cost_row(1, 720))
        append_cost_row(self.path, _cost_row(2, 1440, peak_private_gb=float('nan')))
        with open(self.path, encoding='utf-8') as f:
            lines = f.read().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0], ','.join(COST_COLUMNS))
        self.assertTrue(lines[1].startswith('1,720,12,'))
        self.assertTrue(lines[1].endswith(',0'))
        self.assertIn(',NaN,', lines[2])
        df = pd.read_csv(self.path)
        self.assertEqual(list(df.columns), COST_COLUMNS)
        self.assertEqual(list(df['episode']), [1, 2])
        self.assertEqual(list(df['step']), [720, 1440])
        self.assertEqual(list(df['lost']), [0, 0])
        self.assertAlmostEqual(df['wall_s'][0], 1400.5)
        self.assertTrue(math.isnan(df['peak_private_gb'][1]))
        self.assertFalse(os.path.exists(self.path + '.tmp'))

    def test_resume_appends_to_the_existing_rows(self):
        append_cost_row(self.path, _cost_row(1, 720))
        before = _read(self.path)
        append_cost_row(self.path, _cost_row(2, 1440))
        self.assertTrue(_read(self.path).startswith(before))

    def test_rejects_bad_rows_and_foreign_files(self):
        with self.assertRaises(ValueError):
            append_cost_row(self.path, {k: v for k, v in _cost_row(1, 720).items() if k != 'seed'})
        with self.assertRaises(ValueError):
            append_cost_row(self.path, _cost_row(1, 720, extra=1))
        self.assertFalse(os.path.exists(self.path))
        with open(self.path, 'w') as f:
            f.write('agent,step\nma2c,720\n')
        with self.assertRaises(ValueError):
            append_cost_row(self.path, _cost_row(1, 720))


class MemorySamplerTests(unittest.TestCase):
    def test_process_memory_is_read(self):
        # An earlier ad-hoc ctypes probe here returned zeros (missing restype/argtypes).
        private, working_set = process_memory_bytes()
        self.assertGreater(private, 20 * MIB)
        self.assertGreater(working_set, 20 * MIB)
        self.assertLess(working_set, 512 * 2 ** 30)

    def test_take_without_start_still_samples(self):
        private_gb, working_set_gb = MemorySampler().take_episode_peaks()
        self.assertGreater(private_gb, 0)
        self.assertGreater(working_set_gb, 0)

    def test_peaks_cover_only_the_time_since_the_last_take(self):
        sampler = MemorySampler()
        self.assertTrue(sampler.daemon)
        sampler.start()
        self.addCleanup(sampler.stop)
        blob = b'\x01' * (400 * MIB)  # every page written, so it is private and resident
        time.sleep(2.5)
        del blob  # freed before the first take: only the thread's earlier samples saw it
        private_1, working_set_1 = sampler.take_episode_peaks()
        time.sleep(1.2)
        private_2, working_set_2 = sampler.take_episode_peaks()
        for value in (private_1, working_set_1, private_2, working_set_2):
            self.assertGreater(value, 0)
        self.assertGreater(private_1 - private_2, 0.25)
        self.assertGreater(working_set_1 - working_set_2, 0.25)
        sampler.stop()
        sampler.join(5)
        self.assertFalse(sampler.is_alive())


# --------------------------------------------------------------------------------------
# Trainer: a fake env and model, small enough that an episode is 4 steps (2 explore calls).
# --------------------------------------------------------------------------------------
class FakeEnv:
    agent = 'ma2c'
    T = 4

    def __init__(self, seed=12):
        self.seed = seed
        self.episode_seed = None
        self.train_mode = True
        self.t = 0

    def reset(self, gui=False, test_ind=0, load=False):
        self.episode_seed = self.seed
        self.seed += 1
        self.t = 0
        return [np.zeros(2)]

    def update_fingerprint(self, policy):
        pass

    def step(self, action):
        self.t += 1
        # next_ob, reward, done, global_reward, arrived, departed, waiting, travel, agent_reward
        return [np.zeros(2)], [1.0], self.t >= self.T, 1.0, 2, 3, 4.0, 5.0, [1.0, 2.0]

    def terminate(self):
        pass


class FakeSess:
    def run(self, *args, **kwargs):
        return b''


class FakeWriter:
    def add_summary(self, summary, global_step=None):
        pass

    def flush(self):
        pass


class FakeModel:
    n_step = 2
    n_agent = 2

    def __init__(self, events, fail_save=False):
        self.sess = FakeSess()
        self.events = events
        self.fail_save = fail_save

    def reset(self):
        pass

    def forward(self, ob, done, out_type='pv'):
        if out_type == 'v':
            return [0.0, 0.0]
        return [np.array([0.5, 0.5]), np.array([0.5, 0.5])], [0.0, 0.0]

    def add_transition(self, ob, action, reward, value, done):
        pass

    def backward(self, R, summary_writer, global_step):
        time.sleep(0.01)

    def save(self, model_dir, global_step):
        self.events.append('save')
        if self.fail_save:
            raise RuntimeError('disk full')
        prefix = model_dir + 'checkpoint-%d' % global_step
        with open(prefix + '.data-00000-of-00001', 'wb') as f:
            f.write(b'\0' * (3 * MIB))
        with open(prefix + '.meta', 'wb') as f:
            f.write(b'\0' * (MIB // 2))
        with open(prefix + '.index', 'wb') as f:
            f.write(b'\0')


class FakeSampler:
    def __init__(self):
        self.calls = 0

    def take_episode_peaks(self):
        self.calls += 1
        return 1.5, 0.75


class _EventHandler(logging.Handler):
    def __init__(self, events):
        super().__init__(logging.INFO)
        self.events = events

    def emit(self, record):
        message = record.getMessage()
        if message.startswith('Checkpoint saved'):
            self.events.append('log:' + message)


class TrainerEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.store = _temp_store()
        self.addCleanup(shutil.rmtree, self.store, True)
        self.events = []
        root = logging.getLogger()
        handler = _EventHandler(self.events)
        old_level = root.level
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        self.addCleanup(root.setLevel, old_level)
        self.addCleanup(root.removeHandler, handler)
        for name in ('atomic_write_csv', 'atomic_savetxt'):
            self._record(name, lambda args: 'csv:' + os.path.basename(args[1]))
        self._record('append_cost_row', lambda args: 'cost')

    def _record(self, name, label):
        real = getattr(utils, name)

        def wrapper(*args, **kwargs):
            self.events.append(label(args))
            return real(*args, **kwargs)
        patcher = mock.patch.object(utils, name, side_effect=wrapper)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _trainer(self, total_step=8, fail_save=False, strict_save=False, sampler=None):
        counter = utils.Counter(total_step, 10 ** 9, 10 ** 9)
        return utils.Trainer(FakeEnv(), FakeModel(self.events, fail_save), 'signal_route', counter,
                             FakeWriter(), False, store_dir=self.store, strict_save=strict_save,
                             sampler=sampler)

    def _episode_events(self, step):
        return ['csv:train_reward.csv', 'csv:agent_reward.csv', 'save', 'cost',
                'log:Checkpoint saved at step %d -> %smodel/' % (step, self.store)]

    def test_episode_order_and_files(self):
        sampler = FakeSampler()
        trainer = self._trainer(sampler=sampler)
        self.assertEqual(trainer.last_saved_step, 0)
        trainer.run()
        # Every episode: CSVs, then the save, then the cost row, then the log line; nothing after.
        self.assertEqual(self.events, self._episode_events(4) + self._episode_events(8))
        self.assertEqual(trainer.last_saved_step, 8)
        self.assertEqual(sampler.calls, 2)

        train = pd.read_csv(self.store + 'train_reward.csv', index_col=0)
        self.assertEqual(list(train['step']), [4, 8])
        self.assertEqual(list(train['seed']), [12, 13])
        self.assertEqual(np.loadtxt(self.store + 'agent_reward.csv', delimiter=',').shape, (2, 2))

        cost = pd.read_csv(self.store + 'episode_cost.csv')
        self.assertEqual(list(cost.columns), COST_COLUMNS)
        self.assertEqual(list(cost['episode']), [1, 2])
        self.assertEqual(list(cost['step']), [4, 8])
        self.assertEqual(list(cost['seed']), [12, 13])
        self.assertEqual(list(cost['lost']), [0, 0])
        self.assertEqual(list(cost['peak_private_gb']), [1.5, 1.5])
        self.assertEqual(list(cost['peak_working_set_gb']), [0.75, 0.75])
        self.assertEqual(list(cost['ckpt_data_mb']), [3.0, 3.0])
        self.assertEqual(list(cost['ckpt_meta_mb']), [0.5, 0.5])
        for _, row in cost.iterrows():
            self.assertGreaterEqual(row['update_s'], 0.02)  # two backward calls of 10 ms
            self.assertGreater(row['rollout_s'], 0)
            self.assertGreater(row['save_s'], 0)
            # 4 decimals in the file, so allow the rounding
            self.assertGreaterEqual(row['wall_s'] + 2e-4,
                                    row['rollout_s'] + row['update_s'] + row['save_s'])
        self.assertEqual([n for n in os.listdir(self.store) if n.endswith('.tmp')], [])

    def test_without_sampler_peaks_are_nan(self):
        self._trainer(total_step=4).run()
        cost = pd.read_csv(self.store + 'episode_cost.csv')
        self.assertTrue(math.isnan(cost['peak_private_gb'][0]))
        self.assertTrue(math.isnan(cost['peak_working_set_gb'][0]))

    def test_strict_save_reraises_before_cost_and_log(self):
        trainer = self._trainer(fail_save=True, strict_save=True)
        with self.assertRaises(RuntimeError):
            trainer.run()
        self.assertEqual(self.events, ['csv:train_reward.csv', 'csv:agent_reward.csv', 'save'])
        self.assertEqual(trainer.last_saved_step, 0)
        self.assertFalse(os.path.exists(self.store + 'episode_cost.csv'))

    def test_non_strict_save_failure_warns_and_carries_on(self):
        trainer = self._trainer(fail_save=True, strict_save=False)
        with self.assertLogs(level='WARNING') as logs:
            trainer.run()
        self.assertTrue(any('Per-episode checkpoint save failed' in m for m in logs.output))
        self.assertEqual(self.events, ['csv:train_reward.csv', 'csv:agent_reward.csv', 'save', 'cost'] * 2)
        self.assertEqual(trainer.last_saved_step, 0)
        cost = pd.read_csv(self.store + 'episode_cost.csv')
        self.assertTrue(cost['ckpt_data_mb'].isna().all())

    def test_store_dir_is_required(self):
        for store in (None, '', 'Malta/runs/x'):
            with self.subTest(store=store):
                with self.assertRaises(ValueError):
                    utils.Trainer(FakeEnv(), FakeModel([]), 'signal_route', utils.Counter(8, 1, 1),
                                  FakeWriter(), False, store_dir=store)


class InitDirLogTests(unittest.TestCase):
    def setUp(self):
        self.dir = _temp_store()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_init_dir_makes_nested_dirs(self):
        base = self.dir + 'runs/x'
        dirs = utils.init_dir(base, pathes=['model', 'logs'])
        self.assertEqual(dirs, {'model': base + '/model/', 'logs': base + '/logs/'})
        self.assertTrue(os.path.isdir(base + '/model') and os.path.isdir(base + '/logs'))
        self.assertEqual(utils.init_dir(base, pathes=['model', 'logs']), dirs)  # again: no error
        self.assertEqual(utils.init_dir(base + '/', pathes=['sim']), {'sim': base + '/sim/'})

    def test_init_log_replaces_earlier_handlers(self):
        root = logging.getLogger()
        saved_handlers, saved_level = root.handlers[:], root.level

        def restore():
            for h in root.handlers[:]:
                root.removeHandler(h)
                h.close()
            for h in saved_handlers:
                root.addHandler(h)
            root.setLevel(saved_level)
        self.addCleanup(restore)
        first, second = self.dir + 'log1/', self.dir + 'log2/'
        os.makedirs(first)
        os.makedirs(second)
        utils.init_log(first)
        utils.init_log(second)
        logging.info('init_log marker')
        for h in root.handlers:
            h.flush()
        logs = os.listdir(second)
        self.assertEqual(len(logs), 1)
        with open(second + logs[0]) as f:
            self.assertIn('init_log marker', f.read())
        with open(first + os.listdir(first)[0]) as f:
            self.assertNotIn('init_log marker', f.read())


if __name__ == '__main__':
    unittest.main()
