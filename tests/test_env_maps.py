"""Neighbour maps per comm_arch (RQ2 Task 4).

Each test class builds one MaltaEnv, which parses the network and launches SUMO
(about a minute each). Run from the project root (the config's data_path is relative):

    python -m unittest tests.test_env_maps -v

Port 20 (TraCI 27520, Malta/data/exp_20.*); port 0 is reserved for the training queue.
"""
import ast
import configparser
import os
import re
import shutil
import tempfile
import unittest

import numpy as np

from envs.Malta_env import MaltaEnv
from envs.zones import DISTRICTS, global_order, load_zones, neighbour_hash

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, 'config', 'config_ma2c_Malta.ini')
RQ1_LOG = os.path.join(ROOT, 'training_run_bg_entropy_100ep_r9.log')
ZONES_CSV = os.path.join(ROOT, 'Malta', 'data', 'zones_nso.csv')
PORT = 20
PROBLEM = 'signal_route'
DIM_LINE = re.compile(r'Training: s dim: \d+, a dim \d+, '
                      r's dim ls: (\[[^\]]*\]), a dim ls: (\[[^\]]*\])')


def env_config(**env_keys):
    """The RQ1 ENV_CONFIG section, with env_keys set in memory (the file is not written)."""
    cfg = configparser.ConfigParser()
    if not cfg.read(CONFIG):
        raise FileNotFoundError(CONFIG)
    for key, value in env_keys.items():
        cfg['ENV_CONFIG'][key] = value
    return cfg['ENV_CONFIG']


def rq1_dims():
    """(s dim ls, a dim ls) from the RQ1 training log's 'Training: s dim ...' line."""
    with open(RQ1_LOG, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = DIM_LINE.search(line)
            if m:
                return ast.literal_eval(m.group(1)), ast.literal_eval(m.group(2))
    raise AssertionError('no "Training: s dim ... s dim ls" line in %s' % RQ1_LOG)


def old_local_reward_step(env, reward):
    """_local_reward_step as it was before Task 4 (commit 3c220dd): name lookups per neighbour."""
    new_reward = []
    if env.problem != 'route':
        for i in range(len(env.node_names)):
            node_name = env.node_names[i]
            cur_reward = reward[i]
            for nnode_name in env.nodes[node_name].relevant_ss:
                j = env.node_names.index(nnode_name)
                cur_reward += env.coop_gamma_ss * reward[j]
            if env.problem == 'signal_route':
                for rrc_name in env.nodes[node_name].relevant_sr:
                    j = env.rc_names.index(rrc_name)
                    cur_reward += env.coop_gamma_sr * env.nodes[node_name].relevant_sr[rrc_name] * reward[len(env.node_names) + j]
            new_reward.append(cur_reward)
    if env.problem != 'signal':
        for i in range(len(env.rc_names)):
            rc_name = env.rc_names[i]
            if env.problem == 'signal_route':
                cur_reward = reward[len(env.node_names) + i]
                for rrc_name in env.rcs[rc_name].relevant_rr:
                    j = env.rc_names.index(rrc_name)
                    cur_reward += env.coop_gamma_rr * reward[len(env.node_names) + j]
                for nnode_name in env.rcs[rc_name].relevant_rs:
                    j = env.node_names.index(nnode_name)
                    cur_reward += env.coop_gamma_rs * env.rcs[rc_name].relevant_rs[nnode_name] * reward[j]
            else:
                cur_reward = reward[i]
                for rrc_name in env.rcs[rc_name].relevant_rr:
                    j = env.rc_names.index(rrc_name)
                    cur_reward += env.coop_gamma_rr * reward[j]
            new_reward.append(cur_reward)
    return np.array(new_reward)


class _EnvMixin:
    """Builds one env per class; shared checks for every comm_arch."""
    env_keys = {}
    env_kwargs = {}

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix='test_env_maps_')
        output_path = cls.tmp.replace('\\', '/') + '/'
        cls.env = MaltaEnv(env_config(**cls.env_keys), PROBLEM, output_path=output_path,
                           port=PORT, **cls.env_kwargs)
        cls.n = len(cls.env.node_names)
        cls.m = len(cls.env.rc_names)

    @classmethod
    def tearDownClass(cls):
        # TrafficSimulator.__init__ ends with terminate(): SUMO is already closed here.
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _raw_reward(self, seed):
        """A realistic raw reward vector: signals about -1..-500, routes 1000 x arrivals."""
        rng = np.random.default_rng(seed)
        signals = -rng.uniform(0.0, 500.0, self.n)
        routes = 1000.0 * rng.integers(0, 4, self.m)
        return np.concatenate([signals, routes])

    def test_rq1_key_defaults(self):
        self.assertEqual(self.env.reward_share, 'sum')
        self.assertEqual(self.env.norm_summary_wait, 60.0)
        self.assertEqual(self.env.demand_factor, 1.0)

    def test_index_arrays_match_names(self):
        env, n = self.env, self.n
        self.assertEqual(len(env._ss_idx), n)
        self.assertEqual(len(env._sr_idx), n)
        self.assertEqual(len(env._rr_idx), self.m)
        self.assertEqual(len(env._rs_idx), self.m)
        for i, name in enumerate(env.node_names):
            node = env.nodes[name]
            self.assertEqual(list(env._ss_idx[i]), [env.node_names.index(s) for s in node.relevant_ss])
            self.assertEqual(list(env._sr_idx[i]), [n + env.rc_names.index(r) for r in node.relevant_sr])
            self.assertTrue(np.issubdtype(np.asarray(env._ss_idx[i]).dtype, np.integer))
            self.assertTrue(np.issubdtype(np.asarray(env._sr_idx[i]).dtype, np.integer))
        for k, name in enumerate(env.rc_names):
            rc = env.rcs[name]
            self.assertEqual(list(env._rr_idx[k]), [n + env.rc_names.index(r) for r in rc.relevant_rr])
            self.assertEqual(list(env._rs_idx[k]), [env.node_names.index(s) for s in rc.relevant_rs])
        self.assertEqual(len(env._district_idx), n)
        self.assertTrue(np.issubdtype(np.asarray(env._district_idx).dtype, np.integer))

    def test_reward_step_identical_to_old_formula(self):
        env, n = self.env, self.n
        for seed in range(3):
            reward = self._raw_reward(seed)
            before = reward.copy()
            new = env._local_reward_step(reward)
            old = old_local_reward_step(env, reward)
            self.assertEqual(new.tobytes(), old.tobytes())
            self.assertEqual(reward.tobytes(), before.tobytes(), 'input reward was mutated')
            self.assertIsNot(new, reward)

    def test_reward_step_partial_problems(self):
        # _get_reward(get_parameter=True) calls it with problem 'signal' / 'route' and partial arrays.
        env, n = self.env, self.n
        reward = self._raw_reward(7)
        try:
            for problem, part in (('signal', reward[:n]), ('route', reward[n:])):
                env.problem = problem
                new = env._local_reward_step(part.copy())
                old = old_local_reward_step(env, part.copy())
                self.assertEqual(len(new), len(part))
                self.assertEqual(new.tobytes(), old.tobytes(), problem)
        finally:
            env.problem = PROBLEM

    def test_neighbour_hash(self):
        env = self.env
        expected = neighbour_hash(env.relevant_ss_map, env.relevant_sr_map,
                                  env.relevant_rr_map, env.relevant_rs_map)
        self.assertEqual(env.neighbour_hash(), expected)
        self.assertRegex(expected, r'^[0-9a-f]{64}$')


class TestLocalMaps(_EnvMixin, unittest.TestCase):
    """RQ1 config untouched: comm_arch falls back to 'local' (today's maps)."""

    def test_comm_arch_default_local(self):
        self.assertEqual(self.env.comm_arch, 'local')
        self.assertIs(self.env.generate_traffic, True)

    def test_dims_equal_rq1_training_log(self):
        s_ls, a_ls = rq1_dims()
        self.assertEqual(self.env.n_s_ls, s_ls)
        self.assertEqual(self.env.n_a_ls, a_ls)

    def test_route_signal_links_empty(self):
        for name in self.env.node_names:
            self.assertEqual(len(self.env.nodes[name].relevant_sr), 0, name)
        for name in self.env.rc_names:
            self.assertEqual(len(self.env.rcs[name].relevant_rs), 0, name)

    def test_no_districts_under_local(self):
        self.assertEqual(list(self.env._district_idx), [-1] * self.n)


class TestFogMaps(_EnvMixin, unittest.TestCase):
    """comm_arch = fog: every other agent in the same NSO district, in global order."""
    env_keys = {'comm_arch': 'fog'}
    env_kwargs = {'generate_traffic': False}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.zones = load_zones(ZONES_CSV)
        cls.order = global_order(cls.env.node_names, cls.env.rc_names)

    def _same_district(self, agent, pool):
        return [a for a in self.order if a != agent and a in pool
                and self.zones[a] == self.zones[agent]]

    def test_comm_arch_and_flags(self):
        self.assertEqual(self.env.comm_arch, 'fog')
        self.assertIs(self.env.generate_traffic, False)

    def test_ss_is_other_district_signals_in_global_order(self):
        signals = set(self.env.node_names)
        for name in self.env.node_names:
            expected = self._same_district(name, signals)
            self.assertEqual(self.env.relevant_ss_map[name], expected, name)
            self.assertEqual(self.env.nodes[name].relevant_ss, expected, name)

    def test_sr_rr_rs_are_district_agents(self):
        signals, routes = set(self.env.node_names), set(self.env.rc_names)
        for name in self.env.node_names:
            sr = self.env.nodes[name].relevant_sr
            self.assertEqual(list(sr), self._same_district(name, routes), name)
            self.assertTrue(all(w == 1.0 for w in sr.values()), name)
        for name in self.env.rc_names:
            rc = self.env.rcs[name]
            self.assertEqual(rc.relevant_rr, self._same_district(name, routes), name)
            self.assertEqual(list(rc.relevant_rs), self._same_district(name, signals), name)
            self.assertTrue(all(w == 1.0 for w in rc.relevant_rs.values()), name)

    def test_edge_keyed_rs_map_replaced(self):
        signals = set(self.env.node_names)
        self.assertEqual(list(self.env.relevant_rs_map), self.env.rc_names)
        for name, rs in self.env.relevant_rs_map.items():
            self.assertTrue(set(rs) <= signals, name)

    def test_district_idx(self):
        expected = [DISTRICTS.index(self.zones[s]) for s in self.env.node_names]
        self.assertEqual(list(self.env._district_idx), expected)

    def test_state_dims_follow_maps(self):
        env = self.env
        for i, name in enumerate(env.node_names):
            node = env.nodes[name]
            waves = node.num_state + sum(env.nodes[s].num_state for s in node.relevant_ss)
            self.assertEqual(env.n_s_ls[i], waves, name)
            self.assertEqual(env.n_o_ls[i], sum(env.rcs[r].num_state_arrival for r in node.relevant_sr), name)

    def test_some_route_signal_sharing_exists(self):
        # Under fog, every route agent shares a district with at least one signal.
        for name in self.env.rc_names:
            self.assertGreater(len(self.env.rcs[name].relevant_rs), 0, name)


class TestConfigValidation(unittest.TestCase):
    """Unknown values are rejected before SUMO starts."""

    def _build(self, **keys):
        tmp = tempfile.mkdtemp(prefix='test_env_maps_bad_')
        try:
            MaltaEnv(env_config(**keys), PROBLEM, output_path=tmp.replace('\\', '/') + '/', port=PORT)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_unknown_comm_arch(self):
        with self.assertRaises(ValueError):
            self._build(comm_arch='mesh')

    def test_unknown_reward_share(self):
        with self.assertRaises(ValueError):
            self._build(reward_share='median')

    def test_non_positive_norm_summary_wait(self):
        with self.assertRaises(ValueError):
            self._build(norm_summary_wait='0')

    def test_non_positive_demand_factor(self):
        with self.assertRaises(ValueError):
            self._build(demand_factor='0')


if __name__ == '__main__':
    unittest.main()
