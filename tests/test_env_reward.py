"""Mean reward sharing, hybrid district summary, episode seed and demand factor (RQ2 Task 5).

Stub-based: each env is object.__new__(MaltaEnv) with only the attributes the methods under
test read, so no SUMO starts (importing envs.Malta_env loads malta_cache_v2.pkl). Run from
the project root:

    python -m unittest tests.test_env_reward -v

The one config-validation test builds a real MaltaEnv that raises before SUMO starts (port 20).
"""
import os
import shutil
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np

import envs.env as env_module
import envs.Malta_env as malta_env
from envs.env import Node, Rc
from envs.Malta_env import MaltaEnv
from envs.zones import DISTRICTS, district_summary
from Malta.build_file import BG_PERIOD, read_corridor_total
from tests.test_env_maps import env_config, old_local_reward_step

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, 'tests', 'data', 'flows_f1_golden.xml')
BASE_BG = os.path.join(ROOT, 'Malta', 'data', 'bg_12_p1.2.rou.xml')
N_SUMMARY = 2 * len(DISTRICTS)
PORT = 20


# ---------------------------------------------------------------------------------------------
# Reward stubs
# ---------------------------------------------------------------------------------------------

def reward_stub(ss, sr, rr, rs, gammas, share='mean', problem='signal_route',
                sr_w=None, rs_w=None):
    """A MaltaEnv with only what the reward methods read.

    ss/sr hold one neighbour list per signal, rr/rs one per route, as global indices
    (signal i -> i, route k -> len(ss) + k). gammas = (ss, sr, rr, rs). sr_w/rs_w are the
    relevance weights the sum branch reads (default 1.0 each).
    """
    env = object.__new__(MaltaEnv)
    env.node_names = ['s%d' % i for i in range(len(ss))]
    env.rc_names = ['rc%d' % (k + 1) for k in range(len(rr))]

    def idx(lists):
        return [np.array(a, dtype=np.int64) for a in lists]

    def w(lists, given):
        if given is None:
            given = [[1.0] * len(a) for a in lists]
        return [np.array(g, dtype=np.float64) for g in given]

    env._ss_idx, env._sr_idx, env._rr_idx, env._rs_idx = idx(ss), idx(sr), idx(rr), idx(rs)
    env._sr_w, env._rs_w = w(sr, sr_w), w(rs, rs_w)
    env.coop_gamma_ss, env.coop_gamma_sr, env.coop_gamma_rr, env.coop_gamma_rs = gammas
    env.reward_share = share
    env.problem = problem
    return env


def brief_stub(**kw):
    """The brief's example: signals [s0, s1], route [rc1]."""
    return reward_stub(ss=[[1], [0]], sr=[[2], []], rr=[[]], rs=[[0, 1]],
                       gammas=(0.3, 0.0, 0.0, 0.1), **kw)


def two_route_stub(**kw):
    """Signals [s0, s1], routes [rc1, rc2], every gamma non-zero (so the rr term counts)."""
    return reward_stub(ss=[[1], [0]], sr=[[2, 3], []], rr=[[3], [2]], rs=[[0, 1], [1]],
                       gammas=(0.3, 0.2, 0.5, 0.1), **kw)


def named_stub(share):
    """A names-based stub (Node/Rc objects, relevance weights != 1) whose index arrays come
    from the real _build_index_arrays, so the pre-change name-lookup formula can run on it."""
    env = object.__new__(MaltaEnv)
    env.comm_arch = 'local'
    env.zone_of = None
    env.node_names = ['s0', 's1', 's2']
    env.rc_names = ['rc1', 'rc2']
    env.nodes = {
        's0': Node('s0', relevant_ss=['s1', 's2'], relevant_sr={'rc1': 0.5, 'rc2': 0.25}),
        's1': Node('s1', relevant_ss=['s0'], relevant_sr={}),
        's2': Node('s2', relevant_ss=[], relevant_sr={'rc2': 1.0}),
    }
    env.rcs = {
        'rc1': Rc('rc1', relevant_rr=['rc2'], relevant_rs={'s0': 0.5, 's2': 0.125}),
        'rc2': Rc('rc2', relevant_rr=['rc1'], relevant_rs={}),
    }
    env.coop_gamma_ss, env.coop_gamma_sr, env.coop_gamma_rr, env.coop_gamma_rs = 0.3, 0.2, 0.4, 0.1
    env.reward_share = share
    env.problem = 'signal_route'
    env._build_index_arrays()
    return env


def ref_mean_by_names(env, reward):
    """Spec §6 mean sharing, written independently with name lookups and no weights."""
    n = len(env.node_names)
    pos = {name: i for i, name in enumerate(env.node_names)}
    pos.update({name: n + k for k, name in enumerate(env.rc_names)})

    def mean(names):
        return float(np.mean([reward[pos[a]] for a in names])) if len(names) else 0.0

    out = []
    for name in env.node_names:
        node = env.nodes[name]
        out.append(reward[pos[name]] + env.coop_gamma_ss * mean(node.relevant_ss)
                   + env.coop_gamma_sr * mean(list(node.relevant_sr)))
    for name in env.rc_names:
        rc = env.rcs[name]
        out.append(reward[pos[name]] + env.coop_gamma_rr * mean(rc.relevant_rr)
                   + env.coop_gamma_rs * mean(list(rc.relevant_rs)))
    return np.array(out)


def drive_step(env, raw):
    """Run the real step() with everything around the reward stubbed out.

    _clip_reward is the identity, so step()'s second return value is the shared reward
    before normalisation. Returns (shared, agent_reward).
    """
    env._set_route = lambda a: None
    env._set_phase = lambda a: None
    env._simulate = lambda: (0, 0, 0, 0)
    env._get_state = lambda: []
    env._get_reward = lambda: raw
    env._clip_reward = lambda r: r
    env.cur_sec = 5
    env.episode_length_sec = 3600
    env.is_record = False
    env.train_mode = True
    out = env.step([0] * (len(env.node_names) + len(env.rc_names)))
    return out[1], out[8]


class TestMeanRewardStep(unittest.TestCase):

    def test_brief_example(self):
        env = brief_stub()
        raw = np.array([-1.0, -3.0, 500.0])
        before = raw.copy()
        out = env._mean_reward_step(raw)
        np.testing.assert_allclose(out, [-1.9, -3.3, 499.8], rtol=0, atol=1e-12)
        self.assertEqual(raw.tobytes(), before.tobytes(), 'input reward was mutated')
        self.assertIsInstance(out, np.ndarray)
        self.assertIsNot(out, raw)
        self.assertFalse(np.shares_memory(out, raw))

    def test_rr_and_sr_terms(self):
        # s0:  -2 + 0.3*(-4) + 0.2*mean(1000, 3000)          = 396.8
        # s1:  -4 + 0.3*(-2)  (no route neighbours)           = -4.6
        # rc1: 1000 + 0.5*3000 + 0.1*mean(-2, -4)             = 2499.7
        # rc2: 3000 + 0.5*1000 + 0.1*(-4)                     = 3499.6
        env = two_route_stub()
        out = env._mean_reward_step(np.array([-2.0, -4.0, 1000.0, 3000.0]))
        np.testing.assert_allclose(out, [396.8, -4.6, 2499.7, 3499.6], rtol=0, atol=1e-9)

    def test_relevance_weights_not_applied(self):
        raw = np.array([-2.0, -4.0, 1000.0, 3000.0])
        plain = two_route_stub()._mean_reward_step(raw)
        weighted = two_route_stub(sr_w=[[0.5, 0.25], []],
                                  rs_w=[[0.125, 0.5], [0.25]])._mean_reward_step(raw)
        self.assertEqual(plain.tobytes(), weighted.tobytes())

    def test_matches_spec_formula_by_names(self):
        env = named_stub('mean')
        rng = np.random.default_rng(3)
        for _ in range(5):
            raw = np.concatenate([-rng.uniform(0, 500, 3), 1000.0 * rng.integers(0, 4, 2)])
            np.testing.assert_allclose(env._mean_reward_step(raw), ref_mean_by_names(env, raw),
                                       rtol=0, atol=1e-9)

    def test_empty_sets_add_zero(self):
        env = reward_stub(ss=[[], []], sr=[[], []], rr=[[], []], rs=[[], []],
                          gammas=(0.3, 0.3, 0.5, 0.1))
        raw = np.array([-1.0, -2.0, 100.0, 200.0])
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # no 'Mean of empty slice' RuntimeWarning
            out = env._mean_reward_step(raw)
        self.assertEqual(out.tobytes(), raw.tobytes())
        self.assertTrue(np.all(np.isfinite(out)))

    def test_no_route_agents(self):
        env = reward_stub(ss=[[1], [0]], sr=[[], []], rr=[], rs=[], gammas=(0.3, 0.3, 0.0, 0.1))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            out = env._mean_reward_step(np.array([-1.0, -3.0]))
        np.testing.assert_allclose(out, [-1.9, -3.3], rtol=0, atol=1e-12)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_partial_problems(self):
        # problem 'signal': signal-only array, no sr term (its index 2 would be out of range).
        env = brief_stub(problem='signal')
        env.coop_gamma_sr = 0.5
        np.testing.assert_allclose(env._mean_reward_step(np.array([-1.0, -3.0])),
                                   [-1.9, -3.3], rtol=0, atol=1e-12)
        # problem 'route': route-only array, rr indices shifted back by n, no rs term.
        env = two_route_stub(problem='route')
        np.testing.assert_allclose(env._mean_reward_step(np.array([1000.0, 3000.0])),
                                   [2500.0, 3500.0], rtol=0, atol=1e-9)


class TestSumUnchangedAndDispatch(unittest.TestCase):

    def test_sum_equals_pre_change_formula(self):
        env = named_stub('sum')
        rng = np.random.default_rng(5)
        for _ in range(5):
            raw = np.concatenate([-rng.uniform(0, 500, 3), 1000.0 * rng.integers(0, 4, 2)])
            before = raw.copy()
            self.assertEqual(env._local_reward_step(raw).tobytes(),
                             old_local_reward_step(env, raw).tobytes())
            self.assertEqual(raw.tobytes(), before.tobytes())

    def test_step_uses_mean_under_mean(self):
        env = brief_stub(share='mean')
        raw = np.array([-1.0, -3.0, 500.0])
        before = raw.copy()
        shared, agent_reward = drive_step(env, raw)
        np.testing.assert_allclose(shared, [-1.9, -3.3, 499.8], rtol=0, atol=1e-12)
        self.assertEqual(agent_reward.tobytes(), before.tobytes())
        self.assertEqual(raw.tobytes(), before.tobytes())

    def test_step_uses_sum_under_sum(self):
        env = brief_stub(share='sum')
        raw = np.array([-1.0, -3.0, 500.0])
        expected = env._local_reward_step(raw.copy())
        shared, agent_reward = drive_step(env, raw)
        self.assertEqual(np.asarray(shared).tobytes(), expected.tobytes())
        # sum: rc1 = 500 + 0.1*(-1) + 0.1*(-3) = 499.6, not mean's 499.8
        self.assertAlmostEqual(shared[2], 499.6)
        self.assertEqual(agent_reward.tobytes(), np.array([-1.0, -3.0, 500.0]).tobytes())

    def test_step_named_stub_both_modes(self):
        raw = np.array([-10.0, -20.0, -30.0, 2000.0, 1000.0])
        env = named_stub('sum')
        shared, _ = drive_step(env, raw.copy())
        self.assertEqual(np.asarray(shared).tobytes(), old_local_reward_step(env, raw).tobytes())
        env = named_stub('mean')
        shared, _ = drive_step(env, raw.copy())
        np.testing.assert_allclose(shared, ref_mean_by_names(env, raw), rtol=0, atol=1e-9)

    def test_shared_terms(self):
        raw = np.array([-1.0, -3.0, 500.0])
        before = raw.copy()
        terms = brief_stub(share='mean')._shared_terms(raw)
        np.testing.assert_allclose(terms, [-0.9, -0.3, -0.2], rtol=0, atol=1e-9)
        self.assertEqual(raw.tobytes(), before.tobytes())
        env = brief_stub(share='sum')
        np.testing.assert_allclose(env._shared_terms(raw), env._local_reward_step(raw) - raw,
                                   rtol=0, atol=0)

    def test_get_parameter_path_keeps_sum(self):
        env = named_stub('mean')
        env.coef_rreward = 1000.0
        for i, name in enumerate(env.node_names):
            node = env.nodes[name]
            node.reward, node.wait, node.vehwait = -(i + 1.0), 5.0 * i, 2.0 * i
        for k, name in enumerate(env.rc_names):
            env.rcs[name].reward = env.rcs[name].arrived = k + 1
        raw = np.array([-1.0, -2.0, -3.0, 1000.0, 2000.0])
        summed = old_local_reward_step(env, raw)
        meaned = ref_mean_by_names(env, raw)
        _, totals = env._get_reward(get_parameter=True)
        self.assertEqual(env.problem, 'signal_route')
        self.assertAlmostEqual(totals['sr_signal'], np.mean(summed[:3]))
        self.assertAlmostEqual(totals['sr_route'], np.mean(summed[3:]))
        self.assertNotAlmostEqual(totals['sr_route'], np.mean(meaned[3:]))


# ---------------------------------------------------------------------------------------------
# Hybrid summary in the observation
# ---------------------------------------------------------------------------------------------

def state_stub(comm_arch):
    """Three signals and one route with fixed raw measurements (before _clip_state).

    norms: wave 2, wait 8, arrival 1, occupancy 4; every clip 2; norm_summary_wait 60.
      s0: 3 controlled links on 2 lanes; wave [2,4] -> [1,2]; wait [8,16] -> [1,2] (mean 1.5);
          vehwait 180 -> 180 / 3 links / 60 = 1.0.                 district 0 (Southern Harbour)
      s1: 1 link; wave [6] -> [2] (clipped); wait [24] -> [2] (clipped); vehwait 600 -> 10 -> 2.
                                                                    district 5 (Gozo & Comino)
      s2: no lanes or detectors (wait mean is NaN, ignored); vehwait 0. district 0
      rc1: arrival [1]; occupancy [4, 8] -> [1, 2], weighted by down_edge (1, 0.5) -> [1, 1].
    Expected summary: q = [1.5, 0, 0, 0, 0, 2], w = [mean(1, 0), 0, 0, 0, 0, 2].
    """
    env = object.__new__(MaltaEnv)
    env.comm_arch = comm_arch
    env.problem = 'signal_route'
    env.norm_summary_wait = 60.0
    env.norms = {'wave': 2.0, 'wait': 8.0, 'arrival': 1.0, 'occupancy': 4.0}
    env.clips = {'wave': 2.0, 'wait': 2.0, 'arrival': 2.0, 'occupancy': 2.0, 'reward': 2.0}
    env.node_names = ['s0', 's1', 's2']
    env.rc_names = ['rc1']

    s0 = Node('s0', relevant_ss=['s1'], relevant_sr={'rc1': 1.0})
    s0.lanes_in, s0.ilds_in = ['a_0', 'a_0', 'b_0'], ['e2_a', 'e2_b']
    s0.wave_state, s0.wait_state, s0.vehwait = [2, 4], [8, 16], 180.0
    s0.n_a, s0.fingerprint = 3, np.array([0.25, 0.25])
    s1 = Node('s1', relevant_ss=['s0'], relevant_sr={})
    s1.lanes_in, s1.ilds_in = ['c_0'], ['e2_c']
    s1.wave_state, s1.wait_state, s1.vehwait = [6], [24], 600.0
    s1.n_a, s1.fingerprint = 2, np.array([0.5])
    s2 = Node('s2', relevant_ss=[], relevant_sr={})
    s2.lanes_in, s2.ilds_in = [], []
    s2.wave_state, s2.wait_state, s2.vehwait = [], [], 0
    s2.n_a, s2.fingerprint = 2, np.array([0.5])
    rc1 = Rc('rc1', relevant_rr=[], relevant_rs={'s0': 1.0, 's1': 1.0},
             down_edge={'e1': 1.0, 'e2': 0.5})
    rc1.arrival_state, rc1.occupancy_state = [1], [4, 8]
    rc1.n_a, rc1.fingerprint = 2, np.array([0.5])
    for node in (s0, s1, s2):
        node.num_state = len(node.ilds_in)
        node.num_fingerprint = node.n_a - 1
    rc1.num_state_arrival, rc1.num_state_occupancy, rc1.num_fingerprint = 1, 2, 1

    env.nodes = {'s0': s0, 's1': s1, 's2': s2}
    env.rcs = {'rc1': rc1}
    env._district_idx = np.array([0, 5, 0], dtype=np.int64)
    env._reset_state = lambda: None  # _init_state_space calls it; the counts are set above
    env._init_state_space()
    return env


EXPECTED_SUMMARY = np.array([1.5, 0, 0, 0, 0, 2.0, 0.5, 0, 0, 0, 0, 2.0])


class TestHybridSummary(unittest.TestCase):

    def test_n_o_plus_12_for_every_agent(self):
        fog = state_stub('fog')
        hybrid = state_stub('hybrid')
        self.assertEqual(fog.n_o_ls, [1, 0, 0, 3])
        self.assertEqual(hybrid.n_o_ls, [o + N_SUMMARY for o in fog.n_o_ls])
        self.assertEqual(hybrid.n_o_ls_signal, [o + N_SUMMARY for o in fog.n_o_ls_signal])
        self.assertEqual(hybrid.n_o_ls_route, [o + N_SUMMARY for o in fog.n_o_ls_route])
        for name in ('n_s_ls', 'n_w_ls', 'n_f_ls'):
            self.assertEqual(getattr(hybrid, name), getattr(fog, name), name)
        self.assertEqual(hybrid.n_s, fog.n_s + N_SUMMARY * 4)
        self.assertEqual(hybrid.n_s, sum(hybrid.n_s_ls + hybrid.n_w_ls + hybrid.n_o_ls + hybrid.n_f_ls))

    def test_local_and_fog_have_no_summary(self):
        for arch in ('local', 'fog'):
            env = state_stub(arch)
            ob = env._get_state()
            self.assertEqual([len(o) for o in ob], [3 + 2 + 1 + 2, 3 + 1 + 0 + 2, 0, 1 + 2 + 3 + 3])

    def test_layout_and_values(self):
        env = state_stub('hybrid')
        ob = env._get_state()
        self.assertEqual(len(ob), 4)
        for i, o in enumerate(ob):
            n_s, n_w, n_o, n_f = env.n_s_ls[i], env.n_w_ls[i], env.n_o_ls[i], env.n_f_ls[i]
            self.assertEqual(len(o), n_s + n_w + n_o + n_f, i)
            a = n_s + n_w + n_o
            np.testing.assert_allclose(o[a - N_SUMMARY:a], EXPECTED_SUMMARY, rtol=0, atol=1e-12,
                                       err_msg='agent %d' % i)
            self.assertTrue(np.all(np.isfinite(o)), i)
            self.assertEqual(o.dtype, np.float64, i)
        # s0: [waves 1,2 | s1 wave 2 | wait 1,2 | rc1 arrival 1 | summary | fp s1 0.5 | fp rc1 0.5]
        np.testing.assert_allclose(ob[0][:6], [1, 2, 2, 1, 2, 1])
        np.testing.assert_allclose(ob[0][6 + N_SUMMARY:], [0.5, 0.5])
        # rc1: [arrival 1 | occupancy 1,1 | s0 waves 1,2, s1 wave 2 | summary | fp s0, fp s1]
        np.testing.assert_allclose(ob[3][:6], [1, 1, 1, 1, 2, 2])
        np.testing.assert_allclose(ob[3][6 + N_SUMMARY:], [0.25, 0.25, 0.5])

    def test_removing_summary_gives_fog_observation(self):
        fog_ob = state_stub('fog')._get_state()
        env = state_stub('hybrid')
        hyb_ob = env._get_state()
        for i, (h, f) in enumerate(zip(hyb_ob, fog_ob)):
            a = env.n_s_ls[i] + env.n_w_ls[i] + env.n_o_ls[i]
            stripped = np.concatenate([h[:a - N_SUMMARY], h[a:]])
            self.assertEqual(stripped.tobytes(), np.asarray(f, dtype=np.float64).tobytes(), i)

    def test_summary_computed_once_per_call(self):
        env = state_stub('hybrid')
        with mock.patch.object(env_module, 'district_summary', wraps=district_summary) as spy:
            env._get_state()
        self.assertEqual(spy.call_count, 1)
        args, kwargs = spy.call_args
        params = dict(zip(('wait_means', 'vehwait_per_lane', 'district_idx', 'norm_wait',
                           'clip_wait'), args))
        params.update(kwargs)
        # wait_means from the clipped wait_state (NaN for s2, which has no detectors)
        np.testing.assert_allclose(params['wait_means'], [1.5, 2.0, np.nan])
        # vehwait / len(lanes_in): s0 has 3 controlled links on 2 lanes; s2 has none (max(1, 0))
        np.testing.assert_allclose(params['vehwait_per_lane'], [60.0, 600.0, 0.0])
        self.assertEqual(list(params['district_idx']), [0, 5, 0])
        self.assertEqual(params['norm_wait'], 60.0)
        self.assertEqual(params['clip_wait'], 2.0)

    def test_hybrid_needs_signal_route(self):
        # Rejected in TrafficSimulator.__init__ before SUMO starts.
        tmp = tempfile.mkdtemp(prefix='test_env_reward_')
        try:
            with self.assertRaises(ValueError):
                MaltaEnv(env_config(comm_arch='hybrid'), 'signal',
                         output_path=tmp.replace('\\', '/') + '/', port=PORT)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------------------------
# reset(): episode seed and vehwait
# ---------------------------------------------------------------------------------------------

def reset_stub(train_mode):
    env = object.__new__(MaltaEnv)
    env.train_mode = train_mode
    env.seed = 12
    env.test_seeds = [10000, 20000]
    env.agent = 'ma2c'
    env.cur_episode = 0
    env.node_names = ['s0', 's1']
    env.nodes = {'s0': Node('s0'), 's1': Node('s1')}
    env.nodes['s0'].vehwait, env.nodes['s1'].vehwait = 250.0, 30.0
    env.calls = []
    env._reset_state = lambda: env.calls.append('reset_state')
    env._init_sim = lambda seed, gui=False, load=False: env.calls.append(('init_sim', seed))
    env._init_policy = lambda: []
    env.update_fingerprint = lambda policy: None
    env._init_sim_traffic = lambda: None

    def get_state(reset=False, problem=None):
        env.calls.append(('get_state', reset, [env.nodes[n].vehwait for n in env.node_names]))
        return ['ob']

    env._get_state = get_state
    return env


class TestReset(unittest.TestCase):

    def test_train_reset(self):
        env = reset_stub(train_mode=True)
        self.assertEqual(env.reset(), ['ob'])
        self.assertEqual(env.episode_seed, 12)
        self.assertEqual(env.seed, 13)
        self.assertIn(('init_sim', 12), env.calls)
        self.assertEqual(env.calls[-1], ('get_state', True, [0, 0]))
        env.reset()
        self.assertEqual(env.episode_seed, 13)
        self.assertEqual(env.seed, 14)

    def test_test_reset_uses_test_seed(self):
        env = reset_stub(train_mode=False)
        env.reset(test_ind=1)
        self.assertEqual(env.episode_seed, 20000)
        self.assertIn(('init_sim', 20000), env.calls)
        self.assertEqual(env.calls[-1], ('get_state', True, [0, 0]))


# ---------------------------------------------------------------------------------------------
# _init_sim_config: demand factor, background period and the generation flag
# ---------------------------------------------------------------------------------------------

def sim_config_stub(factor, generate=True, data_path='D/'):
    env = object.__new__(MaltaEnv)
    env.data_path = data_path
    env.demand_flows = malta_env._cache['flows']
    env.sim_thread = 21
    env.demand_factor = factor
    env.generate_traffic = generate
    return env


class TestInitSimConfig(unittest.TestCase):

    def _call(self, env, load=False):
        with mock.patch.object(malta_env, 'gen_rou_file', return_value='cfg') as gen:
            self.assertEqual(env._init_sim_config(12, load), 'cfg')
        self.assertEqual(gen.call_count, 1)
        return gen.call_args

    def test_base_demand_matches_pre_change_call(self):
        env = sim_config_stub(1.0)
        args, kwargs = self._call(env)
        self.assertEqual(args, ('D/', env.demand_flows))
        self.assertEqual(kwargs, {'seed': 12, 'thread': 21, 'load': False,
                                  'period': BG_PERIOD, 'generate_bg': True})
        self.assertEqual('%s' % kwargs['period'], '1.2')

    def test_incoming_load_ignored(self):
        _, kwargs = self._call(sim_config_stub(1.0), load=1.5)
        self.assertIs(kwargs['load'], False)
        _, kwargs = self._call(sim_config_stub(1.10), load=False)
        self.assertEqual(kwargs['load'], 1.10)

    def test_levels(self):
        tags = {1.00: 'p1.2', 1.06: 'p1.132', 1.10: 'p1.091', 1.13: 'p1.062',
                1.20: 'p1.0', 2.00: 'p0.6'}
        for factor, tag in tags.items():
            _, kwargs = self._call(sim_config_stub(factor, generate=False))
            self.assertEqual('p%s' % kwargs['period'], tag, factor)
            self.assertEqual(kwargs['load'], False if factor == 1.0 else factor, factor)
            self.assertIs(kwargs['generate_bg'], False, factor)


class TestDemandFactorRouteFile(unittest.TestCase):
    """The real gen_rou_file, into a temp data_path. output_additional (which parses the 248 MB
    network) is patched out; the background file is a copy, so randomTrips never runs."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='test_env_reward_')
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = self.tmp.replace('\\', '/') + '/'
        for tag in ('bg_12_p1.2', 'bg_12_p1.091'):
            shutil.copy(BASE_BG, self.path + tag + '.rou.xml')

    def _write(self, factor):
        env = sim_config_stub(factor, generate=False, data_path=self.path)
        with mock.patch('Malta.build_file.output_additional', return_value='<additional/>\n'):
            cfg = env._init_sim_config(12, False)
        with open(cfg) as f:
            return f.read()

    def test_factor_1_route_file_unchanged(self):
        cfg = self._write(1.0)
        with open(self.path + 'exp_21.rou.xml') as f, open(GOLDEN) as g:
            self.assertEqual(f.read(), g.read())
        self.assertIn('exp_21.rou.xml,bg_12_p1.2.rou.xml', cfg)

    def test_factor_1_10_scaled(self):
        cfg = self._write(1.10)
        self.assertEqual(read_corridor_total(self.path + 'exp_21.rou.xml'), 10738)
        self.assertIn('exp_21.rou.xml,bg_12_p1.091.rou.xml', cfg)


if __name__ == '__main__':
    unittest.main()
