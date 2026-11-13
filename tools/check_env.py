"""SUMO-level environment checks for one RQ2 arm (spec section 12). No TensorFlow.

Usage (from any directory; the script switches to the project root, where the config's
data_path 'Malta/data/' resolves):

    python tools/check_env.py --arm local|fog|cloud|hybrid [--port 20] [--steps 100]

The arm's ENV_CONFIG is the RQ1 config (config/config_ma2c_Malta.ini) with the spec section 4
RQ2 keys applied in memory, the same keys the RQ2 run configs use; no config file is written.
'local' is the RQ1 config unchanged. One MaltaEnv (problem signal_route, TraCI port
27500 + port, files Malta/data/exp_<port>.*) is reset once in training mode and stepped
with uniform random actions. States come only from what reset() and step() return; the
env's fields are read right after each call.

Checks (each printed as PASS or FAIL):
  neighbour maps    lists in global order; fog/hybrid: every other agent of the same
                    district; cloud: every other agent (local: counts are only reported)
  RQ1 dims          local only: n_s_ls and n_a_ls equal the RQ1 log's 's dim ls'/'a dim ls'
  state length      len(state[i]) == n_s + n_w + n_o + n_f, every agent, every step
  hybrid summary    state[i][n_s+n_w+n_o-12 : n_s+n_w+n_o] equals zones.district_summary,
                    recomputed from the env's fields, for every agent at every step
  finite            observations, the summary and every reward (raw, shared, final)
  shared terms      clip(norm(raw + _shared_terms(raw))) reproduces step()'s reward
  same scale        spec section 6: per agent type, mean |shared term| <= mean |own raw
                    reward| over the run

Output is printed and saved to Malta/runs/_queue/check_env_<arm>.txt.
Exit code: 0 if every check passes, 1 if any check fails or the run errors, 2 on bad usage.
"""
import argparse
import ast
import configparser
import datetime
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.Malta_env import MaltaEnv, ZONES_FILE
from envs.zones import DISTRICTS, district_summary, load_zones

CONFIG = os.path.join(ROOT, 'config', 'config_ma2c_Malta.ini')
RQ1_LOG = os.path.join(ROOT, 'training_run_bg_entropy_100ep_r9.log')
QUEUE_DIR = os.path.join(ROOT, 'Malta', 'runs', '_queue')
PROBLEM = 'signal_route'
ARMS = ('local', 'fog', 'cloud', 'hybrid')
SUMMARY_LEN = 2 * len(DISTRICTS)
MIN_PORT = 20  # port 0 belongs to the training queue; checks use 20 or higher

# Spec section 4, "RQ2 runs" column, ENV_CONFIG keys: the values every RQ2 run config sets.
# comm_arch is the arm itself. MODEL_CONFIG's save_optimizer does not reach the env.
RQ2_ENV_KEYS = (
    ('reward_share', 'mean'),
    ('demand_factor', '1.0'),
    ('seed_continue', 'true'),
    ('eval_levels', '1.00,1.06,1.10,1.13,1.20,2.00'),
    ('norm_summary_wait', '60'),
    ('coop_gamma_sr', '0'),
)

DIM_LINE = re.compile(r'Training: s dim: \d+, a dim \d+, '
                      r's dim ls: (\[[^\]]*\]), a dim ls: (\[[^\]]*\])')


class Report:
    """Prints each line and appends it to the arm's text file as it goes."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, 'w', encoding='utf-8')
        self.results = []  # (name, ok)

    def __call__(self, text=''):
        text = text.rstrip()
        print(text, flush=True)
        self.f.write(text + '\n')
        self.f.flush()

    def check(self, name, ok, detail):
        self.results.append((name, bool(ok)))
        self('%s  %s: %s' % ('PASS' if ok else 'FAIL', name, detail))

    def close(self):
        self.f.close()


def arm_config(arm):
    """(ENV_CONFIG section, {key: value} applied) for an arm, built in memory."""
    cfg = configparser.ConfigParser()
    if not cfg.read(CONFIG):
        raise FileNotFoundError(CONFIG)
    applied = {}
    if arm != 'local':
        applied['comm_arch'] = arm
        applied.update(RQ2_ENV_KEYS)
        for key, value in applied.items():
            cfg['ENV_CONFIG'][key] = value
    return cfg['ENV_CONFIG'], applied


def rq1_dims():
    """(s dim ls, a dim ls) from the RQ1 training log, or None if the log is missing."""
    if not os.path.exists(RQ1_LOG):
        return None
    with open(RQ1_LOG, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = DIM_LINE.search(line)
            if m:
                return ast.literal_eval(m.group(1)), ast.literal_eval(m.group(2))
    return None


def git_head():
    try:
        out = subprocess.run(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return '?'


def mmm(values):
    """'min/median/max' of a list of numbers ('-' if empty)."""
    if len(values) == 0:
        return '-'
    v = np.asarray(values, dtype=np.float64)
    med = np.median(v)
    med_s = ('%d' % med) if med == int(med) else ('%.1f' % med)
    return '%d/%s/%d' % (v.min(), med_s, v.max())


def neighbour_section(out, env, arm, zone_of):
    """Per-district and per-agent neighbour counts; checks the maps' order and size."""
    n, m = len(env.node_names), len(env.rc_names)
    sig_district = [zone_of.get(s, '(none)') for s in env.node_names]
    rc_district = [zone_of.get(r, '(none)') for r in env.rc_names]
    ss = [len(env.nodes[s].relevant_ss) for s in env.node_names]
    sr = [len(env.nodes[s].relevant_sr) for s in env.node_names]
    rr = [len(env.rcs[r].relevant_rr) for r in env.rc_names]
    rs = [len(env.rcs[r].relevant_rs) for r in env.rc_names]
    lanes = [len(env.nodes[s].lanes_in) for s in env.node_names]
    dets = [len(env.nodes[s].ilds_in) for s in env.node_names]

    out('Neighbour counts per district (min/median/max over the district\'s agents)')
    out('%-18s %4s %4s %4s  %-12s %-12s %-12s %-12s' % (
        'district', 'sig', 'det0', 'rc', 'SS', 'SR', 'RR', 'RS'))
    names = list(DISTRICTS)
    if '(none)' in sig_district or '(none)' in rc_district:
        names.append('(none)')
    for d in names + ['all']:
        si = [i for i in range(n) if d == 'all' or sig_district[i] == d]
        ri = [k for k in range(m) if d == 'all' or rc_district[k] == d]
        out('%-18s %4d %4d %4d  %-12s %-12s %-12s %-12s' % (
            d, len(si), sum(1 for i in si if dets[i] == 0), len(ri),
            mmm([ss[i] for i in si]), mmm([sr[i] for i in si]),
            mmm([rr[k] for k in ri]), mmm([rs[k] for k in ri])))
    out('  (sig = signal agents, det0 = signals without detectors, rc = route agents;')
    out('   SS/SR = a signal\'s signal/route neighbours, RR/RS = a route\'s route/signal neighbours)')
    for k, r in enumerate(env.rc_names):
        out('  %s (%s): RR %d, RS %d' % (r, rc_district[k], rr[k], rs[k]))
    out('Agents with no neighbours of a type: SS=0: %d signals, SR=0: %d signals, '
        'RR=0: %d routes, RS=0: %d routes' % (
            sum(1 for x in ss if x == 0), sum(1 for x in sr if x == 0),
            sum(1 for x in rr if x == 0), sum(1 for x in rs if x == 0)))
    out('Signals: %d with no detectors, %d with no controlled lanes'
        % (sum(1 for x in dets if x == 0), sum(1 for x in lanes if x == 0)))
    out('neighbour_hash %s' % env.neighbour_hash())

    # Order: every neighbour list ascends in global agent index (spec section 5).
    unordered = []
    for i, s in enumerate(env.node_names):
        for idx in (env._ss_idx[i], env._sr_idx[i]):
            if np.any(np.diff(idx) <= 0):
                unordered.append(s)
    for k, r in enumerate(env.rc_names):
        for idx in (env._rr_idx[k], env._rs_idx[k]):
            if np.any(np.diff(idx) <= 0):
                unordered.append(r)

    if arm == 'local':
        out.check('neighbour maps', True,
                  'local maps are the RQ1 maps; counts reported above, no expected sizes')
        return

    sig_count = {d: sig_district.count(d) for d in set(sig_district)}
    rc_count = {d: rc_district.count(d) for d in set(rc_district)}
    wrong = []
    for i, s in enumerate(env.node_names):
        d = sig_district[i]
        want = (n - 1, m) if arm == 'cloud' else (sig_count.get(d, 0) - 1, rc_count.get(d, 0))
        if (ss[i], sr[i]) != want:
            wrong.append('%s SS/SR %d/%d != %d/%d' % (s, ss[i], sr[i], want[0], want[1]))
    for k, r in enumerate(env.rc_names):
        d = rc_district[k]
        want = (m - 1, n) if arm == 'cloud' else (rc_count.get(d, 0) - 1, sig_count.get(d, 0))
        if (rr[k], rs[k]) != want:
            wrong.append('%s RR/RS %d/%d != %d/%d' % (r, rr[k], rs[k], want[0], want[1]))
    rule = 'every other agent' if arm == 'cloud' else 'every other agent of the same district'
    if '(none)' in sig_district or '(none)' in rc_district:
        wrong.append('agents without a district in the zone file')
    detail = '%s; lists in global order' % rule
    if wrong or unordered:
        detail = '%d size mismatches %s; %d unordered lists %s' % (
            len(wrong), wrong[:5], len(unordered), unordered[:5])
    out.check('neighbour maps', not wrong and not unordered, detail)


def dims_section(out, env, arm, n_sig):
    lists = (('n_s_ls', env.n_s_ls), ('n_w_ls', env.n_w_ls), ('n_o_ls', env.n_o_ls),
             ('n_f_ls', env.n_f_ls), ('n_a_ls', env.n_a_ls))
    width = [a + b + c + d for a, b, c, d in zip(env.n_s_ls, env.n_w_ls, env.n_o_ls, env.n_f_ls)]
    out('Dimension lists (%d agents: %d signals + %d routes)'
        % (len(env.n_s_ls), n_sig, len(env.n_s_ls) - n_sig))
    out('%-8s %8s   %-20s %-20s' % ('list', 'total', 'signal min/med/max', 'route min/med/max'))
    for name, ls in lists + (('width', width),):
        ls = list(ls)
        out('%-8s %8d   %-20s %-20s' % (name, sum(ls), mmm(ls[:n_sig]), mmm(ls[n_sig:])))
    out('  (width = n_s + n_w + n_o + n_f per agent; env.n_s = %d)' % int(env.n_s))
    for name, ls in lists:
        out('%s = %s' % (name, [int(x) for x in ls]))
    ok_len = all(len(ls) == len(env.node_names) + len(env.rc_names) for _, ls in lists)
    ok_total = int(env.n_s) == sum(width)
    out.check('dimension lists', ok_len and ok_total,
              'all five lists have one entry per agent; env.n_s == sum of widths'
              if ok_len and ok_total else 'list lengths %s, env.n_s %d vs %d' % (
                  [len(ls) for _, ls in lists], int(env.n_s), sum(width)))
    if arm == 'local':
        dims = rq1_dims()
        if dims is None:
            out('SKIP  RQ1 dims: %s missing or without a "Training: s dim" line' % RQ1_LOG)
        else:
            s_ok = list(env.n_s_ls) == list(dims[0])
            a_ok = list(env.n_a_ls) == list(dims[1])
            out.check('RQ1 dims', s_ok and a_ok,
                      'n_s_ls %s and n_a_ls %s the RQ1 log (%s)' % (
                          'equals' if s_ok else 'DIFFERS from',
                          'equals' if a_ok else 'DIFFERS from', os.path.basename(RQ1_LOG)))
    return width


def expected_summary(env, district_idx, norm_wait, clip_wait):
    """The 12 hybrid values from the env's fields as they stand after reset()/step()."""
    n = len(env.node_names)
    wait_means = np.empty(n)
    vehwait_per_lane = np.empty(n)
    for i, name in enumerate(env.node_names):
        node = env.nodes[name]
        wait = np.asarray(node.wait_state, dtype=np.float64)
        wait_means[i] = wait.mean() if wait.size else np.nan
        vehwait_per_lane[i] = node.vehwait / max(1, len(node.lanes_in))
    return district_summary(wait_means, vehwait_per_lane, district_idx, norm_wait, clip_wait)


class ObsCheck:
    """Accumulates the per-step observation checks."""

    def __init__(self, env, width, hybrid, district_idx, norm_wait, clip_wait):
        self.env = env
        self.width = width
        self.hybrid = hybrid
        self.district_idx = district_idx
        self.norm_wait = norm_wait
        self.clip_wait = clip_wait
        self.end_o = [a + b + c for a, b, c in zip(env.n_s_ls, env.n_w_ls, env.n_o_ls)]
        self.n_agents = len(width)
        self.calls = 0
        self.len_bad = []
        self.nonfinite = []
        self.summary_bad = []
        self.summary_nonfinite = []
        self.summary_max_diff = 0.0
        self.summary_lo = np.full(SUMMARY_LEN, np.inf)
        self.summary_hi = np.full(SUMMARY_LEN, -np.inf)
        self.last_summary = None

    def __call__(self, step, ob):
        self.calls += 1
        if len(ob) != self.n_agents:
            self.len_bad.append('step %d: %d observations, expected %d' % (step, len(ob), self.n_agents))
            return
        for i, s in enumerate(ob):
            s = np.asarray(s)
            if len(s) != self.width[i]:
                self.len_bad.append('step %d agent %d: %d != %d' % (step, i, len(s), self.width[i]))
            if not np.all(np.isfinite(s)):
                self.nonfinite.append('step %d agent %d' % (step, i))
        if not self.hybrid:
            return
        want = expected_summary(self.env, self.district_idx, self.norm_wait, self.clip_wait)
        self.last_summary = want
        if not np.all(np.isfinite(want)):
            self.summary_nonfinite.append('step %d' % step)
        self.summary_lo = np.minimum(self.summary_lo, want)
        self.summary_hi = np.maximum(self.summary_hi, want)
        for i, s in enumerate(ob):
            got = np.asarray(s)[self.end_o[i] - SUMMARY_LEN:self.end_o[i]]
            if got.shape != want.shape or not np.array_equal(got, want):
                self.summary_bad.append('step %d agent %d' % (step, i))
                if got.shape == want.shape:
                    self.summary_max_diff = max(self.summary_max_diff,
                                                float(np.nanmax(np.abs(got - want))))


def run(args, out):
    arm = args.arm
    env_cfg, applied = arm_config(arm)
    out('check_env  arm=%s  port=%d (TraCI %d)  steps=%d  action_seed=%d  HEAD=%s  %s' % (
        arm, args.port, 27500 + args.port, args.steps, args.action_seed, git_head(),
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    out('config: %s, ENV_CONFIG with %s' % (
        os.path.relpath(CONFIG, ROOT).replace('\\', '/'),
        ', '.join('%s=%s' % kv for kv in applied.items()) if applied else 'no changes (RQ1)'))
    out()

    sim_dir = tempfile.mkdtemp(prefix='check_env_%s_' % arm)
    env = None
    try:
        t0 = time.time()
        env = MaltaEnv(env_cfg, PROBLEM, output_path=sim_dir.replace('\\', '/') + '/',
                       port=args.port, generate_traffic=False)
        t_build = time.time() - t0
        n_sig = len(env.node_names)
        out('env: comm_arch=%s reward_share=%s coop_gamma ss/sr/rr/rs=%g/%g/%g/%g '
            'norm_summary_wait=%g demand_factor=%g  (built in %.1f s)' % (
                env.comm_arch, env.reward_share, env.coop_gamma_ss, env.coop_gamma_sr,
                env.coop_gamma_rr, env.coop_gamma_rs, env.norm_summary_wait,
                env.demand_factor, t_build))
        out()

        zone_path = os.path.join(env.data_path, ZONES_FILE)
        zone_of = load_zones(zone_path) if os.path.exists(zone_path) else {}
        if not zone_of:
            out('note: %s missing; districts shown as (none)' % zone_path)
        neighbour_section(out, env, arm, zone_of)
        out()
        width = dims_section(out, env, arm, n_sig)
        out()

        hybrid = arm == 'hybrid'
        district_idx = np.array([DISTRICTS.index(zone_of[s]) if s in zone_of else -1
                                 for s in env.node_names], dtype=np.int64)
        if hybrid:
            same_idx = np.array_equal(district_idx, env._district_idx)
            out.check('hybrid district index', same_idx, 'env._district_idx %s the zone file'
                      % ('matches' if same_idx else 'DIFFERS from'))
        norm_wait = float(env_cfg.get('norm_summary_wait', '60'))
        clip_wait = float(env_cfg['clip_wait'])
        obs_check = ObsCheck(env, width, hybrid, district_idx, norm_wait, clip_wait)

        n = n_sig
        norm_sig = float(env_cfg['norm_reward_sr_signal'])
        norm_route = float(env_cfg['norm_reward_sr_route'])
        clip_reward = float(env_cfg['clip_reward'])
        rng = np.random.RandomState(args.action_seed)
        high = np.asarray(env.n_a_ls, dtype=np.int64)

        t0 = time.time()
        ob = env.reset()
        t_reset = time.time() - t0
        obs_check(0, ob)

        reward_nonfinite = []
        consistency_bad = []
        consistency_max = 0.0
        abs_shared = {'signal': [], 'route': []}
        abs_raw = {'signal': [], 'route': []}
        raw_signed = {'signal': [], 'route': []}
        step_times = []
        steps_done = 0
        for t in range(1, args.steps + 1):
            action = [int(a) for a in rng.randint(0, high)]
            t0 = time.time()
            (ob, reward, done, global_reward, arrived, departed,
             wait, travel, agent_reward) = env.step(action)
            step_times.append(time.time() - t0)
            steps_done = t
            obs_check(t, ob)

            raw = np.asarray(agent_reward, dtype=np.float64)
            shared = env._shared_terms(raw)
            final = np.asarray(reward, dtype=np.float64)
            for name, arr in (('raw', raw), ('shared', shared), ('final', final)):
                if not np.all(np.isfinite(arr)):
                    reward_nonfinite.append('step %d %s agents %s' % (
                        t, name, list(np.where(~np.isfinite(arr))[0][:5])))
            if not np.isfinite(global_reward):
                reward_nonfinite.append('step %d global_reward' % t)

            rebuilt = np.concatenate([
                np.clip((raw[:n] + shared[:n]) / norm_sig, -clip_reward, clip_reward),
                np.clip((raw[n:] + shared[n:]) / norm_route, -clip_reward, clip_reward)])
            if rebuilt.shape != final.shape or not np.allclose(rebuilt, final, rtol=1e-9, atol=1e-12):
                consistency_bad.append('step %d' % t)
            if rebuilt.shape == final.shape:
                consistency_max = max(consistency_max, float(np.max(np.abs(rebuilt - final))))

            abs_shared['signal'].append(np.abs(shared[:n]))
            abs_shared['route'].append(np.abs(shared[n:]))
            abs_raw['signal'].append(np.abs(raw[:n]))
            abs_raw['route'].append(np.abs(raw[n:]))
            raw_signed['signal'].append(raw[:n])
            raw_signed['route'].append(raw[n:])
            if done:
                break

        out('Run: reset %.1f s; %d steps in %.1f s (mean %.3f s/step, max %.3f s)' % (
            t_reset, steps_done, sum(step_times),
            float(np.mean(step_times)) if step_times else 0.0,
            float(np.max(step_times)) if step_times else 0.0))
        out()

        out.check('state length', not obs_check.len_bad,
                  'len(state[i]) == n_s+n_w+n_o+n_f for all %d agents at reset + %d steps'
                  % (obs_check.n_agents, steps_done) if not obs_check.len_bad else
                  '%d mismatches, first %s' % (len(obs_check.len_bad), obs_check.len_bad[:5]))
        out.check('observations finite', not obs_check.nonfinite,
                  'every value of every observation' if not obs_check.nonfinite else
                  '%d non-finite, first %s' % (len(obs_check.nonfinite), obs_check.nonfinite[:5]))
        if hybrid:
            checked = obs_check.calls * obs_check.n_agents
            # An all-zero summary would match zeros anywhere, so placement needs non-zero values.
            nonzero = bool(np.any(obs_check.summary_hi > 0))
            if obs_check.summary_bad:
                detail = '%d/%d slices differ (max |diff| %.3g), first %s' % (
                    len(obs_check.summary_bad), checked, obs_check.summary_max_diff,
                    obs_check.summary_bad[:5])
            elif not nonzero:
                detail = 'the summary was 0 at every step, so its placement was not tested'
            else:
                detail = ('state[i][end_o-12:end_o] == district_summary for every agent at '
                          'reset + %d steps (%d slices)' % (steps_done, checked))
            out.check('hybrid summary', not obs_check.summary_bad and nonzero, detail)
            out.check('summary finite', not obs_check.summary_nonfinite,
                      'all 12 values finite at every step' if not obs_check.summary_nonfinite
                      else 'non-finite at %s' % obs_check.summary_nonfinite[:5])
            out('  summary at the last step [q_1..q_6 | w_1..w_6]:')
            s = obs_check.last_summary
            out('    q %s' % ' '.join('%.4f' % v for v in s[:len(DISTRICTS)]))
            out('    w %s' % ' '.join('%.4f' % v for v in s[len(DISTRICTS):]))
            out('  range over the run: q %.4f..%.4f, w %.4f..%.4f' % (
                obs_check.summary_lo[:len(DISTRICTS)].min(), obs_check.summary_hi[:len(DISTRICTS)].max(),
                obs_check.summary_lo[len(DISTRICTS):].min(), obs_check.summary_hi[len(DISTRICTS):].max()))

        # Coverage notes for the finite check: smallest district, agents with no neighbours.
        counts = {d: sum(1 for a in list(env.node_names) + list(env.rc_names)
                         if zone_of.get(a) == d) for d in DISTRICTS}
        smallest = min(DISTRICTS, key=lambda d: counts[d]) if zone_of else '(none)'
        empty_types = (sum(1 for s in env.node_names if not env.nodes[s].relevant_ss),
                       sum(1 for s in env.node_names if not env.nodes[s].relevant_sr),
                       sum(1 for r in env.rc_names if not env.rcs[r].relevant_rr),
                       sum(1 for r in env.rc_names if not env.rcs[r].relevant_rs))
        out.check('rewards finite', not reward_nonfinite,
                  'raw, shared, final and global rewards over %d steps; covers the smallest '
                  'district (%s, %d agents) and SS/SR/RR/RS-empty agents (%d/%d/%d/%d)' % (
                      (steps_done, smallest, counts.get(smallest, 0)) + empty_types)
                  if not reward_nonfinite else '%d problems, first %s' % (
                      len(reward_nonfinite), reward_nonfinite[:5]))
        out.check('shared terms', not consistency_bad,
                  'clip(norm(raw + _shared_terms(raw))) == step() reward (max |diff| %.3g)'
                  % consistency_max if not consistency_bad else
                  '%d steps differ (max |diff| %.3g), first %s' % (
                      len(consistency_bad), consistency_max, consistency_bad[:5]))

        out()
        out('Same-scale rule (spec section 6), over %d steps x agents of each type:' % steps_done)
        same_ok = True
        parts = []
        for kind in ('signal', 'route'):
            if not abs_raw[kind]:
                continue
            m_shared = float(np.mean(np.concatenate(abs_shared[kind])))
            m_raw = float(np.mean(np.concatenate(abs_raw[kind])))
            m_signed = float(np.mean(np.concatenate(raw_signed[kind])))
            ok = m_shared <= m_raw
            same_ok = same_ok and ok
            ratio = (m_shared / m_raw) if m_raw > 0 else float('nan')
            out('  %-6s mean |shared| %12.4f  <=  mean |own raw| %12.4f  ratio %.4f  %s'
                '  (mean raw %+.4f)' % (kind, m_shared, m_raw, ratio, 'ok' if ok else 'VIOLATED',
                                        m_signed))
            parts.append('%s %.3g <= %.3g' % (kind, m_shared, m_raw) if ok
                         else '%s %.3g > %.3g' % (kind, m_shared, m_raw))
        out.check('same scale', same_ok and bool(parts), '; '.join(parts) or 'no steps run')
    finally:
        if env is not None:
            try:
                env.terminate()
            except Exception:
                pass
        shutil.rmtree(sim_dir, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--port', type=int, default=20,
                        help='env port offset (TraCI 27500 + port); 20 or higher')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--action-seed', type=int, default=0,
                        help='seed of the uniform random actions (default 0)')
    args = parser.parse_args(argv)
    if args.port < MIN_PORT:
        parser.error('--port must be %d or higher (port 0 belongs to the queue)' % MIN_PORT)
    if args.steps < 1:
        parser.error('--steps must be at least 1')

    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass
    os.chdir(ROOT)  # data_path 'Malta/data/' is relative to the project root
    os.makedirs(QUEUE_DIR, exist_ok=True)
    out = Report(os.path.join(QUEUE_DIR, 'check_env_%s.txt' % args.arm))
    errored = False
    try:
        run(args, out)
    except Exception:
        errored = True
        out('ERROR: the run stopped with an exception:')
        for line in traceback.format_exc().rstrip().split('\n'):
            out(line)
    failed = [name for name, ok in out.results if not ok]
    out()
    if errored or failed:
        out('RESULT FAIL (%s)  arm=%s' % (
            ', '.join(failed + (['run error'] if errored else [])), args.arm))
    else:
        out('RESULT PASS  arm=%s  (%d checks)' % (args.arm, len(out.results)))
    out('saved %s' % os.path.relpath(out.path, ROOT).replace('\\', '/'))
    out.close()
    return 1 if (errored or failed) else 0


if __name__ == '__main__':
    sys.exit(main())
