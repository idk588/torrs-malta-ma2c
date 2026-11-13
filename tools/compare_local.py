"""Local-unchanged check (spec section 12): under the RQ1 config, the current code's env
gives exactly the pre-change code's states and rewards. SUMO level, no TensorFlow.

Usage (from any directory):

    python tools/compare_local.py --base-worktree <dir> [--port 21] [--seed 12] [--steps 50]
                                  [--actions tests/data/actions_50.json] [--base-commit bc7115c]
    python tools/compare_local.py --make-actions [--port 21] [--steps 50] [--actions ...]

The base is a git worktree of the pre-change commit at <dir>. It is created (detached) if
<dir> does not exist. malta.net.xml, malta_cache_v2.pkl and bg_<seed>_p1.2.rou.xml/.trips.xml,
which git does not track, are copied (not linked) into its Malta/data/.

Both versions are driven only through MaltaEnv, by this script's driver mode, each in a
fresh python process whose working directory is that version's root (the config's data_path
'Malta/data/' is relative), one after the other on the same port, never in parallel. The
driver builds MaltaEnv(RQ1 ENV_CONFIG, 'signal_route', port=<port>), resets in training mode
(seed --seed), applies the fixed action list step by step and pickles every observation and
step output. The comparison is exact: np.array_equal per agent per step for states, bitwise
for rewards.

--make-actions writes the action list once: numpy RandomState(0), one uniform integer in
[0, n_a) per agent per step, with n_a_ls taken from the current code's RQ1 env.

Prints 'IDENTICAL states S/S, rewards S/S' and exits 0 only if the dimensions, the reset
observation, every state, every reward and every other step output match; exits 1 otherwise
and 2 on bad usage or a failed setup.
"""
import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, 'config', 'config_ma2c_Malta.ini')
DATA_DIR = os.path.join('Malta', 'data')
BASE_COMMIT = 'bc7115c'
ACTION_SEED = 0
MIN_PORT = 20  # port 0 belongs to the training queue
PROBLEM = 'signal_route'


# ---------------------------------------------------------------- driver (one version)

def drive(args):
    """Runs inside one version's root: build the env, reset, step, pickle everything."""
    root = os.path.abspath(args.root)
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    # Import envs/ and Malta/ from this version's root only, not from the script's tree.
    sys.path[:] = [root] + [p for p in sys.path
                            if p and os.path.normcase(os.path.abspath(p)) != os.path.normcase(here)]
    import configparser
    import inspect
    from envs.Malta_env import MaltaEnv
    loaded_from = os.path.dirname(os.path.dirname(os.path.abspath(sys.modules['envs.Malta_env'].__file__)))
    if os.path.normcase(loaded_from) != os.path.normcase(root):
        raise RuntimeError('envs imported from %s, expected %s' % (loaded_from, root))

    cfg = configparser.ConfigParser()
    if not cfg.read(args.config):
        raise FileNotFoundError(args.config)
    env_cfg = cfg['ENV_CONFIG']
    env_cfg['seed'] = str(args.seed)
    kwargs = {}
    if 'generate_traffic' in inspect.signature(MaltaEnv.__init__).parameters:
        kwargs['generate_traffic'] = False  # a missing background file is an error, not a rebuild
    sim_dir = os.path.abspath(args.sim_dir).replace('\\', '/') + '/'

    t0 = time.time()
    env = MaltaEnv(env_cfg, PROBLEM, output_path=sim_dir, port=args.port, **kwargs)
    out = {'root': root, 'build_sec': time.time() - t0,
           # Evidence of which code ran: the pre-change env has no comm_arch attribute.
           'comm_arch': getattr(env, 'comm_arch', '(none: pre-change code)'),
           'node_names': list(env.node_names), 'rc_names': list(env.rc_names),
           'n_s_ls': [int(x) for x in env.n_s_ls], 'n_a_ls': [int(x) for x in env.n_a_ls],
           'steps': [], 'step_sec': []}
    if args.steps > 0:
        with open(args.actions, encoding='utf-8') as f:
            actions = json.load(f)['actions']
        if len(actions) < args.steps:
            raise ValueError('%s has %d steps, %d requested' % (args.actions, len(actions), args.steps))
        try:
            seed_before = env.seed
            t0 = time.time()
            ob = env.reset()
            out['reset_sec'] = time.time() - t0
            out['reset_seed'] = seed_before
            out['reset_state'] = [np.array(s, copy=True) for s in ob]
            for t in range(args.steps):
                action = [int(a) for a in actions[t]]
                t0 = time.time()
                (ob, reward, done, global_reward, arrived, departed,
                 wait, travel, agent_reward) = env.step(action)
                out['step_sec'].append(time.time() - t0)
                out['steps'].append({
                    'state': [np.array(s, copy=True) for s in ob],
                    'reward': np.array(reward, copy=True),
                    'agent_reward': np.array(agent_reward, copy=True),
                    'global_reward': global_reward,
                    'other': (done, arrived, departed, wait, travel),
                })
        finally:
            env.terminate()
    with open(args.drive, 'wb') as f:
        pickle.dump(out, f, protocol=4)
    return 0


# ---------------------------------------------------------------- controller

def git(*cmd, cwd=ROOT):
    res = subprocess.run(['git'] + list(cmd), cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError('git %s failed: %s' % (' '.join(cmd), res.stderr.strip()))
    return res.stdout.strip()


def setup_base(path, commit, seed):
    """Create or reuse the base worktree and copy the untracked data files into it."""
    want = git('rev-parse', '%s^{commit}' % commit)
    if not os.path.isdir(path):
        print('creating worktree %s at %s' % (path, commit), flush=True)
        git('worktree', 'add', '--detach', path, commit)
    head = git('rev-parse', 'HEAD', cwd=path)
    if head != want:
        raise RuntimeError('worktree %s is at %s, expected %s (%s)' % (path, head[:7], want[:7], commit))
    dirty = git('status', '--porcelain', '--untracked-files=no', cwd=path)
    if dirty:
        raise RuntimeError('worktree %s has tracked changes:\n%s' % (path, dirty))
    names = ['malta.net.xml', 'malta_cache_v2.pkl',
             'bg_%d_p1.2.rou.xml' % seed, 'bg_%d_p1.2.trips.xml' % seed]
    for name in names:
        src = os.path.join(ROOT, DATA_DIR, name)
        dst = os.path.join(path, DATA_DIR, name)
        if not os.path.isfile(src):
            raise FileNotFoundError(src)
        if os.path.isfile(dst) and not os.path.islink(dst):
            s, d = os.stat(src), os.stat(dst)
            if s.st_size == d.st_size and int(s.st_mtime) == int(d.st_mtime):
                continue
        print('copying %s -> %s' % (name, os.path.join(path, DATA_DIR)), flush=True)
        shutil.copy2(src, dst)
    return head


def run_driver(label, root, args, actions_path, steps, work):
    """One version in its own python process (cwd = root); returns the loaded pickle."""
    out_pkl = os.path.join(work, '%s.pkl' % label)
    sim_dir = os.path.join(work, '%s_sim' % label)
    os.makedirs(sim_dir, exist_ok=True)
    cmd = [sys.executable, os.path.abspath(__file__), '--drive', out_pkl, '--root', root,
           '--config', CONFIG, '--port', str(args.port), '--seed', str(args.seed),
           '--steps', str(steps), '--actions', actions_path, '--sim-dir', sim_dir]
    print('--- %s: %s (cwd %s)' % (label, 'driver', root), flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=root)
    wall = time.time() - t0
    if res.returncode != 0:
        raise RuntimeError('%s driver exited %d' % (label, res.returncode))
    with open(out_pkl, 'rb') as f:
        data = pickle.load(f)
    data['wall_sec'] = wall
    return data


def bitwise_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def states_equal(sa, sb):
    """(equal, first differing agent or None, max |diff| there)."""
    if len(sa) != len(sb):
        return False, 'count %d vs %d' % (len(sa), len(sb)), float('nan')
    for i, (a, b) in enumerate(zip(sa, sb)):
        if a.shape != b.shape:
            return False, 'agent %d shape %s vs %s' % (i, a.shape, b.shape), float('nan')
        if not np.array_equal(a, b):
            return False, 'agent %d' % i, float(np.max(np.abs(a.astype(np.float64) - b)))
    return True, None, 0.0


def compare(base, new, steps):
    """Prints the comparison; returns True if everything is identical."""
    ok = True
    for key in ('node_names', 'rc_names', 'n_s_ls', 'n_a_ls'):
        same = base[key] == new[key]
        ok = ok and same
        print('%-11s %s' % (key, 'identical (%d)' % len(new[key]) if same else 'DIFFERENT'))
    same_seed = base['reset_seed'] == new['reset_seed']
    ok = ok and same_seed
    print('reset seed  %s' % ('identical (%d)' % new['reset_seed'] if same_seed else
                              'DIFFERENT %r vs %r' % (base['reset_seed'], new['reset_seed'])))
    same, where, diff = states_equal(base['reset_state'], new['reset_state'])
    ok = ok and same
    print('reset state %s' % ('identical' if same else 'DIFFERENT at %s (max |diff| %g)' % (where, diff)))

    n_base, n_new = len(base['steps']), len(new['steps'])
    if n_base != steps or n_new != steps:
        print('step count  DIFFERENT: base %d, new %d, expected %d' % (n_base, n_new, steps))
        ok = False
    states_same = rewards_same = other_same = 0
    first = {}
    for t, (b, n) in enumerate(zip(base['steps'], new['steps']), start=1):
        same, where, diff = states_equal(b['state'], n['state'])
        if same:
            states_same += 1
        else:
            first.setdefault('state', 'step %d %s (max |diff| %g)' % (t, where, diff))
        r_ok = (bitwise_equal(b['reward'], n['reward'])
                and bitwise_equal(b['agent_reward'], n['agent_reward'])
                and bitwise_equal(np.float64(b['global_reward']), np.float64(n['global_reward'])))
        if r_ok:
            rewards_same += 1
        else:
            first.setdefault('reward', 'step %d (max |diff| reward %g, agent_reward %g)' % (
                t, float(np.max(np.abs(np.asarray(b['reward'], dtype=np.float64) - n['reward']))),
                float(np.max(np.abs(np.asarray(b['agent_reward'], dtype=np.float64) - n['agent_reward'])))))
        if b['other'] == n['other']:
            other_same += 1
        else:
            first.setdefault('other', 'step %d: base %r, new %r' % (t, b['other'], n['other']))
    print('other step outputs (done, arrived, departed, wait, travel): %d/%d identical'
          % (other_same, steps))
    for key, text in first.items():
        print('first %s difference: %s' % (key, text))
    ok = ok and states_same == steps and rewards_same == steps and other_same == steps
    word = 'IDENTICAL' if ok else 'DIFFERENT'
    print('%s states %d/%d, rewards %d/%d' % (word, states_same, steps, rewards_same, steps))
    return ok


def make_actions(args, actions_path):
    if os.path.exists(actions_path) and not args.force:
        print('%s exists; it is generated once (pass --force to replace it)' % actions_path)
        return 2
    work = tempfile.mkdtemp(prefix='compare_local_actions_')
    try:
        dims = run_driver('new_dims', ROOT, args, actions_path, 0, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    n_a_ls = dims['n_a_ls']
    rng = np.random.RandomState(ACTION_SEED)
    actions = rng.randint(0, np.asarray(n_a_ls, dtype=np.int64), size=(args.steps, len(n_a_ls)))
    lines = ['{',
             '  "description": "Fixed actions for tools/compare_local.py: numpy RandomState(%d), '
             'one uniform integer in [0, n_a) per agent per step; agents in global order '
             '(sorted signals, then rc1..rc4)",' % ACTION_SEED,
             '  "numpy_seed": %d,' % ACTION_SEED,
             '  "steps": %d,' % args.steps,
             '  "n_a_ls": %s,' % json.dumps(n_a_ls),
             '  "actions": [']
    rows = [json.dumps([int(a) for a in row]) for row in actions]
    lines += ['    %s%s' % (row, ',' if k < len(rows) - 1 else '') for k, row in enumerate(rows)]
    lines += ['  ]', '}']
    os.makedirs(os.path.dirname(actions_path), exist_ok=True)
    with open(actions_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote %s: %d steps x %d agents' % (actions_path, args.steps, len(n_a_ls)))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--base-worktree', help='git worktree of the pre-change commit')
    parser.add_argument('--base-commit', default=BASE_COMMIT)
    parser.add_argument('--port', type=int, default=21)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--actions', default=os.path.join('tests', 'data', 'actions_50.json'))
    parser.add_argument('--make-actions', action='store_true',
                        help='write the fixed action list (once) and exit')
    parser.add_argument('--force', action='store_true', help='with --make-actions: overwrite')
    # Driver mode (internal): one version, run by the controller in a subprocess.
    parser.add_argument('--drive', help=argparse.SUPPRESS)
    parser.add_argument('--root', help=argparse.SUPPRESS)
    parser.add_argument('--config', default=CONFIG, help=argparse.SUPPRESS)
    parser.add_argument('--sim-dir', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.port < MIN_PORT:
        parser.error('--port must be %d or higher (port 0 belongs to the queue)' % MIN_PORT)
    if args.drive:
        return drive(args)
    if args.steps < 1:
        parser.error('--steps must be at least 1')

    actions_path = os.path.normpath(args.actions if os.path.isabs(args.actions)
                                    else os.path.join(ROOT, args.actions))
    if args.make_actions:
        return make_actions(args, actions_path)
    if not args.base_worktree:
        parser.error('--base-worktree is required')
    with open(actions_path, encoding='utf-8') as f:
        spec = json.load(f)
    if len(spec['actions']) < args.steps:
        parser.error('%s has %d steps, %d requested' % (actions_path, len(spec['actions']), args.steps))

    base_root = os.path.abspath(args.base_worktree)
    try:
        head = setup_base(base_root, args.base_commit, args.seed)
    except (RuntimeError, OSError) as e:
        print('setup failed: %s' % e)
        return 2
    new_head = git('rev-parse', 'HEAD')
    dirty = git('status', '--porcelain', '--untracked-files=no', '--', 'envs', 'Malta/build_file.py')
    print('base %s (%s), new %s%s; port %d (TraCI %d), seed %d, %d steps, actions %s' % (
        head[:7], base_root, new_head[:7], ' + uncommitted env changes' if dirty else '',
        args.port, 27500 + args.port, args.seed, args.steps, actions_path), flush=True)

    work = tempfile.mkdtemp(prefix='compare_local_')
    try:
        base = run_driver('base', base_root, args, actions_path, args.steps, work)
        new = run_driver('new', ROOT, args, actions_path, args.steps, work)
        for label, d in (('base', base), ('new', new)):
            print('%-4s env comm_arch %s; build %.1f s, reset %.1f s, %d steps %.1f s '
                  '(%.3f s/step), process %.1f s' % (
                      label, d['comm_arch'], d['build_sec'], d['reset_sec'], len(d['step_sec']),
                      sum(d['step_sec']), float(np.mean(d['step_sec'])), d['wall_sec']))
        same = compare(base, new, args.steps)
        for label, d in (('base', base), ('new', new)):
            if d['n_a_ls'] != spec['n_a_ls']:
                print('FAIL: the %s env\'s n_a_ls differs from the action file\'s' % label)
                same = False
        return 0 if same else 1
    except RuntimeError as e:
        print('run failed: %s' % e)
        return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
