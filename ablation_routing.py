# -*- coding: utf-8 -*-
"""Route-redistribution ablation.

Evaluates one trained MA2C model twice under identical conditions:

  routing_on  — route agents choose freely (full TORRS)
  routing_off — route agents forced to alternative A (signal control only)

The difference isolates what route redistribution contributes on top of
signal control. Both arms use the same checkpoint, the same evaluation
seeds, and the same action-sampling stream, so routing is the only variable.

Usage:
    python ablation_routing.py              # stochastic sampling (policy_type 'default')
    python ablation_routing.py deterministic  # argmax action selection
"""
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import configparser
import numpy as np
import pandas as pd
import tensorflow as tf
tf = tf.compat.v1
tf.disable_v2_behavior()

from envs.Malta_env import MaltaEnv
from agents.models import MA2C

DETERMINISTIC = len(sys.argv) > 1 and sys.argv[1].startswith('det')

CONFIG = './config/config_ma2c_Malta.ini'
MODEL_DIR = 'Malta/signal_route/model/'
OUT_DIR = ('Malta/signal_route/ablation_data_deterministic/' if DETERMINISTIC
           else 'Malta/signal_route/ablation_data/')
PROBLEM = 'signal_route'
PORT = 5                      # avoid colliding with any training run
SEEDS = list(range(10000, 100001, 10000))
ACTION_SEED = 20260919        # identical sampling stream for both arms


def build(config):
    env = MaltaEnv(config['ENV_CONFIG'], PROBLEM, port=PORT)
    env.init_test_seeds(SEEDS)
    env.train_mode = False
    model = MA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, env.n_f_ls, 0,
                 config['MODEL_CONFIG'], PROBLEM,
                 agent_type=env.agent_type, n_o_ls=env.n_o_ls)
    if not model.load(MODEL_DIR):
        raise SystemExit('no checkpoint found in %s' % MODEL_DIR)
    return env, model


def run_arm(env, model, route_off, label):
    """One evaluation pass over all seeds. Returns per-trip DataFrame."""
    n_signal = len(env.node_names)
    n_route = len(env.rc_names)
    out = os.path.join(OUT_DIR, label) + os.sep
    os.makedirs(out, exist_ok=True)

    np.random.seed(ACTION_SEED)
    env.cur_episode = 0
    env.init_data(True, out)

    swaps = {'b_chosen': 0, 'total': 0}
    route_probs = []          # P(alternative B) per route agent decision
    for test_ind in range(len(SEEDS)):
        ob = env.reset(gui=False, test_ind=test_ind)
        done = True
        model.reset()
        while True:
            policy = model.forward(ob, done, 'p')
            env.update_fingerprint(policy)
            if DETERMINISTIC:
                action = [int(np.argmax(np.array(pi))) for pi in policy]
            else:
                action = [int(np.random.choice(np.arange(len(pi)), p=pi))
                          for pi in policy]
            route_actions = action[n_signal:n_signal + n_route]
            swaps['total'] += len(route_actions)
            swaps['b_chosen'] += sum(1 for a in route_actions if a == 1)
            for pi in policy[n_signal:n_signal + n_route]:
                if len(pi) > 1:
                    route_probs.append(float(pi[1]))
            if route_off:
                action[n_signal:n_signal + n_route] = [0] * n_route
            ob, reward, done, global_reward, arrived, departed, wait, travel, _ = env.step(action)
            if done:
                break
        env.terminate()
        import time
        time.sleep(1)
        env.collect_tripinfo()
        print('  [%s] seed %d done' % (label, SEEDS[test_ind]), flush=True)
    env.output_data()

    trip = pd.read_csv(out + 'Malta_%s_trip.csv' % PROBLEM)
    pct_b = 100.0 * swaps['b_chosen'] / max(swaps['total'], 1)
    print('  [%s] route agents chose alternative B on %.1f%% of decisions'
          % (label, pct_b))
    if route_probs:
        p = np.array(route_probs)
        # How decisive is the route policy? A near-uniform policy sits at 0.5
        # with little spread; a policy that has actually learned something
        # pushes mass toward 0 or 1 depending on conditions.
        print('  [%s] P(alt B): mean=%.3f std=%.3f  |  decisive calls '
              '(p<0.2 or p>0.8): %.1f%%'
              % (label, p.mean(), p.std(),
                 100.0 * np.mean((p < 0.2) | (p > 0.8))))
    return trip, pct_b


def summarise(trip, label):
    per_seed = trip.groupby('episode').agg({'wait_sec': 'mean', 'duration_sec': 'mean'})
    return {
        'arm': label,
        'trips': len(trip),
        'wait_mean': trip['wait_sec'].mean(),
        'wait_std': per_seed['wait_sec'].std(),
        'travel_mean': trip['duration_sec'].mean(),
        'travel_std': per_seed['duration_sec'].std(),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    config = configparser.ConfigParser()
    config.read(CONFIG)
    env, model = build(config)

    results = []
    for route_off, label in [(False, 'routing_on'), (True, 'routing_off')]:
        print('\n=== %s ===' % label, flush=True)
        trip, pct_b = run_arm(env, model, route_off, label)
        r = summarise(trip, label)
        r['pct_alt_b'] = pct_b
        results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR + 'ablation_summary.csv', index=False)

    print('\n=== ROUTE-REDISTRIBUTION ABLATION ===')
    for r in results:
        print('%-12s trips=%d  wait=%.2f +/- %.2f  travel=%.2f +/- %.2f'
              % (r['arm'], r['trips'], r['wait_mean'], r['wait_std'],
                 r['travel_mean'], r['travel_std']))
    on, off = results[0], results[1]
    for metric in ('wait', 'travel'):
        o, f = on[metric + '_mean'], off[metric + '_mean']
        print('%s: routing contributes %.2fs (%.1f%% vs signal-only)'
              % (metric, f - o, (f - o) / f * 100 if f else 0))
    print('\nsaved %sablation_summary.csv' % OUT_DIR)


if __name__ == '__main__':
    main()
