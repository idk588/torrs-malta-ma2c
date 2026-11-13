import itertools
import logging
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import pandas as pd


def check_dir(cur_dir):
    if not os.path.exists(cur_dir):
        return False
    return True

import os
import shutil

from run_files import append_cost_row, atomic_savetxt, atomic_write_csv, file_size_mb

def copy_file(src_dir, tar_dir, suffix='.ini'):

    os.makedirs(tar_dir, exist_ok=True)

    src = src_dir
    if os.path.isdir(src_dir):
        src = find_file(src_dir, suffix)
        if src is None:
            raise FileNotFoundError(f'Cannot find {suffix} under {src_dir}')

    shutil.copy(src, tar_dir)

def find_file(cur_dir, suffix='.ini'):
    for file in os.listdir(cur_dir):
        if file.endswith(suffix):
            return cur_dir + '/' + file
    logging.error('Cannot find %s file' % suffix)
    return None


def init_dir(base_dir, pathes=['log', 'data', 'model']):
    """Create base_dir/<path>/ for each path (parents too) and return {path: 'base_dir/<path>/'}.
    A base_dir given with a trailing '/' (a store such as 'Malta/runs/<name>/') is not doubled."""
    os.makedirs(base_dir, exist_ok=True)
    base = base_dir[:-1] if base_dir.endswith('/') else base_dir
    dirs = {}
    for path in pathes:
        cur_dir = base + '/%s/' % path
        os.makedirs(cur_dir, exist_ok=True)
        dirs[path] = cur_dir
    return dirs


def init_log(log_dir):
    # force=True: replace (and close) any handlers an earlier init_log or import installed,
    # so this run's log file always receives the log.
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler('%s/%d.log' % (log_dir, time.time())),
                            logging.StreamHandler()
                        ],
                        force=True)


def init_test_flag(test_mode):
    if test_mode == 'no_test':
        return False, False
    if test_mode == 'in_train_test':
        return True, False
    if test_mode == 'after_train_test':
        return False, True
    if test_mode == 'all_test':
        return True, True
    return False, False


def plot_train(data_dirs, labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metrics = ['reward', 'average waiting time', 'average travel time']
    ylabels = ['Cumulative Reward', 'Avg Waiting Time (s)', 'Avg Travel Time (s)']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))

    for data_dir, label in zip(data_dirs, labels):
        csv_path = os.path.join(data_dir, 'train_reward.csv')
        if not os.path.exists(csv_path):
            logging.warning('plot_train: no file at %s' % csv_path)
            continue
        df = pd.read_csv(csv_path)
        train_df = df[df['test_id'] == -1].copy()
        for ax, metric, ylabel in zip(axes, metrics, ylabels):
            ax.plot(train_df['step'], train_df[metric], label=label, linewidth=1.5)
            ax.set_xlabel('Training Step')
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(data_dirs[0], 'train_plot.png')
    plt.savefig(out, dpi=150)
    plt.close()
    logging.info('Saved training plot to %s' % out)


def plot_evaluation(data_dirs, labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    records = []
    for data_dir, label in zip(data_dirs, labels):
        # try eval CSV first, fall back to train_reward.csv
        found = False
        for fname in os.listdir(data_dir):
            if fname.endswith('_trip.csv') or fname.endswith('_traffic.csv'):
                df = pd.read_csv(os.path.join(data_dir, fname))
                if 'average waiting time' in df.columns:
                    records.append({
                        'label': label,
                        'avg_wait': df['average waiting time'].mean(),
                        'avg_travel': df['average travel time'].mean(),
                    })
                    found = True
                    break
        if not found:
            csv_path = os.path.join(data_dir, 'train_reward.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                records.append({
                    'label': label,
                    'avg_wait': df['average waiting time'].mean(),
                    'avg_travel': df['average travel time'].mean(),
                })

    if not records:
        logging.warning('plot_evaluation: no data found in provided dirs')
        return

    labs = [r['label'] for r in records]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, key, title in zip(axes,
                              ['avg_wait', 'avg_travel'],
                              ['Avg Waiting Time (s)', 'Avg Travel Time (s)']):
        vals = [r[key] for r in records]
        bars = ax.bar(labs, vals, color=['steelblue', 'coral', 'seagreen'][:len(vals)])
        ax.set_title(title)
        ax.set_ylabel('Seconds')
        ax.grid(True, axis='y', alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    '%.1f' % v, ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out = os.path.join(data_dirs[0], 'eval_plot.png')
    plt.savefig(out, dpi=150)
    plt.close()
    logging.info('Saved evaluation plot to %s' % out)


class Counter:
    def __init__(self, total_step,test_step, log_step):
        self.counter = itertools.count(1)
        self.cur_step = 0
        self.cur_test_step = 0
        self.total_step = total_step
        self.test_step = test_step
        self.log_step = log_step
        self.stop = False

    def next(self):
        self.cur_step = next(self.counter)
        return self.cur_step

    def resume(self, start_step):
        """Fast-forward the counter to continue from a loaded checkpoint's step."""
        self.counter = itertools.count(start_step + 1)
        self.cur_step = start_step
        self.cur_test_step = start_step

    def should_test(self):
        test = False
        if (self.cur_step - self.cur_test_step) >= self.test_step:
            test = True
            self.cur_test_step = self.cur_step
        return test

    def should_log(self):
        return (self.cur_step % self.log_step == 0)

    def should_stop(self):
        if self.cur_step >= self.total_step:
            return True
        return self.stop


class Trainer():
    def __init__(self, env, model, problem,global_counter, summary_writer, run_test, output_path=None,
                 store_dir=None, strict_save=False, sampler=None):
        """store_dir: the run store every per-episode file goes to (train_reward.csv,
        agent_reward.csv, episode_cost.csv, model/), with a trailing '/':
        'Malta/signal_route/' for RQ1, 'Malta/runs/<name>/' for RQ2.
        strict_save: re-raise a failed checkpoint save instead of logging a warning.
        sampler: a started run_files.MemorySampler for the cost row's memory peaks, or None.
        """
        if not isinstance(store_dir, str) or not store_dir.endswith('/'):
            raise ValueError("Trainer needs store_dir ending in '/', got %r" % (store_dir,))
        self.store_dir = store_dir
        self.strict_save = strict_save
        self.sampler = sampler
        self.last_saved_step = 0  # global step of the newest checkpoint this Trainer saved
        self.cur_step=0
        self.global_counter = global_counter
        self.problem=problem
        self.env = env
        self.agent = self.env.agent
        self.model = model
        self.sess = self.model.sess
        self.n_step = self.model.n_step
        self.summary_writer = summary_writer
        self.run_test = run_test
        assert self.env.T % self.n_step == 0
        self.data = []
        self.agent_reward=[]
        self.output_path = output_path
        if run_test:
            self.test_num = self.env.test_num
            logging.info('Testing: total test num: %d' % self.test_num)
        self._init_summary()

    def _init_summary(self):
        self.train_reward = tf.placeholder(tf.float32, [])
        self.train_summary = tf.summary.scalar('train_reward', self.train_reward)
        self.test_reward = tf.placeholder(tf.float32, [])
        self.test_summary = tf.summary.scalar('test_reward', self.test_reward)

    def _add_summary(self, reward, global_step, is_train=True):
        if is_train:
            summ = self.sess.run(self.train_summary, {self.train_reward: reward})
        else:
            summ = self.sess.run(self.test_summary, {self.test_reward: reward})
        self.summary_writer.add_summary(summ, global_step=global_step)

    def explore(self, prev_ob,prev_done):
        ob = prev_ob
        done = prev_done
        rewards = 0
        total_arrived_number = 0
        total_departed_number = 0
        total_waiting_time = 0
        total_travel_time = 0
        agent_reward=[]
        for i in range(self.n_step):

            policy, value = self.model.forward(ob, done)
            # need to update fingerprint before calling step
            self.env.update_fingerprint(policy)
            action = []
            for pi in policy:
                action.append(np.random.choice(np.arange(len(pi)), p=pi))
            next_ob, reward, done, global_reward,arrived_number, departed_number,waiting_time,travel_time,agent_reward0 = self.env.step(action)
            total_arrived_number += arrived_number
            total_departed_number += departed_number
            total_waiting_time += waiting_time
            total_travel_time += travel_time

            rewards += global_reward
            #if agent_reward==[]:
            if agent_reward is None or np.size(agent_reward) == 0:
                agent_reward=np.array(agent_reward0)
            else:
                agent_reward += np.array(agent_reward0)

            global_step = self.global_counter.next()
            self.cur_step +=1


            self.model.add_transition(ob, action, reward, value, done)
            if self.global_counter.should_log():
                logging.info('''Training: global step %d, current step %d, r: %.2f, train r: %.2f, done: %r''' %
                         (global_step, self.cur_step, global_reward, np.mean(reward), done))

            if done:
                break
            ob = next_ob

        if done:
            R = [0] * self.model.n_agent
        else:
            R = self.model.forward(ob, False, 'v')

        return ob, done, R, rewards,agent_reward,total_arrived_number, total_departed_number,total_waiting_time,total_travel_time

    def perform(self, test_ind, demo=True, policy_type='default',load=False, after_reset=None):
        ob = self.env.reset(gui=demo, test_ind=test_ind,load=load)
        if after_reset is not None:
            after_reset()  # the episode's route files now exist (Evaluator.run's first-reset hook)
        # note this done is pre-decision to reset LSTM states!
        done = True
        self.model.reset()
        rewards = 0
        total_arrived_number = 0
        total_departed_number = 0
        total_waiting_time = 0
        total_travel_time = 0

        total_agent_reward=[]
        while True:
            policy = self.model.forward(ob, done, 'p')
            self.env.update_fingerprint(policy)
            action = []
            for pi in policy:
                if policy_type != 'deterministic':
                    action.append(np.random.choice(np.arange(len(pi)), p=pi))
                else:
                    action.append(np.argmax(np.array(pi)))

            next_ob, reward, done, global_reward, arrived_number, departed_number,waiting_time,travel_time,agent_reward0 = self.env.step(action)

            total_arrived_number += arrived_number
            total_departed_number += departed_number
            total_waiting_time += waiting_time
            total_travel_time += travel_time

            rewards += global_reward
            if total_agent_reward is None or np.size(total_agent_reward) == 0:
                total_agent_reward=np.array(agent_reward0)
            else:
                total_agent_reward += np.array(agent_reward0)

            if done:
                break
            ob = next_ob

        return rewards,total_agent_reward,total_arrived_number, total_departed_number,total_waiting_time,total_travel_time

    def run(self):
        while not self.global_counter.should_stop():
            if self.run_test and self.global_counter.should_test():
                rewards = 0
                total_arrived_number = 0
                total_departed_number = 0
                total_waiting_time = 0
                total_travel_time = 0

                total_agent_reward=[]
                global_step = self.global_counter.cur_step
                self.env.train_mode = False
                for test_ind in range(self.test_num):
                    cur_rewards, _,arrived_number, departed_number,waiting_time,travel_time= self.perform(test_ind)
                    self.env.terminate()
                    rewards+=cur_rewards

                    total_arrived_number += arrived_number
                    total_departed_number += departed_number
                    total_waiting_time += waiting_time
                    total_travel_time += travel_time

                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'reward': rewards,
                            'total arrived vehicle number': total_arrived_number,
                           'total departured vehicle number': total_departed_number,
                           'total waiting time': total_waiting_time,
                           'total travel time': total_travel_time,
                           'average waiting time': total_waiting_time / total_departed_number,
                           'average travel time': total_travel_time / total_departed_number,
                           'seed': self.env.episode_seed}

                    self.data.append(log)
                self._add_summary(rewards, global_step, is_train=False)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, rewards))

            # train
            episode_start = time.perf_counter()  # wall_s: the whole episode, save included
            rollout_s = 0.0
            update_s = 0.0
            self.env.train_mode = True
            ob = self.env.reset()

            done=True
            self.model.reset()

            rewards = 0
            total_arrived_number=0
            total_departed_number=0
            total_waiting_time =0
            total_travel_time = 0
            total_agent_reward=[]
            self.cur_step = 0
            while True:
                t0 = time.perf_counter()
                ob, done, R, cur_rewards,agent_reward,arrived_number, departed_number,waiting_time,travel_time = self.explore(ob,done)
                rollout_s += time.perf_counter() - t0
                global_step = self.global_counter.cur_step
                rewards += cur_rewards
                total_arrived_number += arrived_number
                total_departed_number += departed_number
                total_waiting_time+=waiting_time
                total_travel_time+=travel_time

                if total_agent_reward is None or np.size(total_agent_reward) == 0:
                    total_agent_reward=np.array(agent_reward)
                else:
                    total_agent_reward+=np.array(agent_reward)

                t0 = time.perf_counter()
                self.model.backward(R, self.summary_writer, global_step)
                update_s += time.perf_counter() - t0
                # termination
                if done:
                    #到达3600s
                    #结束 关闭SUMO文件
                    self.env.terminate()
                    break

            log = {'agent': self.agent,
                   'step': global_step,
                   'test_id': -1,
                   'reward': rewards,
                   'total arrived vehicle number':total_arrived_number,
                   'total departured vehicle number': total_departed_number,
                   'total waiting time':total_waiting_time,
                   'total travel time': total_travel_time,
                   'average waiting time':total_waiting_time/total_departed_number,
                   'average travel time': total_travel_time / total_departed_number,
                   'seed': self.env.episode_seed
                   }

            self.data.append(log)
            self.agent_reward.append(total_agent_reward)

            self._add_summary(rewards, global_step)
            self.summary_writer.flush()

            self._persist_episode(global_step, episode_start, rollout_s, update_s)

    def _persist_episode(self, global_step, episode_start, rollout_s, update_s):
        """Write this episode's files to the store, in the order the queue runner relies on
        (spec section 7), so an interrupted run is still useful:
          1. train_reward.csv and agent_reward.csv (atomic);
          2. the checkpoint, <store>model/checkpoint-<step>;
          3. a row appended to episode_cost.csv;
          4. the log line 'Checkpoint saved at step <N> -> <store>model/'.
        A failed save re-raises when strict_save is set (no cost row, no log line);
        otherwise it logs a warning, and the cost row is still written without the line.
        """
        store = self.store_dir
        model_dir = store + 'model/'
        os.makedirs(store, exist_ok=True)

        # 1. Training history: the formats main.py's resume reads back (read_csv index_col=0,
        #    loadtxt delimiter=',').
        atomic_write_csv(pd.DataFrame(self.data), store + 'train_reward.csv')
        atomic_savetxt(self.agent_reward, store + 'agent_reward.csv', delimiter=',')

        # 2. Checkpoint.
        save_start = time.perf_counter()
        try:
            os.makedirs(model_dir, exist_ok=True)
            self.model.save(model_dir, global_step)
            saved = True
        except Exception as e:
            if self.strict_save:
                raise
            logging.warning('Per-episode checkpoint save failed: %s' % e)
            saved = False
        save_end = time.perf_counter()
        if saved:
            self.last_saved_step = global_step

        # 3. Compute cost of this episode.
        if self.sampler is not None:
            peak_private_gb, peak_working_set_gb = self.sampler.take_episode_peaks()
        else:
            peak_private_gb = peak_working_set_gb = float('nan')
        prefix = model_dir + 'checkpoint-%d' % global_step
        append_cost_row(store + 'episode_cost.csv', {
            'episode': global_step // int(self.env.T),
            'step': global_step,
            'seed': self.env.episode_seed,
            'wall_s': save_end - episode_start,
            'rollout_s': rollout_s,
            'update_s': update_s,
            'save_s': save_end - save_start,
            'peak_private_gb': peak_private_gb,
            'peak_working_set_gb': peak_working_set_gb,
            'ckpt_data_mb': file_size_mb(prefix + '.data-00000-of-00001') if saved else float('nan'),
            'ckpt_meta_mb': file_size_mb(prefix + '.meta') if saved else float('nan'),
            'lost': 0,
        })

        # 4. The line the queue runner parses (exact format).
        if saved:
            logging.info('Checkpoint saved at step %d -> %s' % (global_step, model_dir))


class Tester(Trainer):
    def __init__(self, env, model,problem,global_counter, summary_writer, output_path):
        super().__init__(env, model, global_counter, summary_writer)
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.problem=problem
        self.data = []
        logging.info('Testing: total test num: %d' % self.test_num)

    def _init_summary(self):
        self.reward = tf.placeholder(tf.float32, [])
        self.summary = tf.summary.scalar('test_reward', self.reward)

class Evaluator(Tester):
    def __init__(self, env, model, output_path, demo=False, policy_type='default'):
        self.env = env
        self.model = model
        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.demo = demo
        self.policy_type = policy_type

    def run(self, after_first_reset=None):
        """Evaluate every test seed, recording into self.output_path: SUMO's per-seed files and
        the three CSVs (episodes 1..n). The RQ2 sweep sets output_path to each demand level's
        folder before calling run; it must be a folder path ending in '/'.

        after_first_reset, if given, is called once, right after the first seed's env.reset
        (its route files are written) and before that seed's 'test 0' line.
        """
        if self.output_path and not self.output_path.endswith(('/', '\\')):
            raise ValueError('Evaluator output_path must end with a path separator: %r'
                             % self.output_path)
        is_record = True
        self.env.cur_episode = 0
        self.env.init_data(is_record, self.output_path)
        time.sleep(1)

        for test_ind in range(self.test_num):
            hook = after_first_reset if test_ind == 0 else None
            reward, _,_,_,_,_ = self.perform(test_ind, demo=self.demo, policy_type=self.policy_type,
                                             after_reset=hook)
            self.env.terminate()
            logging.info('test %i, avg reward %.2f' % (test_ind, reward))
            time.sleep(2)
            self.env.collect_tripinfo()
        self.env.output_data()

    def run_diff_com(self):
        numbers = np.arange(0, 1.1, 0.1)
        for i in numbers:
            self.env.CAV_rate=i

            folder_path="Sioux/%s/eva_data/%.1f/" % (self.env.problem, i)
            os.mkdir(folder_path)

            is_record = True
            self.env.cur_episode = 0
            self.env.init_data(is_record, self.output_path,diff=i)
            time.sleep(1)
            for test_ind in range(self.test_num):
                reward, _, _, _, _, _ = self.perform(test_ind, demo=False, policy_type=self.policy_type)
                self.env.terminate()
                logging.info('test %i, avg reward %.2f' % (test_ind, reward))
                time.sleep(2)
                self.env.collect_tripinfo()
            self.env.output_data()

    def run_diff_load(self):
        numbers = np.arange(0.2, 2.2, 0.2)
        for i in numbers:

            folder_path="Sioux/%s/eva_data/%.1f/" % (self.env.problem, i)
            os.mkdir(folder_path)

            is_record = True
            self.env.cur_episode = 0
            self.env.init_data(is_record, self.output_path,diff=i)
            time.sleep(1)
            for test_ind in range(self.test_num):
                reward, _, _, _, _, _ = self.perform(test_ind, demo=False, policy_type=self.policy_type,load=i)
                self.env.terminate()
                logging.info('test %i, avg reward %.2f' % (test_ind, reward))
                time.sleep(2)
                self.env.collect_tripinfo()
            self.env.output_data()

class Evaluator_separate(Evaluator):
    def __init__(self, env, model_signal,model_route, output_path, demo=False, policy_type='default'):
        self.env = env
        self.model_signal = model_signal
        self.model_route=model_route

        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.demo = demo
        self.policy_type = policy_type

    def run(self):
        is_record = True
        self.env.cur_episode = 0
        self.env.init_data(is_record, self.output_path)
        time.sleep(1)

        for test_ind in range(self.test_num):
            self.perform(test_ind, demo=True, policy_type=self.policy_type)
            self.env.terminate()
            time.sleep(2)
            self.env.collect_tripinfo()
        self.env.output_data()

    def perform(self, test_ind, demo=True, policy_type='default',load=False):
        ob = self.env.reset(gui=demo, test_ind=test_ind,load=load,separate=True)
        # note this done is pre-decision to reset LSTM states!
        done = True
        self.model_signal.reset()
        self.model_route.reset()
        while True:
            ob_signal=ob[:17]
            ob_route=ob[17:]

            policy_signal = self.model_signal.forward(ob_signal, done, 'p')
            policy_route=self.model_route.forward(ob_route, done, 'p')
            policy=policy_signal+policy_route

            self.env.update_fingerprint(policy)
            action_signal = []
            for pi in policy_signal:
                if policy_type != 'deterministic':
                    action_signal.append(np.random.choice(np.arange(len(pi)), p=pi))
                else:
                    action_signal.append(np.argmax(np.array(pi)))

            action_route = []
            for pi in policy_route:
                if policy_type != 'deterministic':
                    action_route.append(np.random.choice(np.arange(len(pi)), p=pi))
                else:
                    action_route.append(np.argmax(np.array(pi)))

            next_ob, done= self.env.step_separate(action_signal,action_route)

            if done:
                break
            ob=next_ob
