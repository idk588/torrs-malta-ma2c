import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Force CPU-only by default. Originally needed because the per-call
# Python<->GPU copy cost per agent dominated for batch-size-1 LSTM
# inference. After the IA2C.forward/backward batching refactor (one
# sess.run for all 190 agents), the GPU may actually win — try toggling
# this off (comment the next line) once a CPU-only run is verified.
#
# Note (Sep 2026): the installed tensorflow build here is CPU-only
# (no native Windows CUDA support since TF 2.11), so this stays forced
# off until that's addressed separately.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Enable Intel oneDNN CPU math kernels (typically 2-4x speedup on CPU
# matmul/conv/LSTM compared to the stock TF kernels).
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Let oneDNN/OpenMP use every core. os.cpu_count() returns logical cores.
_n_cores = str(os.cpu_count() or 8)
os.environ.setdefault('OMP_NUM_THREADS', _n_cores)
os.environ.setdefault('KMP_BLOCKTIME', '0')  # don't spin between calls
os.environ.setdefault('KMP_AFFINITY', 'granularity=fine,compact,1,0')

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import configparser
import sys
import tensorflow as tf
# Report what TF can see — useful to confirm oneDNN + GPU detection.
print('>>> TF version:', tf.__version__, '| logical CPUs:', _n_cores, flush=True)
try:
    _gpus = tf.config.list_physical_devices('GPU')
    print('>>> TF GPUs visible:', _gpus or 'none (CPU-only run)', flush=True)
except Exception as _e:
    print('>>> TF GPU probe failed:', _e, flush=True)
tf = tf.compat.v1
tf.disable_v2_behavior()
import threading

from envs.Malta_env import MaltaEnv as SiouxEnv, MaltaController as SiouxController
from agents.models import MA2C
from utils import *
from run_config import assert_not_rq1, other_run_pids, parse_args, prepare_store
from run_files import MemorySampler

# Exit code when another `main.py --run` python is alive (spec section 4, one instance).
EXIT_OTHER_RUN = 6


def init_env(config, port=0, naive_policy=False, output_path='', generate_traffic=True):
    """output_path: where SUMO writes its trip/log files ('' = the working directory, as RQ1).
    generate_traffic: False makes a missing background traffic file an error (RQ2 runs)."""
    env = SiouxEnv(config, problem, output_path=output_path, port=port,
                   generate_traffic=generate_traffic)
    if not naive_policy:
        return env
    policy = SiouxController(env.node_names, env.rc_names)
    return env, policy


def get_parameter(scenario):
    config_dir = default_config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)
    env = SiouxEnv(config['ENV_CONFIG'], problem, port=0)
    env.get_parameter(scenario)
    env.terminate()


def _read_resume_history(store, resumed_step):
    """The training history a resumed run carries forward: train_reward.csv as records and
    agent_reward.csv as rows, cut back to the checkpoint.

    Rows with step > resumed_step belong to an episode whose CSVs were written but whose
    checkpoint was not (a crash between the two); that episode is trained again, so its rows
    are dropped here, with the matching agent_reward rows (one per training row, in order).
    """
    data, agent_reward = [], []
    train_csv = store + 'train_reward.csv'
    agent_csv = store + 'agent_reward.csv'
    if os.path.exists(train_csv):
        data = pd.read_csv(train_csv, index_col=0).to_dict('records')
    if os.path.exists(agent_csv):
        arr = np.loadtxt(agent_csv, delimiter=',')
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        agent_reward = list(arr)
    kept = [r for r in data if r['step'] <= resumed_step]
    n_train = sum(1 for r in kept if r.get('test_id', -1) == -1)
    dropped_agent = max(0, len(agent_reward) - n_train)
    agent_reward = agent_reward[:n_train]
    if len(kept) < len(data) or dropped_agent:
        logging.warning('Resume: dropped %d train_reward row(s) and %d agent_reward row(s) '
                        'past checkpoint step %d' % (len(data) - len(kept), dropped_agent,
                                                     resumed_step))
    if len(agent_reward) < n_train:
        logging.warning('Resume: agent_reward.csv has %d row(s) for %d training episode(s)'
                        % (len(agent_reward), n_train))
    return kept, agent_reward


def train(spec):
    """Train MA2C for spec, resuming from the newest checkpoint in <store>model/ if any.

    --run (RQ2): everything stays in the run store spec.store_dir ('Malta/runs/<name>/'):
    model/, logs/ and sim/ (SUMO files). It refuses while another `main.py --run` python is
    alive and refuses RQ1 paths; prepare_store copies the config, zone file and neighbour-map
    hash on first launch and checks them on resume. A checkpoint-save error is raised.
    --rq1: the RQ1 layout under Malta/signal_route/ (log/, data/, model/), as before.

    The last line printed is '>>> TRAINING COMPLETE step <N>', N being the step reached
    (total_step), also when the checkpoint already is at total_step.
    """
    store = spec.store_dir
    rq2 = not spec.rq1
    if rq2:
        others = other_run_pids(os.getpid())
        if others:
            print('refusing: another main.py --run is running (pid %s)'
                  % ', '.join(str(p) for p in others), file=sys.stderr, flush=True)
            raise SystemExit(EXIT_OTHER_RUN)
    model_dir = store + 'model/'
    assert_not_rq1(store, spec.rq1)
    assert_not_rq1(model_dir, spec.rq1)
    config_path = spec.config_path
    if not os.path.isfile(config_path):
        raise SystemExit('refusing: config %s not found' % config_path)
    if rq2:
        dirs = init_dir(store, pathes=['model', 'logs', 'sim'])
        log_dir, sim_dir = dirs['logs'], dirs['sim']
    else:
        dirs = init_dir(store)  # RQ1: log/, data/, model/
        log_dir, sim_dir = dirs['log'], ''
    init_log(log_dir)
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8-sig')  # as run_config compares it; a BOM is ignored
    in_test, _ = init_test_flag('no_test')  # no in-training tests, as before

    # init env
    print(">>> Initializing env (launches SUMO for dimensions)...", flush=True)
    env = init_env(config['ENV_CONFIG'], port=0, output_path=sim_dir, generate_traffic=not rq2)
    logging.info('Training: s dim: %d, a dim %d, s dim ls: %r, a dim ls: %r' %
                 (env.n_s, env.n_a, env.n_s_ls, env.n_a_ls))
    if rq2:
        first = prepare_store(spec, env.neighbour_hash())
        logging.info('Run %s: %s; store %s, config %s' % (
            spec.name, 'first launch' if first else 'resume, store checked', store, config_path))

    print(">>> Building MA2C model...", flush=True)

    # init step counter
    total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
    test_step = int(config.getfloat('TRAIN_CONFIG', 'test_interval'))
    log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
    global_counter = Counter(total_step, test_step, log_step)

    # init centralized or multi agent
    seed = config.getint('ENV_CONFIG', 'seed')

    if problem == 'signal_route':
        model = MA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, env.n_f_ls, total_step,
                     config['MODEL_CONFIG'], problem, agent_type=env.agent_type, n_o_ls=env.n_o_ls, seed=seed)
    else:
        model = MA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, env.n_f_ls, total_step,
                     config['MODEL_CONFIG'], problem, seed=seed)

    # ------------------------------------------------------------------
    # Resume from the latest checkpoint if one exists, instead of always
    # training from scratch. Also carries the training-history CSVs
    # forward so plots/analysis over the full run stay contiguous.
    # model.load raises if save_optimizer is set and the checkpoint has
    # no Adam state; that stops training (no fallback).
    # ------------------------------------------------------------------
    resume_data, resume_agent_reward = [], []
    if model.load(model_dir):
        resumed_step = model.cur_step
        if resumed_step >= total_step:
            print(">>> Checkpoint already at step %d >= target total_step %d; nothing to do." %
                  (resumed_step, total_step), flush=True)
            print('>>> TRAINING COMPLETE step %d' % resumed_step, flush=True)
            return
        global_counter.resume(resumed_step)
        # The LR/entropy scheduler is a plain Python object, not a saved TF
        # variable, so model.load() does not restore its progress. Without
        # this, every resume would restart the decay schedule from n=0,
        # sawtoothing the learning rate instead of decaying smoothly across
        # a multi-session run. Units match: both count cumulative env steps.
        model.lr_scheduler.n = resumed_step
        # seed_continue (spec section 7): the run's episode k trains on seed + k - 1 across
        # resumes too, so a full run uses each training seed once. reset() then increments it.
        if config.getboolean('ENV_CONFIG', 'seed_continue', fallback=False):
            env.seed = seed + resumed_step // int(env.T)
            logging.info('Resume: training seeds continue at %d' % env.seed)
        resume_data, resume_agent_reward = _read_resume_history(store, resumed_step)
        print(">>> Resumed from checkpoint at step %d (target total_step: %d); carried forward %d prior episode(s)." %
              (resumed_step, total_step, len(resume_data)), flush=True)
    else:
        print(">>> No existing checkpoint found; starting training from scratch.", flush=True)

    summary_writer = tf.summary.FileWriter(log_dir)
    sampler = None
    if rq2:
        sampler = MemorySampler()
        sampler.start()
    trainer = Trainer(env, model, problem, global_counter, summary_writer, in_test,
                      store_dir=store, strict_save=rq2, sampler=sampler)
    trainer.data = resume_data
    trainer.agent_reward = resume_agent_reward
    print(">>> Starting trainer.run() — first episode loading SUMO...", flush=True)
    try:
        trainer.run()
    finally:
        if sampler is not None:
            sampler.stop()

    # save model, unless the per-episode loop already saved this step
    final_step = global_counter.cur_step
    if trainer.last_saved_step == final_step:
        logging.info('Training: final model already saved at step %d' % final_step)
    else:
        logging.info('Training: save final model at step %d ...' % final_step)
        model.save(model_dir, final_step)
    summary_writer.close()
    print('>>> TRAINING COMPLETE step %d' % final_step, flush=True)


def evaluate_fn(agent, output_dir, seeds, port, demo, policy_type):
    # load config file for env
    config_dir = default_config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)

    # init env
    env, greedy_policy = init_env(config['ENV_CONFIG'], port=port, naive_policy=True)
    logging.info('Evaluation: s dim: %d, a dim %d, s dim ls: %r, a dim ls: %r' %
                 (env.n_s, env.n_a, env.n_s_ls, env.n_a_ls))
    env.init_test_seeds(seeds)
    if separate_train == True:
        n_s_ls = [46, 36, 14, 22, 38, 38, 30, 22, 28, 34, 28, 30, 12, 14, 14, 28, 24]
        n_a_ls = [5, 4, 2, 2, 4, 4, 2, 2, 2, 4, 2, 4, 2, 2, 2, 4, 2]
        n_w_ls = [10, 8, 6, 6, 8, 8, 6, 6, 6, 8, 6, 8, 6, 6, 6, 8, 6]
        n_f_ls = [11, 7, 3, 6, 9, 9, 8, 6, 7, 6, 7, 7, 1, 3, 3, 5, 7]
        model_signal = MA2C(n_s_ls, n_a_ls, n_w_ls, n_f_ls, 0,
                             config['MODEL_CONFIG'], 'signal', agent_type=env.agent_type, n_o_ls=env.n_o_ls)
        n_s_ls = [1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1]
        n_a_ls = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        n_w_ls = [6, 7, 10, 7, 6, 10, 7, 9, 7, 10, 6, 9]
        n_f_ls = [0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0]
        model_route = MA2C(n_s_ls, n_a_ls, n_w_ls, n_f_ls, 0,
                            config['MODEL_CONFIG'], 'route', agent_type=env.agent_type, n_o_ls=env.n_o_ls)
        if not model_signal.load(default_base_dir + '/signal/model/'):
            return
        if not model_route.load(default_base_dir + '/route/model/'):
            return
        env.agent = agent
        evaluator = Evaluator_separate(env, model_signal, model_route, output_dir, demo=demo, policy_type=policy_type)
        evaluator.run()
        return

    if problem == 'signal_route':
        model = MA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, env.n_f_ls, 0,
                      config['MODEL_CONFIG'], problem, agent_type=env.agent_type, n_o_ls=env.n_o_ls)
    else:
        model = MA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, env.n_f_ls, 0,
                      config['MODEL_CONFIG'], problem)

    if not model.load(default_store_dir + '/model/'):
        return

    env.agent = agent
    evaluator = Evaluator(env, model, output_dir, demo=demo, policy_type=policy_type)
    if diff == False:
        evaluator.run()
    else:
        if compliance == True:
            evaluator.run_diff_com()
        elif load == True:
            evaluator.run_diff_load()


def run_greedy(port=0):
    """Run one full episode with the MaltaController (greedy/naive policy) and save metrics.

    `port` offsets the SUMO TraCI port (DEFAULT_PORT + port) so this can run
    alongside an active training/eval run on the default port without colliding.
    """
    import pandas as pd

    config = configparser.ConfigParser()
    config.read(default_config_dir)
    env, policy = init_env(config['ENV_CONFIG'], port=port, naive_policy=True)

    store_dir = default_store_dir
    dirs = init_dir(store_dir, pathes=['greedy_data', 'log'])
    init_log(dirs['log'])

    seeds_str = config.get('ENV_CONFIG', 'test_seeds', fallback='10000,20000')
    seeds = [int(s) for s in seeds_str.split(',')]
    env.init_test_seeds(seeds)
    env.train_mode = False
    # Record tripinfo exactly as Evaluator does, so greedy and MA2C share one metric pipeline.
    env.cur_episode = 0
    env.init_data(True, dirs['greedy_data'])

    data = []
    for test_ind in range(env.test_num):
        ob = env.reset(gui=False, test_ind=test_ind)
        done = False
        total_reward = 0
        total_arrived = 0
        total_departed = 0
        total_wait = 0
        total_travel = 0
        signal_n_a = env.n_a_ls[:len(env.node_names)]
        step_count = 0
        while not done:
            actions = policy.forward(ob)
            # clip each signal action to its node's valid phase range
            actions = [min(a, n - 1) for a, n in zip(actions, signal_n_a)]
            # route agents get a fixed action (stay on current route) when using greedy signal only
            full_actions = actions + [0] * len(env.rc_names)
            ob, reward, done, global_reward, arrived, departed, wait, travel, _ = env.step(full_actions)
            step_count += 1
            if step_count % 72 == 0:
                print('  episode %d | sim time %ds / %ds | arrived %d' % (
                    test_ind, step_count * 5, 3600, total_arrived), flush=True)
            total_reward += global_reward
            total_arrived += arrived
            total_departed += departed
            total_wait += wait
            total_travel += travel

        env.terminate()
        time.sleep(2)
        env.collect_tripinfo()
        if total_departed > 0:
            avg_wait = total_wait / total_departed
            avg_travel = total_travel / total_departed
        else:
            avg_wait = avg_travel = float('nan')

        log = {
            'agent': 'greedy',
            'test_id': test_ind,
            'reward': total_reward,
            'total arrived': total_arrived,
            'total departed': total_departed,
            'average waiting time': avg_wait,
            'average travel time': avg_travel,
        }
        data.append(log)
        logging.info('Greedy test %d: avg_wait=%.2f s, avg_travel=%.2f s' % (test_ind, avg_wait, avg_travel))

    env.output_data()
    out_path = os.path.join(dirs['greedy_data'], 'greedy_baseline.csv')
    pd.DataFrame(data).to_csv(out_path, index=False)
    logging.info('Greedy baseline saved to %s' % out_path)


def evaluate():
    store_dir = default_store_dir
    dirs = init_dir(store_dir, pathes=['eva_data', 'eva_log'])
    init_log(dirs['eva_log'])
    agents = {'ma2c'}
    seeds = ','.join([str(i) for i in range(10000, 100001, 10000)])
    policy_type = 'default'
    logging.info('Evaluation: policy type: %s, random seeds: %s' % (policy_type, seeds))
    if not seeds:
        seeds = []
    else:
        seeds = [int(s) for s in seeds.split(',')]
    threads = []
    for i, agent in enumerate(agents):
        demo = False
        thread = threading.Thread(target=evaluate_fn,
                                   args=(agent, dirs['eva_data'], seeds, i, demo, policy_type))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


def evaluate_sweep(spec):
    """Evaluate run spec.name at its demand levels (spec section 8)."""
    raise SystemExit('evaluate_sweep: implemented in Task 11')


# Defaults for the RQ1 functions (evaluate, run_greedy, get_parameter); main() sets
# default_config_dir, default_store_dir and option from the command line before dispatch.
default_config_dir = './config/config_ma2c_Malta.ini'
scenario = 'Malta'

option = 'train_then_evaluate'  # 'train', 'evaluate', 'train_then_evaluate', 'greedy', 'get_parameter'

separate_train = False

diff = False
compliance = False
load = False

problem = 'signal_route'
default_base_dir = scenario
if separate_train == True:
    default_store_dir = default_base_dir + '/signal_route(separate_training)'
else:
    default_store_dir = default_base_dir + '/' + problem


def main(argv=None):
    """python main.py --run <name> [--option train|evaluate] [--levels 1.10,2.00]
    or            python main.py --rq1 [--option train|evaluate|train_then_evaluate|greedy|get_parameter]

    Without --run or --rq1 it refuses (exit 2). Returns 0; refusals and errors exit through
    SystemExit.
    """
    global default_config_dir, default_store_dir, option
    spec = parse_args(sys.argv[1:] if argv is None else argv)
    option = spec.option
    default_config_dir = spec.config_path
    default_store_dir = spec.store_dir.rstrip('/')  # the RQ1 functions add '/...' themselves

    if not spec.rq1:
        if option == 'train':
            train(spec)
        elif option == 'evaluate':
            evaluate_sweep(spec)
        return 0

    if option == 'train':
        train(spec)
    elif option == 'train_then_evaluate':
        print(">>> Stage 1/2: training...", flush=True)
        train(spec)
        print(">>> Stage 2/2: evaluating freshly-trained model...", flush=True)
        evaluate()
        print(">>> train_then_evaluate complete.", flush=True)
    elif option == 'evaluate':
        evaluate()
    elif option == 'greedy':
        run_greedy()
    elif option == 'get_parameter':
        get_parameter(scenario)
    return 0


if __name__ == '__main__':
    sys.exit(main())
