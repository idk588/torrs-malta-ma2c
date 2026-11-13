"""MA2C checkpoints: optimiser state and explicit-step load (spec section 7).

A tiny signal_route MA2C (one signal agent, one route agent) built with the RQ1
MODEL_CONFIG. Checkpoints go to a temp dir. TF startup makes this take about a minute.
"""
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import configparser
import shutil
import tempfile
import unittest

import numpy as np
import tensorflow.compat.v1 as tf

from agents.models import MA2C

CONFIG = 'config/config_ma2c_Malta.ini'
N_S_LS = [4, 2]
N_A_LS = [2, 2]
N_W_LS = [2, 1]
N_F_LS = [1, 1]
N_O_LS = [0, 0]
N_SIGNAL, N_ROUTE = 1, 1
TOTAL_STEP = 288


def agent_type_ls(n_signal, n_route):
    # As env.agent_type builds it for signal_route: every signal agent, then every route agent.
    return ['signal'] * n_signal + ['route'] * n_route


def model_config(save_optimizer=None):
    """The RQ1 MODEL_CONFIG; save_optimizer=None leaves the key absent, as in RQ1."""
    config = configparser.ConfigParser()
    if not config.read(CONFIG):
        raise FileNotFoundError(CONFIG)
    if save_optimizer is not None:
        config['MODEL_CONFIG']['save_optimizer'] = save_optimizer
    return config['MODEL_CONFIG']


def ckpt_names(prefix):
    return [name for name, _ in tf.train.list_variables(prefix)]


def graph_var(model, name):
    for v in model.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if v.op.name == name:
            return v
    raise KeyError(name)


def graph_names(model):
    return [v.op.name for v in model.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]


def set_value(model, name, value):
    v = graph_var(model, name)
    v.load(np.full(v.shape.as_list(), value, np.float32), model.sess)


def get_value(model, name):
    return model.sess.run(graph_var(model, name))


def is_adam(name):
    return name.endswith('/Adam') or name.endswith('/Adam_1') or 'beta1_power' in name \
        or 'beta2_power' in name


class TestModelSaver(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        # A2C.save/load concatenate paths, so the model dir must end with '/'.
        self.model_dir = tmp.replace('\\', '/') + '/'

    def build(self, total_step, save_optimizer=None):
        model = MA2C(N_S_LS, N_A_LS, N_W_LS, N_F_LS, total_step,
                     model_config(save_optimizer), 'signal_route',
                     agent_type=agent_type_ls(N_SIGNAL, N_ROUTE), n_o_ls=N_O_LS)
        self.addCleanup(model.sess.close)
        return model

    def test_checkpoint_holds_adam_state(self):
        model = self.build(TOTAL_STEP, 'true')
        self.assertTrue(model.training)
        self.assertTrue(model.save_optimizer)
        model.save(self.model_dir, 144)
        names = ckpt_names(self.model_dir + 'checkpoint-144')
        self.assertTrue(any(n.endswith('/Adam') for n in names), names)
        self.assertTrue(any(n.endswith('beta1_power') for n in names), names)

    def test_resume_restores_adam_state(self):
        model = self.build(TOTAL_STEP, 'true')
        slot = sorted(n for n in graph_names(model) if n.endswith('/Adam'))[0]
        set_value(model, slot, 0.25)
        model.save(self.model_dir, 144)

        resumed = self.build(TOTAL_STEP, 'true')
        self.assertTrue(resumed.load(self.model_dir))
        self.assertEqual(resumed.cur_step, 144)
        np.testing.assert_array_equal(get_value(resumed, slot), get_value(model, slot))

    def test_eval_model_restores_policy_from_full_checkpoint(self):
        model = self.build(TOTAL_STEP, 'true')
        weight = sorted(v.op.name for v in model.sess.graph.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES))[0]
        set_value(model, weight, 0.5)
        model.save(self.model_dir, 144)
        set_value(model, weight, 0.75)
        model.save(self.model_dir, 288)  # newer, so loading 144 must honour the step asked for

        evaluator = self.build(0, 'true')
        self.assertFalse(evaluator.training)
        self.assertFalse(any(is_adam(n) for n in graph_names(evaluator)))
        self.assertTrue(evaluator.load(self.model_dir, checkpoint=144))
        self.assertEqual(evaluator.cur_step, 144)
        np.testing.assert_array_equal(get_value(evaluator, weight),
                                      np.full(graph_var(evaluator, weight).shape.as_list(),
                                              0.5, np.float32))

    def test_training_refuses_checkpoint_without_adam(self):
        old = self.build(TOTAL_STEP, 'false')
        old.save(self.model_dir, 144)
        self.assertFalse(any(is_adam(n) for n in ckpt_names(self.model_dir + 'checkpoint-144')))

        model = self.build(TOTAL_STEP, 'true')
        expected = ('checkpoint %scheckpoint-144 has no optimiser state; refusing to resume'
                    % self.model_dir)
        with self.assertRaises(RuntimeError) as cm:
            model.load(self.model_dir)
        self.assertEqual(str(cm.exception), expected)
        with self.assertRaises(RuntimeError) as cm:
            model.load(self.model_dir, checkpoint=144)
        self.assertEqual(str(cm.exception), expected)

    def test_rq1_config_saver_has_no_adam(self):
        model = self.build(TOTAL_STEP)  # save_optimizer absent, as in config_ma2c_Malta.ini
        self.assertTrue(model.training)
        self.assertFalse(model.save_optimizer)
        # The optimiser exists in the graph; the Saver just does not cover it.
        self.assertTrue(any(n.endswith('/Adam') for n in graph_names(model)))
        model.save(self.model_dir, 144)
        self.assertFalse(any(is_adam(n) for n in ckpt_names(self.model_dir + 'checkpoint-144')))

        resumed = self.build(TOTAL_STEP)
        self.assertTrue(resumed.load(self.model_dir))
        self.assertEqual(resumed.cur_step, 144)
        evaluator = self.build(0)  # as ablation_routing.py builds it
        self.assertTrue(evaluator.load(self.model_dir))
        self.assertEqual(evaluator.cur_step, 144)


if __name__ == '__main__':
    unittest.main()
