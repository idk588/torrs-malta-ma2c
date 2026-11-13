import json
import os
import tempfile
import unittest

import run_config as rc


class ParseArgsTests(unittest.TestCase):
    def test_defaults(self):
        s = rc.parse_args(['--run', 'cloud_base'])
        self.assertEqual(s.option, 'train')
        self.assertEqual(s.store_dir, 'Malta/runs/cloud_base/')
        self.assertEqual(s.config_path, 'config/runs/cloud_base.ini')
        self.assertFalse(s.rq1)

    def test_levels_and_smoke(self):
        s = rc.parse_args(['--run', '_smoke_fog', '--option', 'evaluate',
                           '--levels', '1.10,2.00'])
        self.assertEqual(s.levels, [1.10, 2.00])
        self.assertEqual(s.option, 'evaluate')

    def test_bad_cli(self):
        for argv in (['--run', '../signal_route'], ['--run', 'Cloud'],
                     ['--run', 'x', '--rq1'], ['--run', 'x', '--option', 'greedy'],
                     [], ['--option', 'train'], ['--run', '_smoke_a1']):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit) as cm:
                    rc.parse_args(argv)
                self.assertEqual(cm.exception.code, 2)

    def test_rq1(self):
        s = rc.parse_args(['--rq1'])
        self.assertEqual(s.option, 'train_then_evaluate')
        self.assertEqual(s.store_dir, 'Malta/signal_route/')
        self.assertEqual(s.config_path, 'config/config_ma2c_Malta.ini')
        self.assertTrue(s.rq1)
        self.assertEqual(rc.parse_args(['--rq1', '--option', 'greedy']).option, 'greedy')

    def test_kernel_arg_ignored(self):
        s = rc.parse_args(['--run', 'x', '-f', 'kernel.json'])
        self.assertEqual(s.name, 'x')


class GuardTests(unittest.TestCase):
    def test_true(self):
        for p in ('Malta/signal_route', 'Malta\\signal_route\\',
                  './malta/Signal_Route/model/', 'Malta/signal_route/x'):
            with self.subTest(p=p):
                self.assertTrue(rc.is_rq1_path(p))

    def test_false(self):
        for p in ('Malta/signal_route_copy', 'Malta/signal_route(separate_training)',
                  'Malta/runs/cloud_base/', 'Malta/signal_route_100ep_fixedroutes_backup'):
            with self.subTest(p=p):
                self.assertFalse(rc.is_rq1_path(p))

    def test_assert(self):
        rc.assert_not_rq1('Malta/runs/x/', False)
        rc.assert_not_rq1('Malta/signal_route/', True)
        with self.assertRaises(SystemExit) as cm:
            rc.assert_not_rq1('Malta/signal_route/model/', False)
        self.assertEqual(cm.exception.code, 3)

    def test_store_string_with_backslash_cwd(self):
        s = rc.parse_args(['--run', 'cloud_base'])
        self.assertNotIn('\\', s.store_dir)
        self.assertTrue(s.store_dir.endswith('/'))
        self.assertFalse(os.path.isabs(s.store_dir))


class ConfigsEqualTests(unittest.TestCase):
    def _w(self, d, name, text, nl='\n'):
        p = os.path.join(d, name)
        with open(p, 'w', newline='') as f:
            f.write(text.replace('\n', nl))
        return p

    def test_equal_ignoring_noise(self):
        with tempfile.TemporaryDirectory() as d:
            a = self._w(d, 'a.ini', '[ENV_CONFIG]\n# c\ncomm_arch = cloud\n')
            b = self._w(d, 'b.ini', '; other\n[ENV_CONFIG]\ncomm_arch=cloud  \n', '\r\n')
            self.assertEqual(rc.configs_equal(a, b), (True, []))

    def test_diff(self):
        with tempfile.TemporaryDirectory() as d:
            a = self._w(d, 'a.ini', '[ENV_CONFIG]\ncomm_arch = cloud\nx = 1\n[S]\nk = 1\n')
            b = self._w(d, 'b.ini', '[ENV_CONFIG]\ncomm_arch = fog\n')
            ok, diffs = rc.configs_equal(a, b)
            self.assertFalse(ok)
            self.assertEqual(diffs, ['ENV_CONFIG.comm_arch', 'ENV_CONFIG.x', 'S'])


class PidParseTests(unittest.TestCase):
    def test_list_and_dict_and_null(self):
        j = ('[{"ProcessId":5,"CommandLine":"python main.py --run a"},'
             '{"ProcessId":6,"CommandLine":null},'
             '{"ProcessId":7,"CommandLine":"python main.py --option train"},'
             '{"ProcessId":8,"CommandLine":"python  main.py  --run b_1"}]')
        self.assertEqual(rc.parse_pid_json(j, 99), [5, 8])
        self.assertEqual(rc.parse_pid_json(j, 5), [8])
        self.assertEqual(rc.parse_pid_json(
            '{"ProcessId":5,"CommandLine":"python main.py --run a"}', 99), [5])
        self.assertEqual(rc.parse_pid_json('', 1), [])

    def test_run_arg_forms(self):
        def one(cmd):
            return rc.parse_pid_json(
                '{"ProcessId":5,"CommandLine":%s}' % json.dumps(cmd), 99)
        self.assertEqual(one('python main.py --option evaluate --run x'), [5])
        self.assertEqual(one('python main.py --run=x'), [5])
        self.assertEqual(one('python main.py --option train'), [])
        self.assertEqual(one('python other.py --run x'), [])
        self.assertEqual(one('python notmain.py --run x'), [])

    def test_parent_chain_excluded(self):
        cmd = 'python main.py --run a'
        j = json.dumps([
            {'ProcessId': 10, 'ParentProcessId': 1, 'CommandLine': cmd},
            {'ProcessId': 11, 'ParentProcessId': 10, 'CommandLine': cmd},
            {'ProcessId': 12, 'ParentProcessId': 11, 'CommandLine': cmd},
            {'ProcessId': 20, 'ParentProcessId': 1, 'CommandLine': 'python main.py --run b'},
        ])
        self.assertEqual(rc.parse_pid_json(j, 12), [20])


class PrepareStoreTests(unittest.TestCase):
    def test_first_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            ini = os.path.join(d, 'r.ini')
            with open(ini, 'w') as f:
                f.write('[ENV_CONFIG]\ncomm_arch = cloud\n')
            store = os.path.join(d, 'store') + '/'
            spec = rc.RunSpec('r', 'train', None, False, ini, store)
            self.assertTrue(rc.prepare_store(spec, 'abc'))
            for n in ('config.ini', 'zones_nso.csv', 'neighbours.sha256'):
                self.assertTrue(os.path.isfile(os.path.join(store, n)), n)
            self.assertFalse(os.path.exists(os.path.join(store, 'config.ini.tmp')))
            with open(os.path.join(store, 'neighbours.sha256')) as f:
                self.assertEqual(f.read(), 'abc\n')
            self.assertFalse(rc.prepare_store(spec, 'abc'))
            with self.assertRaises(SystemExit) as cm:
                rc.prepare_store(spec, 'def')
            self.assertEqual(cm.exception.code, 4)
            with open(ini, 'w') as f:
                f.write('[ENV_CONFIG]\ncomm_arch = fog\n')
            with self.assertRaises(SystemExit) as cm:
                rc.prepare_store(spec, 'abc')
            self.assertEqual(cm.exception.code, 4)


if __name__ == '__main__':
    unittest.main()
