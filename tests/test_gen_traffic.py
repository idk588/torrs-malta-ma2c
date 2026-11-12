import unittest

from tools.gen_traffic import parse_seeds, needed_files


class TestGenTraffic(unittest.TestCase):
    def test_parse_range(self):
        self.assertEqual(parse_seeds('12-14'), [12, 13, 14])

    def test_parse_list(self):
        self.assertEqual(parse_seeds('10000,20000'), [10000, 20000])

    def test_parse_test(self):
        self.assertEqual(parse_seeds('test'), list(range(10000, 100001, 10000)))

    def test_needed_files(self):
        self.assertEqual(needed_files([12], [10000], [1.0, 1.10]),
                         [(12, 1.2), (10000, 1.2), (10000, 1.091)])

    def test_needed_files_dedup(self):
        self.assertEqual(needed_files([12, 12], [10000], [1.0, 1.0]),
                         [(12, 1.2), (10000, 1.2)])


if __name__ == '__main__':
    unittest.main()
