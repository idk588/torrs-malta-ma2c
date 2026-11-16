import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parent.parent / "tools" / "rq1_manifest.py"


def run(root, cmd):
    return subprocess.run([sys.executable, str(TOOL), "--root", str(root), cmd],
                          capture_output=True, text=True)


def write(root, rel, text):
    p = Path(root) / "Malta" / "signal_route" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


class RQ1ManifestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write(self.root, "model/checkpoint-72000.index", "idx")
        write(self.root, "model/checkpoint-72000.meta", "meta")
        write(self.root, "model/checkpoint-48000.index", "older")
        write(self.root, "train_reward.csv", "a,b\n1,2\n")
        write(self.root, "eva_data/x.csv", "x")
        write(self.root, "baseline_data/fixed_static/y.csv", "y")
        write(self.root, "baseline_data/actuated_rerouting/z.csv", "z")
        write(self.root, "stats/s.csv", "s")
        write(self.root, "ablation_data_foo/a.csv", "a")
        bk = write(self.root, "_rq1_final_backup/b.txt", "b")
        os.chmod(bk, stat.S_IREAD)
        self.frozen = self.root / "Malta/signal_route/eva_data/x.csv"
        self.allowed = self.root / "Malta/signal_route/stats/s.csv"

    def tearDown(self):
        for p in self.root.rglob("*"):
            if p.is_file():
                os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
        self.tmp.cleanup()

    def test_record_check_ok_and_backup(self):
        self.assertEqual(run(self.root, "record").returncode, 0)
        q = self.root / "Malta/runs/_queue/rq1_manifest.csv"
        b = self.root / "Malta/signal_route/_rq1_final_backup/rq1_manifest.csv"
        self.assertTrue(q.exists())
        self.assertTrue(b.exists())
        self.assertFalse(os.access(b, os.W_OK))
        text = q.read_text()
        self.assertTrue(text.startswith("part,path,size,mtime,sha256"))
        self.assertNotIn("checkpoint-48000", text)
        self.assertNotIn("_rq1_final_backup/rq1_manifest.csv", text)
        self.assertIn("allowed,Malta/signal_route/ablation_data_foo/a.csv", text)
        self.assertEqual(run(self.root, "record").returncode, 0)  # re-record over read-only copy
        self.assertEqual(run(self.root, "check").returncode, 0)

    def test_allowed_change_ok(self):
        run(self.root, "record")
        self.allowed.write_text("changed content")
        r = run(self.root, "check")
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_frozen_change_fails(self):
        run(self.root, "record")
        self.frozen.write_text("changed content")
        r = run(self.root, "check")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("eva_data/x.csv", r.stdout)

    def test_frozen_delete_fails(self):
        run(self.root, "record")
        self.frozen.unlink()
        self.assertEqual(run(self.root, "check").returncode, 1)

    def test_missing_manifest(self):
        self.assertEqual(run(self.root, "check").returncode, 2)

    def test_old_manifest_moved_aside(self):
        q = self.root / "Malta/runs/_queue"
        q.mkdir(parents=True)
        (q / "rq1_manifest.csv").write_text("path,size,mtime,sha256\nold,1,2,3\n")
        run(self.root, "record")
        self.assertTrue((q / "rq1_manifest_2026-11-08.csv").exists())
        self.assertTrue((q / "rq1_manifest.csv").read_text().startswith("part,"))


if __name__ == "__main__":
    unittest.main()
