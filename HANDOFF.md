# Handoff: continuing TORRS on the home PC

Written 6 November 2026, when the project moved from the laptop (Ryzen 5 5600H) to the home PC (Ryzen 7 7700X, 32 GB). A new Claude Code session should read this file first, then `TORRS_Progress_and_Plan.docx` for the full done/left list and the weekly plan to 31 December 2026.

## Where things stand

- **Training run:** MA2C with background traffic, 63 of 100 episodes done (step 45360 of 72000). Stopped cleanly at `checkpoint-45360`. `Malta/signal_route/train_reward.csv` and `agent_reward.csv` hold episodes 1–63.
- **Training trend:** average wait fell from 79.0 s (episodes 1–10) to 28.4 s (episodes 51–63); completion rose from 79.6% to about 82.5%. These are training numbers with exploration on, not evaluation results.
- **Baselines** on the background-traffic network, 10 seeds each, in `Malta/signal_route/baseline_data/`: fixed_static (reference), actuated_static, fixed_rerouting and greedy are complete. actuated_rerouting has 9 of 10 seeds and no trip CSV; it was killed by a Windows Update restart on 28 October and needs re-running.
- **Statistics:** `stats_compare.py` writes `Malta/signal_route/stats/`. Comparisons against fixed_static exist for greedy and actuated_static only.
- **Pace on the laptop:** about 57 min per episode, plus about 70 min of TensorFlow start-up after every restart. The home PC should be faster; measure the first few episodes.

## Setting up the home PC

1. Install SUMO **1.26** (same version, so simulations match). Set `SUMO_HOME` to the install folder and add `%SUMO_HOME%\tools` to `PYTHONPATH` (traci and sumolib are imported from there).
2. Install Python 3.11, then `pip install -r requirements.txt`.
3. Copy the whole project folder from the laptop (it is a git checkout with the GitHub remote already set). It holds the files git ignores, which a fresh clone would lack:
   - `Malta/data/malta.net.xml` (the network; required), `malta_cache_v2.pkl`
   - `Malta/data/bg_*` (cached background demand per seed; rebuilt automatically if missing)
   - `Malta/signal_route/model/checkpoint` and `checkpoint-45360.*` (the training checkpoint; older `checkpoint-*` files there are not needed)
   - `Malta/signal_route/baseline_data/` and `greedy_data/` per-seed edge and summary data (needed for the spatial analysis), plus the uncommitted rerouting results and stats CSVs
   - `claude_context/` (gitignored): the SOI, the Claude memory notes from the laptop, and the exported laptop chat transcript
4. Pause Windows Update before long runs. A forced restart killed two runs on 28 October.

## Resuming training

```bash
python main.py > training_run_bg_entropy_100ep_r6.log 2>&1
```

It prints `Resumed from checkpoint at step 45360` and continues from episode 64. A checkpoint and both reward CSVs are saved at the end of every episode, so stop only right after a `Checkpoint saved at step N` line. Only one machine may run this training run; the laptop must not resume it.

## Next steps (week 2 of the plan onwards)

1. Finish the run (37 episodes), commit the CSVs.
2. Evaluate MA2C on the 10 test seeds; re-run the missing actuated_rerouting seed; re-run the routing ablation on the background network.
3. Paired statistics of MA2C against all five baselines (RQ1).
4. RQ3 compliance sweep: adapt `run_diff_com` to the Malta paths and sweep `CAV_rate`.
5. Peak-season demand (candidate `BG_PERIOD = 0.6`, needs an NSO source), then the RQ2 architecture runs.

## Working rules (from the laptop sessions)

- Keep `TORRS_MA2C_Complete.ipynb` cells in sync with the `.py` files they mirror whenever those change.
- Commit and push every change to `origin main`. Checkpoints and run logs stay gitignored.
- Run one heavy job at a time; a second SUMO/TensorFlow job slows both.
- The dissertation must follow the approved SOI (in `claude_context/`): RQ1 learning vs rule-based and shortest-path routing; RQ2 centralised vs fog vs hybrid under base and peak season; RQ3 minimum compliance. The draft's RQ numbering still differs from the SOI.
- Deadlines: experiments done by end of February 2027; submission May 2027.
