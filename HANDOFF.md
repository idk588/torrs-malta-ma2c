# Handoff: continuing TORRS on the home PC

Written 6 November 2026, when the project moved from the laptop (Ryzen 5 5600H) to the home PC (Ryzen 7 7700X, 32 GB); last updated 7 November 2026 (21:52), when training was paused at episode 85. A new Claude Code session should read this file first, then `TORRS_Progress_and_Plan.docx` for the full done/left list and the weekly plan to 31 December 2026.

## Where things stand

- **Training run:** MA2C with background traffic, 85 of 100 episodes done (step 61200 of 72000). Episodes 64–85 ran on the home PC. It was paused on request twice: at `checkpoint-52560` (6 November, 20:19) and at `checkpoint-61200` (7 November, 21:51, partway through episode 86, which is lost and will re-run). `Malta/signal_route/train_reward.csv` and `agent_reward.csv` hold episodes 1–85.
- **Training trend:** average wait fell from 79.0 s (episodes 1–10) to 28.4 s (episodes 51–63); completion rose from 79.6% to about 82.5%. Episodes 64–85 averaged 31.5 s and 82.0%, but vary a lot from episode to episode (12.6 s to 83.0 s). These are training numbers with exploration on, not evaluation results.
- **Baselines** on the background-traffic network, 10 seeds each, in `Malta/signal_route/baseline_data/`: fixed_static (reference), actuated_static, fixed_rerouting and greedy are complete. actuated_rerouting has 9 of 10 seeds and no trip CSV; it was killed by a Windows Update restart on 28 October and needs re-running.
- **Statistics:** `stats_compare.py` writes `Malta/signal_route/stats/`. Comparisons against fixed_static exist for greedy and actuated_static only.
- **Pace on the laptop:** about 57 min per episode, plus about 70 min of TensorFlow start-up after every restart.
- **Pace on the home PC:** about 28–31 min per episode (episodes 65–85). Games slow it badly: after Roblox started on 7 November, Windows paged most of the training process out of RAM, and episode 86 was only about 40% done after 40 min. Don't game while it trains. After a restart, start-up takes about 16 min and the first episode about 83 min. The training process commits about 43 GB of memory (about 50 GB while saving a checkpoint), more than the 32 GB of RAM, so Windows must grow the pagefile. Shortly after a reboot the pagefile is small, and the first resume attempt died this way; a fixed pagefile (e.g. 48–64 GB) avoids it.

## Setting up the home PC

1. Install SUMO **1.26** (same version, so simulations match). Set `SUMO_HOME` to the install folder and add `%SUMO_HOME%\tools` to `PYTHONPATH` (traci and sumolib are imported from there).
2. Install Python 3.11, then `pip install -r requirements.txt`.
3. Copy the whole project folder from the laptop (it is a git checkout with the GitHub remote already set). It holds the files git ignores, which a fresh clone would lack:
   - `Malta/data/malta.net.xml` (the network; required), `malta_cache_v2.pkl`
   - `Malta/data/bg_*` (cached background demand per seed; rebuilt automatically if missing)
   - `Malta/signal_route/model/checkpoint` and the newest `checkpoint-N.*` set it names (now `checkpoint-61200.*`; older `checkpoint-*` files there are not needed)
   - `Malta/signal_route/baseline_data/` and `greedy_data/` per-seed edge and summary data (needed for the spatial analysis), plus the uncommitted rerouting results and stats CSVs
   - `claude_context/` (gitignored): the SOI, the Claude memory notes from the laptop, and the exported laptop chat transcript
4. Pause Windows Update before long runs. A forced restart killed two runs on 28 October.

## Resuming training

```bash
python main.py > training_run_bg_entropy_100ep_r8.log 2>&1
```

It prints `Resumed from checkpoint at step 61200` and continues from episode 86. On the home PC the 15 remaining episodes take about 8 h including start-up, followed by the 10-seed evaluation. A checkpoint and both reward CSVs are saved at the end of every episode, so stop only right after a `Checkpoint saved at step N` line. Only one machine may run this training run; the laptop must not resume it.

On the home PC, a crash-recovery supervisor (`claude_context/supervisor/train_supervisor.ps1`, gitignored) relaunches training after a crash and cleans up any half-saved checkpoint; a Task Scheduler task, "TORRS training supervisor", starts it at logon. While `claude_context/supervisor/STOP` exists it launches nothing. To resume through it, delete the STOP file and run `Start-ScheduledTask 'TORRS training supervisor'` in PowerShell; it starts `main.py` as the next `_rN` log (r8). Remove the task once the run and its evaluation are finished.

## Next steps (week 2 of the plan onwards)

1. Finish the run (15 episodes left), commit the CSVs.
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
