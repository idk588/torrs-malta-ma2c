# Handoff: continuing TORRS on the home PC

Written 6 November 2026, when the project moved from the laptop (Ryzen 5 5600H) to the home PC (Ryzen 7 7700X, 32 GB). Last updated 8 November 2026 (08:10), after the 100-episode background-traffic run and its evaluation finished. A new Claude Code session should read this file first, then `TORRS_Progress_and_Plan.docx` for the full list of what is done and left, and the weekly plan to 31 December 2026.

## Where things stand

- **Training run: complete.** MA2C with background traffic, 100 of 100 episodes (step 72000), finished on 8 November at 07:21. Episodes 64–100 ran on the home PC; it was paused on request at `checkpoint-52560` and `checkpoint-61200` and resumed from both. `Malta/signal_route/train_reward.csv` and `agent_reward.csv` hold episodes 1–100 (commit 517dc1b). The final model is `Malta/signal_route/model/checkpoint-72000.*`.
- **Training trend:** the average wait was 79.0 s in episodes 1–10, 28.4 s in 51–63, 31.5 s in 64–85 and 27.7 s in 86–100. Completion went from 79.6% to 82.5%, 82.0% and 82.2% over the same blocks. Waits vary a lot from episode to episode, from 12 s to 83 s. These are training numbers with exploration on.
- **MA2C evaluation: complete.** It ran on 10 seeds (10000–100000) with checkpoint-72000, in the same process as training, between 07:25 and 08:02 on 8 November (commit aec9800). `Malta/signal_route/eva_data/Malta_signal_route_{trip,traffic,control}.csv` now hold this result, replacing the 19 September fixed-route, no-background result. The per-seed `edgedata_<seed>.xml` and `summary_<seed>.xml` files are in the same folder but are gitignored. Means over the 10 seeds:

  | Arm | Completion | Travel | Wait | Halted veh-h | Congested min |
  |---|---|---|---|---|---|
  | MA2C | 82.4% | 539.9 s | 23.2 s | 26.0 | 5.9 |
  | fixed_static | 81.1% | 555.5 s | 42.8 s | 48.4 | 11.5 |
  | actuated_static | 82.2% | 533.8 s | 20.6 s | 24.1 | 7.7 |
  | fixed_rerouting | 83.3% | 524.2 s | 9.8 s | 11.9 | 0.2 |
  | greedy | 54.1% | 982.2 s | 539.7 s | 537.5 | 56.1 |

  These figures were computed in memory with `stats_compare.py`'s functions, and the stats CSVs have not been written yet.
  - **Against fixed_static:** MA2C cuts the mean wait from 42.8 s to 23.2 s. The paired Wilcoxon test gives p = 0.049, but the paired t-test gives only p = 0.21, because seed 40000 congested badly (96.7 s wait, 38.5 congested minutes); seeds 60000 and 90000 also run 26–28 s.
  - **Against actuated_static:** no significant difference.
  - **Against fixed_rerouting:** fixed_rerouting has the shorter travel time (paired t p = 0.041).

  stats_compare excludes actuated_rerouting, which has no trip CSV.
- **Baselines** on the background-traffic network, 10 seeds each, in `Malta/signal_route/baseline_data/`. fixed_static (the reference), actuated_static, fixed_rerouting and greedy are complete. actuated_rerouting has 9 of 10 seeds and no trip CSV, because a Windows Update restart killed it on 28 October; it needs re-running. Its folder holds only `trip_90000.xml` from that aborted run, which is not a result.
- **Statistics:** `stats_compare.py` writes to `Malta/signal_route/stats/`. The comparisons against fixed_static there (uncommitted) cover greedy and actuated_static only. The MA2C comparison (`python stats_compare.py`, reference ma2c) has not been run yet.
- **Pace on the laptop:** about 57 min per training episode, plus about 70 min of TensorFlow start-up after every restart.
- **Pace on the home PC:**
  - Training episodes: about 23 min each in episodes 87–100, and 28–31 min earlier.
  - After a restart: start-up takes about 16 min and the first episode 80–90 min.
  - Evaluation: about 3 min per seed, plus about 6 min to build and restore the model.
  - Games slow it badly: with Roblox running on 7 November, Windows paged most of the training process out of RAM. Don't game while it trains.
  - Memory: the training process commits about 43 GB (about 50 GB while saving a checkpoint), more than the 32 GB of RAM, so Windows must grow the pagefile. Shortly after a reboot the pagefile is small, and one resume attempt died that way. A fixed pagefile (e.g. 48–64 GB) avoids it.

## Setting up the home PC

1. Install SUMO **1.26** (same version, so simulations match). Set `SUMO_HOME` to the install folder and add `%SUMO_HOME%\tools` to `PYTHONPATH` (traci and sumolib are imported from there).
2. Install Python 3.11, then `pip install -r requirements.txt`.
3. Copy the whole project folder from the laptop (it is a git checkout with the GitHub remote already set). It holds the files git ignores, which a fresh clone would lack:
   - `Malta/data/malta.net.xml` (the network; required) and `malta_cache_v2.pkl`
   - `Malta/data/bg_*` (cached background demand per seed; rebuilt automatically if missing)
   - `Malta/signal_route/model/checkpoint` and the newest `checkpoint-N.*` set it names (now `checkpoint-72000.*`, the final model; older `checkpoint-*` files there are not needed)
   - `Malta/signal_route/eva_data/`, `baseline_data/` and `greedy_data/` per-seed edge and summary XML (needed for the congestion metrics and the spatial analysis), plus the uncommitted rerouting results and stats CSVs
   - `claude_context/` (gitignored): the SOI, the Claude memory notes from the laptop, the exported laptop chat transcript, and the crash-recovery supervisor
4. Pause Windows Update before long runs. A forced restart killed two runs on 28 October.

## Re-running the evaluation

Training is finished, so `python main.py` (option `train_then_evaluate`, `total_step` 72000) skips training and only re-evaluates checkpoint-72000, overwriting `eva_data/`. Its `>>> train_then_evaluate complete.` line is printed even if the evaluation thread failed. Before trusting `eva_data/`, check the run log for:

- `Checkpoint loaded: checkpoint-72000`
- all ten `test N, avg reward` lines
- no `Traceback` or `Exception in thread`

Also check that the three CSVs are newly written. Evaluation log lines go to the run log and `Malta/signal_route/log/<epoch>.log`, not to `eva_log/`.

The crash-recovery supervisor (`claude_context/supervisor/train_supervisor.ps1`) is retired. `claude_context/supervisor/STOP` exists, and the "TORRS training supervisor" logon task was removed on 8 November. To use it for a future long run, delete STOP, edit the log prefix in the script, and run it detached.

## Next steps (week 2 of the plan onwards)

1. Run `python stats_compare.py` (reference ma2c) and commit the stats, together with the uncommitted fixed_static comparison and `baseline_data/fixed_rerouting/Malta_signal_route_trip.csv`.
2. Re-run the missing actuated_rerouting seed so all five baselines have 10 seeds, then the paired statistics of MA2C against all five (RQ1). Look into why seed 40000 congests under MA2C.
3. Re-run the routing ablation on the background network.
4. RQ3 compliance sweep: adapt `run_diff_com` to the Malta paths and sweep `CAV_rate`.
5. Peak-season demand (candidate `BG_PERIOD = 0.6`, needs an NSO source), then the RQ2 architecture runs.

## Working rules (from the laptop sessions)

- Keep `TORRS_MA2C_Complete.ipynb` cells in sync with the `.py` files they mirror whenever those change.
- Commit and push every change to `origin main`. Checkpoints and run logs stay gitignored.
- Run one heavy job at a time; a second SUMO/TensorFlow job slows both.
- The dissertation must follow the approved SOI (in `claude_context/`): RQ1 learning vs rule-based and shortest-path routing; RQ2 centralised vs fog vs hybrid under base and peak season; RQ3 minimum compliance. The draft's RQ numbering still differs from the SOI.
- Deadlines: experiments done by end of February 2027; submission May 2027.
