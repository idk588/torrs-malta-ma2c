# Handoff: continuing TORRS on the home PC

Written 6 November 2026, when the project moved from the laptop (Ryzen 5 5600H) to the home PC (Ryzen 7 7700X, 32 GB). Last updated 16 November 2026, after the RQ2 training queue was built (it has not been launched yet; see "RQ2 training queue" below). A new Claude Code session should read this file first, then `TORRS_Progress_and_Plan.docx` for the full list of what is done and left, and the weekly plan to 31 December 2026.

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

  The paired statistics are in `Malta/signal_route/stats/{per_seed,comparison}_vs_ma2c.csv` (commit 816ca94).
  - **Against fixed_static:** MA2C cuts the mean wait from 42.8 s to 23.2 s and has the lower wait on 9 of 10 seeds. The paired Wilcoxon test gives p = 0.049, but the paired t-test gives only p = 0.21, because seed 40000 congested badly (96.7 s wait, 38.5 congested minutes); seeds 60000 and 90000 also run 26–28 s.
  - **Against actuated_static:** no significant difference.
  - **Against fixed_rerouting:** fixed_rerouting has the shorter travel time (paired t p = 0.041).

  stats_compare excludes actuated_rerouting, which has no trip CSV.
- **Baselines** on the background-traffic network, 10 seeds each, in `Malta/signal_route/baseline_data/`. fixed_static (the reference), actuated_static, fixed_rerouting and greedy are complete. actuated_rerouting is not usable and has no trip CSV: a Windows Update restart killed it on 28 October, and seeds 10000–80000 kept no trip rows, 90000 is partial and 100000 is missing (spec §10). All ten seeds must be re-run (`python baselines.py actuated_rerouting`, about 2.5 h), and only while the RQ2 queue is not running. Its folder holds only the partial `trip_90000.xml` from the aborted run, which is not a result.
- **Statistics:** `stats_compare.py` writes to `Malta/signal_route/stats/`. The `*_vs_ma2c.csv` files cover MA2C against greedy, fixed_static, fixed_rerouting and actuated_static (816ca94). The fixed_rerouting arm reads `baseline_data/fixed_rerouting/Malta_signal_route_trip.csv` (f4a3bc5). The comparisons against fixed_static there (also uncommitted) cover greedy and actuated_static only.
- **RQ2 training queue: built, not launched.** The code, configs and queue runner for the cloud, fog and hybrid runs were finished on 16 November. The smoke queue and the author's gates come first; see "RQ2 training queue" below.
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

Training is finished, so `python main.py --rq1` (option `train_then_evaluate`, `total_step` 72000) skips training and only re-evaluates checkpoint-72000, overwriting `eva_data/`. Its `>>> train_then_evaluate complete.` line is printed even if the evaluation thread failed. Before trusting `eva_data/`, check the run log for:

- `Checkpoint loaded: checkpoint-72000`
- all ten `test N, avg reward` lines
- no `Traceback` or `Exception in thread`

Also check that the three CSVs are newly written. Evaluation log lines go to the run log and, since `init_log` now replaces the earlier log handler, to `Malta/signal_route/eva_log/<epoch>.log` (before the RQ2 changes they went to `log/<epoch>.log`).

Since the RQ2 changes, `main.py` refuses to run without `--rq1` (or `--run <name>`), and nothing writes into `Malta/signal_route/` without `--rq1`. `eva_data/` is in the frozen part of the RQ1 manifest, so don't re-run the evaluation while the RQ2 queue exists, and after a re-run record the manifest again (`python tools/rq1_manifest.py record`).

The crash-recovery supervisor (`claude_context/supervisor/train_supervisor.ps1`) is retired. `claude_context/supervisor/STOP` exists, and the "TORRS training supervisor" logon task was removed on 8 November. To use it for a future long run, delete STOP, edit the log prefix in the script, and run it detached.

## RQ2 training queue

Built between 12 and 16 November to the design in `docs/superpowers/specs/2026-11-08-rq2-training-queue-design.md` (the spec) and the plan `docs/superpowers/plans/2026-11-12-rq2-training-queue.md`. It trains the three RQ2 architectures one after the other, unattended: cloud (every agent sees every other agent), fog (every agent in the same NSO district) and hybrid (fog plus a 12-value district summary). It then evaluates each at 1.00, 1.06, 1.10, 1.13, 1.20 and 2.00× base demand; 1.10× is peak season. **It has not been launched yet:** no scheduled task is registered, and `Malta/runs/` holds only `_queue/`. Tasks 1–16 of the plan are done; Task 17 (smoke queue, author gates, launch) is next.

**Runs.** `config/queue.txt` lists `cloud_base`, `fog_base` and `hybrid_base`, in that order. Each run's config is `config/runs/<name>.ini`: the RQ1 config with `comm_arch` (cloud, fog or hybrid), `reward_share = mean`, `coop_gamma_sr = 0`, `seed_continue = true`, `eval_levels`, `norm_summary_wait = 60` and `save_optimizer = true`; everything else, including `total_step = 72000` and `seed = 12`, is as in RQ1. A full run trains on seeds 12–111, once each. `config/queue_smoke.txt` lists the smoke runs `_smoke_hybrid`, `_smoke_fog` and `_smoke_cloud` (`config/runs/_smoke_<arm>.ini`: 2 episodes of 720 s, test seeds 10000 and 20000, levels 1.00 and 1.10). The spec estimates about 40 h of training plus 3–4 h of evaluation per run, about 5.5–6 days for all three, and about 1 h per restart.

**Where runs live** (spec §3). Each run has its own store, `Malta/runs/<name>/`:
- `config.ini`, `zones_nso.csv` and `neighbours.sha256`: copies of what the run used, written at its first launch. A resume or evaluation refuses (exit 4) if the run's config or neighbour maps differ from them.
- `train_reward.csv` (with a `seed` column), `agent_reward.csv` and `episode_cost.csv` (per-episode time, peak memory and checkpoint size).
- `eva_data/f1.00/` … `eva_data/f2.00/`: the evaluation CSVs per demand level. `eva_data/EVAL_OK` is written when every check passes; a failed level is moved to `eva_data/_failed_<stamp>/` before its retry.
- `DONE` or `FAILED`: the runner's terminal marker.
- `model/` (checkpoints), `logs/` (`launch_rN.log`, `eval_rN.log`, exit files, code hashes, python and TF logs) and `sim/` (SUMO files): gitignored.

The runner never runs git. Commit a run's tracked files (the first four bullets) only within 10 minutes after a `Checkpoint saved` line or after the run's `DONE`. Smoke stores (`Malta/runs/_smoke_*/`) are gitignored.

**`main.py`.** `python main.py --run <name> [--option train|evaluate] [--levels 1.10,2.00]`. With `--run`, `--option` is `train` (the default; it resumes from the newest checkpoint) or `evaluate` (all of the store config's `eval_levels`, or only `--levels`). Without `--run`, `main.py` refuses (exit 2) unless `--rq1` is given. **RQ1 guard:** any path in `Malta/signal_route/` is refused (exit 3) without `--rq1`, and `--rq1` cannot be combined with `--run`. Only one `main.py --run` may run at a time (exit 6). In normal use only the queue runs `main.py`.

**The runner.** `tools/run_queue.ps1` (with its library `tools/queue_lib.ps1`) is started only by the scheduled task "TORRS training queue", never from a Claude shell. `tools/register_queue_task.ps1` registers the task (elevated PowerShell; it never starts it; `-DryRun` prints the definition, `-Unregister` removes it). The task runs at startup and at logon (2-minute delay), as kylea with S4U (whether or not anyone is logged on), and is restarted every 5 min, up to 99 times, if it fails. For each run in the queue file, the runner:
- skips a `DONE` run and stops the whole queue at a `FAILED` one (a reboot never skips past it);
- evaluates a run whose training is complete;
- otherwise runs Recovery (moves half-saved checkpoints aside and trims the CSVs to the last good checkpoint; it never deletes) and trains.

Before every launch it waits, with status `WAITING_<reason>` and a re-check every 5 min, until:
- STOP is absent;
- the machine is free: no other `main.py`, `baselines.py`, `ablation_routing.py`, `gen_traffic.py`, `build_zones.py`, randomTrips or Jupyter kernel; no SUMO, duarouter or netconvert outside the queue; TCP 27500 free;
- the commit limit minus the current commit is at least the arm's smoke-test peak + 8 GB (60 + 8 GB until the smoke numbers are entered in `state.json`);
- at least 60 GB is free on E: and 10 GB on C:;
- every traffic file the queue needs is valid.

A run is marked FAILED after 3 launches in a row without progress or 12 launches in all (STUCK), a non-finite reward (NAN), a code change since its first launch (CODE_CHANGED), a second failed evaluation attempt or 6 evaluation launches in all, counting those stopped by STOP or a reboot (EVAL_FAILED), a refusal by `main.py` that a relaunch cannot fix (REFUSED), or CSVs and checkpoints Recovery cannot reconcile (NEEDS_REVIEW). When the queue finishes or fails, the runner disables the task as its last action; unregister it afterwards (`tools/register_queue_task.ps1 -Unregister`, elevated).

**Watchdog and time limits.** A training launch is killed (and counted as a death) if its first `Checkpoint saved` line takes longer than `arms.<arm>.first_checkpoint_min` (default 240 min) or two checkpoint lines are further apart than `arms.<arm>.checkpoint_gap_min` (default 90 min; the STOP wait uses the same limit). Both keys are set by hand in `state.json`, per arm, after the smoke queue. In evaluation, each level up to 1.20 gets 3 × the f1.00 level time (at least 90 min); f2.00 gets 8 × (at least 360 min), because double demand with teleporting off may gridlock (a deliberate deviation from spec §8). Training counts as complete when the launch printed `>>> TRAINING COMPLETE step 72000` and every other check passes, even if python then exits non-zero (a TensorFlow crash at teardown; logged as a WARNING). `queue.log` also gets a `WARNING:` line, never a failure, when a run's first launch uses different code from the previous DONE run's first launch, or when a hybrid run's `neighbours.sha256` differs from the fog run's.

**After a FAILED run.** The queue stops at the run and the task is disabled.
1. Inspect: `Get-Content Malta\runs\<name>\FAILED` (the first line is the reason code), `Get-Content Malta\runs\_queue\queue.log -Tail 40`, and the run's newest `logs\launch_rN.log` or `logs\eval_rN.log`.
2. Fix the cause. STUCK, EVAL_FAILED: find out why the launches died or the levels failed. NEEDS_REVIEW: repair the CSVs or checkpoints by hand (Recovery's copies are in `_queue/recovery_<stamp>/`). NAN: the run probably needs rolling back to an earlier checkpoint. REFUSED: fix the config or neighbour-map mismatch. CODE_CHANGED: restore the code to the hashes in the run's `logs\launch_r1.code.txt`; a reset alone fails again.
3. `powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_queue.ps1 -ResetRun <name>`. It refuses (exit 1) while a runner holds the queue mutex or a queue python is alive. Otherwise it resets the run's launch counters (`launches`, `no_progress`) and evaluation counters (`eval_attempts`, `eval_launches`, `eval_levels_next`) in `state.json`, keeps the old values in `runs.<name>.resets`, moves `FAILED` to `logs\FAILED_<stamp>.txt` (nothing is deleted), logs to `queue.log`, prints what it changed and exits 0. It launches nothing.
4. `Enable-ScheduledTask 'TORRS training queue'` (the runner disabled it at the FAILED stop; use an elevated PowerShell if access is denied), then `Start-ScheduledTask 'TORRS training queue'`.

**Start, stop and resume.**
- **Start:** `Start-ScheduledTask 'TORRS training queue'`, only after Task 17's gates. `status.txt` then moves through the start-up checks to `TRAINING` for `cloud_base`.
- **Stop:** only through STOP, never by shutting down, restarting or logging off the PC mid-run (that kills python between checkpoints and costs a restart and Recovery). Create `Malta/runs/_queue/STOP` (`New-Item Malta\runs\_queue\STOP -ItemType File`), and shut down only once `status.txt` shows `STOPPED`. During training the runner waits for the next `Checkpoint saved` line (at most the checkpoint-gap limit, default 90 min), waits 15 s, kills python and its SUMO, runs Recovery, writes `STOPPED` and exits. During evaluation it stops at once, and the attempt does not count. Deleting STOP while the runner waits cancels the stop. The runner never deletes STOP; while it exists, a reboot only starts a runner that exits `STOPPED`.
- **Resume:** delete STOP, then `Start-ScheduledTask 'TORRS training queue'`. Training resumes from the last good checkpoint.

**Status** (all in `Malta/runs/_queue/`, gitignored):
- `status.txt`: one line, rewritten every 60 s: `<STATE> <detail> | heartbeat <time> | runner pid <pid>`. Typical states are `TRAINING`, `EVALUATING`, `RECOVERY`, `WAITING_<reason>`, `STOPPING` and `STOPPED`. The runner ends at `QUEUE_DONE`, `FAILED`, `STUCK` or `ENV_ERROR`; `ERROR` is logged and retried after 60 s. A stale heartbeat means the runner is not running.
- `queue.log`: one timestamped line per event.
- `state.json`: per-run launch counters, the launch in flight, measured times and peak commit, `-ResetRun` history (`resets`), and the per-arm thresholds set by hand after the smoke queue (`arms.<arm>.peak_commit_gb`, `first_checkpoint_min`, `checkpoint_gap_min`).
- `quarantine_<stamp>/` and `recovery_<stamp>/`: files Recovery moved aside.
- The live training log is `Malta/runs/<name>/logs/launch_rN.log` (evaluation: `eval_rN.log`).

To read it: `Get-Content Malta\runs\_queue\status.txt`, then `Get-Content Malta\runs\_queue\queue.log -Tail 20`.

**While the queue exists** (from registering the real task until it is unregistered):
- **Code freeze:** do not edit `main.py`, `utils.py`, `run_config.py`, `run_files.py`, `agents/`, `envs/` or `Malta/build_file.py` in the main working tree. The runner hashes them at every launch, and a relaunch whose hashes differ from the run's first launch fails the run (CODE_CHANGED). That includes a fix merged while the queue is stopped, if a run is part-trained or trained but not yet evaluated. Make any fix in a git worktree and merge it only while the queue is stopped. No git stash, checkout, restore, reset or pull in the main tree.
- Don't run `main.py`, the notebook or any other heavy job alongside it. A test or check uses TraCI port offset 20 or higher: port 0 (TraCI 27500) and `Malta/data/exp_0.*` belong to the queue.
- No gaming, and don't open run CSVs in Excel while a run trains.
- Stop the queue only through `Malta/runs/_queue/STOP` (see "Start, stop and resume"); never shut the PC down mid-run.
- Windows Fast Startup stays OFF (otherwise "Shut down" is a hibernate, and the task's startup trigger does not fire at the next power-on), and Sleep is Never on AC power (the runner keeps the PC awake only while a job runs, not in `WAITING_<reason>` states).
- `python tools/rq1_manifest.py check` confirms the frozen RQ1 files are unchanged (exit 0).

**Tools** (Python, run from the project root; mirrored in the notebook, sections 17–21):
- `tools/gen_traffic.py`: pre-generates and validates the background traffic files, for example `--seeds test --factors 1.00,1.06,1.10,1.13,1.20,2.00 --check-only`. All of them exist: 110 base files (`p1.2`: training seeds 12–111 and the 10 test seeds) and the 50 sweep files (the test seeds at `p1.132`, `p1.091`, `p1.062`, `p1.0` and `p0.6`).
- `tools/build_zones.py`: builds `Malta/data/zones_nso.csv` (each agent's nearest of the 68 councils in `Malta/data/councils_nso.csv`, and its district) and prints the per-district and per-council counts for the author's approval.
- `tools/check_env.py --arm local|fog|cloud|hybrid --port 20`: SUMO-level checks per arm, without TensorFlow; output in `Malta/runs/_queue/check_env_<arm>.txt`. All four arms passed on 13 November (commit 1d9ff6b).
- `tools/compare_local.py --base-worktree <dir> --port 21`: checks that the RQ1 (`local`) env gives the same states and rewards as the pre-change commit bc7115c.
- `tools/rq1_manifest.py record|check`: the manifest of `Malta/signal_route/`, in `Malta/runs/_queue/rq1_manifest.csv` with a read-only copy in `_rq1_final_backup/`. Frozen: checkpoint-72000, the reward CSVs, `eva_data/`, `greedy_data/`, three baseline folders and `_rq1_final_backup/`. Allowed to change: `stats/`, `ablation_data*/` and `baseline_data/actuated_rerouting/`.

Tests: `python -m unittest discover -s tests -t . -v` (SUMO tests use ports 20 and up), and the runner tests `powershell -NoProfile -ExecutionPolicy Bypass -File tests/test_queue_lib.ps1` and `tests/test_run_queue.ps1`. `tools/run_queue.ps1 -DryRun` prints each run's decision and the preflight without launching training; it does start the environment check, so don't use it while a job runs.

## Next steps (week 2 of the plan onwards)

1. **Launch the RQ2 queue (Task 17 of the plan).**
   - Re-record the RQ1 manifest (`python tools/rq1_manifest.py record`, then `check`).
   - Run the smoke queue (`config/queue_smoke.txt`) through a temporary task "TORRS smoke queue", with the fault, second-copy and STOP tests. Record the smoke numbers in `state.json`, then unregister the smoke task and move the smoke stores to the scratchpad.
   - Author gates: a fixed pagefile on C: (initial = maximum ≥ 65,536 MB, then a reboot; Claude checks read-only for a commit limit of at least 95 GB); Windows Update paused; Windows Fast Startup OFF (Control Panel > Power Options > Choose what the power buttons do), so the startup trigger fires after a power-on; Sleep = Never on AC power, so the PC does not sleep while the runner waits; optionally, BIOS "restore on AC power loss"; the per-arm limits (`peak_commit_gb`, `first_checkpoint_min`, `checkpoint_gap_min`) entered in `state.json` from the smoke numbers, scaled to 3600 s episodes; and approval of the per-district counts from `tools/build_zones.py`. Show the author route agent rc3 (its up-edge junction is in Northern Harbour by about 70 m, while most of its corridor is in Western) and the agents near a district boundary.
   - Register the real task (`tools/register_queue_task.ps1 -TaskName 'TORRS training queue' -QueueFile config/queue.txt`) and run `Start-ScheduledTask 'TORRS training queue'`. About 1.5 h in, at the first checkpoint, check: `rq1_manifest.py check` exits 0; only `Malta/runs/cloud_base/`, `_queue/` and the `exp_0.*` files changed; `.meta` and `.data` are each ≤ 1.5 GB; `episode_cost.csv` has a valid row.
2. Decide whether to commit or regenerate the uncommitted fixed_static comparison (`stats/*_vs_fixed_static.csv`, from 28 October).
3. Optional RQ1 jobs, before the queue starts or after it finishes, never during it. Re-run actuated_rerouting (`python baselines.py actuated_rerouting`, about 2.5 h). It needs all ten seeds: 10000–80000 kept no trip rows, 90000 is partial and 100000 is missing. Re-run the routing ablation on the background network (about 1.5 h). Then re-run `python stats_compare.py` for MA2C against all five baselines (RQ1). Look into why seed 40000 congests under MA2C.
4. RQ3 compliance sweep: adapt `run_diff_com` to the Malta paths and sweep `CAV_rate`.
5. After the queue: peak-season baselines at 1.10×, RQ2 paired statistics, and the thesis text (base demand described as a scaled, typical term-time weekday peak hour, NSO NR 182/2026 cited for the 1.10× peak factor, and the protocol changes against RQ1; spec §14).

## Working rules (from the laptop sessions)

- Keep `TORRS_MA2C_Complete.ipynb` cells in sync with the `.py` files they mirror whenever those change.
- Commit and push every change to `origin main`. Checkpoints and run logs stay gitignored.
- Run one heavy job at a time; a second SUMO/TensorFlow job slows both.
- While the RQ2 queue exists, keep to its code freeze and port rules (see "RQ2 training queue").
- The dissertation must follow the approved SOI (in `claude_context/`): RQ1 learning vs rule-based and shortest-path routing; RQ2 centralised vs fog vs hybrid under base and peak season; RQ3 minimum compliance. The draft's RQ numbering still differs from the SOI.
- Deadlines: experiments done by end of February 2027; submission May 2027.
