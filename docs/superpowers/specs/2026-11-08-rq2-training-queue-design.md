# RQ2 training queue: design

Date: 8 November 2026. Status: agreed in conversation section by section, then revised after a three-reviewer check against the code (74 findings). Awaiting the author's review of this written spec.

## 1. Goal

Train the three RQ2 information architectures as soon as possible, unattended, on the home PC (Ryzen 7 7700X, 31 GB RAM). Each run is kept in its own organised folder, and the finished RQ1 run is never touched.

RQ2, as the approved SOI words it, compares centralised cloud, decentralised fog (Maltese zones) and hybrid architectures. The comparison covers scalability, stability and performance under base-season and peak-season demand, with compute cost and training stability among the metrics.

## 2. Decisions (made by the author on 8 November 2026)

| Topic | Decision |
|---|---|
| Arms | Cloud, fog and hybrid only. No "local" arm. The RQ1 MA2C run is not an RQ2 arm. |
| Peak season | No peak training runs. Each base-trained arm is evaluated at demand levels 1.00, 1.06, 1.10, 1.13, 1.20 and 2.00× base (Section 8). |
| Peak level | 1.10× is "peak season". It is cited to NSO *Tourism Intensity: 2025* (NR 182/2026) and cross-checked against Eurostat nrg_cb_oilm road-fuel deliveries. 1.06–1.13× is the NSO-supported range, 1.20× is an upper sensitivity bound, and 2.00× is a stress test only. |
| Cloud | Global-state MA2C. Every agent has every other agent as a neighbour (state and shared reward), and each junction keeps its own policy. |
| Fog zones | The six NSO districts: Southern Harbour, Northern Harbour, South Eastern, Western, Northern, and Gozo & Comino. Each agent is assigned through its nearest local council. |
| Fog information | Full zone state. Every other agent in the same district is a neighbour. |
| Hybrid | Fog, plus a 12-value global summary: a queue measure and a waiting-time measure for each district. |
| Protocol | Training seeds carry on across restarts. Optimiser state is saved. Neighbour rewards are averaged (with the same-scale rule in Section 6). Per-episode cost is logged. |
| Between runs | The next run starts automatically after the previous run's checks pass. A failed run stops the queue. |
| Reboots | A scheduled task restarts the queue runner after a reboot. It is removed when the queue finishes. |
| Build approach | One codebase, one config file per run, a `--run` CLI and a queue runner. |

Out of scope: peak-season baselines, RQ2 paired statistics, the RQ3 compliance sweep and thesis text changes (Section 14).

## 3. Folder layout

```
Malta/
├── signal_route/                 RQ1 run; from Section 10 step 6 on, nothing writes here
│   └── _rq1_final_backup/        read-only copy of checkpoint-72000 (+ index), RQ1 CSVs and the
│                                 eva_data / baseline_data / greedy_data folders        [ignored]
├── runs/
│   ├── _queue/                   status.txt, queue.log, state.json, STOP, quarantine_*,
│   │                             rq1_manifest.csv                                      [ignored]
│   ├── cloud_base/
│   │   ├── config.ini            copy of the config the run used                      [git]
│   │   ├── zones_nso.csv         copy of the zone file the run used                   [git]
│   │   ├── neighbours.sha256     hash of the four neighbour maps                       [git]
│   │   ├── train_reward.csv      per-episode results, plus a seed column               [git]
│   │   ├── agent_reward.csv      per-agent results                                    [git]
│   │   ├── episode_cost.csv      per-episode time and memory                          [git]
│   │   ├── eva_data/
│   │   │   ├── f1.00/ … f2.00/   evaluation CSVs per demand level                     [git]
│   │   │   │                     (per-seed edgedata/summary XML and sumo_*.log: ignored)
│   │   │   ├── _failed_<stamp>/  levels moved aside before a retry                    [ignored]
│   │   │   └── EVAL_OK           written by the runner when every check passes        [git]
│   │   ├── model/                checkpoints                                          [ignored]
│   │   ├── logs/                 launch_rN.log, eval_rN.log, exit files, code hashes,
│   │   │                         python + TF logs                                     [ignored]
│   │   ├── sim/                  SUMO trip/log files written during training          [ignored]
│   │   └── DONE | FAILED         terminal marker written by the runner                [git]
│   ├── fog_base/
│   └── hybrid_base/
└── data/                         shared: network, traffic files, zones_nso.csv [git]
config/
├── config_ma2c_Malta.ini         unchanged (RQ1)
├── runs/cloud_base.ini, fog_base.ini, hybrid_base.ini                                [git]
├── runs/_smoke_<arm>.ini         smoke-test configs (Section 12)                      [git]
├── queue.txt                     one run name per line; blank and # lines ignored     [git]
└── queue_smoke.txt               the smoke queue                                      [git]
tools/
├── build_zones.py                builds Malta/data/zones_nso.csv                      [git]
├── gen_traffic.py                pre-generates and validates background traffic files [git]
├── check_env.py                  environment-level checks for each arm (Section 12)   [git]
└── run_queue.ps1                 queue runner                                         [git]
```

**Git markers.** [git] = tracked. Claude commits these files at check-in, and only within 10 minutes after a `Checkpoint saved` line or after a run's DONE. The runner writes the DONE, FAILED and EVAL_OK markers but never runs git. Run names starting with `_` are smoke runs and never appear in `config/queue.txt`.

**`.gitignore` additions:**
- `Malta/runs/_queue/`
- `Malta/runs/*/model/`
- `Malta/runs/*/logs/`
- `Malta/runs/*/sim/`
- `Malta/runs/*/eva_data/_failed_*/`
- `Malta/runs/_smoke_*/`

The per-seed XML and `sumo_*.log` files are already ignored by existing patterns.

**Shared scratch files.** Each thread uses its own `Malta/data/exp_<thread>.rou.xml`, `.sumocfg` and `.add.xml`; `gen_rou_file` writes `exp_<thread>.add.xml` instead of the shared `exp.add.xml`. Thread and port 0 (TraCI 27500) are reserved for the queue. Every test or debug run started while the queue exists uses port 20 or higher.

## 4. `main.py` CLI and guards

- **Usage:** `python main.py [--run <name>] [--option train|evaluate|train_then_evaluate|greedy|get_parameter] [--levels 1.10,2.00] [--rq1]`. Parsing uses `parse_known_args()` so the notebook cell still runs.
- **With `--run`:**
  - the name must match `^[a-z][a-z0-9_]*$`, or `^_smoke_[a-z]+$` for smoke runs;
  - `--option` defaults to `train` and must be `train` or `evaluate`;
  - `--rq1` is an error;
  - the config is `config/runs/<name>.ini` and the store is `Malta/runs/<name>/`.
- **Without `--run`:** `--option` defaults to `train_then_evaluate` (as today), the config is `config/config_ma2c_Malta.ini` and the store is `Malta/signal_route`. Every option is refused unless `--rq1` is given.
- **RQ1 guard:** `main.py` and the runner refuse, unless `--rq1` is given, any store or `model.load` directory whose `os.path.normcase(os.path.realpath(...))` equals or lies under `normcase(realpath("Malta/signal_route"))`.
- **One instance:** with `--run`, `main.py` refuses to start while another `main.py --run` python is alive.
- **Config files.** First launch means `<store>/config.ini` does not exist yet. The run's `.ini` and `Malta/data/zones_nso.csv` are copied into the store, and `neighbours.sha256` is written.
  - Resume and evaluate parse `config/runs/<name>.ini` and `<store>/config.ini` with configparser. They refuse if any section, key or value differs; whitespace, comments and line endings are ignored.
  - Both also refuse if the store's neighbour-map hash differs from the maps they build.
  - Evaluate takes every setting from `<store>/config.ini`.
- **Paths:** every path derives from the store.
  - `train()` no longer calls `init_dir('Malta')` or copies the config into `Malta/data/`.
  - `init_env(config, port, output_path)` passes `output_path = <store>/sim/` to MaltaEnv.
  - Train creates `model/`, `logs/` and `sim/`; evaluate creates `eva_data/` and `logs/`. No `log/`, `data/` or `eva_log/` folder is created under `Malta/runs/<run>/`.
  - The Trainer receives `store_dir` and writes `<store>/train_reward.csv`, `agent_reward.csv`, `episode_cost.csv` and `<store>/model/`. Every `output_path.replace('data', problem)` is removed.
  - `init_log` uses `logging.basicConfig(..., force=True)` and writes to `<store>/logs/`, and so does the TF FileWriter.
- **Training end:** `train()` skips the final `model.save` when the per-episode loop has already saved `total_step`. Its last line is `>>> TRAINING COMPLETE step <N>`, then it exits 0. With `--run`, a checkpoint-save error is raised, not logged as a warning.
- **New config keys.** Every key is read with a fallback equal to its RQ1 behaviour, so `config_ma2c_Malta.ini` needs no edits.

| Section | Key | Default (RQ1) | RQ2 runs |
|---|---|---|---|
| ENV_CONFIG | `comm_arch` | `local` | `cloud`, `fog`, `hybrid` |
| ENV_CONFIG | `reward_share` | `sum` | `mean` |
| ENV_CONFIG | `demand_factor` | `1.0` | `1.0` (training) |
| ENV_CONFIG | `seed_continue` | `false` | `true` |
| ENV_CONFIG | `eval_levels` | `1.0` | `1.00,1.06,1.10,1.13,1.20,2.00` |
| ENV_CONFIG | `norm_summary_wait` | (unused) | `60` (hybrid only) |
| ENV_CONFIG | `coop_gamma_sr` | `0.3` | `0` (Section 6) |
| MODEL_CONFIG | `save_optimizer` | `false` | `true` |

All other keys in the run configs are copied unchanged from `config_ma2c_Malta.ini`: learning-rate and entropy schedules, `coop_gamma_ss` 0.3, `coop_gamma_rs` 0.1, `coop_gamma_rr` 0, norms, clips, `total_step = 72000`, `seed = 12` and `test_seeds`.

## 5. The three architectures

All three arms use the existing MA2C model (`FPLstmACPolicySR`). They differ in the four neighbour maps that `_init_map` builds (`relevant_ss`, `relevant_sr`, `relevant_rs`, `relevant_rr`), and for hybrid only, in the 12 summary values. The maps decide each agent's state width (neighbour waves or arrivals), its fingerprints (neighbours' policies) and its shared reward. Agents never see a neighbour's wait or occupancy, as in RQ1.

### Zones (`tools/build_zones.py` → `Malta/data/zones_nso.csv`, UTF-8)

- **Columns:** `agent_id, type (signal|route), x, y, lon, lat, island, council, district`.
- **Points.** A signal's point is the junction with the agent's id. A route agent's point is the from-junction of its `up_edge`.
- **Reading the network.** The script iterparses only `<location>`, `<junction>` and `<edge>` from `malta.net.xml`; the 248 MB network is not loaded with sumolib. It converts x/y to lon/lat with the network's own projection (UTM 33N, `projParameter` and `netOffset`).
- **Assignment.**
  - The island is decided first: Gozo & Comino if lat > 36.005, otherwise Malta.
  - The nearest council is then chosen among that island's councils only. The table has 68 rows keyed by (council, district), with lat/lon, so the two Rabats and two Żebbuġs stay distinct.
  - Comino belongs to Għajnsielem.
- **The script exits non-zero if:**
  - any agent is more than 5 km from its council centre;
  - any agent lies outside lon 14.15–14.60, lat 35.78–36.10;
  - the signal ids differ from the env's `node_names`, or the route ids from `rc_names`.
- **Output for review.** It prints counts per district and per council, and lists corridors whose edges cross districts. A route agent belongs only to its `up_edge` district. The author approves the per-district counts before launch.

### Maps

- **local** (default): built as today.
- **fog:** each agent's neighbours are every other agent, signal or route, in the same district. These replace the local maps, so an adjacent junction in another district is not a neighbour.
- **cloud:** each agent's neighbours are every other agent in the network, including across Malta and Gozo. The thesis notes that the two islands have no road link.
- **hybrid:** the fog maps, plus the summary below.
- **Format and order.**
  - `relevant_ss` and `relevant_rr` are lists. `relevant_sr` and `relevant_rs` are dicts `{id: 1.0}`. The existing `tl_set` cleaning is skipped for fog, cloud and hybrid.
  - Every neighbour list is ordered by global agent index (sorted `node_names`, then route ids in order). Maps are never built by iterating a set.
  - Integer index arrays for state, fingerprint and reward gathering are built at the end of `_init_agents`, so there is no `node_names.index()` per step.

### Hybrid summary (12 values)

The summary is computed in `_get_state` after `_clip_state()`, from signal agents only, in the fixed district order Southern Harbour, Northern Harbour, South Eastern, Western, Northern, Gozo & Comino, as `[q_1..q_6, w_1..w_6]`:

- `q_d` (queue): the mean over district d's signal agents of `mean(node.wait_state)`. That is the halting-based value, already normalised and clipped, so it is not normalised again.
- `w_d` (waiting time): the mean over district d's signal agents of `clip(node.vehwait / max(1, n_incoming_lanes) / norm_summary_wait, 0, clip_wait)`. This uses the waiting time measured for the reward in the previous step; it is 0 at reset.
- A district with no signal agents gives 0 for both.
- **Placement:** the 12 values go at the end of each agent's `n_o` segment, before the fingerprints. For a signal that is after its `relevant_sr` arrival states; for a route agent, after its `relevant_rs` wave states. `n_o_ls` rises by 12 for every agent. The policy splits observations by position, so placement is tested (Section 12).

## 6. Rewards

Under `reward_share = mean`, each agent's reward is built from raw per-agent rewards, before normalisation:

- **Signal agent:** `r_i' = r_i + γ_ss · mean_{j∈SS(i)} r_j + γ_sr · mean_{j∈SR(i)} r_j`
- **Route agent:** `r_i' = r_i + γ_rr · mean_{j∈RR(i)} r_j + γ_rs · mean_{j∈RS(i)} r_j`

A mean over an empty set contributes 0. Relevance weights are not applied under `mean`. The existing normalisation and `clip_reward` clip then apply as today.

**Same-scale rule.** Raw route rewards (1000 × arrivals, about +548 per step on average) are about 500 times larger than raw signal rewards (about −1.2 per step). If signals shared route rewards, every signal's reward would become nearly the same exogenous number, and RQ2 would measure a scale artefact.

- RQ2 runs therefore set `coop_gamma_sr = 0`. Signals still see route agents in their state, but do not share their reward.
- `γ_ss = 0.3`, `γ_rs = 0.1` and `γ_rr = 0` are unchanged.
- In the cloud arm this means each signal shares the network-wide mean of the other signals' rewards.

`tools/check_env.py` enforces the rule: over 100 random-action steps, the mean |shared term| must not exceed the mean |own reward| for any agent type.

## 7. Training protocol, files and cost logging

- **Length and schedules.** 100 episodes of 720 steps (`total_step = 72000`) with RQ1's learning-rate and entropy schedules.
- **Seeds (`seed_continue = true`).** After `model.load` and before `trainer.run`: `env.seed = seed + resumed_step // int(env.T)`. A full run therefore uses training seeds 12–111, once each.
  - `train_reward.csv` gains a `seed` column.
  - Action sampling is already seeded per episode, because `output_flows` calls `np.random.seed(seed)` on every reset.
- **Optimiser state (`save_optimizer = true`).** When `total_step > 0`, the MA2C Saver is created after the optimiser, so checkpoints include the Adam slots, and resumes are exact up to floating-point reduction order.
  - Evaluation (`total_step = 0`) builds a policy-only Saver, which restores its subset of a larger checkpoint by name.
  - If `save_optimizer` is true and the newest checkpoint lacks Adam variables, training stops with a clear error. It does not fall back.
- **Files per episode, in this order:**
  1. `train_reward.csv` and `agent_reward.csv`;
  2. the checkpoint save;
  3. a row appended to `episode_cost.csv` (header only when the file is new);
  4. the log line `Checkpoint saved at step <N> -> <store>/model/`.

  CSVs are written to `<file>.tmp` and then `os.replace`d. On `PermissionError` (for example a file open in Excel) the write is retried 6 times over 60 s, then raised. On resume, `episode_cost.csv` is appended to, not rewritten.
- **`episode_cost.csv` columns:**

  ```
  episode, step, seed, wall_s, rollout_s, update_s, save_s, peak_private_gb, peak_working_set_gb, ckpt_data_mb, ckpt_meta_mb, lost
  ```

  `lost` is 0 normally, and 1 on the placeholder row that Recovery appends when a cost row was lost.

  A daemon thread samples the python process's private bytes and working set every 1 s (`psutil` if installed, otherwise `GetProcessMemoryInfo` through ctypes) and keeps the per-episode maximum. SUMO is not included.
- **Traffic files.**
  - `gen_background_file` and `tools/gen_traffic.py` write into `Malta/data/_tmp_<tag>_<pid>/`, validate, then `os.replace` into place. No code writes a final `bg_*.rou.xml` directly.
  - A file is valid when it parses fully, closes with `</routes>`, and has a vehicle count of at least 95% of `3600 / period`. The threshold is confirmed against the 110 existing files.
  - One helper, `build_file.bg_period(factor) = round(1.2 / factor, 3)`, formatted with `%s`, names every file. This gives `p1.2`, `p1.132`, `p1.091`, `p1.062`, `p1.0` and `p0.6`.
  - Nothing is generated inside a TensorFlow process.

## 8. Evaluation sweep

- **Launch.** The runner starts `main.py --run <name> --option evaluate [--levels …]` as a separate process, after training is complete (Section 9). It logs to `<store>/logs/eval_rN.log`.
- **Model load.** It restores with `model.load(<store>/model/, checkpoint=total_step)` and raises if that fails. It builds the env once, at factor 1.0, and reads `test_seeds` from the config.
- **Each level** (folder `'f%.2f' % level`):
  1. Move any existing `eva_data/f<level>/` to `eva_data/_failed_<stamp>/`.
  2. Set `env.demand_factor = level`.
  3. Log `EVAL_LEVEL_START f<level> bg_tag=<tag> corridor_total=<veh/h>`. The corridor total is read back from `exp_0.rou.xml` for the first seed.
  4. Run the test seeds, then call `init_data` and `output_data` into `eva_data/f<level>/`.
  5. Log `EVAL_LEVEL_DONE f<level> seeds=<n>`.

  After the last level it logs `>>> EVALUATE COMPLETE levels=<k>` and exits 0. Any error exits non-zero. Evaluation runs in the main thread, so there is no swallowed thread exception.
- **Demand scaling.** MaltaEnv passes `load = (False if factor == 1.0 else factor)` and `bg_period(factor)` through `_init_sim_config` → `gen_rou_file` → `gen_background_file`.
  - Corridor flows become `floor(flow × f + 0.5)` when f ≠ 1.0; at 1.0 the base files are byte-identical to today's.
  - The defaults keep `baselines.py` and RQ1 unchanged.
- **Checks (by the runner, on this attempt's log and files only).** The attempt passes when:
  - the process exited 0 and logged `EVALUATE COMPLETE`;
  - the log has `Restoring parameters from <store>/model/checkpoint-<total_step>` before the first START;
  - there is no line starting with `Traceback` or `Exception in thread`, and no `Can not find old checkpoint`.

  A level passes when:
  - its START…DONE block holds exactly `test 0` … `test n-1`;
  - its three CSVs, n `edgedata_<seed>.xml` and n `summary_<seed>.xml` files are newer than the attempt's launch and close properly;
  - the CSVs cover episodes 1–n;
  - the logged `bg_tag` equals `p<bg_period(f)>`, and the corridor total equals the rounded scaled base total;
  - mean trips per episode does not decrease as the level rises.

  When every level passes, the runner writes `eva_data/EVAL_OK`.
- **Retry.** A failed attempt re-runs only the failed levels (`--levels`), once. A second failure writes `FAILED` (EVAL_FAILED).
- **Time limit.** Each level gets 3 × the measured f1.00 level time (at least 90 min); f2.00 may gridlock, because teleporting is off.

## 9. Queue runner (`tools/run_queue.ps1`)

### Start-up and environment

- **Starting.** The runner is started only through the scheduled task (`Start-ScheduledTask 'TORRS training queue'`), never from a Claude shell.
- **Single instance.** It holds the named mutex `Global\TORRS_queue_runner`. A second copy blocks in `WaitOne` and takes over if the first copy dies. `AbandonedMutexException` counts as ownership. On taking ownership, the runner adopts any live `python.exe` whose command line contains `main.py --run <name>` and launches nothing new.
- **Environment check.** At start it runs, the same detached way, `python -c "import tensorflow, traci, sumolib, os; print(os.environ['SUMO_HOME'])"` and checks that `sumo.exe` exists. On failure it records ENV_ERROR and launches nothing.
- **Launch command.** Every python launch uses:

  ```
  cmd.exe /v:on /c "set PYTHONIOENCODING=utf-8&& set PYTHONUNBUFFERED=1&& "C:\Program Files\Python311\python.exe" -X faulthandler main.py --run <name> --option <stage> > "<log>" 2>&1 & echo EXIT=!errorlevel! at !date! !time! > "<exitfile>""
  ```

  It is started with `Invoke-CimMethod Win32_Process Create`, with the working directory set to the project root.
- **Code hashes.** At each launch the runner writes SHA-256 hashes of `main.py`, `utils.py`, `agents/*.py`, `envs/*.py`, `Malta/build_file.py` and `<store>/config.ini` to `<store>/logs/launch_rN.code.txt`. A relaunch whose hashes differ from the run's first launch is refused (FAILED: CODE_CHANGED).
- **Config.** The runner reads `total_step`, `test_seeds` and `eval_levels` from `<store>/config.ini`. It never hard-codes 72000 or 10.

### Before every launch

All of these must pass; otherwise the runner sets a WAITING status and re-checks every 5 min.

- STOP is absent.
- **The machine is free:**
  - no other python runs `main.py`, `ablation_routing.py`, `baselines.py`, `gen_traffic.py`, `randomTrips.py`, `build_zones.py` or an ipykernel;
  - no `sumo`, `sumo-gui`, `duarouter` or `netconvert` runs outside the queue's process tree;
  - nothing listens on TCP 27500.
- **Resources:**
  - commit limit minus current commit ≥ the arm's smoke-test peak + 8 GB;
  - free space ≥ 60 GB on E: and ≥ 10 GB on C:.
- **Traffic files:** every traffic file the queue needs exists and passes the validity test.

### Per run (`config/queue.txt` is re-read before each run)

| State | Action |
|---|---|
| `DONE` | Skip. |
| `FAILED` | Exit at once. A reboot never retries or skips past a failed run. |
| Training complete, no `EVAL_OK` | Evaluate (Section 8). |
| Otherwise | Recovery, then Train. |

**Training is complete** when all of these hold:
- the training python exited 0 after printing `TRAINING COMPLETE step <total_step>`;
- no python or SUMO for the run is alive;
- `checkpoint-<total_step>` `.data`, `.index` and `.meta` exist;
- both reward CSVs have exactly `total_step / T` rows, with every reward and average wait finite;
- the seed column runs 12, 13, … with no repeats.

The runner never launches `--option train` for a complete run, and never launches evaluation while a training process for the run is alive.

### Recovery

Recovery uses the old supervisor's logic, scoped to `<store>`, and only ever moves files, never deletes them:

- **Good checkpoint.** The good checkpoint is the newest one that has all three files and its exact `Checkpoint saved at step <N> -> <store>/model/` line in this run's launch logs.
- **Half-saved checkpoints.** Checkpoint files above the good one, including `.tempstate*` and `.tmp*` files, move to `_queue/quarantine_<stamp>/`.
- **Trimming.**
  - `train_reward.csv`, `agent_reward.csv` and `episode_cost.csv` are trimmed to rows with step ≤ the good checkpoint. Originals are copied to `_queue/recovery_<stamp>/`.
  - If the cost CSV is exactly one row short, a NaN row with `lost = 1` is appended.
  - If a reward CSV is short or fails to parse, the run is marked FAILED (NEEDS_REVIEW).
- **From scratch.** A run restarts from scratch only when `model/` has no `checkpoint-*` file and `train_reward.csv` is absent or empty. If checkpoint files exist but none is logged, the run is marked FAILED (NEEDS_REVIEW).
- **Corrupt checkpoint.** If a launch log shows restore errors (DataLossError, "Unable to open table file"), that checkpoint is quarantined and the run resumes from the next older good one.
- **Cleanup before every launch, and whenever no queue python is alive:**
  - `taskkill /T /F` on the wrapper or python;
  - kill every `sumo.exe` whose command line contains `--remote-port 27500`, and every `duarouter.exe` or randomTrips python writing under `Malta\data`;
  - wait up to 2 min for port 27500 to be free.

### Failure, watchdog and STOP

- **Death counters.** These persist in `_queue/state.json`, written to a temp file and then moved. A launch made progress if its good checkpoint rose.
  - Three consecutive launches without progress, or 12 launches of one run in total, mark the run FAILED (STUCK).
  - An interruption (reboot or logoff, shown by exit code 0x40010004 or a missing exit file with a boot time after the launch) counts only toward the 12.
- **Watchdog.** A launch is treated as a death and its process tree killed if:
  - no `Checkpoint saved` line appears within the first-checkpoint limit (default 4 h, set from the smoke test);
  - more than 90 min pass after the previous checkpoint line;
  - the log contains a checkpoint-save error.

  During evaluation, a level that exceeds its time limit fails that attempt.
- **NaN check.** At each new checkpoint the runner reads the last `train_reward.csv` row. A non-finite value marks the run FAILED (NAN).
- **STOP** (`Malta/runs/_queue/STOP`). It is checked on start and before every launch; if present, the runner launches nothing, writes STOPPED and exits.
  - During training, the runner waits for the next `Checkpoint saved` line (or 90 min at most), waits 15 s, kills python and its SUMO children, runs Recovery and exits.
  - During evaluation it kills at once, and that attempt does not count as a retry.
  - The runner never deletes STOP. To resume, the author deletes STOP and runs `Start-ScheduledTask 'TORRS training queue'`.
- **Robustness.**
  - The main loop catches every exception, logs it, sleeps 60 s and continues; only terminal states exit.
  - Logs are read incrementally from a saved byte offset, with shared read access.
  - `status.txt` is rewritten every poll with a heartbeat time.
  - Files are written as UTF-8 without a BOM.
  - While a job runs, `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` keeps the PC awake.

### Scheduled task

- **Name:** "TORRS training queue".
- **Triggers:** at startup and at logon of kylea, each with a 2-minute delay.
- **Principal:** kylea, logon type S4U. It runs whether or not anyone is logged on, and no password is stored.
- **Settings:**
  - no time limit;
  - IgnoreNew for a second instance;
  - start when available;
  - battery restrictions off;
  - restart every 5 min, up to 99 times.
- **Action:** `powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File "<project>\tools\run_queue.ps1"`, with the project as working directory.
- **Removal.** When the queue finishes or fails, the runner disables the triggers as its last action. Claude unregisters the task afterwards.

### Git and code freeze

The runner never runs git. While the queue exists, Claude does not edit `main.py`, `utils.py`, `agents/`, `envs/` or `Malta/build_file.py` in the main working tree. Any fix is made in a git worktree and merged only while the queue is stopped. Claude never runs git stash, checkout, restore, reset or pull in the main tree during the queue.

## 10. Pre-launch work, in order

Heavy steps run strictly one at a time. Writing code may overlap with them; running it may not.

1. ✅ **Done 8 Nov:** RQ1 backup to `_rq1_final_backup/` (143 files, 1.7 GB, verified, read-only), and a first manifest of all 395 RQ1 files.
2. ✅ **Done 8 Nov:** base traffic files for training seeds 50–111 (62 files, all valid). All of seeds 12–111 plus the 10 test seeds now exist at `p1.2`.
3. Generate the 50 sweep traffic files: the 10 test seeds at `p1.132`, `p1.091`, `p1.062`, `p1.0` and `p0.6`, 4 at a time.
4. **Optional RQ1 jobs, if the author wants them before the queue:**
   - the actuated-rerouting baseline needs all ten seeds (`python baselines.py actuated_rerouting`, about 2.5 h), because seeds 10000–80000 kept no trip rows, 90000 is partial and 100000 is missing;
   - the routing ablation (TensorFlow, about 1.5 h).

   They write only into `Malta/signal_route/{baseline_data/actuated_rerouting, ablation_data*, stats}`.
5. **Code changes** (Sections 3–9), in small commits.
   - Re-sync `TORRS_MA2C_Complete.ipynb` for every changed `.py` file, including the existing drift (`Counter.resume`).
   - Add notebook cells for `config/runs/*.ini` and the `tools/*.py` files.
   - Update the notebook intro to document `--run`, and say not to run the notebook while the queue exists.
6. **Re-record the RQ1 manifest in two parts**, with a copy kept in `_rq1_final_backup/`:
   - **frozen:** `model/checkpoint-72000.*`, `train_reward.csv`, `agent_reward.csv`, `eva_data/`, `greedy_data/`, `baseline_data/{fixed_static,fixed_rerouting,actuated_static}/` and `_rq1_final_backup/`;
   - **allowed:** `stats/`, `ablation_data*/` and `baseline_data/actuated_rerouting/`.

   Live checks compare only the frozen part.
7. Run the Section 12 checks and the smoke queue.
8. **Author gates:**
   - a fixed pagefile on C: (initial = maximum ≥ 65,536 MB), with a reboot; Claude confirms, read-only, a commit limit of at least 95 GB;
   - Windows Update paused;
   - optionally, BIOS "restore on AC power loss" set to On;
   - the per-district zone counts approved;
   - no gaming during the queue, and run CSVs not opened in Excel while training.
9. Register the scheduled task, with the author's approval as given, then `Start-ScheduledTask`.

## 11. Timeline

| Date | Work |
|---|---|
| 8 Nov | Steps 1–2 done; sweep traffic files and coding start. |
| 9 Nov | Code finished; checks and smoke queue (about 5 h); author gates. |
| Night of 9 Nov or 10 Nov | `cloud_base` starts. |

Each run takes about 40 h of training plus about 3–4 h of sweep. The f2.00 level may take longer, because the network can gridlock. That is about 5.5–6 days for all three, finishing around 15–16 November if nothing goes wrong. Each restart costs about 1 h.

## 12. Testing

**`tools/check_env.py --arm <arm> --port 20`** (no TensorFlow; 100 random-action SUMO steps) checks:
- per-district and per-agent neighbour counts;
- `n_s_ls`, `n_a_ls`, `n_w_ls`, `n_o_ls`, `n_f_ls`;
- `len(state[i]) == n_s + n_w + n_o + n_f` for every agent at every step;
- for hybrid, that `state[i][n_s+n_w+n_o-12 : n_s+n_w+n_o]` equals the computed summary for one signal and one route agent;
- the summary is finite (including districts with no signal lanes);
- rewards are finite (including the smallest district and agents with no neighbours of a type);
- the Section 6 same-scale rule.

**Guards:**
- `main.py` refuses without `--rq1`, and refuses `--run ../signal_route`, `Malta\signal_route`, `./malta/Signal_Route/` and `Malta/signal_route/x`;
- a 2-step training run of the local arm into a temp store adds no files outside it, the shared `exp_<thread>.*` files and `Malta/runs/_queue/`.

**`local` unchanged:**
- The new code's `local` dimensions equal the RQ1 values in `training_run_bg_entropy_100ep_r9.log` (`s dim ls`, `a dim ls`).
- For one seed and a fixed action list, per-step states and rewards match the pre-change code.
  - The pre-change copy is a git worktree of HEAD in the scratchpad, driven only through MaltaEnv from a test script, never through its `main.py`.
  - It runs on port 21, after copying `malta.net.xml`, `malta_cache_v2.pkl` and the bg file into its `Malta/data/`.

**Demand factor:** at 1.0 the route file is byte-identical to today's; at 1.10 the corridor flows are `floor(flow × 1.1 + 0.5)` and the background tag is `p1.091`.

**Smoke queue** (`config/queue_smoke.txt`, run through a temporary scheduled task, with nothing else running).
- **Configs:** `_smoke_hybrid`, `_smoke_fog` and `_smoke_cloud`. Each is its arm's config with:
  - `episode_length_sec = 720`;
  - `total_step = 288` (2 episodes);
  - `test_seeds = 10000,20000`;
  - `eval_levels = 1.00,1.10`.
- **Fault test:** python is killed once during episode 2.
- **Pass when:**
  - recovery resumes from the first checkpoint with `env.seed = 13`, with the CSVs and cost file consistent;
  - `tf.train.list_variables` shows Adam slots in the checkpoint;
  - evaluation restores it with the policy-only Saver and passes the Section 8 checks, writing `EVAL_OK` and `DONE`;
  - STOP, a second runner copy and a FAILED marker behave as specified;
  - the frozen RQ1 manifest is unchanged.
- **Recorded per arm:** peak commit, TF build time, seconds per step, `.meta` and `.data` size, first-checkpoint time. These set the runner's watchdog and memory thresholds.
- **Afterwards:** the smoke folders move to the scratchpad.

**Live, at each run's first checkpoint:**
- only `Malta/runs/<run>/`, `Malta/runs/_queue/` and the shared `exp_<thread>.*` files changed;
- the frozen RQ1 manifest matches;
- `.meta` ≤ 1.5 GB and `.data` ≤ 1.5 GB (RQ1: 743 MB and 151 MB);
- the episode time and peak commit are recorded;
- `episode_cost.csv` has a valid row.

An episode over 35 min, or a peak commit within 2 GB of the limit, is flagged to the author.

## 13. Risks

- **Memory.** Each run commits about 43 GB, or about 52 GB while saving; the Adam slots and the cloud arm's wider inputs add more, measured in the smoke test. The fixed pagefile gate addresses this.
- **Power cuts.** With the startup trigger, the queue resumes after a reboot without a logon. Without the BIOS setting, the PC stays off after a power cut until someone turns it on.
- **Similar results across arms.** The arms may give similar results. That is a legitimate RQ2 finding, and the demand sweep adds discrimination.
- **Training stability from one seed.** One training seed per arm gives a thin basis for "training stability". Episode-to-episode variance and the cost log are reported, and replicate runs can be queued later.
- **One failure stops the queue,** as the author decided. A failure overnight leaves the PC idle until Claude or the author checks. The smoke queue exists to make that unlikely.

## 14. Follow-ups (not in this design)

- Peak-season baselines at 1.10× in `Malta/peak/baseline_data/`.
- RQ2 paired statistics in `Malta/rq2_stats/`.
- The RQ3 compliance sweep (port `run_diff_com` to Malta).
- **Thesis text:**
  - describe the base demand as a scaled, typical term-time weekday peak hour (the current text claims an NSO calibration the code does not do);
  - cite NR 182/2026 for the peak factor;
  - note the protocol changes against RQ1: seed carry-over, optimiser state, mean reward sharing with `coop_gamma_sr = 0`;
  - note the cloud arm's Malta–Gozo link.
