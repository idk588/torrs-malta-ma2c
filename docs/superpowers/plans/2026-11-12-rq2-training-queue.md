# RQ2 Training Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the cloud, fog and hybrid MA2C arms one after another, unattended, each in its own `Malta/runs/<run>/` folder, then evaluate each arm at six demand levels. The RQ1 run in `Malta/signal_route/` is never touched.

**Architecture:** The three arms are the existing MA2C model with different neighbour maps, built from an NSO-district zone file (`envs/zones.py`). `main.py` gains a `--run` CLI (`run_config.py`) that resolves a per-run config and store, and guards RQ1. Training writes atomically and logs per-episode cost (`run_files.py`). Evaluation runs as its own process, sweeping demand levels. A PowerShell queue runner (`tools/run_queue.ps1` + `tools/queue_lib.ps1`), started by a scheduled task, launches, recovers, checks and advances runs.

**Tech Stack:**
- Python 3.11 standard library plus the existing TF 2.18 (compat.v1), numpy 2.0.2, pandas, SUMO 1.26 traci/sumolib;
- Windows PowerShell 5.1;
- `unittest` for Python tests and plain-assert `.ps1` scripts for PowerShell tests.

**Spec:** `docs/superpowers/specs/2026-11-08-rq2-training-queue-design.md`. Read it with this plan; section numbers below (§n) refer to it.

## Global Constraints

- **No new dependencies.** `pytest`, `psutil` and `pyproj` are NOT installed and must not be added. Use `unittest`, `ctypes` and `math`.
- **Python:** `C:\Program Files\Python311\python.exe`. **PowerShell:** 5.1 only; no `&&`, `??`, `?:` or `?.`, and `ConvertFrom-Json` returns a PSCustomObject.
- **RQ1 defaults:** every new config key is read with a fallback equal to RQ1 behaviour, so `config/config_ma2c_Malta.ini` stays unedited. `baselines.py`, `ablation_routing.py` and `stats_compare.py` must run unchanged.
- **Never write into `Malta/signal_route/`** from new code paths; RQ1 access needs `--rq1`.
- **Path strings:**
  - Run stores are written exactly `Malta/runs/<name>/`: relative, forward slashes, trailing `/`.
  - Model dir `<store>model/`, SUMO dir `<store>sim/`, level dir `<store>eva_data/f%.2f/`.
  - The runner matches log lines built from these exact strings.
- **Files:** route and config files are written in text mode (CRLF), as today. Files the runner writes are UTF-8 without a BOM, via `[IO.File]` with absolute paths.
- **Ports:** port/thread 0 (TraCI 27500, `Malta/data/exp_0.*`) is reserved for the queue. Every test or check uses port 20 or higher.
- **Demand levels** (exact): `1.00,1.06,1.10,1.13,1.20,2.00`. Background period: `round(1.2 / factor, 3)`. Tags: `p1.2, p1.132, p1.091, p1.062, p1.0, p0.6`.
- **Districts, in this order** (exact strings): `Southern Harbour, Northern Harbour, South Eastern, Western, Northern, Gozo & Comino`.
- **RQ2 run configs:** `comm_arch` ∈ {cloud, fog, hybrid}; `reward_share = mean`; `coop_gamma_sr = 0`; `seed_continue = true`; `save_optimizer = true` (MODEL_CONFIG); `norm_summary_wait = 60`; `total_step = 72000`; `seed = 12`. Everything else is copied from the RQ1 config.
- **Commits:**
  - Commit after every task, with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`, and push to `origin main`.
  - Re-sync `TORRS_MA2C_Complete.ipynb` in Task 16, not per task, but before the queue launches.
- **Pre-change baseline commit:** `bc7115c`, used by the local-unchanged comparison in Task 6.
- **PowerShell numbers:** parse and format with `[cultureinfo]::InvariantCulture`, so level folders are always `f1.10`, never `f1,10`.
- **Code freeze:** once the real queue task is registered (Task 17 Step 6), do not edit `main.py`, `utils.py`, `run_config.py`, `run_files.py`, `agents/`, `envs/` or `Malta/build_file.py` in the main working tree (§9).

## Review Focus

1. **A run CSV open in Excel during an episode write.** Expect the write to retry for 60 s, then raise rather than silently skip. Test in Task 9.
2. **A power cut in the middle of a checkpoint save**, leaving `.tempstate*` / `.meta.tmp*` files and CSVs one row ahead. Expect recovery to quarantine the partial step and trim the CSVs to the last logged checkpoint. Test in Task 12.
3. **A district with no signal agents, or a node with no detectors**, in the hybrid summary. Expect 0, never NaN. Test in Task 3 (`district_summary`), and check end to end in Task 6.
4. **The store written in a different textual form** (backslashes, absolute path, missing trailing slash). Expect `main.py` to normalise to `Malta/runs/<name>/`, so the runner's string matches still hold. Test in Task 8.
5. **A reboot while python is still running, or a second runner copy.** Expect the runner to adopt the live process, not start a second one. Exercised in the Task 17 smoke queue (fault test plus a second copy).

---

### Task 1: Traffic generation and flow scaling in `Malta/build_file.py`

**Files:**
- Modify: `Malta/build_file.py:308-348` (`output_flows`), `:351-361` (`output_config`), `:423` (detector output name in `output_additional`), `:435-470` (`BG_PERIOD`, `gen_background_file`, `gen_rou_file`)
- Modify: `.gitignore` (add `Malta/data/exp_*.add.xml`, `Malta/data/ild_out_*.xml`, `Malta/data/_tmp_*/`)
- Create: `tests/__init__.py` (empty), `tests/test_build_file.py`, `tests/data/flows_f1_golden.xml`

**Interfaces:**
- Produces:
  - `bg_period(factor: float) -> float`
  - `bg_tag(seed: int, period: float) -> str`
  - `validate_bg_file(rou_path: str, period: float) -> tuple[bool, int]`
  - `gen_background_file(path: str, seed, period: float = BG_PERIOD, generate: bool = True) -> str` (bare `.rou.xml` name)
  - `output_flows(demand_flows, seed=None, load=False) -> str`
  - `gen_rou_file(path, demand_flows, seed=None, thread=None, load=False, period=BG_PERIOD, generate_bg=True) -> None`
  - `expected_corridor_total(demand_flows, factor: float) -> float`
  - `read_corridor_total(rou_path: str) -> float`

- [ ] **Step 1: Capture the golden flows before changing anything.** Write `output_flows(get_flows(), seed=12)` from the current code to `tests/data/flows_f1_golden.xml` in text mode.
- [ ] **Step 2: Write the failing tests** in `tests/test_build_file.py`:

```python
class TestBuildFile(unittest.TestCase):
    def test_bg_period_levels(self):
        self.assertEqual([bg_period(f) for f in (1.0, 1.06, 1.10, 1.13, 1.20, 2.00)],
                         [1.2, 1.132, 1.091, 1.062, 1.0, 0.6])
    def test_bg_tag(self):
        self.assertEqual(bg_tag(12, 1.2), 'bg_12_p1.2')
        self.assertEqual(bg_tag(10000, bg_period(1.10)), 'bg_10000_p1.091')
    def test_flows_unscaled_byte_identical(self):
        golden = open('tests/data/flows_f1_golden.xml').read()
        self.assertEqual(output_flows(get_flows(), seed=12), golden)
    def test_flows_scaled_half_up(self):
        txt = output_flows(get_flows(), seed=12, load=1.10)
        self.assertEqual(sum_vehs_per_hour(txt), 10738)   # per-flow half-up total
        self.assertEqual(expected_corridor_total(get_flows(), 1.10), 10738)
        self.assertEqual(expected_corridor_total(get_flows(), 1.0), sum_vehs_per_hour(open('tests/data/flows_f1_golden.xml').read()))
    def test_validate_bg_file(self):
        # existing base file: 3000 vehicles at p1.2
        self.assertEqual(validate_bg_file('Malta/data/bg_12_p1.2.rou.xml', 1.2), (True, 3000))
        # a truncated copy in a temp dir must fail
    def test_missing_bg_without_generate_raises(self):
        with self.assertRaises(FileNotFoundError):
            gen_background_file(tmpdir + '/', 999999, 1.2, generate=False)
    def test_per_thread_add_file(self):
        # gen_rou_file into a temp copy dir with thread=21 writes exp_21.add.xml whose
        # detector file attribute is ild_out_21.xml, and exp_21.sumocfg references exp_21.add.xml
```

`sum_vehs_per_hour` is a test helper that sums the `vehsPerHour` attributes in the text.

- [ ] **Step 3: Run the tests and confirm they fail.**
  Run: `python -m unittest tests.test_build_file -v`
  Expected: errors on `bg_period`, `bg_tag` and `validate_bg_file` not defined.
- [ ] **Step 4: Implement.**
  - **`output_flows`:** replace `int(flow_val * load)` with `math.floor(flow_val * load + 0.5)`, keeping `if load:`.
  - **`expected_corridor_total`:** applies the same per-flow expression (or no change at 1.0).
  - **`gen_background_file`:**
    - Generate into `path + '_tmp_<tag>_<pid>/'`, validate (full `iterparse`, closes `</routes>`, count ≥ `0.95 * ceil(3600/period)`), then `os.replace` the trips file and then the rou file into `path`.
    - When the final file is missing and `generate=False`, raise `FileNotFoundError(path + rou)`.
    - Keep returning the bare name, because `baselines.py:75` relies on it.
  - **`gen_rou_file`:**
    - Write `exp_%d.add.xml % int(thread)`, with the detector `file="ild_out_%d.xml" % int(thread)`, via a new `output_additional(..., det_file='ild_out.xml')` parameter; pass that add-file name to `output_config(add_file=...)`.
    - Forward `period` and `generate_bg` to `gen_background_file`.
  - **`read_corridor_total`:** sums `vehsPerHour` in a written `.rou.xml`.
- [ ] **Step 5: Run the tests and confirm they pass.**
  Run: `python -m unittest tests.test_build_file -v`
  Expected: all PASS.
- [ ] **Step 6: Check that RQ1 callers still work.**
  Run: `python -c "from Malta.build_file import gen_background_file as g; print(g('Malta/data/', 12))"`
  Expected: prints `bg_12_p1.2.rou.xml`, with no generation and no change to that file's mtime.
- [ ] **Step 7: Commit** with `git add Malta/build_file.py .gitignore tests/` and the message "Make traffic generation per-period, validated and per-thread".

### Task 2: `tools/gen_traffic.py` (pre-generation and validation)

**Files:**
- Create: `tools/__init__.py` (empty), `tools/gen_traffic.py`, `tests/test_gen_traffic.py`

**Interfaces:**
- Consumes (Task 1): `bg_period`, `bg_tag`, `validate_bg_file`, `gen_background_file(..., generate=True)`
- Produces:
  - CLI `python tools/gen_traffic.py --seeds 12-111 --factors 1.0 [--workers 4] [--check-only]`. It exits 0 only if every requested file is valid, and prints one line per file plus a summary.
  - `parse_seeds(text: str) -> list[int]` (accepts `12-111`, `10000,20000`, or `test` = the config `test_seeds`)
  - `needed_files(train_seeds, test_seeds, levels) -> list[tuple[int, float]]` (seed, period)

- [ ] **Step 1: Write the failing tests:**
  - `parse_seeds('12-14') == [12, 13, 14]`;
  - `parse_seeds('10000,20000') == [10000, 20000]`;
  - `needed_files([12], [10000], [1.0, 1.10]) == [(12, 1.2), (10000, 1.2), (10000, 1.091)]` (training seeds at 1.0 only).
- [ ] **Step 2: Run** `python -m unittest tests.test_gen_traffic -v` and confirm FAIL.
- [ ] **Step 3: Implement.** Use a `ThreadPoolExecutor` (work happens in subprocesses) with disjoint seed lists per worker. `--check-only` validates without generating.
- [ ] **Step 4: Run the tests and confirm PASS.**
- [ ] **Step 5: Check every file the queue will need:**

  Run: `python tools/gen_traffic.py --seeds 12-111 --factors 1.0 --check-only`, then `python tools/gen_traffic.py --seeds test --factors 1.00,1.06,1.10,1.13,1.20,2.00 --check-only`

  Expected: both exit 0. That covers 100 + 60 files: the 110 base files exist and the 50 sweep files were generated on 12 Nov. Any missing or invalid file is regenerated by re-running without `--check-only`.
- [ ] **Step 6: Commit** with the message "Add traffic pre-generation and validation tool".

### Task 3: Zones (`envs/zones.py`, `tools/build_zones.py`, council table)

**Files:**
- Create: `envs/zones.py`, `tools/build_zones.py`, `Malta/data/councils_nso.csv`, `Malta/data/zones_nso.csv` (generated), `tests/test_zones.py`

**Interfaces:**
- Produces, in `envs/zones.py`:
  - `DISTRICTS: tuple[str, ...]` (the six names, in the Global Constraints order)
  - `load_zones(csv_path: str) -> dict[str, str]` (agent_id → district, read as UTF-8)
  - `build_maps(comm_arch: str, signal_ids: list[str], route_ids: list[str], zone_of: dict[str, str]) -> tuple[dict, dict, dict, dict]`, returning `(ss, sr, rr, rs)` with these shapes:
    - `ss = {sig: [sig, ...]}`
    - `sr = {sig: {rc: 1.0}}`
    - `rr = {rc: [rc, ...]}`
    - `rs = {rc: {sig: 1.0}}`
  - `global_order(signal_ids, route_ids) -> list[str]` = `sorted(signal_ids) + list(route_ids)`
  - `neighbour_hash(ss, sr, rr, rs) -> str` (SHA-256 hex of `json.dumps([ss, sr, rr, rs], ensure_ascii=False)` with insertion order preserved)
  - `district_summary(wait_means: np.ndarray, vehwait_per_lane: np.ndarray, district_idx: np.ndarray, norm_wait: float, clip_wait: float) -> np.ndarray`. It returns shape (12,) = `[q_1..q_6, w_1..w_6]`; a district with no members gives 0.
- Produces, in `tools/build_zones.py`:
  - `utm_to_lonlat(e: float, n: float, zone: int = 33) -> tuple[float, float]`
  - `lonlat_to_utm(lon, lat, zone=33) -> tuple[float, float]`
  - `read_net(net_path) -> tuple[tuple[float, float], dict[str, tuple[float, float]], dict[str, str]]`, i.e. (netOffset, junction xy, edge → from-junction)
  - CLI `python tools/build_zones.py`, which writes `Malta/data/zones_nso.csv`, prints counts per district and per council and the corridors that cross districts, and exits 2 on any violation listed in §5.

- [ ] **Step 1: Build `Malta/data/councils_nso.csv`.**
  - **Columns:** `council,district,island,lat,lon,source`.
  - **District membership** follows the NSO LAU1 districts, 68 councils in all:
    - Southern Harbour 14: Valletta, Floriana, Marsa, Paola, Tarxien, Fgura, Santa Luċija, Luqa, Kalkara, Birgu, Isla, Bormla, Xgħajra, Żabbar;
    - Northern Harbour 13: Birkirkara, Gżira, Ħamrun, Msida, Pembroke, Pietà, Qormi, San Ġwann, Santa Venera, San Ġiljan, Swieqi, Ta' Xbiex, Sliema;
    - South Eastern 11: Birżebbuġa, Gudja, Għaxaq, Kirkop, Marsaskala, Marsaxlokk, Mqabba, Qrendi, Safi, Żejtun, Żurrieq;
    - Western 10: Attard, Balzan, Dingli, Iklin, Lija, Mdina, Mtarfa, Rabat, Siġġiewi, Żebbuġ;
    - Northern 6: Għargħur, Mellieħa, Mġarr, Mosta, Naxxar, San Pawl il-Baħar;
    - Gozo & Comino 14: Fontana, Għajnsielem, Għarb, Għasri, Kerċem, Munxar, Nadur, Qala, Rabat (Victoria), San Lawrenz, Sannat, Xagħra, Xewkija, Żebbuġ (Gozo).

    Ħamrun belongs to Northern Harbour only. Confirm the list against NSO's *Regional Statistics Malta* before committing, and record the edition in `source`.
  - **Points:** lat/lon are each locality's OSM `place` node, fetched once from the public Overpass or Nominatim API at ≤ 1 request/s and recorded in `source`.
- [ ] **Step 2: Write the failing tests** in `tests/test_zones.py`:
  - **UTM:**
    - `lonlat_to_utm` then `utm_to_lonlat` round-trips 5 points in Malta to within 1e-7°;
    - each of the four net corners (`convBoundary` 0..35083.37 × 0..30711.08 shifted by `netOffset` −426538.16, −3962602.48) converts to lon 14.15–14.60, lat 35.78–36.10.
  - **Council table:** `councils_nso.csv` has 68 rows, unique (council, district) pairs, and district counts 14/13/11/10/6/14. Every row with island == Gozo has lat > 36.005.
  - **Maps:**
    - `build_maps('fog', ['b', 'a', 'c'], ['rc1'], {'a': 'Western', 'b': 'Western', 'c': 'Northern', 'rc1': 'Western'})` gives `ss['a'] == ['b']`, `sr['a'] == {'rc1': 1.0}`, `rs['rc1'] == {'a': 1.0, 'b': 1.0}`, `ss['c'] == []`, `rr['rc1'] == []`.
    - `build_maps('cloud', ...)` gives every agent all the others, in `global_order`, excluding itself.
    - `neighbour_hash` is stable across two calls and changes if one neighbour list is reordered.
  - **Summary:** `district_summary` with district 5 empty and one node with NaN `wait_means` (no detectors) returns 12 finite values, with indices 5 and 11 equal to 0.0. A node's `w` is `clip(vehwait_per_lane / norm, 0, clip)`.
- [ ] **Step 3: Run** `python -m unittest tests.test_zones -v` and confirm FAIL.
- [ ] **Step 4: Implement.**
  - **UTM:** WGS84 transverse Mercator (k0 0.9996, false easting 500000, central meridian 15°), using the standard series.
  - **`read_net`:** iterparses only `location`, `junction` and `edge`.
  - **Assignment:** island first (Gozo & Comino if lat > 36.005), then the nearest council of that island by great-circle distance.
  - **Route points:** each route agent's point is the from-junction of its `up_edge`, read from `MaltaRoute` in `envs/Malta_env.py`.
  - **Exit 2 if:**
    - any agent is > 5 km from its council;
    - any point is out of bounds;
    - signal ids ≠ `sorted(NODE)`, or route ids ≠ `RC_NAMES`.
- [ ] **Step 5: Run the tests and confirm PASS**, then run `python tools/build_zones.py` and keep its printed counts for the author's approval (Task 17).
- [ ] **Step 6: Commit** `envs/zones.py tools/build_zones.py Malta/data/councils_nso.csv Malta/data/zones_nso.csv tests/test_zones.py` with the message "Add NSO district zones for fog/hybrid maps".

### Task 4: Communication maps in the env (`comm_arch`)

**Files:**
- Modify: `envs/env.py:123-166` (read the new keys before `_init_map`), `:424-438` (end of `_init_agents`: index arrays)
- Modify: `envs/Malta_env.py:312-314` (`__init__`), `:469-511` (`_init_map`)

**Interfaces:**
- Consumes (Task 3): `load_zones`, `build_maps`, `global_order`, `neighbour_hash`
- Produces:
  - Attributes on `TrafficSimulator`: `comm_arch: str` (fallback `'local'`), `reward_share: str` (`'sum'`), `norm_summary_wait: float` (60.0)
  - `MaltaEnv.__init__(self, config, problem, output_path='', is_record=False, port=0, generate_traffic=True)`, which sets `self.demand_factor` (config fallback 1.0) and `self.generate_traffic` before `super().__init__`
  - `env.neighbour_hash() -> str`
  - Index arrays built at the end of `_init_agents` by `_build_index_arrays()`: `self._ss_idx, self._sr_idx` (one int array per signal, global indices: signal i → i, route k → n_signals + k) and `self._rr_idx, self._rs_idx` (one per route)
  - `self._district_idx: np.ndarray` (per signal, index into `DISTRICTS`; -1 if absent)

- [ ] **Step 1: Write the failing test** in `tests/test_env_maps.py`:
  - **local:** build `MaltaEnv(config_section, 'signal_route', output_path=tmp/, port=20)` from the RQ1 config. It gives `env.n_s_ls` and `env.n_a_ls` equal to the lists in `training_run_bg_entropy_100ep_r9.log` (the `s dim ls` / `a dim ls` line, parsed by the test), and every `relevant_sr` / `relevant_rs` is empty.
  - **fog:** the same env with `comm_arch = fog` (a config copy with the key set) gives every signal's `relevant_ss` equal to the other signals of its district in `global_order` order, and `env.neighbour_hash()` equals `neighbour_hash` of the four maps.
- [ ] **Step 2: Run** `python -m unittest tests.test_env_maps -v` and confirm FAIL. The test takes about a minute, since it starts SUMO.
- [ ] **Step 3: Implement.**
  - Under `local`, `_init_map` runs today's body unchanged, including the `tl_set` cleaning.
  - For the other arms, set `relevant_ss_map`, `relevant_sr_map`, `relevant_rr_map` and `relevant_rs_map` from `build_maps(comm_arch, NODE, RC_NAMES, load_zones(data_path + 'zones_nso.csv'))`, and skip the cleaning.
  - Replace the `node_names.index()` / `rc_names.index()` lookups in `_local_reward_step` with the index arrays, keeping the sum branch's accumulation order (loop, `+=`) so the floats stay identical.
- [ ] **Step 4: Run the test and confirm PASS.**
- [ ] **Step 5: Commit** with the message "Build fog/cloud/hybrid neighbour maps from NSO zones".

### Task 5: Mean reward sharing, hybrid summary and episode seed

**Files:**
- Modify: `envs/env.py:282-345` (`_get_state`: summary insertion), `:539-609` (`_init_state_space`: n_o +12), `:1020-1051` (`_local_reward_step`: mean branch)
- Modify: `envs/Malta_env.py:280-293` (`reset`: zero `vehwait`, set `episode_seed`), `:515-529` (`_init_sim_config`: demand factor and generation flag)
- Create: `tests/test_env_reward.py`

**Interfaces:**
- Consumes (Task 4): the index arrays and `_district_idx`; (Task 3): `district_summary`; (Task 1): `bg_period`, `gen_rou_file(..., period, generate_bg)`
- Produces:
  - `env.episode_seed: int` (the seed the last `reset` used)
  - `_mean_reward_step(reward: np.ndarray) -> np.ndarray`, which returns a new array
  - `_shared_terms(reward) -> np.ndarray` (r' − r before clipping, for `check_env`)
  - Under hybrid, `n_o_ls[i]` = old value + 12 for every agent

- [ ] **Step 1: Write the failing tests.** Build a stub with `object.__new__(MaltaEnv)` and only the attributes the methods read: names, index arrays, gammas, `problem = 'signal_route'`.
  - **Mean:** for signals [s0, s1] and route [rc1], `_ss_idx = [[1], [0]]`, `_sr_idx = [[2], []]`, `_rs_idx = [[0, 1]]`, γ_ss 0.3, γ_sr 0, γ_rs 0.1, γ_rr 0, and raw reward `[-1.0, -3.0, 500.0]`, `_mean_reward_step` returns `[-1.9, -3.3, 499.8]`. The input array is unchanged afterwards (no aliasing).
  - **Empty sets:** an agent with no neighbours of a type adds 0, and the result is finite.
  - **Sum unchanged:** under `reward_share = 'sum'`, `_local_reward_step` returns exactly what the pre-change formula gives for the same stub.
  - **Hybrid layout:** for one signal, the 12 values sit at `ob[n_s+n_w+n_o-12 : n_s+n_w+n_o]`, and `len(ob) == n_s + n_w + n_o + n_f`. Use a stub whose `_get_state` inputs are fixed arrays.
- [ ] **Step 2: Run** `python -m unittest tests.test_env_reward -v` and confirm FAIL.
- [ ] **Step 3: Implement.**
  - **`step()`:** call `_mean_reward_step` when `reward_share == 'mean'`, otherwise `_local_reward_step`. The `get_parameter` path keeps the sum branch.
  - **Hybrid summary:** compute it once per `_get_state` call, after `_clip_state()`, from signal agents only:
    - `wait_means = mean(node.wait_state)` per node (NaN-safe);
    - `vehwait_per_lane = node.vehwait / max(1, len(node.lanes_in))`.

    `lanes_in` is the same list `vehwait` is summed over; document this choice in a comment. Insert the summary after the `relevant_sr` arrivals (signals) or the `relevant_rs` waves (routes), before the fingerprints, only when `comm_arch == 'hybrid'`.
  - **`MaltaEnv.reset`:** set `node.vehwait = 0` for every node before `_get_state(reset=True)`, and set `self.episode_seed = seed`.
  - **`_init_sim_config`:** ignore the incoming `load` and call `gen_rou_file(..., load=(False if self.demand_factor == 1.0 else self.demand_factor), period=bg_period(self.demand_factor), generate_bg=self.generate_traffic)`.
- [ ] **Step 4: Run** `python -m unittest tests.test_env_reward tests.test_env_maps -v` and confirm PASS.
- [ ] **Step 5: Commit** with the message "Add mean reward sharing, hybrid district summary and demand factor".

### Task 6: `tools/check_env.py` and `tools/compare_local.py` (SUMO-level checks, no TF)

**Files:**
- Create: `tools/check_env.py`, `tools/compare_local.py`

**Interfaces:**
- Consumes (Tasks 4–5): env attributes, `_shared_terms`, `neighbour_hash`
- Produces:
  - `python tools/check_env.py --arm local|fog|cloud|hybrid --port 20 [--steps 100]`. It prints per-district and per-agent neighbour counts and the dimension lists, and exits non-zero on any §12 check failure.
  - `python tools/compare_local.py --base-worktree <dir> --port 21 --seed 12 --steps 50 --actions tests/data/actions_50.json`. It exits 0 only if per-step states and rewards are equal under the RQ1 config.

- [ ] **Step 1: Implement `check_env.py`.** It builds the arm's config by applying that arm's keys from §4 (the same keys Task 14's run configs use) to a copy of the RQ1 config, then checks:
  - `len(state[i]) == n_s + n_w + n_o + n_f` for every agent at every step;
  - the hybrid slice equals `district_summary`, recomputed from the env's own fields after the step;
  - all rewards and summaries are finite;
  - the same-scale rule (mean |shared term| ≤ mean |own reward| per agent type over the run).

  States are taken only from `reset()` / `step()` returns, never from a second `_get_state` call.
- [ ] **Step 2: Run** `python tools/check_env.py --arm fog --port 20`, then `--arm cloud`, `--arm hybrid` and `--arm local`. Expected: exit 0 each time, with the counts printed. Save the output to `Malta/runs/_queue/check_env_<arm>.txt` for the author.
- [ ] **Step 3: Implement `compare_local.py`.**
  - Create the worktree with `git worktree add <scratchpad>/base bc7115c`, then copy `malta.net.xml`, `malta_cache_v2.pkl` and `bg_12_p1.2.*` into `<worktree>/Malta/data/`.
  - Drive both versions only through `MaltaEnv` from this script; the base runs in a subprocess with `cwd = worktree`. Run them one after the other, never in parallel.
  - Generate the fixed action list once with numpy seed 0 and save it to `tests/data/actions_50.json`.
- [ ] **Step 4: Run** `python tools/compare_local.py --base-worktree <scratchpad>/base --port 21 --seed 12 --steps 50 --actions tests/data/actions_50.json`. Expected: `IDENTICAL states 50/50, rewards 50/50`, exit 0.
- [ ] **Step 5: Commit** `tools/check_env.py tools/compare_local.py tests/data/actions_50.json` with the message "Add environment checks for each arm and the local-unchanged comparison".

### Task 7: Optimiser state and explicit checkpoint load (`agents/models.py`)

**Files:**
- Modify: `agents/models.py:127-150` (`A2C.load`), `:379-423` (`MA2C.__init__` Saver placement)
- Create: `tests/test_models_saver.py`

**Interfaces:**
- Produces:
  - `MA2C.save_optimizer: bool` (MODEL_CONFIG `save_optimizer`, fallback False)
  - `MA2C.training: bool` (= `bool(total_step)`)
  - `load(model_dir=None, checkpoint=None) -> bool`. With `checkpoint=N`, `cur_step == N`. Training with `save_optimizer` on a checkpoint that has no `/Adam` variables raises `RuntimeError('checkpoint <prefix> has no optimiser state; refusing to resume')`.

- [ ] **Step 1: Write the failing tests.** Use a tiny MA2C: two agents (one signal, one route), widths `n_s_ls = [4, 2]`, `n_a_ls = [2, 2]`, `n_w_ls = [2, 1]`, `n_f_ls = [1, 1]`, `n_o_ls = [0, 0]`, `total_step = 288`, the RQ1 MODEL_CONFIG plus `save_optimizer = true`, and `agent_type` built the way `env.agent_type` builds it.
  - **Adam saved:** after `model.save(tmp + '/', 144)`, `tf.train.list_variables(tmp + '/checkpoint-144')` contains names ending `/Adam` and `beta1_power`.
  - **Policy-only restore:** an eval model (`total_step = 0`) built with the same widths restores it with `load(tmp + '/', checkpoint=144)` → True and `cur_step == 144`.
  - **Refuse Adam-less resume:** a training model with `save_optimizer = true` refuses a checkpoint saved by a `save_optimizer = false` model (RuntimeError).
  - **RQ1 unchanged:** with `save_optimizer` absent, the Saver's var_list has no Adam variables.
- [ ] **Step 2: Run** `python -m unittest tests.test_models_saver -v` and confirm FAIL. It takes about a minute (TF startup).
- [ ] **Step 3: Implement.**
  - When `total_step and self.save_optimizer`, build the Saver after `_init_train` and before the variable initialiser; otherwise leave the Saver where it is today.
  - In `load`, set `save_step = int(checkpoint)` in the explicit branch. Run the Adam check only when `self.training and self.save_optimizer`, using `tf.train.list_variables(model_dir + save_file)`.
- [ ] **Step 4: Run the tests and confirm PASS.** Then check RQ1 still restores, read-only:
  `python -c "...build the RQ1 eval MA2C as ablation_routing.py does...; assert model.load('Malta/signal_route/model/')"` loads checkpoint-72000.
- [ ] **Step 5: Commit** with the message "Save optimiser state for RQ2 runs and fix explicit checkpoint load".

### Task 8: `run_config.py` (CLI, RQ1 guard, store set-up)

**Files:**
- Create: `run_config.py`, `tests/test_run_config.py`
- Modify: `.gitignore` (add the six §3 lines: `Malta/runs/_queue/`, `Malta/runs/*/model/`, `Malta/runs/*/logs/`, `Malta/runs/*/sim/`, `Malta/runs/*/eva_data/_failed_*/`, `Malta/runs/_smoke_*/`, plus `Malta/runs/*/*.tmp`)

**Interfaces:**
- Produces:
  - `RunSpec` dataclass: `name: str | None; option: str; levels: list[float] | None; rq1: bool; config_path: str; store_dir: str`. `store_dir` is `'Malta/runs/<name>/'` exactly, or `'Malta/signal_route/'` for RQ1.
  - `parse_args(argv: list[str]) -> RunSpec` (uses `parse_known_args`; raises `SystemExit(2)` on invalid combinations)
  - `is_rq1_path(path: str) -> bool` (normcase/realpath, path-component aware)
  - `assert_not_rq1(path: str, rq1: bool) -> None` (raises `SystemExit(3)` with a message)
  - `configs_equal(a: str, b: str) -> tuple[bool, list[str]]` (configparser; differences listed as `section.key`)
  - `other_run_pids(own_pid: int) -> list[int]` (python.exe processes whose command line matches `main\.py\s+--run\s+\S+`; uses PowerShell `Get-CimInstance` through `subprocess`)
  - `prepare_store(spec, maps_hash: str) -> bool` (True on first launch). On first launch it writes `zones_nso.csv` and `neighbours.sha256` first, then `config.ini` last (atomic). On resume it compares the config and the hash, and raises `SystemExit(4)` on any difference.

- [ ] **Step 1: Write the failing tests:**
  - **Name and option rules:**
    - `parse_args(['--run', 'cloud_base'])` gives option `train`, store `Malta/runs/cloud_base/` and config `config/runs/cloud_base.ini`;
    - `parse_args(['--run', '_smoke_fog', '--option', 'evaluate', '--levels', '1.10,2.00'])` gives levels `[1.10, 2.00]`;
    - `--run ../signal_route`, `--run Cloud`, `--run x --rq1`, and `--run x --option greedy` all raise SystemExit;
    - no `--run` and no `--rq1` raises SystemExit;
    - `['--rq1']` gives option `train_then_evaluate` and store `Malta/signal_route/`;
    - `parse_args(['--run', 'x', '-f', 'kernel.json'])` ignores the kernel argument.
  - **Guard:**
    - `is_rq1_path` is True for `Malta/signal_route`, `Malta\\signal_route\\`, `./malta/Signal_Route/model/` and `Malta/signal_route/x`;
    - it is False for `Malta/signal_route_copy`, `Malta/signal_route(separate_training)` and `Malta/runs/cloud_base/`.
  - **Store string:** `RunSpec.store_dir` is forward-slash, relative, and ends with `/` even when the cwd is given with backslashes.
  - **Config comparison:** `configs_equal` ignores comments, whitespace and line endings, and reports a changed value as `ENV_CONFIG.comm_arch`.
  - **`prepare_store`** on a temp store: the first call returns True and writes the three files; the second call with the same hash returns False; a different hash raises SystemExit(4).
- [ ] **Step 2: Run** `python -m unittest tests.test_run_config -v` and confirm FAIL.
- [ ] **Step 3: Implement** as specified.
- [ ] **Step 4: Run the tests and confirm PASS.**
- [ ] **Step 5: Commit** `run_config.py tests/test_run_config.py .gitignore` with the message "Add run CLI, RQ1 guard and store set-up".

### Task 9: Atomic files, cost logging and the Trainer (`run_files.py`, `utils.py`)

**Files:**
- Create: `run_files.py`, `tests/test_run_files.py`
- Modify: `utils.py:38-56` (`init_dir`, `init_log`), `:193-225` (`Trainer.__init__`), `:355-436` (`Trainer.run`)

**Interfaces:**
- Produces, in `run_files.py`:
  - `atomic_write_text(path, text, retries=6, wait_s=10.0) -> None` (write `path + '.tmp'`, then `os.replace`; on `PermissionError`, retry `retries` times with `wait_s` waits, then re-raise)
  - `atomic_write_csv(df, path, **to_csv_kwargs)`, `atomic_savetxt(arr, path, **kw)`
  - `COST_COLUMNS = ['episode', 'step', 'seed', 'wall_s', 'rollout_s', 'update_s', 'save_s', 'peak_private_gb', 'peak_working_set_gb', 'ckpt_data_mb', 'ckpt_meta_mb', 'lost']`
  - `append_cost_row(path, row: dict) -> None` (atomic rewrite of the existing text plus one line; header only when new; `lost` defaults to 0)
  - `class MemorySampler(threading.Thread)` (daemon, 1 s interval, ctypes `PROCESS_MEMORY_COUNTERS_EX` with explicit `restype` / `argtypes`): `take_episode_peaks() -> tuple[float, float]` returns (private GB, working set GB) and resets the maxima
- Produces, in `utils.py`:
  - `init_dir(base_dir, pathes=['log', 'data', 'model'])` uses `os.makedirs(exist_ok=True)`
  - `init_log(log_dir)` uses `basicConfig(..., force=True)`
  - `Trainer(env, model, problem, global_counter, summary_writer, run_test, output_path=None, store_dir=None, strict_save=False, sampler=None)` exposes `last_saved_step: int`

- [ ] **Step 1: Write the failing tests:**
  - **Locked file:** `atomic_write_text` onto a file held open by another handle with an exclusive share (via `msvcrt.locking`, or by holding `path + '.tmp'` open) retries, then raises `PermissionError` after the configured attempts. With `retries=2, wait_s=0.01` it finishes in under 1 s.
  - **Header once:** `append_cost_row` called twice gives one header plus two rows in `COST_COLUMNS` order, with `lost` written as 0.
  - **Memory sampler:** `MemorySampler` started for 2.5 s gives `take_episode_peaks()` > 0 for both values, and a second call reflects only the time since the first.
  - **Episode order:** a stub Trainer episode records (CSV write, save, cost row, log line) in that order. Use a fake model whose `save` records a call, and capture logging.
- [ ] **Step 2: Run** `python -m unittest tests.test_run_files -v` and confirm FAIL.
- [ ] **Step 3: Implement.**
  - **Per-episode order in `Trainer.run`:**
    1. `train_reward.csv` and `agent_reward.csv` (atomic);
    2. `model.save(store_dir + 'model/', global_step)`;
    3. `append_cost_row(store_dir + 'episode_cost.csv', …)`;
    4. `logging.info('Checkpoint saved at step %d -> %s' % (global_step, store_dir + 'model/'))`.
  - **Episode log dict:** add `'seed': self.env.episode_seed`.
  - **Strict save:** when `strict_save` is True, a save exception is re-raised; otherwise the old warning is kept.
  - **Remove** the post-loop CSV rewrite (`utils.py:431-436`) and every `output_path.replace('data', ...)`. `store_dir` is required, and `main.py` passes `'Malta/signal_route/'` for RQ1.
  - **Timers:** time the rollout (`explore`), the update (`backward`) and the save with `time.perf_counter()`.
- [ ] **Step 4: Run the tests and confirm PASS.**
- [ ] **Step 5: Commit** with the message "Write run files atomically and log per-episode cost".

### Task 10: `main.py` training path

**Files:**
- Modify: `main.py:44-50` (`init_env`), `:62-148` (`train`), `:307-340` (globals and `__main__`)

**Interfaces:**
- Consumes:
  - `run_config.parse_args`, `assert_not_rq1`, `prepare_store`, `other_run_pids` (Task 8)
  - `Trainer(..., store_dir, strict_save, sampler)`, `MemorySampler` (Task 9)
  - `load(checkpoint=...)`, the Adam check (Task 7)
  - `env.neighbour_hash()`, `env.episode_seed` (Tasks 4–5)
- Produces:
  - `init_env(config, port=0, naive_policy=False, output_path='', generate_traffic=True)`
  - `train(spec: RunSpec) -> None`, which prints `>>> TRAINING COMPLETE step <N>` as its last line
  - `main(argv=None) -> int`

- [ ] **Step 1: Write the failing test** in `tests/test_main_cli.py`:
  - **Refusal:** `python main.py` with no arguments exits non-zero with "refusing: pass --run <name> or --rq1" (run as a subprocess).
  - **One instance:** `main(['--run', 'x'])` while another `main.py --run` python is alive (a dummy subprocess with that command line) exits non-zero.
- [ ] **Step 2: Run** `python -m unittest tests.test_main_cli -v` and confirm FAIL.
- [ ] **Step 3: Implement.**
  - **With `--run`, `train(spec)`:**
    1. Create `model/`, `logs/` and `sim/` under `spec.store_dir`, and `init_log(store + 'logs/')`.
    2. Build the env with `output_path = store + 'sim/'`, `port = 0` and `generate_traffic = False`.
    3. Call `prepare_store(spec, env.neighbour_hash())`.
    4. Build the model, then `model.load(store + 'model/')`. On resume, set `env.seed = seed + resumed_step // int(env.T)` when `seed_continue` is true; keep `model.lr_scheduler.n = resumed_step`.
    5. Run the Trainer with `strict_save = True` and the sampler started.
    6. Skip the final save when `trainer.last_saved_step == total_step`.
    7. Print the completion line.
  - **`nothing to do` path:** when the checkpoint is already at `total_step`, also print `>>> TRAINING COMPLETE step <N>` and return 0.
  - **No more copies:** drop `init_dir('Malta')` and `copy_file(..., 'Malta/data')`.
  - **The RQ1 path (`--rq1`)** keeps today's behaviour, with `store_dir = 'Malta/signal_route/'`.
- [ ] **Step 4: Run** `python -m unittest tests.test_main_cli tests.test_run_config tests.test_run_files -v` and confirm PASS.
- [ ] **Step 5: Commit** with the message "Train RQ2 runs from per-run stores with --run".

### Task 11: `main.py` evaluation sweep

**Files:**
- Modify: `main.py:151-202, 284-304` (replace threaded `evaluate` / `evaluate_fn` for `--run`; keep the RQ1 path behind `--rq1`)
- Modify: `utils.py:453-476` (`Evaluator.run`: honour a per-level `output_path`)

**Interfaces:**
- Consumes (Task 1): `bg_period`, `bg_tag`, `expected_corridor_total`, `read_corridor_total`; (Task 7): `load(..., checkpoint=total_step)`
- Produces:
  - `level_dir(store: str, level: float) -> str` = `store + 'eva_data/f%.2f/' % level`
  - `evaluate_sweep(spec: RunSpec) -> None`, which runs in the main thread and exits non-zero on any error
  - **Log lines (exact):**
    - `EVAL_LEVEL_START f%.2f bg_tag=p%s corridor_total=%g corridor_expected=%g`
    - `EVAL_LEVEL_DONE f%.2f seeds=%d`
    - `>>> EVALUATE COMPLETE levels=%d`

- [ ] **Step 1: Write the failing tests:**
  - `level_dir('Malta/runs/x/', 1.1) == 'Malta/runs/x/eva_data/f1.10/'`;
  - with `--levels 2.0` the sweep runs only f2.00 (test the level-list resolution as a pure function `resolve_levels(spec, cfg) -> list[float]`);
  - a missing `config.ini` in the store raises SystemExit.
- [ ] **Step 2: Run the tests and confirm FAIL.**
- [ ] **Step 3: Implement.**
  - **Setup:**
    - read every setting from `<store>/config.ini`;
    - `assert_not_rq1`;
    - build the env once (factor 1.0, `output_path = store + 'logs/'`, port 0, `generate_traffic = False`);
    - build the eval MA2C (`total_step = 0`);
    - `load(store + 'model/', checkpoint=total_step)`; a False return raises SystemExit(5).
  - **For each level:**
    1. Move any existing level dir to `eva_data/_failed_<YYYYmmdd-HHMMSS>/`.
    2. Set `env.demand_factor = level` and `env.cur_episode = 0`.
    3. Call `env.init_data(True, level_dir)` before the first reset.
    4. Set `evaluator.output_path = level_dir`.
    5. Run the test seeds.
    6. After the first seed's reset, read the corridor total back from `Malta/data/exp_0.rou.xml`.
    7. Write `output_data`.
    8. Log the START and DONE lines.
  - **End:** log `EVALUATE COMPLETE` and return 0. Any exception propagates, so the exit is non-zero.
- [ ] **Step 4: Run the tests and confirm PASS.**
- [ ] **Step 5: Commit** with the message "Evaluate RQ2 runs in-process across demand levels".

### Task 12: `tools/queue_lib.ps1` (runner decisions, tested)

**Files:**
- Create: `tools/queue_lib.ps1`, `tests/test_queue_lib.ps1`

**Interfaces:**
- Produces (dot-sourced by `run_queue.ps1`):
  - `Write-Utf8NoBom($Path, $Text)`, `Add-Utf8NoBomLine($Path, $Line)` (absolute paths)
  - `Read-RunConfig($IniPath) -> [pscustomobject]@{ TotalStep; T; TestSeeds; EvalLevels }`. T = ceil(`episode_length_sec` / `control_interval_sec`); the reader trims around `=` and skips `;` and `#` lines.
  - `Get-QueueRuns($QueueFile) -> string[]` (skips blank and `#` lines)
  - `Get-GoodCheckpoint($Store) -> int` (-1 if none). Newest step with all three files and an exact `Checkpoint saved at step <N> -> <store>model/` line in `<store>logs/launch_r*.log`.
  - `Test-TrainingComplete($Store, $Cfg) -> [pscustomobject]@{ Ok; Reason }`, covering the five §9 conditions. The exit code comes from the newest `launch_rN.exit`.
  - `Get-RunStage($Store, $Cfg) -> 'DONE' | 'FAILED' | 'EVALUATE' | 'TRAIN'`
  - `Invoke-Recovery($Store, $Cfg, $QueueDir) -> 'RESUME' | 'SCRATCH' | 'NEEDS_REVIEW:<reason>'`. It only moves files, to `$QueueDir\quarantine_<stamp>\` and `$QueueDir\recovery_<stamp>\`.
  - `Test-EvalAttempt($Store, $Cfg, $LogPath, $LaunchTime) -> [pscustomobject]@{ Ok; FailedLevels; Reason }`, covering every §8 check and comparing `corridor_total` with `corridor_expected`.
  - `Read-QueueState($QueueDir)`, `Save-QueueState($QueueDir, $State)` (temp file then `Move-Item -Force`)
  - `Update-LaunchCounters($State, $Run, $GoodBefore, $GoodAfter, $Interrupted) -> 'OK' | 'STUCK'`

- [ ] **Step 1: Write `tests/test_queue_lib.ps1`.** It uses a fake store under `$env:TEMP`, plain asserts (`if (-not (...)) { throw '...' }`) and `exit 0` on success. Cases:
  - `Read-RunConfig` on a smoke-style ini gives `TotalStep = 288`, `T = 144` and `EvalLevels = @(1.0, 1.1)`;
  - **Good checkpoint:** a store with checkpoint-144 (three files plus its log line) and a checkpoint-288 `.index` plus a `.data...tempstate123` (no log line) gives `Get-GoodCheckpoint = 144`;
  - **Recovery:** `Invoke-Recovery` moves the 288 files to quarantine, trims `train_reward.csv` from 2 rows to 1, and appends a `lost = 1` cost row when `episode_cost.csv` is exactly one row short; it returns `RESUME`;
  - an empty `model/` with an empty `train_reward.csv` gives `SCRATCH`;
  - checkpoint files without any log line give `NEEDS_REVIEW:`;
  - **Eval check:** a fake eval log with two START/DONE blocks, `test 0`–`test 1` in each, matching bg tags and totals, plus new CSV/XML files gives `Ok`; an `EVAL_LEVEL_START f1.10 ... corridor_total=10000 corridor_expected=10738` line gives `FailedLevels = 1.10`;
  - **Counters:** three consecutive no-progress launches give `STUCK`; an interrupted launch counts only toward the total of 12;
  - **No BOM:** a file written by `Write-Utf8NoBom` has no BOM (first bytes ≠ EF BB BF).
- [ ] **Step 2: Run** `powershell -NoProfile -ExecutionPolicy Bypass -File tests/test_queue_lib.ps1`. Expected: it fails, because the functions are not defined.
- [ ] **Step 3: Implement `tools/queue_lib.ps1`.** Reuse the patterns from `claude_context/supervisor/train_supervisor.ps1`: `Get-SavedSteps`, `Move-UnsafeCheckpoints` and `Repair-RewardCsvs` (copied, not referenced; that folder is gitignored). Logs are read with `FileShare.ReadWrite`.
- [ ] **Step 4: Run the test script again.** Expected: exit 0, and it prints `ALL QUEUE_LIB TESTS PASSED`.
- [ ] **Step 5: Commit** with the message "Add tested queue-runner decision library".

### Task 13: `tools/run_queue.ps1` (main loop)

**Files:**
- Create: `tools/run_queue.ps1`

**Interfaces:**
- Consumes (Task 12): all of `queue_lib.ps1`
- Produces: `run_queue.ps1 [-QueueFile config/queue.txt] [-ProjectRoot <abs>] [-PythonExe 'C:\Program Files\Python311\python.exe'] [-DryRun]`. With `-DryRun` it prints its decision for each run and the preflight results, then exits without launching anything.

- [ ] **Step 1: Implement**, following §9:
  - the named mutex `Global\TORRS_queue_runner` (WaitOne; `AbandonedMutexException` counts as ownership);
  - adopt a live `python.exe` matching `main\.py\s+--run\s+<name>(\s|$)`;
  - the environment check;
  - the preflight: STOP, machine free (CommandLine `$null` counts as busy), commit headroom ≥ the smoke peak + 8 GB (from `state.json`, default 60 GB), free space on E: ≥ 60 GB and on C: ≥ 10 GB, and `gen_traffic.py --check-only`;
  - the launch command exactly as in §9 (Win32_Process Create, cwd = project root), with logs `<store>logs/launch_rN.log` / `eval_rN.log` and exit files `.exit`;
  - the watchdog: first checkpoint within 4 h (or the smoke-derived value), 90 min between checkpoints, and a `save` error in the log;
  - the evaluation time limit: each level gets 3 × the measured f1.00 level time, at least 90 min. Exceeding it fails that attempt.
  - the NaN check on each new checkpoint;
  - STOP handling;
  - the code-hash file per launch, with relaunches refused on change;
  - `SetThreadExecutionState` while a job runs;
  - a heartbeat in `status.txt` every poll (60 s);
  - catch-all → log, sleep 60, continue;
  - at a terminal state, disable the task's triggers (`Disable-ScheduledTask`), then exit.
- [ ] **Step 2: Run a dry run.**
  Run: `powershell -NoProfile -ExecutionPolicy Bypass -File tools/run_queue.ps1 -QueueFile config/queue_smoke.txt -DryRun` (after Task 14 creates the queue files).
  Expected: it prints `_smoke_hybrid: TRAIN (no checkpoint)` and the same for the other runs, and every preflight line shows PASS or a named WAIT reason. Nothing is launched.
- [ ] **Step 3: Commit** with the message "Add queue runner main loop".

### Task 14: Run configs, queue files and the task registration script

**Files:**
- Create:
  - `config/runs/cloud_base.ini`, `fog_base.ini`, `hybrid_base.ini`
  - `config/runs/_smoke_cloud.ini`, `_smoke_fog.ini`, `_smoke_hybrid.ini`
  - `config/queue.txt` (`cloud_base`, `fog_base`, `hybrid_base`, one per line)
  - `config/queue_smoke.txt` (`_smoke_hybrid`, `_smoke_fog`, `_smoke_cloud`)
  - `tools/register_queue_task.ps1`
  - `tests/test_run_configs.py`

**Interfaces:**
- Produces: `register_queue_task.ps1 -TaskName 'TORRS training queue' -QueueFile config/queue.txt [-Unregister]`, using the §9 trigger, principal, settings and action.

- [ ] **Step 1: Write the failing test** `tests/test_run_configs.py`. Every `config/runs/*.ini` must differ from `config_ma2c_Malta.ini` in exactly the §4 RQ2 keys:
  - `comm_arch`, `reward_share`, `seed_continue`, `eval_levels`, `norm_summary_wait`, `coop_gamma_sr`, `save_optimizer`;
  - for smoke configs, also `episode_length_sec = 720`, `total_step = 288`, `test_seeds = 10000,20000` and `eval_levels = 1.00,1.10`.

  `total_step` must be a multiple of T (Trainer requirement).
- [ ] **Step 2: Run** `python -m unittest tests.test_run_configs -v` and confirm FAIL.
- [ ] **Step 3: Create the configs** (a copy of the RQ1 ini plus the keys).
- [ ] **Step 4: Run the test and confirm PASS.**
- [ ] **Step 5: Write `register_queue_task.ps1`.**
  - If S4U registration is refused without elevation, the script prints "run this script in an elevated PowerShell" and exits 1; the author then runs it.
  - The script registers only. Nothing starts until `Start-ScheduledTask`.
- [ ] **Step 6: Commit** with the message "Add RQ2 and smoke run configs, queue files and task registration".

### Task 15: `tools/rq1_manifest.py` (frozen RQ1 check)

**Files:**
- Create: `tools/rq1_manifest.py`, `tests/test_rq1_manifest.py`

**Interfaces:**
- Produces:
  - `python tools/rq1_manifest.py record`, which writes `Malta/runs/_queue/rq1_manifest.csv` (columns `part,path,size,mtime,sha256`; `part` ∈ frozen/allowed) and a copy in `Malta/signal_route/_rq1_final_backup/rq1_manifest.csv`. The write to `_rq1_final_backup` needs the folder's read-only attribute cleared for that one file only.
  - `python tools/rq1_manifest.py check`, which exits 0 only if every frozen row matches size, mtime and sha256.

- [ ] **Step 1: Write the failing test** on a temp tree. `record` then `check` gives 0; touching an allowed file still gives 0; touching a frozen file gives 1.
- [ ] **Step 2: Run** `python -m unittest tests.test_rq1_manifest -v` and confirm FAIL.
- [ ] **Step 3: Implement**, with the frozen and allowed path lists exactly as in §10 step 6.
- [ ] **Step 4: Run the test and confirm PASS.**
- [ ] **Step 5: Commit** with the message "Add RQ1 frozen-manifest tool".

### Task 16: Notebook and handoff sync

**Files:**
- Modify: `TORRS_MA2C_Complete.ipynb`:
  - code cells 14 (build_file), 16 (env), 18 (Malta_env), 24 (models), 26 (utils), 28 (main);
  - markdown cells 0 and 27;
  - new cells for `run_config.py`, `run_files.py`, `envs/zones.py`, `config/runs/*.ini` and `tools/*.py`.
- Modify: `HANDOFF.md` (the RQ2 queue, `--run`, STOP, and the scheduled task)

- [ ] **Step 1: Re-sync each changed `.py` into its cell.**
  - Strip CR and keep the notebook-safe lines (cell 14 `DATA_DIR`, cell 18 `_CACHE_FILE` using `os.getcwd()`).
  - Clear outputs and `execution_count` on changed cells.
  - Update cells 0 and 27 to document `--run`, `--option`, `--levels` and `--rq1`, and to say "do not run this notebook while the queue exists".
- [ ] **Step 2: Verify the sync.** Run a script that compares each mirrored cell's source with its file, ignoring CR and the two documented lines. Expected: 0 differences.
- [ ] **Step 3: Update `HANDOFF.md`:** where RQ2 runs live, how to stop and resume the queue, and how to read status.
- [ ] **Step 4: Run the whole Python suite.**
  Run: `python -m unittest discover -s tests -t . -v`
  Expected: all PASS.
- [ ] **Step 5: Commit** with the message "Sync notebook and handoff for the RQ2 queue".

### Task 17: Smoke queue, author gates and launch (operational)

**Files:** none new. This task produces only results under `Malta/runs/_queue/` (gitignored) and the smoke folders, which move to the scratchpad afterwards.

- [ ] **Step 1: Run the optional RQ1 jobs only if the author asks** (§10 step 4): `python baselines.py actuated_rerouting`, then `python ablation_routing.py`, one at a time, with nothing else running.
- [ ] **Step 2: Re-record the manifest:** `python tools/rq1_manifest.py record`, then `check`. Expected: exit 0.
- [ ] **Step 3: Run the smoke queue** through a temporary task: `tools/register_queue_task.ps1 -TaskName 'TORRS smoke queue' -QueueFile config/queue_smoke.txt`, then `Start-ScheduledTask 'TORRS smoke queue'`.
  - **Fault test:** during `_smoke_hybrid` episode 2, kill its python (`taskkill /PID <python> /T /F`).
  - **Second copy:** start a second runner copy by hand and confirm it blocks.
  - **STOP:** create STOP during `_smoke_fog`; confirm a clean stop after its checkpoint. Delete STOP and restart the task.
  - **Expected:**
    - all three smoke stores reach `DONE` with `eva_data/EVAL_OK`;
    - recovery resumed `_smoke_hybrid` from checkpoint-144 with seed 13 in `train_reward.csv` row 2;
    - `tf.train.list_variables` on `_smoke_hybrid/model/checkpoint-288` shows `/Adam` names;
    - `rq1_manifest.py check` exits 0.
- [ ] **Step 4: Record smoke metrics** in `state.json`: the peak commit, TF build time, seconds per step, `.meta` and `.data` size, and first-checkpoint time per arm. Then unregister the smoke task and move `Malta/runs/_smoke_*` to the scratchpad.
- [ ] **Step 5: Author gates.** The author confirms:
  - the fixed pagefile, with a reboot; Claude checks read-only that `TotalVirtualMemorySize` ≥ 95 GB;
  - Windows Update paused;
  - BIOS AC restore (optional);
  - approval of the district counts from Task 3;
  - no gaming, and no CSVs open in Excel.
- [ ] **Step 6: Launch.** Run `tools/register_queue_task.ps1 -TaskName 'TORRS training queue' -QueueFile config/queue.txt`, then `Start-ScheduledTask 'TORRS training queue'`. Expected: `status.txt` shows `cloud_base TRAIN` within 2 min.
- [ ] **Step 7: First-checkpoint check** (§12 Live), about 1.5 h in:
  - `rq1_manifest.py check` exits 0;
  - only `Malta/runs/cloud_base/`, `Malta/runs/_queue/` and `Malta/data/exp_0.*` / `ild_out_0.xml` changed;
  - `.meta` ≤ 1.5 GB and `.data` ≤ 1.5 GB;
  - `episode_cost.csv` has a valid row.

  Report these to the author, and commit `config.ini`, `zones_nso.csv`, `neighbours.sha256` and the first CSV rows for `cloud_base`.
