<#
tools/run_queue.ps1 - the RQ2 training queue runner.

Spec: docs/superpowers/specs/2026-11-08-rq2-training-queue-design.md, sections 8 and 9.
Decisions come from tools/queue_lib.ps1 (Task 12); this file is the loop that uses them.
Tests: tests/test_run_queue.ps1 (pure helpers), tests/test_queue_lib.ps1 (the library).
Windows PowerShell 5.1, ASCII only.

Usage
  run_queue.ps1 [-QueueFile config/queue.txt] [-ProjectRoot <abs>]
                [-PythonExe 'C:\Program Files\Python311\python.exe']
                [-TaskName 'TORRS training queue'] [-DryRun] [-NoRun]
  The real queue is started only by its scheduled task (tools/register_queue_task.ps1).
  -QueueFile is resolved against -ProjectRoot (default: the parent of tools\).
  -DryRun prints each run's decision and the preflight results and exits 0. It launches no
   training or evaluation and writes nothing in the project; it does run the read-only
   environment check and gen_traffic.py --check-only (their output goes to %TEMP%).
  -NoRun (or dot-sourcing the file) only defines the functions, for the tests.

One pass of the loop (Invoke-RunnerIteration; every pass re-reads the queue file and state.json)
  1. A live python.exe running 'main.py --run <name>' for a queue run (or for a run whose launch
     is in flight in state.json) is adopted and watched until it ends; then its launch is
     completed. Nothing new is launched while it lives.
  2. A launch in flight in state.json whose python is gone is completed.
  3. Cleanup (no queue python is alive). STOP present -> STOPPED, exit.
  4. The queue file's runs, in order:
       DONE      skip (a missing DONE marker is written when eva_data\EVAL_OK exists);
       FAILED    stop the queue (terminal);
       EVALUATE  evaluate (only the failed levels on the one retry);
       TRAIN     Invoke-Recovery (NEEDS_REVIEW -> FAILED), then train.
     Before a launch: code hashes (a change against the run's first launch -> FAILED
     CODE_CHANGED), the preflight (a failed check -> WAITING_<reason>, re-checked every
     5 min), then the environment check (once per runner process, one retry after 60 s; a
     memory error in its output -> WAITING_MEMORY; other failures -> ENV_ERROR, terminal),
     cleanup (TCP 27500 still busy after 2 min -> WAITING_PORT_BUSY), the launch, the watch,
     the completion. Long checks rewrite status.txt every 30 s.
  5. Every run DONE -> QUEUE_DONE (terminal).
  Every pass is wrapped in a catch-all: an exception is logged, the status shows ERROR, the
  runner sleeps 60 s and tries again. Only the terminal states (QUEUE_DONE, FAILED, STUCK,
  ENV_ERROR) and STOPPED end the runner (exit 0). At a terminal state the scheduled task(s) of
  this queue are disabled as the last action (STOPPED leaves them enabled).

Files (all under Malta\runs\_queue\, UTF-8 without BOM)
  status.txt   one line, rewritten every poll: '<STATE> <detail> | heartbeat <time> | runner pid <pid>'
  queue.log    one timestamped line per event
  STOP         created by the author; never deleted by the runner
  checks\      output of the environment and traffic checks (env_check.out/.exit, ...)
  state.json   the library's schema (queue_lib.ps1 header), plus, written by the runner:
    runs.<run>.current_launch: kind, n, good_before, started, plus cmd_pid, py_pid,
                 levels (evaluate: the '--levels' strings, empty = all), last_ckpt_step,
                 last_ckpt_time
    runs.<run>.eval_launches     evaluation launches made (a cap of 6 -> FAILED EVAL_FAILED)
    runs.<run>.eval_levels_next  the failed levels the retry re-runs ('1.10' strings)
    runs.<run>.measured          recorded for the author, never used as a limit:
                 first_checkpoint_min, max_checkpoint_gap_min, peak_commit_gb (the job's
                 process tree, sampled each poll), eval_f100_min, eval_peak_commit_gb
  and read (set by hand after the smoke queue; smoke episodes are 720 s, not 3600 s, so the
  smoke times must be scaled before they are entered):
    arms.<comm_arch>.peak_commit_gb        preflight: headroom >= this + 8 GB (default 60)
    arms.<comm_arch>.first_checkpoint_min  watchdog limit for the first checkpoint, and for
                                           the first EVAL_LEVEL_START (default 240)

Per launch (logs in <store>logs\)
  launch_rN.log / .exit (training), eval_rN.log / .exit (evaluation), <kind>_rN.code.txt (code
  hashes). The command is spec section 9's, started with Win32_Process.Create in the project
  root (hidden window), so it outlives the runner:
    cmd.exe /v:on /c "set PYTHONIOENCODING=utf-8&& set PYTHONUNBUFFERED=1&& "<python>" -X
      faulthandler main.py --run <name> --option <train|evaluate> [--levels a,b] > "<log>" 2>&1
      & echo EXIT=!errorlevel! at !date! !time! > "<exit>""

Watch (one poll every 60 s; the log is read from a saved byte offset with shared access)
  training    first 'Checkpoint saved' line within the first-checkpoint limit, then at most
              90 min between them; a checkpoint-save error (a traceback through
              _persist_episode, or 'Per-episode checkpoint save failed') -> the tree is killed
              after 2 min; at each new checkpoint the last train_reward.csv row must have a
              finite reward and average waiting time, else FAILED (NAN).
  evaluation  the first EVAL_LEVEL_START within the first-checkpoint limit; then each level
              (from the previous START or DONE line) within 3 x the f1.00 level time, at
              least 90 min. Before f1.00 is measured its estimate is (test seeds) x (median
              training rollout_s from episode_cost.csv), else 360 min. Exceeding a limit
              kills the tree and fails the attempt.
  STOP        training: wait for the next checkpoint line (at most 90 min), wait 15 s, kill
              the tree, run Recovery, exit. Evaluation: kill at once; the attempt is not
              counted. Removing STOP while the runner waits cancels it.
  A FAILED marker written while a job runs kills the job (terminal).
  The PC is kept awake (SetThreadExecutionState) while a job runs.

Deviations from the spec text, on purpose
  - Cleanup kills duarouter.exe and randomTrips python under Malta\data only when they are
    orphans or children of a main.py python, never another job's (gen_traffic.py) children.
  - A training launch that exits 2 or 4 with main.py's refusal message at the end of its log
    (bad arguments; config or neighbour-hash mismatch) marks the run FAILED (REFUSED) at once:
    a relaunch cannot fix it. Exit 3 is never a refusal here (python -X faulthandler returns 3
    on any native abort), and neither is an exit 2 or 4 without the message: both are ordinary
    deaths for the counters.
  - A STOP kill counts toward the 12-launch cap but not toward the 3 launches without
    progress (it is passed to Update-LaunchCounters as an interruption).
  - An evaluation ended by a reboot or logoff is not counted as an attempt (like STOP); a cap
    of 6 evaluation launches still ends the run (EVAL_FAILED).
  - The environment check prints 'SUMO_HOME=<path>' instead of the bare path, so the line can
    be told apart from TensorFlow's import messages.
#>
param(
    [string]$QueueFile = 'config/queue.txt',
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonExe = 'C:\Program Files\Python311\python.exe',
    [string]$TaskName = 'TORRS training queue',
    [switch]$DryRun,
    [switch]$NoRun
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'queue_lib.ps1')

# ---------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------
$RunnerMutexName = 'Global\TORRS_queue_runner'
$RunnerRunNamePattern = '^[a-z][a-z0-9_]*$|^_smoke_[a-z]+$'
$RunnerPollSeconds = 60
$RunnerWaitSeconds = 300
$RunnerDefaultPeakGB = 60.0
$RunnerCommitReserveGB = 8.0
$RunnerFreeProjectGB = 60.0
$RunnerFreeSystemGB = 10.0
$RunnerFirstCheckpointMin = 240.0
$RunnerCheckpointGapMin = 90.0
$RunnerEvalLevelFactor = 3.0
$RunnerEvalLevelFloorMin = 90.0
$RunnerEvalLevelDefaultMin = 360.0
$RunnerSaveErrorGraceSec = 120
$RunnerStopGraceSec = 15
$RunnerPortWaitSec = 120
$RunnerCheckStatusSec = 30
$RunnerWrapperWaitSec = 30
$RunnerPythonFindSec = 30
$RunnerEnvTimeoutSec = 600
$RunnerTrafficTimeoutSec = 1800
$RunnerMaxEvalAttempts = 2
$RunnerMaxEvalLaunches = 6
# main.py refusals that a relaunch cannot fix: exit 2 (run_config.parse_args / main._refuse:
# 'invalid ...', 'refusing: ...', argparse 'main.py: error: ...') and exit 4 (run_config
# .prepare_store 'resume refused for ...', main 'evaluate refused for ...'). Exit 3 (the RQ1
# path) is not here: queue stores are always under Malta\runs, and 3 is also what python
# -X faulthandler returns on any native abort, which must be relaunched.
$RunnerRefusalPatterns = @{
    2 = '(?:^|\s)(?:refusing: |invalid: |invalid run name: |invalid --levels: |main\.py: error: )'
    4 = '(?:^|\s)(?:resume|evaluate) refused for '
}
$RunnerCodeFiles = @('main.py', 'utils.py', 'run_config.py', 'run_files.py', 'Malta\build_file.py')
$RunnerCodeDirs = @('agents', 'envs')
$RunnerBusyExeNames = @('sumo.exe', 'sumo-gui.exe', 'duarouter.exe', 'netconvert.exe')
$RunnerBusyPythonPattern = '(?i)(?:^|[\s"''\\/])(?<what>main\.py|ablation_routing\.py|baselines\.py|gen_traffic\.py|randomTrips\.py|build_zones\.py)(?:["''\s]|$)|(?<what>ipykernel)'
$RunnerEnvCheckArgs = '-c "import tensorflow, traci, sumolib, os; print(''SUMO_HOME='' + os.environ[''SUMO_HOME''])"'

$script:Q = $null
$script:LoggedOnce = @{}
$script:KeepAwakeWarned = $false

# ---------------------------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------------------------
function Format-Num([double]$Value, [string]$Format = '0.0') { $Value.ToString($Format, $QueueInvariant) }

function Format-Stamp([datetime]$Time) { $Time.ToString('yyyy-MM-dd HH:mm:ss', $QueueInvariant) }

function ConvertTo-OneLine([string]$Text, [int]$Max = 600) {
    if ($null -eq $Text) { return '' }
    $s = ($Text -replace '\s*\r?\n\s*', ' | ').Trim()
    if ($s.Length -gt $Max) { $s = '...' + $s.Substring($s.Length - $Max) }
    $s
}

function Test-QueueRunName([string]$Name) {
    # Spec 4: ^[a-z][a-z0-9_]*$, or ^_smoke_[a-z]+$ for smoke runs (case-sensitive).
    (-not [string]::IsNullOrEmpty($Name)) -and ($Name -cmatch $RunnerRunNamePattern)
}

function Resolve-QueueFilePath([string]$ProjectRoot, [string]$QueueFile) {
    $p = $QueueFile
    if (-not [IO.Path]::IsPathRooted($p)) { $p = Join-Path $ProjectRoot $p }
    [IO.Path]::GetFullPath($p)
}

function ConvertTo-LocalTime($Value) {
    # state.json keeps times as ISO 8601 strings (ConvertTo-Json in PS 5.1 mangles [datetime]).
    if ($null -eq $Value) { return $null }
    if ($Value -is [datetime]) { return $Value }
    $o = [DateTimeOffset]::MinValue
    if ([DateTimeOffset]::TryParse([string]$Value, $QueueInvariant, [Globalization.DateTimeStyles]::None, [ref]$o)) { return $o.LocalDateTime }
    $null
}

function Get-LogLineTime([string]$Line, [datetime]$Fallback) {
    # Python logging lines start '2026-11-16 10:00:00,123 [INFO] ...' (local time).
    $m = [regex]::Match($Line, '^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
    if ($m.Success) {
        $t = [datetime]::MinValue
        if ([datetime]::TryParseExact($m.Groups[1].Value, 'yyyy-MM-dd HH:mm:ss', $QueueInvariant, [Globalization.DateTimeStyles]::AssumeLocal, [ref]$t)) { return $t }
    }
    $Fallback
}

function ConvertTo-LevelValues($Levels) {
    # '1.10' strings -> [double[]] (invariant culture), or $null for none.
    $list = New-Object System.Collections.Generic.List[double]
    foreach ($s in @($Levels)) {
        if ($null -eq $s -or ([string]$s).Trim().Length -eq 0) { continue }
        $d = 0.0
        if (-not [double]::TryParse(([string]$s).Trim(), $QueueFloatStyle, $QueueInvariant, [ref]$d)) { throw "invalid level '$s'" }
        $list.Add($d)
    }
    if ($list.Count -eq 0) { return $null }
    , $list.ToArray()
}

function Get-StringList($Value) {
    # A state.json array (or $null) -> string[] without empty entries.
    $out = New-Object System.Collections.Generic.List[string]
    if ($null -ne $Value) {
        foreach ($v in @($Value)) { if ($null -ne $v -and ([string]$v).Length -gt 0) { $out.Add([string]$v) } }
    }
    , $out.ToArray()
}

function Get-IniValue([string]$Path, [string]$Section, [string]$Key) {
    $current = $null
    foreach ($raw in ((Read-SharedText $Path) -split "`r?`n")) {
        $line = $raw.Trim()
        if ($line.Length -eq 0 -or $line.StartsWith(';') -or $line.StartsWith('#')) { continue }
        if ($line -match '^\[(.+)\]$') { $current = $Matches[1].Trim(); continue }
        if ($current -ne $Section) { continue }
        $eq = $line.IndexOf('=')
        if ($eq -lt 1) { continue }
        if ($line.Substring(0, $eq).Trim() -eq $Key) { return $line.Substring($eq + 1).Trim() }
    }
    $null
}

# ---------------------------------------------------------------------------------------------
# The launch command (spec 9)
# ---------------------------------------------------------------------------------------------
function Assert-CmdSafe([string]$Text, [string]$What) {
    # Inside cmd /v:on /c "..." a double quote ends the quoting, % and ! expand, and ^ escapes.
    if ([string]::IsNullOrEmpty($Text)) { throw "$What is empty" }
    if ($Text -match '["!%^]') { throw ($What + ' contains a character cmd.exe cannot quote safely (one of " ! % ^): ' + $Text) }
}

function New-LaunchArguments([string]$RunName, [string]$Option, [string[]]$Levels) {
    # The python arguments of a queue launch: -X faulthandler main.py --run <name> --option <o> [--levels a,b]
    if (-not (Test-QueueRunName $RunName)) { throw "invalid run name '$RunName'" }
    if ($Option -cne 'train' -and $Option -cne 'evaluate') { throw "invalid option '$Option'" }
    $a = '-X faulthandler main.py --run ' + $RunName + ' --option ' + $Option
    $list = Get-StringList $Levels      # not @(): the comma-returned array would nest
    if ($list.Count -gt 0) {
        if ($Option -cne 'evaluate') { throw '--levels is only for evaluate' }
        foreach ($l in $list) { if ($l -notmatch '^\d+\.\d\d$') { throw "invalid level '$l'" } }
        $a += ' --levels ' + ($list -join ',')
    }
    $a
}

function New-DetachedCommandLine([string]$PythonExe, [string]$PythonArgs, [string]$OutPath, [string]$ExitPath) {
    # The wrapper sets the two variables itself: a WMI-created process does not inherit the
    # runner's environment. The exit code goes to <exit> after python ends, whatever its code.
    Assert-CmdSafe $PythonExe 'the python path'
    Assert-CmdSafe $OutPath 'the log path'
    Assert-CmdSafe $ExitPath 'the exit-file path'
    if ($PythonArgs -match '[!%^]') { throw "python arguments contain ! % or ^: $PythonArgs" }
    'cmd.exe /v:on /c "set PYTHONIOENCODING=utf-8&& set PYTHONUNBUFFERED=1&& "' + $PythonExe + '" ' + $PythonArgs +
        ' > "' + $OutPath + '" 2>&1 & echo EXIT=!errorlevel! at !date! !time! > "' + $ExitPath + '""'
}

# ---------------------------------------------------------------------------------------------
# Verdicts (pure)
# ---------------------------------------------------------------------------------------------
function Get-BusyReasons($Processes, $ExcludeIds) {
    # Spec 9 'the machine is free'. Processes: objects with ProcessId, Name, CommandLine.
    # A python whose command line cannot be read counts as busy.
    foreach ($p in @($Processes)) {
        if ($null -eq $p) { continue }
        $id = [int]$p.ProcessId
        if ($null -ne $ExcludeIds -and $ExcludeIds.Contains($id)) { continue }
        $name = ([string]$p.Name).ToLowerInvariant()
        if ($name -eq 'python.exe' -or $name -eq 'pythonw.exe') {
            $cl = [string]$p.CommandLine
            if ([string]::IsNullOrEmpty($cl)) { "$name pid $id (command line not readable)"; continue }
            $m = [regex]::Match($cl, $RunnerBusyPythonPattern)
            if ($m.Success) { "$name pid $id runs $($m.Groups['what'].Value)" }
        } elseif ($RunnerBusyExeNames -contains $name) {
            "$name pid $id"
        }
    }
}

function Get-CommitVerdict([double]$LimitBytes, [double]$CommittedBytes, [double]$PeakGB, [double]$ReserveGB) {
    $head = ($LimitBytes - $CommittedBytes) / 1GB
    $need = $PeakGB + $ReserveGB
    $detail = 'commit headroom {0} GB (limit {1} GB), need {2} GB = peak {3} + {4}' -f (Format-Num $head), (Format-Num ($LimitBytes / 1GB)), (Format-Num $need), (Format-Num $PeakGB), (Format-Num $ReserveGB '0')
    [pscustomobject]@{ Ok = ($head -ge $need); Detail = $detail }
}

function Get-DiskVerdict([string]$Drive, [double]$FreeBytes, [double]$NeedGB) {
    $free = $FreeBytes / 1GB
    [pscustomobject]@{ Ok = ($free -ge $NeedGB); Detail = ('{0} {1} GB free, need {2} GB' -f $Drive, (Format-Num $free), (Format-Num $NeedGB '0')) }
}

function Get-TrainWatchdogReason {
    param([datetime]$Now, [datetime]$Started, $LastCheckpoint, [double]$FirstLimitMin, [double]$GapMin, $SaveErrorAt, [int]$SaveErrorGraceSec = 120)
    # -> $null, or why the launch is treated as a death (spec 9 watchdog).
    if ($null -ne $SaveErrorAt -and ($Now - [datetime]$SaveErrorAt).TotalSeconds -ge $SaveErrorGraceSec) {
        return 'a checkpoint-save error in the log'
    }
    if ($null -eq $LastCheckpoint) {
        $m = ($Now - $Started).TotalMinutes
        if ($m -gt $FirstLimitMin) { return ('no Checkpoint saved line {0} min after the launch (limit {1} min)' -f (Format-Num $m '0.#'), (Format-Num $FirstLimitMin '0.##')) }
        return $null
    }
    $g = ($Now - [datetime]$LastCheckpoint).TotalMinutes
    if ($g -gt $GapMin) { return ('no Checkpoint saved line for {0} min (limit {1} min)' -f (Format-Num $g '0'), (Format-Num $GapMin '0')) }
    $null
}

function Get-EvalLevelLimitMinutes($F100Minutes, [double[]]$RolloutSeconds, [int]$Seeds) {
    # Spec 8: 3 x the measured f1.00 level time, at least 90 min. Before f1.00 is measured:
    # 3 x (seeds x median training rollout), at least 90 min; with no data 360 min.
    $base = $null
    if ($null -ne $F100Minutes -and [double]$F100Minutes -gt 0) { $base = [double]$F100Minutes }
    elseif ($null -ne $RolloutSeconds -and $RolloutSeconds.Count -gt 0 -and $Seeds -gt 0) {
        $sorted = [double[]]($RolloutSeconds | Sort-Object)
        $k = $sorted.Count
        if ($k % 2 -eq 1) { $median = $sorted[($k - 1) / 2] } else { $median = ($sorted[$k / 2 - 1] + $sorted[$k / 2]) / 2.0 }
        if ($median -gt 0) { $base = $Seeds * $median / 60.0 }
    }
    if ($null -eq $base) { return $RunnerEvalLevelDefaultMin }
    [Math]::Max($RunnerEvalLevelFloorMin, $RunnerEvalLevelFactor * $base)
}

function Get-EvalTimeoutReason([datetime]$Now, [datetime]$PhaseStart, [bool]$LevelSeen, [string]$Level, [double]$PreLimitMin, [double]$LevelLimitMin) {
    # Phase = since the launch (no level yet) or since the last EVAL_LEVEL_START/DONE line.
    $m = ($Now - $PhaseStart).TotalMinutes
    if (-not $LevelSeen) {
        if ($m -gt $PreLimitMin) { return ('no EVAL_LEVEL_START line {0} min after the launch (limit {1} min)' -f (Format-Num $m '0.#'), (Format-Num $PreLimitMin '0.##')) }
        return $null
    }
    if ($m -gt $LevelLimitMin) {
        $what = 'the level after the last EVAL_LEVEL_DONE'
        if ($Level) { $what = 'level f' + $Level }
        return ('{0} exceeded its time limit ({1} min > {2} min)' -f $what, (Format-Num $m '0'), (Format-Num $LevelLimitMin '0'))
    }
    $null
}

function Get-TrafficCheckArgs($Cfg) {
    # gen_traffic.py --check-only arguments for the files a run needs: its training seeds
    # (seed .. seed + episodes - 1) at 1.0, and its test seeds at every eval level.
    $episodes = [int][Math]::Ceiling([double]$Cfg.TotalStep / [double]$Cfg.T)
    if ($episodes -lt 1) { $episodes = 1 }
    $first = [int]$Cfg.Seed
    $levels = (@($Cfg.EvalLevels) | ForEach-Object { Format-Level $_ }) -join ','
    [pscustomobject]@{
        Train = 'tools/gen_traffic.py --seeds {0}-{1} --factors 1.0 --check-only' -f $first, ($first + $episodes - 1)
        Test  = 'tools/gen_traffic.py --seeds {0} --factors {1} --check-only' -f (@($Cfg.TestSeeds) -join ','), $levels
    }
}

function Test-RefusalExit($ExitCode, [string]$LogTail) {
    # Whether a launch ended in a main.py refusal: exit 2 or 4 AND, among the last 5 non-empty
    # lines of its log, main.py's refusal message for that code. Anything else (exit 3 from a
    # native abort, an exit 2 or 4 without the message) is an ordinary death for the counters.
    if ($null -eq $ExitCode) { return $false }
    $code = [long]$ExitCode
    if (-not ($code -eq 2 -or $code -eq 4)) { return $false }
    $pattern = $RunnerRefusalPatterns[[int]$code]
    $lines = @(([string]$LogTail -split "`r?`n") | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -Last 5)
    foreach ($l in $lines) { if ($l -match $pattern) { return $true } }
    $false
}

function Get-ArmSetting($State, [string]$Arm, [string]$Key, [double]$Default) {
    # state.json arms.<arm>.<key> (a positive number), else the default.
    if ($null -eq $State -or -not $State.Contains('arms') -or $null -eq $State['arms']) { return $Default }
    $arms = $State['arms']
    if (-not $arms.Contains($Arm) -or $null -eq $arms[$Arm]) { return $Default }
    $a = $arms[$Arm]
    if (-not $a.Contains($Key) -or $null -eq $a[$Key]) { return $Default }
    $v = 0.0
    if ([double]::TryParse([string]$a[$Key], $QueueFloatStyle, $QueueInvariant, [ref]$v) -and $v -gt 0 -and -not [double]::IsInfinity($v)) { return $v }
    $Default
}

function Compare-CodeHashLines([string[]]$First, [string[]]$Current) {
    # '<sha256>  <label>' lines -> the labels whose hash differs, appears or disappears.
    $a = @{}; $b = @{}
    foreach ($l in @($First)) { $m = [regex]::Match([string]$l, '^(\S+)\s+(.+?)\s*$'); if ($m.Success) { $a[$m.Groups[2].Value] = $m.Groups[1].Value } }
    foreach ($l in @($Current)) { $m = [regex]::Match([string]$l, '^(\S+)\s+(.+?)\s*$'); if ($m.Success) { $b[$m.Groups[2].Value] = $m.Groups[1].Value } }
    $labels = @($a.Keys) + @($b.Keys) | Sort-Object -Unique
    foreach ($k in $labels) {
        if (-not $a.ContainsKey($k)) { "$k (new)"; continue }
        if (-not $b.ContainsKey($k)) { "$k (gone)"; continue }
        if ($a[$k] -ne $b[$k]) { $k }
    }
}

function Test-TaskRunsQueue($Task, [string]$QueueFileFull, [string]$DefaultRoot) {
    # Whether a scheduled task's action runs run_queue.ps1 with this queue file (its -QueueFile,
    # default config/queue.txt, resolved against its -ProjectRoot or working directory).
    foreach ($a in @($Task.Actions)) {
        if ($null -eq $a) { continue }
        $argText = [string]$a.Arguments
        if ($argText.IndexOf('run_queue.ps1', [StringComparison]::OrdinalIgnoreCase) -lt 0) { continue }
        $qf = 'config/queue.txt'
        $m = [regex]::Match($argText, '-QueueFile\s+(?:"([^"]*)"|(\S+))', 'IgnoreCase')
        if ($m.Success) { $qf = $m.Groups[1].Value + $m.Groups[2].Value }
        $root = $DefaultRoot
        $m = [regex]::Match($argText, '-ProjectRoot\s+(?:"([^"]*)"|(\S+))', 'IgnoreCase')
        if ($m.Success) { $root = $m.Groups[1].Value + $m.Groups[2].Value }
        elseif (-not [string]::IsNullOrEmpty([string]$a.WorkingDirectory)) { $root = [string]$a.WorkingDirectory }
        try { $full = Resolve-QueueFilePath $root $qf } catch { continue }
        if ([string]::Equals($full, $QueueFileFull, [StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    $false
}

# ---------------------------------------------------------------------------------------------
# Context, queue log and status
# ---------------------------------------------------------------------------------------------
function Initialize-RunnerContext {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [string]$QueueFile = 'config/queue.txt',
        [string]$PythonExe = 'C:\Program Files\Python311\python.exe',
        [string]$TaskName = 'TORRS training queue',
        [bool]$TaskNameGiven = $false,
        [bool]$DryRun = $false,
        [string]$ChecksDir = $null
    )
    Assert-AbsolutePath $ProjectRoot 'ProjectRoot'
    $root = [IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\')
    $queueDir = Join-Path $root 'Malta\runs\_queue'
    if (-not $ChecksDir) { $ChecksDir = Join-Path $queueDir 'checks' }
    $script:Q = [pscustomobject]@{
        Root          = $root
        QueueFileArg  = $QueueFile
        QueueFile     = (Resolve-QueueFilePath $root $QueueFile)
        Python        = $PythonExe
        TaskName      = $TaskName
        TaskNameGiven = $TaskNameGiven
        DryRun        = $DryRun
        QueueDir      = $queueDir
        StatusFile    = (Join-Path $queueDir 'status.txt')
        LogFile       = (Join-Path $queueDir 'queue.log')
        StopFile      = (Join-Path $queueDir 'STOP')
        ChecksDir     = $ChecksDir
        RunsDir       = (Join-Path $root 'Malta\runs')
        Rq1Dir        = (Join-Path $root 'Malta\signal_route')
        EnvOk         = $false
    }
}

function Write-QueueLog([string]$Message) {
    $line = (Format-Stamp (Get-Date)) + '  ' + (ConvertTo-OneLine $Message 4000)
    if ($null -ne $Q -and -not $Q.DryRun) {
        try { Add-Utf8NoBomLine $Q.LogFile $line } catch { }
    }
    Write-Host $line
}

function Write-QueueLogOnce([string]$Key, [string]$Message, [string]$Compare = $null) {
    # Logs a message only when it (or -Compare) differs from the last one under this key, so
    # a WAIT loop logs its reason once, not every 5 min.
    if (-not $Compare) { $Compare = $Message }
    if ($script:LoggedOnce.ContainsKey($Key) -and $script:LoggedOnce[$Key] -ceq $Compare) { return }
    $script:LoggedOnce[$Key] = $Compare
    Write-QueueLog $Message
}

function Write-RunnerStatus([string]$State, [string]$Detail = '') {
    $line = $State
    if ($Detail) { $line += ' ' + (ConvertTo-OneLine $Detail 700) }
    $line += ' | heartbeat ' + (Format-Stamp (Get-Date)) + ' | runner pid ' + $PID
    if ($null -eq $Q -or $Q.DryRun) { return }
    try { Write-Utf8NoBom $Q.StatusFile ($line + "`r`n") } catch { }
}

function Format-RunnerError($ErrorRecord) {
    $msg = $ErrorRecord.Exception.Message
    $at = ''
    if ($null -ne $ErrorRecord.InvocationInfo) { $at = $ErrorRecord.InvocationInfo.PositionMessage }
    ConvertTo-OneLine ($msg + ' ' + $at + ' ' + $ErrorRecord.ScriptStackTrace) 2000
}

function Set-KeepAwake([bool]$On) {
    # SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) while a job runs.
    try {
        if (-not ('TorrsQueue.Power' -as [type])) {
            Add-Type -Namespace TorrsQueue -Name Power -MemberDefinition '[System.Runtime.InteropServices.DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
        }
        $flags = [uint32]2147483648                 # ES_CONTINUOUS
        if ($On) { $flags = [uint32]2147483649 }    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        [void][TorrsQueue.Power]::SetThreadExecutionState($flags)
    } catch {
        if (-not $script:KeepAwakeWarned) { $script:KeepAwakeWarned = $true; Write-QueueLog ('keep-awake call failed: ' + $_.Exception.Message) }
    }
}

# ---------------------------------------------------------------------------------------------
# Runs, stores and markers
# ---------------------------------------------------------------------------------------------
function Assert-QueueStore([string]$RunName) {
    # The run's store, Malta\runs\<name>, refusing names that do not match spec 4 and any path
    # under Malta\signal_route (the RQ1 run) or outside Malta\runs.
    if (-not (Test-QueueRunName $RunName)) { throw "refusing run name '$RunName' (it must match $RunnerRunNamePattern)" }
    $store = [IO.Path]::GetFullPath((Get-StoreDir $Q.Root $RunName)).TrimEnd('\')
    $rq1 = [IO.Path]::GetFullPath($Q.Rq1Dir).TrimEnd('\') + '\'
    $runs = [IO.Path]::GetFullPath($Q.RunsDir).TrimEnd('\') + '\'
    if (($store + '\').StartsWith($rq1, [StringComparison]::OrdinalIgnoreCase)) { throw "refusing a store under Malta\signal_route: $store" }
    if (-not ($store + '\').StartsWith($runs, [StringComparison]::OrdinalIgnoreCase)) { throw "refusing a store outside Malta\runs: $store" }
    foreach ($d in $Q.RunsDir, $store) {
        if ([IO.Directory]::Exists($d) -and (([IO.File]::GetAttributes($d) -band [IO.FileAttributes]::ReparsePoint) -ne 0)) {
            throw "refusing a store reached through a junction or link: $d"
        }
    }
    $store
}

function Get-RunIniPath([string]$RunName) {
    # The runner reads <store>config.ini (spec 9); before the first launch, config\runs\<name>.ini.
    $store = Join-Path (Assert-QueueStore $RunName) 'config.ini'
    if ([IO.File]::Exists($store)) { return $store }
    $src = Join-Path $Q.Root ('config\runs\' + $RunName + '.ini')
    if ([IO.File]::Exists($src)) { return $src }
    $null
}

function Get-RunArm([string]$IniPath) {
    $v = Get-IniValue $IniPath 'ENV_CONFIG' 'comm_arch'
    if ([string]::IsNullOrEmpty($v)) { return 'local' }
    $v.ToLowerInvariant()
}

function Get-MarkerLine([string]$Path, [int]$Index = 0) {
    # The marker's non-empty line number $Index (0 = the FAILED reason code), or ''.
    try { $t = Read-SharedText $Path } catch { return '' }
    $lines = @(($t -split "`r?`n") | Where-Object { $_.Trim().Length -gt 0 })
    if ($Index -ge $lines.Count) { return '' }
    $lines[$Index].Trim()
}

function Get-MarkerFirstLine([string]$Path) { Get-MarkerLine $Path 0 }

function Write-RunMarker([string]$RunName, [string]$RelPath, [string]$Text) {
    $path = Join-Path (Assert-QueueStore $RunName) $RelPath
    Write-Utf8NoBom $path $Text
    $path
}

function Set-RunFailed([string]$RunName, [string]$Code, [string]$Detail) {
    # Writes <store>FAILED (the first line is the reason code) unless one exists, and returns
    # the terminal result. The queue stops.
    $path = Join-Path (Assert-QueueStore $RunName) 'FAILED'
    if ([IO.File]::Exists($path)) {
        Write-QueueLog ("{0}: FAILED marker already present ({1}); not rewritten (new reason {2}: {3})" -f $RunName, (Get-MarkerFirstLine $path), $Code, $Detail)
    } else {
        $text = $Code + "`r`n" + (ConvertTo-OneLine $Detail 4000) + "`r`n" + 'written ' + (Get-Date).ToString('o') + " by tools/run_queue.ps1`r`n"
        [void](Write-RunMarker $RunName 'FAILED' $text)
        Write-QueueLog ("{0}: FAILED ({1}): {2}" -f $RunName, $Code, $Detail)
    }
    Get-FailedResult $RunName
}

function Get-FailedResult([string]$RunName) {
    $path = Join-Path (Assert-QueueStore $RunName) 'FAILED'
    $code = Get-MarkerLine $path 0
    $status = 'FAILED'
    if ($code -eq 'STUCK') { $status = 'STUCK' }
    New-RunnerResult 'TERMINAL' $status ("{0} ({1}): {2}" -f $RunName, $code, (Get-MarkerLine $path 1))
}

function New-RunnerResult([string]$Action, [string]$Status = '', [string]$Detail = '') {
    # Action: CONTINUE (next pass), WAIT (re-check in 5 min), STOPPED or TERMINAL (exit).
    [pscustomobject]@{ Action = $Action; Status = $Status; Detail = $Detail }
}

function Get-RunMeasured($State, [string]$RunName) {
    $r = Get-RunCounters $State $RunName
    if (-not $r.Contains('measured') -or $null -eq $r['measured']) { $r['measured'] = [ordered]@{} }
    $r['measured']
}

function Set-MaxMeasured($Measured, [string]$Key, [double]$Value) {
    if ([double]::IsNaN($Value) -or $Value -le 0) { return }
    $old = 0.0
    if ($Measured.Contains($Key) -and [double]::TryParse([string]$Measured[$Key], $QueueFloatStyle, $QueueInvariant, [ref]$old) -and $old -ge $Value) { return }
    $Measured[$Key] = [Math]::Round($Value, 2)
}

# ---------------------------------------------------------------------------------------------
# Code hashes (spec 9)
# ---------------------------------------------------------------------------------------------
function Get-CodeHashLines([string]$RunName) {
    # '<sha256>  <label>' for main.py, utils.py, run_config.py, run_files.py, agents\*.py,
    # envs\*.py, Malta\build_file.py and config.ini (the store's copy; before the first launch
    # config\runs\<name>.ini, which main.py copies byte for byte). Sorted by label.
    $items = New-Object System.Collections.Generic.List[object]
    foreach ($f in $RunnerCodeFiles) { $items.Add(@($f, (Join-Path $Q.Root $f))) }
    foreach ($d in $RunnerCodeDirs) {
        $dir = Join-Path $Q.Root $d
        if (-not [IO.Directory]::Exists($dir)) { $items.Add(@(($d + '\*.py'), $null)); continue }
        foreach ($p in [IO.Directory]::GetFiles($dir)) {
            if ([IO.Path]::GetExtension($p) -ne '.py') { continue }
            $items.Add(@(($d + '\' + [IO.Path]::GetFileName($p)), $p))
        }
    }
    $items.Add(@('config.ini', (Get-RunIniPath $RunName)))
    $lines = foreach ($it in $items) {
        $hash = 'MISSING'
        if ($null -ne $it[1] -and [IO.File]::Exists($it[1])) { $hash = (Get-FileHash -LiteralPath $it[1] -Algorithm SHA256).Hash.ToLowerInvariant() }
        $hash + '  ' + $it[0]
    }
    [string[]]($lines | Sort-Object { ($_ -split '\s+', 2)[1] })
}

function Read-CodeHashFile([string]$Path) {
    (Read-SharedText $Path) -split "`r?`n" | Where-Object { $_.Trim().Length -gt 0 -and -not $_.StartsWith('#') }
}

function Get-FirstCodeHashFile([string]$RunName) {
    # The code file of the run's first training launch (lowest N).
    $logs = Join-Path (Assert-QueueStore $RunName) 'logs'
    if (-not [IO.Directory]::Exists($logs)) { return $null }
    $best = $null; $bestN = [int]::MaxValue
    foreach ($f in [IO.Directory]::GetFiles($logs)) {
        $m = [regex]::Match([IO.Path]::GetFileName($f), '^launch_r(\d{1,9})\.code\.txt$', 'IgnoreCase')
        if ($m.Success -and [int]$m.Groups[1].Value -lt $bestN) { $bestN = [int]$m.Groups[1].Value; $best = $f }
    }
    $best
}

function Test-CodeHashes([string]$RunName) {
    # -> { Ok; Lines; Detail }. A launch whose hashes differ from the run's first launch is refused.
    $lines = Get-CodeHashLines $RunName
    $first = Get-FirstCodeHashFile $RunName
    if ($null -eq $first) { return [pscustomobject]@{ Ok = $true; Lines = $lines; Detail = 'first launch of the run; hashes recorded at launch' } }
    $diff = @(Compare-CodeHashLines @(Read-CodeHashFile $first) $lines)
    if ($diff.Count -eq 0) { return [pscustomobject]@{ Ok = $true; Lines = $lines; Detail = ('same as ' + [IO.Path]::GetFileName($first)) } }
    [pscustomobject]@{ Ok = $false; Lines = $lines; Detail = ('differs from {0}: {1}' -f [IO.Path]::GetFileName($first), ($diff -join ', ')) }
}

# ---------------------------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------------------------
function Get-ProcessTable {
    Get-CimInstance -ClassName Win32_Process -Property ProcessId, ParentProcessId, Name, CommandLine, CreationDate, PrivatePageCount -ErrorAction Stop
}

function Get-TreeIds($Table, [int]$RootId) {
    # The root and its descendants as a HashSet[int]. A child must not be older than its
    # parent, so a reused parent id does not adopt unrelated processes.
    $ids = New-Object 'System.Collections.Generic.HashSet[int]'
    $byId = @{}
    foreach ($p in @($Table)) { if ($null -ne $p) { $byId[[int]$p.ProcessId] = $p } }
    if (-not $byId.ContainsKey($RootId)) { return , $ids }
    [void]$ids.Add($RootId)
    $todo = New-Object System.Collections.Generic.Queue[int]
    $todo.Enqueue($RootId)
    while ($todo.Count -gt 0) {
        $cur = $todo.Dequeue()
        $parent = $byId[$cur]
        foreach ($p in @($Table)) {
            if ($null -eq $p) { continue }
            $id = [int]$p.ProcessId
            if ([int]$p.ParentProcessId -ne $cur -or $ids.Contains($id)) { continue }
            if ($null -ne $parent.CreationDate -and $null -ne $p.CreationDate -and $p.CreationDate -lt $parent.CreationDate) { continue }
            [void]$ids.Add($id)
            $todo.Enqueue($id)
        }
    }
    , $ids
}

function Test-PythonName([string]$Name) {
    $n = ([string]$Name).ToLowerInvariant()
    $n -eq 'python.exe' -or $n -eq 'pythonw.exe'
}

function Stop-ProcessTree([int]$Id, [string]$Why) {
    $ErrorActionPreference = 'Continue'
    $null = & taskkill.exe /T /F /PID $Id 2>&1
    Write-QueueLog ('taskkill /T /F /PID {0} ({1}): exit {2}' -f $Id, $Why, $LASTEXITCODE)
}

function Test-PortListening([int]$Port) {
    $g = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties()
    foreach ($ep in $g.GetActiveTcpListeners()) { if ($ep.Port -eq $Port) { return $true } }
    $false
}

function Test-PortFree([int]$Port) {
    # No listener and no connection on the port other than TIME_WAIT.
    if (Test-PortListening $Port) { return $false }
    $g = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties()
    foreach ($c in $g.GetActiveTcpConnections()) {
        if ($c.LocalEndPoint.Port -eq $Port -and $c.State -ne [System.Net.NetworkInformation.TcpState]::TimeWait) { return $false }
    }
    $true
}

function Test-CleanupTarget($Proc, $ById) {
    # duarouter.exe or a randomTrips python writing under Malta\data that is an orphan (its
    # parent is gone, or the parent id was reused) or a child of a main.py python.
    $name = ([string]$Proc.Name).ToLowerInvariant()
    $cl = [string]$Proc.CommandLine
    $router = ($name -eq 'duarouter.exe') -or ((Test-PythonName $name) -and $cl -match 'randomTrips\.py')
    if (-not $router -or $cl -notmatch 'Malta[\\/]data') { return $false }
    $parent = $ById[[int]$Proc.ParentProcessId]
    if ($null -eq $parent) { return $true }
    if ($null -ne $parent.CreationDate -and $null -ne $Proc.CreationDate -and $parent.CreationDate -gt $Proc.CreationDate) { return $true }
    ([string]$parent.CommandLine) -match '(?:^|[\s"''\\/])main\.py(?:["''\s]|$)'
}

function Wait-PortFree([int]$Port, [int]$TimeoutSec) {
    # Up to $TimeoutSec for the port to be free (Test-PortFree), with a status heartbeat while
    # waiting. -> $true when free.
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while (-not (Test-PortFree $Port)) {
        if ((Get-Date) -ge $deadline) { return $false }
        Write-RunnerStatus 'CLEANUP' ('waiting up to {0} s for TCP {1} to be free' -f $TimeoutSec, $Port)
        Start-Sleep -Seconds ([Math]::Min(5, [Math]::Max(1, $TimeoutSec)))
    }
    $true
}

function Invoke-RunnerCleanup {
    # Spec 9 cleanup, run when no queue python is alive and before every launch: SUMO on the
    # queue's TraCI port, orphaned duarouter / randomTrips under Malta\data, stale queue
    # wrappers; then up to 2 min for port 27500 to be free. -> $true when the port is free
    # (callers that only clean up discard it with [void]).
    $table = @(Get-ProcessTable)
    $byId = @{}
    foreach ($p in $table) { $byId[[int]$p.ProcessId] = $p }
    $storeRoot = $Q.RunsDir + '\'
    foreach ($p in $table) {
        $name = ([string]$p.Name).ToLowerInvariant()
        $cl = [string]$p.CommandLine
        $id = [int]$p.ProcessId
        if ($name -eq 'sumo.exe' -and $cl -match ('--remote-port\s+' + $QueueTrainPort + '(?:\s|$)')) { Stop-ProcessTree $id 'SUMO on the queue port'; continue }
        if (Test-CleanupTarget $p $byId) { Stop-ProcessTree $id 'orphaned route generation under Malta\data'; continue }
        if ($name -eq 'cmd.exe' -and $cl.IndexOf('-X faulthandler main.py --run ', [StringComparison]::Ordinal) -ge 0 -and
            $cl.IndexOf($storeRoot, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
            $kids = @($table | Where-Object { [int]$_.ParentProcessId -eq $id -and (Test-PythonName $_.Name) })
            if ($kids.Count -eq 0) { Stop-ProcessTree $id 'stale queue wrapper (its python is gone)' }
        }
    }
    Wait-PortFree $QueueTrainPort $RunnerPortWaitSec
}

# ---------------------------------------------------------------------------------------------
# Detached python (Win32_Process.Create)
# ---------------------------------------------------------------------------------------------
function Start-Detached([string]$CommandLine) {
    # -> the cmd.exe wrapper's pid. The process is created by WMI, in the project root, with a
    # hidden window, outside the runner's process tree (it outlives the runner).
    $arguments = @{ CommandLine = $CommandLine; CurrentDirectory = $Q.Root }
    try { $arguments['ProcessStartupInformation'] = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 } } catch { }
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments $arguments
    if ($r.ReturnValue -ne 0) { throw ('Win32_Process.Create returned ' + $r.ReturnValue + ' for: ' + $CommandLine) }
    [int]$r.ProcessId
}

function Test-ProcessAlive([int]$Id) {
    $null -ne (Get-Process -Id $Id -ErrorAction SilentlyContinue)
}

function Invoke-DetachedPython([string]$PythonArgs, [string]$Name, [int]$TimeoutSec, [string]$Status = 'CHECKING') {
    # Runs '<python> <args>' through the launch wrapper and waits for it, rewriting status.txt
    # (state $Status) every $RunnerCheckStatusSec so a slow check never looks like a dead runner.
    # -> { Code (Int64 or $null); Output; TimedOut }
    $out = Join-Path $Q.ChecksDir ($Name + '.out')
    $exit = Join-Path $Q.ChecksDir ($Name + '.exit')
    Write-Utf8NoBom $exit "PENDING`r`n"
    Write-Utf8NoBom $out ''
    $started = Get-Date
    Write-RunnerStatus $Status ('{0}: started {1}' -f $Name, (Format-Stamp $started))
    $cmdPid = Start-Detached (New-DetachedCommandLine $Q.Python $PythonArgs $out $exit)
    $deadline = $started.AddSeconds($TimeoutSec)
    $nextStatus = $started.AddSeconds($RunnerCheckStatusSec)
    $timedOut = $false
    while (Test-ProcessAlive $cmdPid) {
        $now = Get-Date
        if ($now -gt $deadline) { $timedOut = $true; Stop-ProcessTree $cmdPid "$Name timed out"; break }
        if ($now -ge $nextStatus) {
            Write-RunnerStatus $Status ('{0}: running for {1} s (limit {2} s)' -f $Name, [int]($now - $started).TotalSeconds, $TimeoutSec)
            $nextStatus = $now.AddSeconds($RunnerCheckStatusSec)
        }
        Start-Sleep -Milliseconds 500
    }
    $text = ''
    if ([IO.File]::Exists($out)) { $text = Read-SharedText $out }
    [pscustomobject]@{ Code = (Read-ExitCode $exit); Output = $text; TimedOut = $timedOut }
}

function Get-TextTail([string]$Text, [int]$Lines = 4) {
    $all = @(($Text -split "`r?`n") | Where-Object { $_.Trim().Length -gt 0 })
    if ($all.Count -eq 0) { return '' }
    ConvertTo-OneLine (($all | Select-Object -Last $Lines) -join "`n") 600
}

function Test-MemoryErrorText([string]$Text) {
    # A failure caused by memory pressure (another program, a small pagefile), not by the
    # environment: the runner then waits (WAITING_MEMORY) instead of stopping (ENV_ERROR).
    if ([string]::IsNullOrEmpty($Text)) { return $false }
    $Text -match '(?i)paging file is too small|MemoryError|out of memory'
}

function Test-RunnerEnvironment {
    # Spec 9: the training python imports tensorflow, traci and sumolib and sees SUMO_HOME,
    # launched the same detached way; sumo.exe exists under SUMO_HOME\bin.
    # -> { Ok; Memory (the failure text names a memory error); Detail }
    $fail = { param([string]$m, [string]$Out = '') [pscustomobject]@{ Ok = $false; Memory = (Test-MemoryErrorText $Out); Detail = $m } }
    if (-not [IO.File]::Exists($Q.Python)) { return (& $fail "python not found: $($Q.Python)") }
    $r = Invoke-DetachedPython $RunnerEnvCheckArgs 'env_check' $RunnerEnvTimeoutSec 'CHECKING_ENVIRONMENT'
    if ($r.TimedOut) { return (& $fail "the environment check timed out after $RunnerEnvTimeoutSec s" $r.Output) }
    if ($null -eq $r.Code) { return (& $fail ('the environment check wrote no exit code: ' + (Get-TextTail $r.Output)) $r.Output) }
    if ($r.Code -ne 0) { return (& $fail ('the environment check exited {0}: {1}' -f $r.Code, (Get-TextTail $r.Output)) $r.Output) }
    $m = [regex]::Matches($r.Output, '(?m)^SUMO_HOME=(.+?)\s*$')
    if ($m.Count -eq 0) { return (& $fail ('no SUMO_HOME line: ' + (Get-TextTail $r.Output)) $r.Output) }
    $sumoHome = $m[$m.Count - 1].Groups[1].Value.Trim()
    $sumo = Join-Path $sumoHome 'bin\sumo.exe'
    if (-not [IO.File]::Exists($sumo)) { return (& $fail "sumo.exe not found: $sumo") }
    [pscustomobject]@{ Ok = $true; Memory = $false; Detail = "tensorflow, traci, sumolib import; SUMO_HOME=$sumoHome; $sumo exists" }
}

function Confirm-RunnerEnvironment {
    # The environment check, once per runner process, with one retry after 60 s. A failure
    # whose output names a memory error returns at once with Memory = $true (the caller waits
    # and re-checks); EnvOk stays false, so the next pass checks again.
    if ($Q.EnvOk) { return [pscustomobject]@{ Ok = $true; Memory = $false; Detail = 'checked earlier' } }
    $e = $null
    for ($i = 1; $i -le 2; $i++) {
        Write-RunnerStatus 'CHECKING_ENVIRONMENT' "try $i of 2"
        $e = Test-RunnerEnvironment
        if ($e.Ok) { $Q.EnvOk = $true; Write-QueueLog ('environment check passed: ' + $e.Detail); return $e }
        Write-QueueLog ("environment check failed (try {0} of 2): {1}" -f $i, $e.Detail)
        if ($e.Memory) { return $e }
        if ($i -lt 2) {
            Write-RunnerStatus 'CHECKING_ENVIRONMENT' ("try 1 of 2 failed; retrying in {0} s" -f $RunnerPollSeconds)
            Start-Sleep -Seconds $RunnerPollSeconds
        }
    }
    $e
}

# ---------------------------------------------------------------------------------------------
# Preflight (spec 9 'Before every launch')
# ---------------------------------------------------------------------------------------------
function Get-CommitBytes {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem -Property TotalVirtualMemorySize, FreeVirtualMemory -ErrorAction Stop
    $limit = [double]$os.TotalVirtualMemorySize * 1024
    [pscustomobject]@{ Limit = $limit; Committed = $limit - [double]$os.FreeVirtualMemory * 1024 }
}

function Invoke-TrafficCheck([string]$PythonArgs, [string]$Name, [hashtable]$Cache) {
    if ($Cache.ContainsKey($PythonArgs)) { return $Cache[$PythonArgs] }
    $r = Invoke-DetachedPython $PythonArgs $Name $RunnerTrafficTimeoutSec 'PREFLIGHT'
    $summary = Get-TextTail $r.Output 1
    $res = $null
    if ($r.TimedOut) { $res = [pscustomobject]@{ Ok = $false; Detail = "timed out after $RunnerTrafficTimeoutSec s" } }
    elseif ($r.Code -eq 0) { $res = [pscustomobject]@{ Ok = $true; Detail = $summary } }
    else {
        $bad = @(($r.Output -split "`r?`n") | Where-Object { $_ -match ': (MISSING|INVALID|FAILED)' } | Select-Object -First 3)
        $res = [pscustomobject]@{ Ok = $false; Detail = ('exit {0}: {1} {2}' -f $r.Code, $summary, ($bad -join '; ')) }
    }
    $Cache[$PythonArgs] = $res
    $res
}

function Invoke-Preflight {
    param([string]$RunName, $Cfg, [string]$Arm, $State, [hashtable]$Cache = $null)
    # -> { Ok; Wait (the first failed check's reason); Lines { Name; Ok; Wait; Detail } }.
    if ($null -eq $Cache) { $Cache = @{} }
    $lines = New-Object System.Collections.Generic.List[object]
    $add = { param($Name, [bool]$Ok, $Wait, $Detail) $lines.Add([pscustomobject]@{ Name = $Name; Ok = $Ok; Wait = $Wait; Detail = [string]$Detail }) }

    if ([IO.File]::Exists($Q.StopFile)) { & $add 'STOP' $false 'STOP' 'Malta\runs\_queue\STOP exists' }
    else { & $add 'STOP' $true $null 'absent' }

    $table = @(Get-ProcessTable)
    $exclude = Get-TreeIds $table $PID
    $busy = @(Get-BusyReasons $table $exclude)
    if ($busy.Count -gt 0) { & $add 'machine free' $false 'MACHINE_BUSY' ($busy -join '; ') }
    else { & $add 'machine free' $true $null 'no other main.py / ablation_routing / baselines / gen_traffic / randomTrips / build_zones / ipykernel python; no sumo, sumo-gui, duarouter or netconvert' }

    if (Test-PortListening $QueueTrainPort) { & $add 'port 27500' $false 'PORT_BUSY' 'something listens on TCP 27500' }
    else { & $add 'port 27500' $true $null 'nothing listens' }

    $peak = Get-ArmSetting $State $Arm 'peak_commit_gb' $RunnerDefaultPeakGB
    $c = Get-CommitBytes
    $v = Get-CommitVerdict $c.Limit $c.Committed $peak $RunnerCommitReserveGB
    $src = 'default'
    if ($peak -ne $RunnerDefaultPeakGB) { $src = 'state.json arms.' + $Arm }
    & $add 'commit' $v.Ok $(if ($v.Ok) { $null } else { 'MEMORY' }) ($v.Detail + ' (peak from ' + $src + ')')

    $projDrive = [IO.Path]::GetPathRoot($Q.Root).TrimEnd('\')
    $sysDrive = ([string]$env:SystemDrive).TrimEnd('\')
    if (-not $sysDrive) { $sysDrive = 'C:' }
    $disks = New-Object System.Collections.Generic.List[object]
    $disks.Add(@($projDrive, $RunnerFreeProjectGB))
    if ($sysDrive -ieq $projDrive) { $disks[0] = @($projDrive, [Math]::Max($RunnerFreeProjectGB, $RunnerFreeSystemGB)) }
    else { $disks.Add(@($sysDrive, $RunnerFreeSystemGB)) }
    foreach ($d in $disks) {
        $free = [IO.DriveInfo]::new($d[0]).AvailableFreeSpace
        $v = Get-DiskVerdict $d[0] $free $d[1]
        & $add ('disk ' + $d[0]) $v.Ok $(if ($v.Ok) { $null } else { 'DISK_' + $d[0].TrimEnd(':') }) $v.Detail
    }

    $ta = Get-TrafficCheckArgs $Cfg
    foreach ($t in @(@('traffic train', $ta.Train, 'traffic_train'), @('traffic eval', $ta.Test, 'traffic_eval'))) {
        $r = Invoke-TrafficCheck $t[1] $t[2] $Cache
        & $add $t[0] $r.Ok $(if ($r.Ok) { $null } else { 'TRAFFIC' }) ($t[1].Replace('tools/gen_traffic.py ', '') + ' -> ' + $r.Detail)
    }

    $all = $lines.ToArray()
    $firstBad = $all | Where-Object { -not $_.Ok } | Select-Object -First 1
    $wait = $null
    if ($null -ne $firstBad) { $wait = $firstBad.Wait }
    [pscustomobject]@{ Ok = ($null -eq $firstBad); Wait = $wait; Lines = $all }
}

function Format-PreflightLine($Line) {
    $verdict = 'PASS'
    if (-not $Line.Ok) { $verdict = 'WAIT_' + $Line.Wait }
    '  {0,-14}: {1} ({2})' -f $Line.Name, $verdict, $Line.Detail
}

# ---------------------------------------------------------------------------------------------
# Jobs: launch, adoption, log reading
# ---------------------------------------------------------------------------------------------
function New-JobRecord {
    param([string]$RunName, $Cfg, [string]$Kind, [int]$N, [datetime]$Started, $CmdPid, $PyPid, [int]$GoodBefore,
          [string[]]$Levels, [bool]$Foreign = $false)
    $logs = Join-Path (Assert-QueueStore $RunName) 'logs'
    $prefix = 'launch'
    if ($Kind -eq 'evaluate') { $prefix = 'eval' }
    [pscustomobject]@{
        Run = $RunName; Cfg = $Cfg; Kind = $Kind; N = $N; Started = $Started; Foreign = $Foreign
        CmdPid = $CmdPid; PyPid = $PyPid; GoodBefore = $GoodBefore; Levels = (Get-StringList $Levels)
        Log = (Join-Path $logs ('{0}_r{1}.log' -f $prefix, $N)); Exit = (Join-Path $logs ('{0}_r{1}.exit' -f $prefix, $N))
        Label = ('{0}_r{1}' -f $prefix, $N)
        CkptRe = ('(?:^|\s)Checkpoint saved at step (\d{1,9}) -> ' + [regex]::Escape((Get-StoreRel $RunName) + 'model/') + '\s*$')
        Offset = [long]0; CkptCount = 0; NewCkpt = 0; LastCkptTime = $null; LastCkptStep = -1
        FirstCkptMin = $null; MaxGapMin = 0.0; InTraceback = $false; SaveErrorAt = $null; NanDetail = $null
        PhaseStart = $Started; LevelSeen = $false; CurrentLevel = $null; LevelStarts = @{}; F100Min = $null; DoneLevels = 0
        StopSince = $null; StopCkptBase = 0; Adopted = $false; Polled = $false; AdoptCkptBase = 0
        EndReason = $null; EndDetail = $null; TimedOutLevel = $null; PeakCommitGB = 0.0
        FirstLimitMin = $RunnerFirstCheckpointMin; Rollouts = $null; StateF100 = $null
    }
}

function Read-NewLogLines($Job, [int]$MaxBytes = 33554432) {
    # The complete lines appended to the job's log since its saved byte offset; advances the
    # offset past the last line end only (a partial last line is read on a later poll).
    if (-not [IO.File]::Exists($Job.Log)) { return }
    $fs = Open-SharedRead $Job.Log
    try {
        if ($fs.Length -lt $Job.Offset) { $Job.Offset = [long]0 }
        $avail = $fs.Length - $Job.Offset
        if ($avail -le 0) { return }
        $n = [int][Math]::Min([long]$MaxBytes, $avail)
        [void]$fs.Seek($Job.Offset, [IO.SeekOrigin]::Begin)
        $buf = New-Object byte[] $n
        $read = 0
        while ($read -lt $n) {
            $k = $fs.Read($buf, $read, $n - $read)
            if ($k -le 0) { break }
            $read += $k
        }
        if ($read -le 0) { return }
        $last = [Array]::LastIndexOf($buf, [byte]10, $read - 1)
        if ($last -lt 0) {
            if ($read -lt $MaxBytes) { return }
            $last = $read - 1
        }
        $text = $QueueUtf8NoBom.GetString($buf, 0, $last + 1)
        $Job.Offset = $Job.Offset + $last + 1
        foreach ($l in ($text -split "`r?`n")) { if ($l.Length -gt 0) { $l } }
    } finally { $fs.Dispose() }
}

function Update-JobFromLine($Job, [string]$Line) {
    $now = Get-Date
    if ($Job.Kind -eq 'train') {
        $m = [regex]::Match($Line, $Job.CkptRe)
        if ($m.Success) {
            $t = Get-LogLineTime $Line $now
            if ($null -eq $Job.LastCkptTime) { $Job.FirstCkptMin = ($t - $Job.Started).TotalMinutes }
            else {
                $gap = ($t - [datetime]$Job.LastCkptTime).TotalMinutes
                if ($gap -gt $Job.MaxGapMin) { $Job.MaxGapMin = $gap }
            }
            $Job.LastCkptTime = $t
            $Job.LastCkptStep = [int]$m.Groups[1].Value
            $Job.CkptCount++
            $Job.NewCkpt++
            $Job.InTraceback = $false
            return
        }
        if ($Line.StartsWith('Traceback')) { $Job.InTraceback = $true }
        if ($Line.Contains('Per-episode checkpoint save failed') -or ($Job.InTraceback -and $Line -match '\bin _persist_episode\b')) {
            if ($null -eq $Job.SaveErrorAt) { $Job.SaveErrorAt = $now; Write-QueueLog ("{0}: checkpoint-save error in {1}: {2}" -f $Job.Run, $Job.Label, $Line.Trim()) }
        }
        return
    }
    $m = [regex]::Match($Line, 'EVAL_LEVEL_START f(\S+) ')
    if ($m.Success) {
        $t = Get-LogLineTime $Line $now
        $Job.PhaseStart = $t; $Job.LevelSeen = $true; $Job.CurrentLevel = $m.Groups[1].Value
        $Job.LevelStarts[$m.Groups[1].Value] = $t
        return
    }
    $m = [regex]::Match($Line, 'EVAL_LEVEL_DONE f(\S+) seeds=')
    if ($m.Success) {
        $t = Get-LogLineTime $Line $now
        $tag = $m.Groups[1].Value
        $Job.PhaseStart = $t; $Job.CurrentLevel = $null; $Job.DoneLevels++
        if ($Job.LevelStarts.ContainsKey($tag)) {
            $mins = ($t - [datetime]$Job.LevelStarts[$tag]).TotalMinutes
            if ($tag -eq (Format-Level 1.0)) { $Job.F100Min = $mins }
            Write-QueueLog ("{0}: {1} level f{2} done in {3} min" -f $Job.Run, $Job.Label, $tag, (Format-Num $mins))
        }
    }
}

function Test-LastTrainRow([string]$Store) {
    # Spec 9 NaN check: the last train_reward.csv row's reward and average waiting time.
    $csv = Read-CsvFile (Join-Path $Store 'train_reward.csv')
    if (-not $csv.Exists -or -not $csv.Ok -or $csv.Rows.Count -eq 0) { return [pscustomobject]@{ Checked = $false; Finite = $true; Detail = 'train_reward.csv not readable now' } }
    $ri = Get-HeaderIndex $csv 'reward'
    $wi = Get-HeaderIndex $csv 'average waiting time'
    if ($ri -lt 0 -or $wi -lt 0) { return [pscustomobject]@{ Checked = $false; Finite = $true; Detail = 'train_reward.csv lacks a reward or average waiting time column' } }
    $f = $csv.Rows[$csv.Rows.Count - 1].Fields
    $ok = (Test-FiniteNumber $f[$ri]) -and (Test-FiniteNumber $f[$wi])
    [pscustomobject]@{ Checked = $true; Finite = $ok; Detail = ("train_reward.csv row {0}: reward '{1}', average waiting time '{2}'" -f $csv.Rows.Count, $f[$ri], $f[$wi]) }
}

function Update-JobFromLog($Job, $State) {
    # Reads the log's new lines; at new checkpoints: the NaN check and state.json bookkeeping.
    $Job.NewCkpt = 0
    while ($true) {
        $lines = @(Read-NewLogLines $Job)
        if ($lines.Count -eq 0) { break }
        foreach ($l in $lines) { Update-JobFromLine $Job $l }
    }
    if ($Job.Kind -ne 'train' -or $Job.NewCkpt -eq 0) { return }
    $nan = Test-LastTrainRow (Assert-QueueStore $Job.Run)
    if ($nan.Checked -and -not $nan.Finite -and $null -eq $Job.NanDetail) { $Job.NanDetail = $nan.Detail }
    if (-not $nan.Checked) { Write-QueueLog ("{0}: NaN check skipped at checkpoint-{1}: {2}" -f $Job.Run, $Job.LastCkptStep, $nan.Detail) }
    $cur = (Get-RunCounters $State $Job.Run)['current_launch']
    if ($null -ne $cur) {
        $cur['last_ckpt_step'] = $Job.LastCkptStep
        $cur['last_ckpt_time'] = ([datetime]$Job.LastCkptTime).ToString('o')
    }
    Save-JobMeasurements $Job $State
    Save-QueueState $Q.QueueDir $State
    Write-QueueLog ("{0}: {1} checkpoint-{2} saved ({3} new)" -f $Job.Run, $Job.Label, $Job.LastCkptStep, $Job.NewCkpt)
}

function Save-JobMeasurements($Job, $State) {
    $m = Get-RunMeasured $State $Job.Run
    if ($Job.Kind -eq 'train') {
        if ($null -ne $Job.FirstCkptMin) { Set-MaxMeasured $m 'first_checkpoint_min' $Job.FirstCkptMin }
        Set-MaxMeasured $m 'max_checkpoint_gap_min' $Job.MaxGapMin
        Set-MaxMeasured $m 'peak_commit_gb' $Job.PeakCommitGB
    } else {
        if ($null -ne $Job.F100Min) { $m['eval_f100_min'] = [Math]::Round([double]$Job.F100Min, 2) }
        Set-MaxMeasured $m 'eval_peak_commit_gb' $Job.PeakCommitGB
    }
}

function Find-PythonChild([int]$CmdPid) {
    $kids = @(Get-CimInstance -ClassName Win32_Process -Filter ("ParentProcessId={0}" -f $CmdPid) -Property ProcessId, Name -ErrorAction Stop)
    foreach ($k in $kids) { if (Test-PythonName $k.Name) { return [int]$k.ProcessId } }
    $null
}

function Wait-PythonChild([int]$CmdPid) {
    $deadline = (Get-Date).AddSeconds($RunnerPythonFindSec)
    while ((Get-Date) -lt $deadline) {
        $py = Find-PythonChild $CmdPid
        if ($null -ne $py) { return $py }
        if (-not (Test-ProcessAlive $CmdPid)) { return $null }
        Start-Sleep -Milliseconds 500
    }
    $null
}

function Start-RunnerLaunch([string]$RunName, $Cfg, [string]$Stage, [string[]]$Levels, $State, [string[]]$CodeLines) {
    if (-not $Q.EnvOk) { throw 'refusing to launch before the environment check passed' }
    $kind = 'train'; $prefix = 'launch'
    if ($Stage -eq 'EVALUATE') { $kind = 'evaluate'; $prefix = 'eval' }
    $store = Assert-QueueStore $RunName
    $n = Get-NextLaunchNumber $Q.Root $RunName $prefix
    $logs = Join-Path $store 'logs'
    if (-not [IO.Directory]::Exists($logs)) { [void][IO.Directory]::CreateDirectory($logs) }
    $log = Join-Path $logs ('{0}_r{1}.log' -f $prefix, $n)
    $exit = Join-Path $logs ('{0}_r{1}.exit' -f $prefix, $n)
    $good = -1
    if ($kind -eq 'train') { $good = Get-GoodCheckpoint $Q.Root $RunName }
    $levelList = Get-StringList $Levels
    $cl = New-DetachedCommandLine $Q.Python (New-LaunchArguments $RunName $kind $levelList) $log $exit
    $header = '# code hashes for {0}_r{1} of {2}, {3}' -f $prefix, $n, $RunName, (Get-Date).ToString('o')
    Write-Utf8NoBom (Join-Path $logs ('{0}_r{1}.code.txt' -f $prefix, $n)) ((((@($header) + @($CodeLines)) -join "`r`n")) + "`r`n")
    $counters = Get-RunCounters $State $RunName
    if ($kind -eq 'evaluate') {
        $counters['eval_attempts'] = [int]$counters['eval_attempts'] + 1
        $el = 0
        if ($counters.Contains('eval_launches') -and $null -ne $counters['eval_launches']) { $el = [int]$counters['eval_launches'] }
        $counters['eval_launches'] = $el + 1
    }
    $script:LoggedOnce.Remove('wait')
    $started = Get-Date
    $cmdPid = Start-Detached $cl
    $counters['current_launch'] = [ordered]@{
        kind = $kind; n = $n; good_before = $good; started = $started.ToString('o')
        cmd_pid = $cmdPid; py_pid = $null; levels = $levelList; last_ckpt_step = -1; last_ckpt_time = $null
    }
    Save-QueueState $Q.QueueDir $State
    $goodText = ''
    if ($kind -eq 'train') { $goodText = '; good checkpoint ' + $good }
    Write-QueueLog ("{0}: launched {1}_r{2} (wrapper pid {3}{4}): {5}" -f $RunName, $prefix, $n, $cmdPid, $goodText, $cl)
    $py = Wait-PythonChild $cmdPid
    if ($null -ne $py) { $counters['current_launch']['py_pid'] = $py; Save-QueueState $Q.QueueDir $State }
    else { Write-QueueLog ("{0}: no python child of wrapper {1} seen (it ended at once, or did not start within {2} s)" -f $RunName, $cmdPid, $RunnerPythonFindSec) }
    New-JobRecord -RunName $RunName -Cfg $Cfg -Kind $kind -N $n -Started $started -CmdPid $cmdPid -PyPid $py -GoodBefore $good -Levels $levelList
}

function Find-LiveJob($Runs, $State, $Table) {
    # Spec 9 adoption: a live python running 'main.py ... --run <name>' for a queue run, or for a
    # run with a launch in flight in state.json. -> a job record, or $null.
    $names = New-Object System.Collections.Generic.List[string]
    foreach ($r in @($Runs)) { if ((Test-QueueRunName $r) -and -not $names.Contains($r)) { $names.Add($r) } }
    foreach ($k in @($State['runs'].Keys)) {
        $entry = $State['runs'][$k]
        if ($null -ne $entry -and $entry.Contains('current_launch') -and $null -ne $entry['current_launch'] -and (Test-QueueRunName $k) -and -not $names.Contains($k)) { $names.Add($k) }
    }
    $byId = @{}
    foreach ($p in @($Table)) { if ($null -ne $p) { $byId[[int]$p.ProcessId] = $p } }
    foreach ($p in @($Table)) {
        if ($null -eq $p -or -not (Test-PythonName $p.Name)) { continue }
        foreach ($name in $names) {
            if (Test-RunCommandLine $name $p.Name $p.CommandLine) { return (New-AdoptedJob $name $p $byId $State) }
        }
    }
    $null
}

function New-AdoptedJob([string]$RunName, $Proc, $ById, $State) {
    # The launch is identified from the wrapper's command line ('<store>\logs\launch_rN.log'),
    # else from state.json's launch in flight with the same python pid; else it is foreign
    # (not started by the queue) and is only watched until it exits.
    $cl = [string]$Proc.CommandLine
    $kind = 'train'; $prefix = 'launch'
    if ($cl -match '--option(?:\s+|=)["'']?evaluate') { $kind = 'evaluate'; $prefix = 'eval' }
    $n = $null; $cmdPid = $null
    $parent = $ById[[int]$Proc.ParentProcessId]
    if ($null -ne $parent -and ([string]$parent.Name).ToLowerInvariant() -eq 'cmd.exe') {
        $logs = (Join-Path (Assert-QueueStore $RunName) 'logs') + '\'
        $m = [regex]::Match([string]$parent.CommandLine, [regex]::Escape($logs) + $prefix + '_r(\d{1,9})\.log', 'IgnoreCase')
        if ($m.Success) { $n = [int]$m.Groups[1].Value; $cmdPid = [int]$parent.ProcessId }
    }
    $cur = (Get-RunCounters $State $RunName)['current_launch']
    if ($null -eq $n -and $null -ne $cur -and $null -ne $cur['py_pid'] -and [int]$cur['py_pid'] -eq [int]$Proc.ProcessId -and $cur['kind'] -eq $kind) {
        $n = [int]$cur['n']
        if ($null -ne $cur['cmd_pid']) { $cmdPid = [int]$cur['cmd_pid'] }
    }
    $ini = Get-RunIniPath $RunName
    $cfg = $null
    if ($null -ne $ini) { $cfg = Read-RunConfig $ini }
    $created = Get-Date
    if ($null -ne $Proc.CreationDate) { $created = [datetime]$Proc.CreationDate }
    if ($null -eq $n -or $null -eq $cfg) {
        return (New-JobRecord -RunName $RunName -Cfg $cfg -Kind $kind -N 0 -Started $created -CmdPid $null -PyPid ([int]$Proc.ProcessId) -GoodBefore -1 -Foreign $true)
    }
    $started = $created; $good = -1
    $levels = @()
    $lm = [regex]::Match($cl, '--levels(?:\s+|=)["'']?([0-9.,]+)')
    if ($lm.Success) { $levels = @($lm.Groups[1].Value.Split(',') | Where-Object { $_.Length -gt 0 }) }
    if ($null -ne $cur -and [int]$cur['n'] -eq $n -and $cur['kind'] -eq $kind) {
        $t = ConvertTo-LocalTime $cur['started']
        if ($null -ne $t) { $started = $t }
        if ($null -ne $cur['good_before']) { $good = [int]$cur['good_before'] }
    }
    $job = New-JobRecord -RunName $RunName -Cfg $cfg -Kind $kind -N $n -Started $started -CmdPid $cmdPid -PyPid ([int]$Proc.ProcessId) -GoodBefore $good -Levels $levels
    $job.Adopted = $true
    $job
}

function ConvertFrom-CurrentLaunch([string]$RunName, $Cur) {
    # A job record for a launch in flight in state.json whose python is gone.
    $ini = Get-RunIniPath $RunName
    if ($null -eq $ini) { throw "run $RunName has a launch in flight but no config" }
    $started = ConvertTo-LocalTime $Cur['started']
    if ($null -eq $started) { $started = Get-Date }
    $good = -1
    if ($null -ne $Cur['good_before']) { $good = [int]$Cur['good_before'] }
    $cmdPid = $null; $pyPid = $null
    if ($null -ne $Cur['cmd_pid']) { $cmdPid = [int]$Cur['cmd_pid'] }
    if ($null -ne $Cur['py_pid']) { $pyPid = [int]$Cur['py_pid'] }
    $job = New-JobRecord -RunName $RunName -Cfg (Read-RunConfig $ini) -Kind ([string]$Cur['kind']) -N ([int]$Cur['n']) -Started $started `
        -CmdPid $cmdPid -PyPid $pyPid -GoodBefore $good -Levels (Get-StringList $Cur['levels'])
    $job.EndReason = 'EXITED'
    $job
}

function Sync-CurrentLaunch($Job, $State) {
    # Makes state.json's launch in flight describe an adopted job, so a later runner can finish it.
    $counters = Get-RunCounters $State $Job.Run
    $cur = $counters['current_launch']
    if ($null -ne $cur -and $cur['kind'] -eq $Job.Kind -and [int]$cur['n'] -eq $Job.N) {
        if ($null -eq $cur['py_pid'] -and $null -ne $Job.PyPid) { $cur['py_pid'] = $Job.PyPid; Save-QueueState $Q.QueueDir $State }
        return
    }
    $counters['current_launch'] = [ordered]@{
        kind = $Job.Kind; n = $Job.N; good_before = $Job.GoodBefore; started = ([datetime]$Job.Started).ToString('o')
        cmd_pid = $Job.CmdPid; py_pid = $Job.PyPid; levels = $Job.Levels; last_ckpt_step = -1; last_ckpt_time = $null
    }
    Save-QueueState $Q.QueueDir $State
}

# ---------------------------------------------------------------------------------------------
# Watching a job
# ---------------------------------------------------------------------------------------------
function Test-JobAlive($Job, $Table) {
    $pyAlive = $false; $cmdAlive = $false
    foreach ($p in @($Table)) {
        if ($null -eq $p) { continue }
        $id = [int]$p.ProcessId
        if ($null -ne $Job.PyPid -and $id -eq [int]$Job.PyPid -and (Test-PythonName $p.Name) -and (Test-RunCommandLine $Job.Run $p.Name $p.CommandLine)) { $pyAlive = $true }
        if ($null -ne $Job.CmdPid -and $id -eq [int]$Job.CmdPid -and ([string]$p.Name).ToLowerInvariant() -eq 'cmd.exe' -and
            ([string]$p.CommandLine).IndexOf($Job.Log, [StringComparison]::OrdinalIgnoreCase) -ge 0) { $cmdAlive = $true }
    }
    if (-not $pyAlive -and $null -eq $Job.PyPid -and $cmdAlive) {
        foreach ($p in @($Table)) {
            if ($null -ne $p -and [int]$p.ParentProcessId -eq [int]$Job.CmdPid -and (Test-PythonName $p.Name)) { $Job.PyPid = [int]$p.ProcessId; $pyAlive = $true }
        }
    }
    $pyAlive -or $cmdAlive
}

function Get-JobCommitGB($Job, $Table) {
    $root = $Job.CmdPid
    if ($null -eq $root) { $root = $Job.PyPid }
    if ($null -eq $root) { return 0.0 }
    $ids = Get-TreeIds $Table ([int]$root)
    $sum = 0.0
    foreach ($p in @($Table)) { if ($null -ne $p -and $ids.Contains([int]$p.ProcessId) -and $null -ne $p.PrivatePageCount) { $sum += [double]$p.PrivatePageCount } }
    $sum / 1GB
}

function Test-WrapperAlive($Job) {
    if ($null -eq $Job.CmdPid) { return $false }
    $p = @(Get-CimInstance -ClassName Win32_Process -Filter ("ProcessId={0}" -f [int]$Job.CmdPid) -Property ProcessId, Name, CommandLine -ErrorAction Stop)
    foreach ($x in $p) {
        if (([string]$x.Name).ToLowerInvariant() -eq 'cmd.exe' -and ([string]$x.CommandLine).IndexOf($Job.Log, [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $true }
    }
    $false
}

function Wait-JobWrapper($Job) {
    # The wrapper writes the exit file right after python ends; give it 30 s, then kill it.
    if ($null -eq $Job.CmdPid) { return }
    $deadline = (Get-Date).AddSeconds($RunnerWrapperWaitSec)
    while ((Get-Date) -lt $deadline) {
        if (-not (Test-WrapperAlive $Job)) { return }
        Start-Sleep -Seconds 1
    }
    Stop-ProcessTree ([int]$Job.CmdPid) ($Job.Label + ' wrapper still alive')
}

function Stop-RunnerJob($Job, [string]$Why) {
    # Kills python and its SUMO children; the wrapper then writes the exit file (EXIT=1).
    Write-QueueLog ("{0}: killing {1}: {2}" -f $Job.Run, $Job.Label, $Why)
    if ($null -ne $Job.PyPid) { Stop-ProcessTree ([int]$Job.PyPid) $Why }
    elseif ($null -ne $Job.CmdPid) { Stop-ProcessTree ([int]$Job.CmdPid) $Why }
    Wait-JobWrapper $Job
}

function Get-JobStatus($Job) {
    $total = ''
    if ($null -ne $Job.Cfg) { $total = '/' + $Job.Cfg.TotalStep }
    if ($Job.Kind -eq 'train') {
        if ($Job.CkptCount -eq 0 -and $Job.LastCkptStep -lt 0) {
            return ('{0} {1} started {2}, no checkpoint yet (python pid {3})' -f $Job.Run, $Job.Label, (Format-Stamp $Job.Started), $Job.PyPid)
        }
        return ('{0} {1} step {2}{3} (last checkpoint {4}, python pid {5})' -f $Job.Run, $Job.Label, $Job.LastCkptStep, $total, (Format-Stamp ([datetime]$Job.LastCkptTime)), $Job.PyPid)
    }
    $lv = 'before the first level'
    if ($Job.CurrentLevel) { $lv = 'level f' + $Job.CurrentLevel }
    elseif ($Job.LevelSeen) { $lv = 'between levels' }
    '{0} {1} {2} ({3} level(s) done, python pid {4})' -f $Job.Run, $Job.Label, $lv, $Job.DoneLevels, $Job.PyPid
}

function Write-WatchError($Job, $ErrorRecord) {
    # A failed poll of a watched job: logged, and status.txt still gets its heartbeat.
    $msg = Format-RunnerError $ErrorRecord
    Write-QueueLog ('watch error (job keeps running): ' + $msg)
    $state = 'TRAINING'
    if ($Job.Foreign) { $state = 'ADOPTED' }
    elseif ($Job.Kind -eq 'evaluate') { $state = 'EVALUATING' }
    Write-RunnerStatus $state ('{0} {1}: poll failed, retrying in {2} s: {3}' -f $Job.Run, $Job.Label, $RunnerPollSeconds, $msg)
}

function Step-RunnerJob($Job, $State) {
    # One poll of a watched job; sets $Job.EndReason when it has ended or was killed.
    Update-JobFromLog $Job $State
    if (-not $Job.Polled) {
        # An adopted job's first read covers its whole log: those checkpoints were saved before
        # this runner watched it, so STOP waits for the next one (base = the count so far).
        $Job.Polled = $true
        if ($Job.Adopted) { $Job.AdoptCkptBase = $Job.CkptCount }
    }
    $table = @(Get-ProcessTable)
    if (-not (Test-JobAlive $Job $table)) { $Job.EndReason = 'EXITED'; return }
    $gb = Get-JobCommitGB $Job $table
    if ($gb -gt $Job.PeakCommitGB) { $Job.PeakCommitGB = $gb }
    if ($null -ne $Job.NanDetail) { Stop-RunnerJob $Job ('non-finite training result: ' + $Job.NanDetail); $Job.EndReason = 'NAN'; return }
    if ([IO.File]::Exists((Join-Path (Assert-QueueStore $Job.Run) 'FAILED'))) { Stop-RunnerJob $Job 'a FAILED marker is present'; $Job.EndReason = 'FAILED_MARKER'; return }
    $now = Get-Date
    $stop = [IO.File]::Exists($Q.StopFile)
    if ($Job.Kind -eq 'evaluate') {
        if ($stop) { Stop-RunnerJob $Job 'STOP during evaluation'; $Job.EndReason = 'STOP'; return }
        $f100 = $Job.F100Min
        if ($null -eq $f100) { $f100 = $Job.StateF100 }
        $limit = Get-EvalLevelLimitMinutes $f100 $Job.Rollouts (@($Job.Cfg.TestSeeds).Count)
        $why = Get-EvalTimeoutReason $now ([datetime]$Job.PhaseStart) $Job.LevelSeen $Job.CurrentLevel $Job.FirstLimitMin $limit
        if ($null -ne $why) {
            $Job.TimedOutLevel = $Job.CurrentLevel
            Stop-RunnerJob $Job $why
            $Job.EndReason = 'TIMEOUT'; $Job.EndDetail = $why
            return
        }
        Write-RunnerStatus 'EVALUATING' (Get-JobStatus $Job)
        Set-KeepAwake $true
        return
    }
    if ($stop) {
        if ($null -eq $Job.StopSince) {
            # A checkpoint read in this same poll counts as the one STOP waits for, unless it
            # came from an adopted job's existing log.
            $Job.StopSince = $now
            $Job.StopCkptBase = [Math]::Max($Job.CkptCount - $Job.NewCkpt, $Job.AdoptCkptBase)
            Write-QueueLog ("{0}: STOP seen; waiting for the next checkpoint of {1} (at most {2} min)" -f $Job.Run, $Job.Label, $RunnerCheckpointGapMin)
        }
        if ($Job.CkptCount -gt $Job.StopCkptBase -or ($now - [datetime]$Job.StopSince).TotalMinutes -ge $RunnerCheckpointGapMin) {
            Write-RunnerStatus 'STOPPING' ('{0}: checkpoint-{1} saved; stopping in {2} s' -f $Job.Run, $Job.LastCkptStep, $RunnerStopGraceSec)
            Start-Sleep -Seconds $RunnerStopGraceSec
            Stop-RunnerJob $Job 'STOP'
            $Job.EndReason = 'STOP'
            return
        }
        Write-RunnerStatus 'STOPPING' ('{0}: STOP seen at {1}; waiting for the next checkpoint' -f (Get-JobStatus $Job), (Format-Stamp ([datetime]$Job.StopSince)))
        Set-KeepAwake $true
        return
    } elseif ($null -ne $Job.StopSince) {
        Write-QueueLog ("{0}: STOP was removed; {1} continues" -f $Job.Run, $Job.Label)
        $Job.StopSince = $null
    }
    $why = Get-TrainWatchdogReason -Now $now -Started $Job.Started -LastCheckpoint $Job.LastCkptTime -FirstLimitMin $Job.FirstLimitMin `
        -GapMin $RunnerCheckpointGapMin -SaveErrorAt $Job.SaveErrorAt -SaveErrorGraceSec $RunnerSaveErrorGraceSec
    if ($null -ne $why) { Stop-RunnerJob $Job ('watchdog: ' + $why); $Job.EndReason = 'WATCHDOG'; $Job.EndDetail = $why; return }
    Write-RunnerStatus 'TRAINING' (Get-JobStatus $Job)
    Set-KeepAwake $true
}

function Get-RolloutSeconds([string]$Store) {
    $csv = Read-CsvFile (Join-Path $Store 'episode_cost.csv')
    if (-not $csv.Exists -or -not $csv.Ok) { return $null }
    $i = Get-HeaderIndex $csv 'rollout_s'
    if ($i -lt 0) { return $null }
    $vals = New-Object System.Collections.Generic.List[double]
    foreach ($r in $csv.Rows) {
        $d = 0.0
        if ((Test-FiniteNumber $r.Fields[$i]) -and [double]::TryParse($r.Fields[$i].Trim(), $QueueFloatStyle, $QueueInvariant, [ref]$d)) { $vals.Add($d) }
    }
    if ($vals.Count -eq 0) { return $null }
    , $vals.ToArray()
}

function Watch-ForeignJob($Job) {
    # A python of a queue run that the queue did not start: watched until it exits, then cleaned
    # up; no counters, no watchdog (its log is unknown). STOP leaves it alone and stops the runner.
    Write-QueueLog ("{0}: adopted python pid {1} that the queue did not start; watching it until it exits" -f $Job.Run, $Job.PyPid)
    while ($true) {
        try {
            if ([IO.File]::Exists($Q.StopFile)) {
                Set-KeepAwake $false
                return (New-RunnerResult 'STOPPED' 'STOPPED' ('STOP present; python pid {0} of {1} (not started by the queue) left running' -f $Job.PyPid, $Job.Run))
            }
            $table = @(Get-ProcessTable)
            $alive = $false
            foreach ($p in $table) { if ([int]$p.ProcessId -eq [int]$Job.PyPid -and (Test-RunCommandLine $Job.Run $p.Name $p.CommandLine)) { $alive = $true } }
            if (-not $alive) { break }
            Write-RunnerStatus 'ADOPTED' ('{0}: python pid {1} not started by the queue' -f $Job.Run, $Job.PyPid)
            Set-KeepAwake $true
        } catch { Write-WatchError $Job $_ }
        Start-Sleep -Seconds $RunnerPollSeconds
    }
    Set-KeepAwake $false
    Write-QueueLog ("{0}: python pid {1} (not started by the queue) has exited" -f $Job.Run, $Job.PyPid)
    [void](Invoke-RunnerCleanup)
    New-RunnerResult 'CONTINUE'
}

function Watch-RunnerJob($Job, $State) {
    if ($Job.Foreign) { return (Watch-ForeignJob $Job) }
    Sync-CurrentLaunch $Job $State
    $ini = Get-RunIniPath $Job.Run
    $Job.FirstLimitMin = Get-ArmSetting $State (Get-RunArm $ini) 'first_checkpoint_min' $RunnerFirstCheckpointMin
    if ($Job.Kind -eq 'evaluate') {
        $Job.Rollouts = Get-RolloutSeconds (Assert-QueueStore $Job.Run)
        $m = Get-RunMeasured $State $Job.Run
        if ($m.Contains('eval_f100_min') -and $null -ne $m['eval_f100_min']) { $Job.StateF100 = [double]$m['eval_f100_min'] }
    }
    Write-QueueLog ("{0}: watching {1} (python pid {2}, wrapper pid {3}, started {4}, first-checkpoint limit {5} min)" -f $Job.Run, $Job.Label, $Job.PyPid, $Job.CmdPid, (Format-Stamp $Job.Started), (Format-Num $Job.FirstLimitMin '0.##'))
    Set-KeepAwake $true
    while ($null -eq $Job.EndReason) {
        try { Step-RunnerJob $Job $State }
        catch { Write-WatchError $Job $_ }
        if ($null -ne $Job.EndReason) { break }
        Start-Sleep -Seconds $RunnerPollSeconds
    }
    Set-KeepAwake $false
    Complete-RunnerJob $Job $State
}

# ---------------------------------------------------------------------------------------------
# Completing a launch
# ---------------------------------------------------------------------------------------------
function Complete-RunnerJob($Job, $State) {
    Write-RunnerStatus 'COMPLETING' ('{0} {1} ({2})' -f $Job.Run, $Job.Label, $Job.EndReason)
    Wait-JobWrapper $Job
    try { Update-JobFromLog $Job $State } catch { Write-QueueLog ('final log read failed: ' + $_.Exception.Message) }
    [void](Invoke-RunnerCleanup)
    Save-JobMeasurements $Job $State
    if ($Job.Kind -eq 'evaluate') { return (Complete-EvalLaunch $Job $State) }
    Complete-TrainLaunch $Job $State
}

function Get-LaunchSummary($Job, $Outcome) {
    $code = 'none'
    if ($null -ne $Outcome.ExitCode) { $code = [string]$Outcome.ExitCode }
    $why = $Job.EndReason
    if ($Job.EndDetail) { $why += ': ' + $Job.EndDetail }
    $tail = ''
    if ([IO.File]::Exists($Job.Log)) { try { $tail = Get-TextTail (Get-FileTail $Job.Log 2048) 4 } catch { } }
    'exit code {0}, interrupted {1}, {2}; log tail: {3}' -f $code, $Outcome.Interrupted, $why, $tail
}

function Complete-TrainLaunch($Job, $State) {
    $run = $Job.Run
    $counters = Get-RunCounters $State $run
    if ($null -ne $Job.NanDetail) {
        $counters['current_launch'] = $null
        Save-QueueState $Q.QueueDir $State
        return (Set-RunFailed $run 'NAN' ($Job.Label + ': ' + $Job.NanDetail))
    }
    if ($Job.EndReason -eq 'FAILED_MARKER') {
        $counters['current_launch'] = $null
        Save-QueueState $Q.QueueDir $State
        return (Get-FailedResult $run)
    }
    $outcome = Get-LaunchOutcome $Q.Root $run $Job.N $Job.Started
    $tc = Test-TrainingComplete $Q.Root $run $Job.Cfg
    if ($tc.Ok) {
        $counters['current_launch'] = $null
        Save-QueueState $Q.QueueDir $State
        Write-QueueLog ("{0}: training complete ({1} exited 0)" -f $run, $Job.Label)
        return (New-RunnerResult 'CONTINUE')
    }
    Write-QueueLog ("{0}: {1} ended without completing training: {2}; not complete because: {3}" -f $run, $Job.Label, (Get-LaunchSummary $Job $outcome), $tc.Reason)
    if ($Job.EndReason -eq 'EXITED' -and $null -ne $outcome.ExitCode) {
        $tail = ''
        if ([IO.File]::Exists($Job.Log)) { $tail = Get-FileTail $Job.Log 4096 }
        if (Test-RefusalExit $outcome.ExitCode $tail) {
            $counters['current_launch'] = $null
            Save-QueueState $Q.QueueDir $State
            return (Set-RunFailed $run 'REFUSED' ('{0}: main.py refused (exit {1}: {2}), which a relaunch cannot fix; see {3}' -f $Job.Label, $outcome.ExitCode, (Get-TextTail $tail 1), $Job.Log))
        }
    }
    Write-RunnerStatus 'RECOVERY' $run
    $rec = Invoke-Recovery $Q.Root $run $Job.Cfg $Q.QueueDir -Log { param($m) Write-QueueLog $m }
    if ($rec.StartsWith('NEEDS_REVIEW:')) {
        $counters['current_launch'] = $null
        Save-QueueState $Q.QueueDir $State
        return (Set-RunFailed $run 'NEEDS_REVIEW' $rec.Substring(13))
    }
    $goodAfter = Get-GoodCheckpoint $Q.Root $run
    $interrupted = [bool]$outcome.Interrupted -or ($Job.EndReason -eq 'STOP')
    $verdict = Update-LaunchCounters $State $run ([int]$Job.GoodBefore) $goodAfter $interrupted
    $counters['current_launch'] = $null
    Save-QueueState $Q.QueueDir $State
    Write-QueueLog ("{0}: recovery {1}; good checkpoint {2} -> {3}; launches {4}/{5}, in a row without progress {6}/{7}, interrupted {8}" -f $run, $rec,
        $Job.GoodBefore, $goodAfter, $counters['launches'], $QueueMaxLaunches, $counters['no_progress'], $QueueMaxNoProgress, $counters['interrupted'])
    if ($verdict -eq 'STUCK') {
        return (Set-RunFailed $run 'STUCK' ('{0} launches, {1} in a row without progress (limits {2} and {3}); last: {4}' -f $counters['launches'], $counters['no_progress'], $QueueMaxLaunches, $QueueMaxNoProgress, $Job.Label))
    }
    New-RunnerResult 'CONTINUE'
}

function Complete-EvalLaunch($Job, $State) {
    $run = $Job.Run
    $counters = Get-RunCounters $State $run
    $notCounted = {
        param([string]$Why)
        $counters['eval_attempts'] = [Math]::Max(0, [int]$counters['eval_attempts'] - 1)
        $counters['current_launch'] = $null
        Save-QueueState $Q.QueueDir $State
        Write-QueueLog ("{0}: {1} {2}; the attempt is not counted" -f $run, $Job.Label, $Why)
    }
    if ($Job.EndReason -eq 'FAILED_MARKER') {
        $counters['current_launch'] = $null
        Save-QueueState $Q.QueueDir $State
        return (Get-FailedResult $run)
    }
    if ($Job.EndReason -eq 'STOP') { & $notCounted 'was stopped by STOP'; return (New-RunnerResult 'CONTINUE') }
    $outcome = Get-LaunchOutcome $Q.Root $run $Job.N $Job.Started -Kind 'eval'
    if ($outcome.Interrupted -and $Job.EndReason -ne 'TIMEOUT') { & $notCounted 'was interrupted (reboot or logoff)'; return (New-RunnerResult 'CONTINUE') }
    $levels = ConvertTo-LevelValues $Job.Levels
    $res = Test-EvalAttempt $Q.Root $run $Job.Cfg $Job.Log $Job.Started -Levels $levels
    foreach ($w in @($res.Warnings)) { if ($w) { Write-QueueLog ("{0}: {1} warning: {2}" -f $run, $Job.Label, $w) } }
    $counters['current_launch'] = $null
    if ($res.Ok -and $Job.EndReason -ne 'TIMEOUT') {
        $all = (@($Job.Cfg.EvalLevels) | ForEach-Object { Format-Level $_ }) -join ','
        $text = 'EVAL_OK ' + (Get-Date).ToString('o') + "`r`n" + 'levels ' + $all + "`r`n" + 'last attempt ' + $Job.Label + "`r`n"
        foreach ($w in @($res.Warnings)) { if ($w) { $text += 'warning: ' + $w + "`r`n" } }
        [void](Write-RunMarker $run 'eva_data\EVAL_OK' $text)
        [void](Write-RunMarker $run 'DONE' ('DONE ' + (Get-Date).ToString('o') + "`r`n"))
        $counters['eval_levels_next'] = @()
        Save-QueueState $Q.QueueDir $State
        Write-QueueLog ("{0}: evaluation passed ({1}); wrote eva_data\EVAL_OK and DONE" -f $run, $Job.Label)
        return (New-RunnerResult 'CONTINUE')
    }
    $failed = New-Object System.Collections.Generic.List[string]
    foreach ($f in @($res.FailedLevels)) { if ($f -and -not $failed.Contains($f)) { $failed.Add($f) } }
    if ($Job.TimedOutLevel -and -not $failed.Contains($Job.TimedOutLevel)) { $failed.Add($Job.TimedOutLevel) }
    $reason = $res.Reason
    if ($Job.EndReason -eq 'TIMEOUT') { $reason = $Job.EndDetail + '; ' + $reason }
    Write-QueueLog ("{0}: {1} failed (attempt {2} of {3}): {4}; failed levels {5}" -f $run, $Job.Label, $counters['eval_attempts'], $RunnerMaxEvalAttempts, $reason, ($failed -join ','))
    if ([int]$counters['eval_attempts'] -ge $RunnerMaxEvalAttempts) {
        Save-QueueState $Q.QueueDir $State
        return (Set-RunFailed $run 'EVAL_FAILED' ('{0}: {1}' -f $Job.Label, $reason))
    }
    $counters['eval_levels_next'] = $failed.ToArray()
    Save-QueueState $Q.QueueDir $State
    New-RunnerResult 'CONTINUE'
}

# ---------------------------------------------------------------------------------------------
# One pass of the loop
# ---------------------------------------------------------------------------------------------
function Get-QueueRunList {
    # -> { Runs; Problem }: the queue file's runs, re-read every pass.
    if (-not [IO.File]::Exists($Q.QueueFile)) { return [pscustomobject]@{ Runs = @(); Problem = ('queue file not found: ' + $Q.QueueFile) } }
    $runs = @(Get-QueueRuns $Q.QueueFile)
    $bad = @($runs | Where-Object { -not (Test-QueueRunName $_) })
    if ($bad.Count -gt 0) { return [pscustomobject]@{ Runs = $runs; Problem = ('invalid run name(s) in the queue file: ' + ($bad -join ', ')) } }
    foreach ($r in $runs) { [void](Assert-QueueStore $r) }
    [pscustomobject]@{ Runs = $runs; Problem = $null }
}

function Start-RunStep([string]$RunName, [string]$IniPath, $Cfg, [string]$Stage, $State) {
    # Recovery (TRAIN), checks, preflight, cleanup, launch and watch for one run.
    $counters = Get-RunCounters $State $RunName
    $levels = $null
    if ($Stage -eq 'TRAIN') {
        Write-RunnerStatus 'RECOVERY' $RunName
        $rec = Invoke-Recovery $Q.Root $RunName $Cfg $Q.QueueDir -Log { param($m) Write-QueueLog $m }
        if ($rec.StartsWith('NEEDS_REVIEW:')) { return (Set-RunFailed $RunName 'NEEDS_REVIEW' $rec.Substring(13)) }
        Write-QueueLogOnce ($RunName + ':decision') ('{0}: TRAIN (recovery {1}, good checkpoint {2}, launches {3}, in a row without progress {4})' -f $RunName, $rec, (Get-GoodCheckpoint $Q.Root $RunName), $counters['launches'], $counters['no_progress'])
        if ([int]$counters['launches'] -ge $QueueMaxLaunches -or [int]$counters['no_progress'] -ge $QueueMaxNoProgress) {
            return (Set-RunFailed $RunName 'STUCK' ('{0} launches, {1} in a row without progress' -f $counters['launches'], $counters['no_progress']))
        }
    } else {
        $el = 0
        if ($counters.Contains('eval_launches') -and $null -ne $counters['eval_launches']) { $el = [int]$counters['eval_launches'] }
        if ([int]$counters['eval_attempts'] -ge $RunnerMaxEvalAttempts) { return (Set-RunFailed $RunName 'EVAL_FAILED' ('{0} evaluation attempts already counted' -f $counters['eval_attempts'])) }
        if ($el -ge $RunnerMaxEvalLaunches) { return (Set-RunFailed $RunName 'EVAL_FAILED' ('{0} evaluation launches (cap {1})' -f $el, $RunnerMaxEvalLaunches)) }
        if ([int]$counters['eval_attempts'] -gt 0) {
            $levels = Get-StringList $counters['eval_levels_next']
            if ($levels.Count -eq 0) { $levels = $null }
        }
        $lv = 'all levels'
        if ($null -ne $levels) { $lv = 'levels ' + ($levels -join ',') }
        Write-QueueLogOnce ($RunName + ':decision') ('{0}: EVALUATE (attempt {1} of {2}, {3})' -f $RunName, ([int]$counters['eval_attempts'] + 1), $RunnerMaxEvalAttempts, $lv)
    }
    $code = Test-CodeHashes $RunName
    if (-not $code.Ok) { return (Set-RunFailed $RunName 'CODE_CHANGED' $code.Detail) }
    Write-RunnerStatus 'PREFLIGHT' $RunName
    $pf = Invoke-Preflight $RunName $Cfg (Get-RunArm $IniPath) $State
    if (-not $pf.Ok) {
        if ($pf.Wait -eq 'STOP') { return (New-RunnerResult 'STOPPED' 'STOPPED' 'STOP present; nothing launched') }
        $bad = @($pf.Lines | Where-Object { -not $_.Ok } | ForEach-Object { $_.Name + ': ' + $_.Detail })
        return (New-RunnerResult 'WAIT' ('WAITING_' + $pf.Wait) ($RunName + ' ' + $Stage + ': ' + ($bad -join '; ')))
    }
    # The environment check (a TensorFlow import) only after the preflight passed, so memory
    # pressure from another program makes the runner wait instead of stopping (ENV_ERROR).
    $envCheck = Confirm-RunnerEnvironment
    if (-not $envCheck.Ok) {
        if ($envCheck.Memory) { return (New-RunnerResult 'WAIT' 'WAITING_MEMORY' ($RunName + ' ' + $Stage + ': environment check hit a memory error: ' + $envCheck.Detail)) }
        return (New-RunnerResult 'TERMINAL' 'ENV_ERROR' $envCheck.Detail)
    }
    Write-RunnerStatus 'CLEANUP' $RunName
    if (-not (Invoke-RunnerCleanup)) {
        return (New-RunnerResult 'WAIT' 'WAITING_PORT_BUSY' ('{0} {1}: TCP {2} still busy after {3} s' -f $RunName, $Stage, $QueueTrainPort, $RunnerPortWaitSec))
    }
    if ([IO.File]::Exists($Q.StopFile)) { return (New-RunnerResult 'STOPPED' 'STOPPED' 'STOP present; nothing launched') }
    $job = Start-RunnerLaunch $RunName $Cfg $Stage $levels $State $code.Lines
    Watch-RunnerJob $job $State
}

function Invoke-RunnerIteration {
    $state = Read-QueueState $Q.QueueDir
    $list = Get-QueueRunList
    if ($null -ne $list.Problem) { return (New-RunnerResult 'WAIT' 'WAITING_QUEUE_FILE' $list.Problem) }
    $runs = $list.Runs

    # 1. Adopt a live queue python.
    $job = Find-LiveJob $runs $state @(Get-ProcessTable)
    if ($null -ne $job) { return (Watch-RunnerJob $job $state) }

    # 2. Complete a launch in flight whose python is gone.
    foreach ($name in @($state['runs'].Keys)) {
        $entry = $state['runs'][$name]
        if ($null -eq $entry -or -not $entry.Contains('current_launch') -or $null -eq $entry['current_launch']) { continue }
        if (-not (Test-QueueRunName $name)) { continue }
        $job = ConvertFrom-CurrentLaunch $name $entry['current_launch']
        Write-QueueLog ("{0}: completing {1}, which ended while no runner watched it" -f $name, $job.Label)
        return (Complete-RunnerJob $job $state)
    }

    # 3. Cleanup and STOP.
    Write-RunnerStatus 'CLEANUP' ''
    [void](Invoke-RunnerCleanup)
    if ([IO.File]::Exists($Q.StopFile)) { return (New-RunnerResult 'STOPPED' 'STOPPED' 'STOP present; nothing launched') }

    # 4. The runs, in order.
    foreach ($run in $runs) {
        $ini = Get-RunIniPath $run
        if ($null -eq $ini) { return (New-RunnerResult 'WAIT' 'WAITING_CONFIG' ("{0}: config\runs\{0}.ini not found" -f $run)) }
        $cfg = Read-RunConfig $ini
        $stage = Get-RunStage $Q.Root $run $cfg
        if ($stage -eq 'DONE') {
            $done = Join-Path (Assert-QueueStore $run) 'DONE'
            if (-not [IO.File]::Exists($done)) {
                [void](Write-RunMarker $run 'DONE' ('DONE ' + (Get-Date).ToString('o') + "`r`n"))
                Write-QueueLog ("{0}: eva_data\EVAL_OK present without DONE; wrote DONE" -f $run)
            }
            continue
        }
        if ($stage -eq 'FAILED') { return (Get-FailedResult $run) }
        return (Start-RunStep $run $ini $cfg $stage $state)
    }
    New-RunnerResult 'TERMINAL' 'QUEUE_DONE' ('every run is DONE: ' + ($runs -join ', '))
}

# ---------------------------------------------------------------------------------------------
# Terminal action: disable the queue's scheduled task(s)
# ---------------------------------------------------------------------------------------------
function Select-QueueTasks($Tasks, [string]$TaskName, [bool]$TaskNameGiven, [string]$QueueFileFull, [string]$Root) {
    # The tasks to disable: -TaskName when it was passed explicitly, and every task whose action
    # runs run_queue.ps1 with this queue file. A smoke runner (default -TaskName) therefore never
    # disables the real queue's task, and the real runner finds its own task by its action.
    foreach ($t in @($Tasks)) {
        if ($null -eq $t) { continue }
        if (($TaskNameGiven -and $t.TaskName -eq $TaskName) -or (Test-TaskRunsQueue $t $QueueFileFull $Root)) { $t }
    }
}

function Disable-QueueTask {
    # Failures are logged, never thrown (the task may not exist, or may not be ours to change).
    $tasks = @()
    try { $tasks = @(Get-ScheduledTask -ErrorAction Stop) } catch { Write-QueueLog ('cannot list scheduled tasks: ' + $_.Exception.Message); return }
    $targets = @(Select-QueueTasks $tasks $Q.TaskName $Q.TaskNameGiven $Q.QueueFile $Q.Root)
    if ($targets.Count -eq 0) { Write-QueueLog ("no scheduled task of this queue found to disable (looked for '{0}' and tasks running run_queue.ps1 with {1})" -f $Q.TaskName, $Q.QueueFile); return }
    foreach ($t in $targets) {
        try {
            Disable-ScheduledTask -TaskName $t.TaskName -TaskPath $t.TaskPath -ErrorAction Stop | Out-Null
            Write-QueueLog ("disabled scheduled task '{0}{1}'" -f $t.TaskPath, $t.TaskName)
        } catch { Write-QueueLog ("could not disable scheduled task '{0}': {1}" -f $t.TaskName, $_.Exception.Message) }
    }
}

# ---------------------------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------------------------
function Enter-RunnerMutex($Mutex, [int]$TimeoutMs) {
    # WaitOne (-1 = block until the owner releases it or dies). An AbandonedMutexException
    # (the previous owner died holding it) counts as ownership.
    try { return [bool]$Mutex.WaitOne($TimeoutMs) }
    catch {
        if ($_.Exception -is [System.Threading.AbandonedMutexException] -or $_.Exception.InnerException -is [System.Threading.AbandonedMutexException]) { return $true }
        throw
    }
}

function Get-MutexProbe {
    $m = New-Object System.Threading.Mutex($false, $RunnerMutexName)
    try {
        $got = Enter-RunnerMutex $m 0
        if ($got) { $m.ReleaseMutex(); return ('free (no runner holds {0})' -f $RunnerMutexName) }
        return ('held: a runner is active ({0})' -f $RunnerMutexName)
    } finally { $m.Dispose() }
}

function Get-RunDecision([string]$RunName, $State) {
    # -> { Stage; Text; Actionable; Ini; Cfg } for the dry run (reads only).
    $ini = Get-RunIniPath $RunName
    if ($null -eq $ini) { return [pscustomobject]@{ Stage = 'NO_CONFIG'; Text = ('WAIT (config\runs\{0}.ini not found)' -f $RunName); Actionable = $false; Ini = $null; Cfg = $null } }
    $cfg = Read-RunConfig $ini
    $alive = Test-RunProcessAlive $RunName
    if ($alive) { return [pscustomobject]@{ Stage = 'ADOPT'; Text = 'ADOPT (a python or SUMO of the run is alive; the runner would watch it)'; Actionable = $false; Ini = $ini; Cfg = $cfg } }
    $stage = Get-RunStage $Q.Root $RunName $cfg -ProcessAlive $false
    $store = Assert-QueueStore $RunName
    $counters = $State['runs'][$RunName]
    $text = $stage
    if ($stage -eq 'DONE') {
        if ([IO.File]::Exists((Join-Path $store 'DONE'))) { $text = 'DONE (skip)' } else { $text = 'DONE (eva_data\EVAL_OK present; the runner would write the DONE marker)' }
    } elseif ($stage -eq 'FAILED') {
        $text = 'FAILED ({0}) - the queue stops here' -f (Get-MarkerFirstLine (Join-Path $store 'FAILED'))
    } elseif ($stage -eq 'EVALUATE') {
        $att = 0; $lv = 'all levels'
        if ($null -ne $counters) {
            $att = [int]$counters['eval_attempts']
            $next = Get-StringList $counters['eval_levels_next']
            if ($att -gt 0 -and $next.Count -gt 0) { $lv = 'levels ' + ($next -join ',') }
        }
        $text = 'EVALUATE (training complete; attempt {0} of {1}, {2})' -f ($att + 1), $RunnerMaxEvalAttempts, $lv
    } else {
        $good = Get-GoodCheckpoint $Q.Root $RunName
        $launches = @(Get-LaunchFiles $Q.Root $RunName 'launch')
        $next = Get-NextLaunchNumber $Q.Root $RunName 'launch'
        if ($good -ge 0) { $text = 'TRAIN (resume from checkpoint-{0}; next launch_r{1})' -f $good, $next }
        elseif ($launches.Count -eq 0) { $text = 'TRAIN (no checkpoint)' }
        else { $text = 'TRAIN (no checkpoint; launch_r1..r{0} exist, so Recovery decides scratch or review)' -f $launches[-1].N }
        if ($null -ne $counters) { $text += (' [launches {0}/{1}, in a row without progress {2}/{3}]' -f $counters['launches'], $QueueMaxLaunches, $counters['no_progress'], $QueueMaxNoProgress) }
    }
    [pscustomobject]@{ Stage = $stage; Text = $text; Actionable = ($stage -eq 'TRAIN' -or $stage -eq 'EVALUATE'); Ini = $ini; Cfg = $cfg }
}

function Invoke-DryRun {
    Write-Host ('run_queue.ps1 -DryRun  ' + (Format-Stamp (Get-Date)) + '  (nothing is launched; nothing is written in the project)')
    Write-Host ('ProjectRoot : ' + $Q.Root)
    Write-Host ('QueueFile   : ' + $Q.QueueFile)
    Write-Host ('PythonExe   : ' + $Q.Python)
    Write-Host ('TaskName    : ' + $Q.TaskName)
    Write-Host ('Mutex       : ' + (Get-MutexProbe))
    $stop = 'absent'
    if ([IO.File]::Exists($Q.StopFile)) { $stop = 'PRESENT - the runner would write STOPPED and exit' }
    Write-Host ('STOP        : ' + $stop)
    $list = Get-QueueRunList
    if ($null -ne $list.Problem) { Write-Host ('Queue file  : WAITING_QUEUE_FILE (' + $list.Problem + ')'); return }
    $runs = $list.Runs
    Write-Host ('Queue       : ' + ($runs -join ', '))
    $state = Read-QueueState $Q.QueueDir
    $live = Find-LiveJob $runs $state @(Get-ProcessTable)
    if ($null -ne $live) { Write-Host ('Live job    : ADOPT {0} {1} (python pid {2}); the runner would watch it and launch nothing' -f $live.Run, $live.Label, $live.PyPid) }
    else { Write-Host 'Live job    : none (no python of a queue run is alive)' }
    $envCheck = Test-RunnerEnvironment
    $ev = 'PASS'
    if (-not $envCheck.Ok) { $ev = 'ENV_ERROR' }
    Write-Host ('Environment : {0} ({1})' -f $ev, $envCheck.Detail)

    $decisions = @()
    foreach ($run in $runs) {
        $d = Get-RunDecision $run $state
        $decisions += , @($run, $d)
        Write-Host ('{0}: {1}' -f $run, $d.Text)
    }
    $nextRun = $null
    foreach ($pair in $decisions) {
        if ($pair[1].Stage -eq 'DONE') { continue }
        $nextRun = $pair
        break
    }
    if ($null -eq $nextRun) { Write-Host 'Next action : QUEUE_DONE (every run is DONE)' }
    else { Write-Host ('Next action : {0} {1}' -f $nextRun[0], $nextRun[1].Stage) }

    $cache = @{}
    foreach ($pair in $decisions) {
        $run = $pair[0]; $d = $pair[1]
        if (-not $d.Actionable) { continue }
        Write-Host ('Preflight for {0} ({1}, arm {2}):' -f $run, $d.Stage, (Get-RunArm $d.Ini))
        $code = Test-CodeHashes $run
        $cv = 'PASS'
        if (-not $code.Ok) { $cv = 'FAILED_CODE_CHANGED' }
        Write-Host ('  {0,-14}: {1} ({2})' -f 'code hashes', $cv, $code.Detail)
        $pf = Invoke-Preflight $run $d.Cfg (Get-RunArm $d.Ini) $state $cache
        foreach ($l in $pf.Lines) { Write-Host (Format-PreflightLine $l) }
        $result = 'PASS - the runner would launch now'
        if (-not $code.Ok) { $result = 'FAILED (CODE_CHANGED) - the runner would stop the queue' }
        elseif (-not $envCheck.Ok) { $result = 'ENV_ERROR - the runner would stop the queue' }
        elseif (-not $pf.Ok) { $result = 'WAITING_' + $pf.Wait + ' - the runner would re-check every 5 min' }
        Write-Host ('  {0,-14}: {1}' -f 'result', $result)
    }
    Write-Host 'Nothing launched (DryRun).'
}

# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------
function Wait-WithHeartbeat($Result) {
    # WAITING_<reason>: status every poll, STOP every poll, re-check after 5 min.
    Write-QueueLogOnce 'wait' ('{0}: {1}' -f $Result.Status, $Result.Detail) $Result.Status
    $polls = [int]($RunnerWaitSeconds / $RunnerPollSeconds)
    for ($i = 0; $i -lt $polls; $i++) {
        Write-RunnerStatus $Result.Status $Result.Detail
        if ([IO.File]::Exists($Q.StopFile)) { return }
        Start-Sleep -Seconds $RunnerPollSeconds
    }
}

function Invoke-RunnerMain {
    if ($Q.DryRun) {
        try { Invoke-DryRun }
        finally { if ([IO.Directory]::Exists($Q.ChecksDir)) { Remove-Item -LiteralPath $Q.ChecksDir -Recurse -Force -ErrorAction SilentlyContinue } }
        return
    }
    $mutex = New-Object System.Threading.Mutex($false, $RunnerMutexName)
    $owned = $false
    try {
        $owned = Enter-RunnerMutex $mutex 0
        if (-not $owned) {
            Write-QueueLog ("runner pid {0}: another runner holds {1}; waiting to take over" -f $PID, $RunnerMutexName)
            $owned = Enter-RunnerMutex $mutex -1
        }
        Write-QueueLog ("runner pid {0} started (queue {1}, python {2}, task '{3}')" -f $PID, $Q.QueueFile, $Q.Python, $Q.TaskName)
        Write-RunnerStatus 'STARTING' ''
        $final = $null
        while ($null -eq $final) {
            $res = $null
            try {
                # The result object, even if a helper ever leaks a value into the pipeline.
                $res = @(Invoke-RunnerIteration) | Where-Object { $null -ne $_ -and $null -ne $_.PSObject.Properties['Action'] } | Select-Object -Last 1
                if ($null -eq $res) { throw 'the loop pass returned no result' }
            }
            catch {
                $msg = Format-RunnerError $_
                Write-QueueLog ('ERROR (retrying in 60 s): ' + $msg)
                Write-RunnerStatus 'ERROR' $msg
                Set-KeepAwake $false
                Start-Sleep -Seconds $RunnerPollSeconds
                continue
            }
            if ($res.Action -eq 'CONTINUE') { Start-Sleep -Seconds 5; continue }
            if ($res.Action -eq 'WAIT') { Wait-WithHeartbeat $res; continue }
            $final = $res
        }
        Write-RunnerStatus $final.Status $final.Detail
        Write-QueueLog ('runner exits: {0} {1}' -f $final.Status, $final.Detail)
    } finally {
        Set-KeepAwake $false
        if ($owned) { try { $mutex.ReleaseMutex() } catch { } }
        $mutex.Dispose()
    }
    if ($final.Action -eq 'TERMINAL') { Disable-QueueTask }
}

if ($NoRun -or $MyInvocation.InvocationName -eq '.') { return }

$checks = $null
if ($DryRun) { $checks = Join-Path $env:TEMP ('torrs_queue_dryrun_' + $PID) }
Initialize-RunnerContext -ProjectRoot $ProjectRoot -QueueFile $QueueFile -PythonExe $PythonExe -TaskName $TaskName `
    -TaskNameGiven ($PSBoundParameters.ContainsKey('TaskName')) -DryRun ([bool]$DryRun) -ChecksDir $checks
Invoke-RunnerMain
exit 0
