# Plain-assert tests for tools/run_queue.ps1, the queue runner's main loop (Task 13).
#
# Run:  powershell -NoProfile -ExecutionPolicy Bypass -File tests/test_run_queue.ps1
#
# The runner is dot-sourced with -NoRun, so only its functions are defined. Every case uses a
# fake project root under $env:TEMP whose name has spaces and parentheses (like the real one);
# nothing in the real project is read or written, except tools\run_queue.ps1 itself.
# Two parts start the real python through the launch wrapper (Win32_Process.Create), always on
# a FAKE main.py in the temp folder: the exact command line end to end, and a simulation of the
# main loop (Invoke-RunnerIteration) through crash/resume, evaluation retry, watchdog -> STUCK,
# NaN, STOP in training and in evaluation, the evaluation time limit and adoption. Both are
# skipped while a real runner holds Global\TORRS_queue_runner. The simulation uses TraCI port
# 27599 and no orphan cleanup, so it cannot touch other processes. No real training,
# evaluation or scheduled task is started. About 80 s. Success prints ALL RUN_QUEUE TESTS
# PASSED and exits 0.

$ErrorActionPreference = 'Stop'
$RealRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $RealRoot 'tools\run_queue.ps1') -NoRun
$ErrorActionPreference = 'Stop'

$TestRoot = Join-Path $env:TEMP ('torrs rq test (' + [guid]::NewGuid().ToString('N').Substring(0, 8) + ')')
$Utf8 = New-Object System.Text.UTF8Encoding $false
$script:Asserts = 0
$exitCode = 1

function Check([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
    $script:Asserts++
}

function Put([string]$Path, [string]$Text) {
    $dir = [IO.Path]::GetDirectoryName($Path)
    if (-not [IO.Directory]::Exists($dir)) { [void][IO.Directory]::CreateDirectory($dir) }
    [IO.File]::WriteAllText($Path, $Text, $Utf8)
}

function Get-SharedTextOrEmpty([string]$Path) {
    if ([IO.File]::Exists($Path)) { return (Read-SharedText $Path) }
    ''
}

$SmokeIni = @'
[MODEL_CONFIG]
save_optimizer = true

[TRAIN_CONFIG]
total_step = 288

[ENV_CONFIG]
control_interval_sec = 5
episode_length_sec = 720
seed = 12
test_seeds = 10000,20000
eval_levels = 1.00,1.10
comm_arch = fog
'@

function New-FakeProject([string]$Name) {
    $p = Join-Path $TestRoot $Name
    [void][IO.Directory]::CreateDirectory((Join-Path $p 'Malta\runs\_queue'))
    foreach ($f in 'main.py', 'utils.py', 'run_config.py', 'run_files.py', 'Malta\build_file.py', 'agents\models.py', 'agents\policies.py', 'envs\env.py') {
        Put (Join-Path $p $f) ("# $f`n")
    }
    Put (Join-Path $p 'agents\notes.txt') 'not code'
    Put (Join-Path $p 'config\runs\_smoke_fog.ini') $SmokeIni
    Put (Join-Path $p 'config\queue_smoke.txt') "# smoke`n_smoke_fog`n"
    $p
}

function New-Proc([int]$Id, [int]$Parent, [string]$Name, [string]$CommandLine, [datetime]$Created, [double]$PrivateBytes = 0) {
    [pscustomobject]@{ ProcessId = [uint32]$Id; ParentProcessId = [uint32]$Parent; Name = $Name; CommandLine = $CommandLine; CreationDate = $Created; PrivatePageCount = [uint64]$PrivateBytes }
}

try {
    [void][IO.Directory]::CreateDirectory($TestRoot)
    $t0 = Get-Date '2026-11-16 10:00:00'

    # -----------------------------------------------------------------------------------------
    # Run names, queue file path
    # -----------------------------------------------------------------------------------------
    foreach ($ok in 'cloud_base', 'fog_base', 'hybrid_base', '_smoke_hybrid', '_smoke_fog', 'a', 'x9_') {
        Check (Test-QueueRunName $ok) "run name '$ok' should be valid"
    }
    foreach ($bad in 'Cloud_base', '_smoke_', '_smoke_x1', '_smoke_Fog', '../signal_route', 'signal route', '_other', '1abc', '', 'a-b', 'Malta\signal_route') {
        Check (-not (Test-QueueRunName $bad)) "run name '$bad' should be refused"
    }
    Check ((Resolve-QueueFilePath 'E:\Thesis notebook (2)\Thesis notebook' 'config/queue.txt') -eq 'E:\Thesis notebook (2)\Thesis notebook\config\queue.txt') 'relative queue file resolves against the project root'
    Check ((Resolve-QueueFilePath 'E:\Thesis notebook (2)\Thesis notebook' 'D:\q\queue_x.txt') -eq 'D:\q\queue_x.txt') 'absolute queue file is kept'

    # -----------------------------------------------------------------------------------------
    # The launch command (spec 9), paths with spaces and parentheses
    # -----------------------------------------------------------------------------------------
    $py = 'C:\Program Files\Python311\python.exe'
    $log = 'E:\Thesis notebook (2)\Thesis notebook\Malta\runs\_smoke_fog\logs\launch_r1.log'
    $exitF = 'E:\Thesis notebook (2)\Thesis notebook\Malta\runs\_smoke_fog\logs\launch_r1.exit'
    $cl = New-DetachedCommandLine $py (New-LaunchArguments '_smoke_fog' 'train' $null) $log $exitF
    $want = 'cmd.exe /v:on /c "set PYTHONIOENCODING=utf-8&& set PYTHONUNBUFFERED=1&& "C:\Program Files\Python311\python.exe" -X faulthandler main.py --run _smoke_fog --option train > "E:\Thesis notebook (2)\Thesis notebook\Malta\runs\_smoke_fog\logs\launch_r1.log" 2>&1 & echo EXIT=!errorlevel! at !date! !time! > "E:\Thesis notebook (2)\Thesis notebook\Malta\runs\_smoke_fog\logs\launch_r1.exit""'
    Check ($cl -ceq $want) "launch command differs:`n$cl`n$want"
    Check ((New-LaunchArguments 'cloud_base' 'evaluate' @('1.10', '2.00')) -ceq '-X faulthandler main.py --run cloud_base --option evaluate --levels 1.10,2.00') 'evaluate with --levels'
    Check ((New-LaunchArguments 'cloud_base' 'evaluate' @()) -ceq '-X faulthandler main.py --run cloud_base --option evaluate') 'no --levels for an empty list'
    Check ((New-LaunchArguments 'cloud_base' 'evaluate' (Get-StringList @('1.10'))) -ceq '-X faulthandler main.py --run cloud_base --option evaluate --levels 1.10') 'one level from Get-StringList'
    foreach ($bad in @(
            { New-LaunchArguments 'Cloud' 'train' $null },
            { New-LaunchArguments 'cloud_base' 'greedy' $null },
            { New-LaunchArguments 'cloud_base' 'train' @('1.10') },
            { New-LaunchArguments 'cloud_base' 'evaluate' @('1.1') },
            { New-LaunchArguments 'cloud_base' 'evaluate' @('1.10 && del x') },
            { New-DetachedCommandLine $py 'x' 'E:\a"b\l.log' $exitF },
            { New-DetachedCommandLine $py 'x' 'E:\a!b\l.log' $exitF },
            { New-DetachedCommandLine $py 'x' $log 'E:\100%\l.exit' },
            { New-DetachedCommandLine 'C:\py^\python.exe' 'x' $log $exitF },
            { New-DetachedCommandLine $py '-c "print(1)" !x!' $log $exitF })) {
        $threw = $false
        try { [void](& $bad) } catch { $threw = $true }
        Check $threw ('should refuse: ' + $bad.ToString().Trim())
    }

    # -----------------------------------------------------------------------------------------
    # Small helpers
    # -----------------------------------------------------------------------------------------
    $lv = ConvertTo-LevelValues @('1.10', '2.00')
    Check ($lv -is [double[]] -and $lv.Count -eq 2 -and $lv[0] -eq 1.1 -and $lv[1] -eq 2.0) 'ConvertTo-LevelValues'
    Check ($null -eq (ConvertTo-LevelValues @())) 'no levels -> $null'
    Check ($null -eq (ConvertTo-LevelValues $null)) '$null levels -> $null'
    $sl = Get-StringList @('a', '', $null, 'b')
    Check ($sl.Count -eq 2 -and $sl[1] -eq 'b') 'Get-StringList drops empty entries'
    Check ((Get-StringList $null).Count -eq 0) 'Get-StringList of $null'
    $lt = Get-LogLineTime '2026-11-16 10:22:33,456 [INFO] Checkpoint saved at step 144 -> x' $t0
    Check ($lt -eq (Get-Date '2026-11-16 10:22:33')) 'log line time parsed'
    Check ((Get-LogLineTime 'test 0, avg reward -1.0' $t0) -eq $t0) 'no timestamp -> fallback'
    $iso = $t0.ToString('o')
    Check ((ConvertTo-LocalTime $iso) -eq $t0) 'ISO time round-trips'
    Check ($null -eq (ConvertTo-LocalTime 'not a time')) 'bad time -> $null'

    # -----------------------------------------------------------------------------------------
    # Verdicts
    # -----------------------------------------------------------------------------------------
    $v = Get-CommitVerdict (96GB) (20GB) 60 8
    Check ($v.Ok -and $v.Detail -match 'headroom 76\.0 GB') "commit 76 GB headroom passes: $($v.Detail)"
    $v = Get-CommitVerdict (63GB) (17GB) 60 8
    Check (-not $v.Ok -and $v.Detail -match 'need 68\.0 GB') "commit 46 GB headroom waits: $($v.Detail)"
    $v = Get-CommitVerdict (96GB) (28GB) 60 8
    Check ($v.Ok) 'exactly the needed headroom passes'
    Check ((Get-DiskVerdict 'E:' (61GB) 60).Ok) 'disk 61 GB passes'
    Check (-not (Get-DiskVerdict 'C:' (9GB) 10).Ok) 'disk 9 GB < 10 GB waits'

    $w = Get-TrainWatchdogReason -Now $t0.AddMinutes(239) -Started $t0 -LastCheckpoint $null -FirstLimitMin 240 -GapMin 90 -SaveErrorAt $null
    Check ($null -eq $w) 'no checkpoint at 239 min of 240 is fine'
    $w = Get-TrainWatchdogReason -Now $t0.AddMinutes(241) -Started $t0 -LastCheckpoint $null -FirstLimitMin 240 -GapMin 90 -SaveErrorAt $null
    Check ($w -match 'no Checkpoint saved line 241 min after the launch') "first-checkpoint limit: $w"
    $w = Get-TrainWatchdogReason -Now $t0.AddMinutes(500) -Started $t0 -LastCheckpoint $t0.AddMinutes(420) -FirstLimitMin 240 -GapMin 90 -SaveErrorAt $null
    Check ($null -eq $w) '80 min after a checkpoint is fine'
    $w = Get-TrainWatchdogReason -Now $t0.AddMinutes(520) -Started $t0 -LastCheckpoint $t0.AddMinutes(420) -FirstLimitMin 240 -GapMin 90 -SaveErrorAt $null
    Check ($w -match 'for 100 min') "gap limit: $w"
    $w = Get-TrainWatchdogReason -Now $t0.AddMinutes(10) -Started $t0 -LastCheckpoint $null -FirstLimitMin 240 -GapMin 90 -SaveErrorAt $t0.AddMinutes(9)
    Check ($null -eq $w) 'a save error gets a 2 min grace'
    $w = Get-TrainWatchdogReason -Now $t0.AddMinutes(12) -Started $t0 -LastCheckpoint $null -FirstLimitMin 240 -GapMin 90 -SaveErrorAt $t0.AddMinutes(9)
    Check ($w -match 'checkpoint-save error') "save error kills: $w"

    Check ((Get-EvalLevelLimitMinutes 50.0 $null 10) -eq 150) '3 x measured f1.00'
    Check ((Get-EvalLevelLimitMinutes 10.0 $null 10) -eq 90) 'at least 90 min'
    Check ((Get-EvalLevelLimitMinutes $null ([double[]]@(300, 600, 900)) 10) -eq 300) 'estimate: 3 x 10 seeds x median 600 s'
    Check ((Get-EvalLevelLimitMinutes $null ([double[]]@(300, 500, 700, 900)) 2) -eq 90) 'even-count median, floor 90'
    Check ((Get-EvalLevelLimitMinutes $null $null 10) -eq 360) 'no data -> 360 min'
    Check ($null -eq (Get-EvalTimeoutReason $t0.AddMinutes(200) $t0 $false $null 240 150)) 'pre-level phase within 240 min'
    Check ((Get-EvalTimeoutReason $t0.AddMinutes(241) $t0 $false $null 240 150) -match 'no EVAL_LEVEL_START') 'pre-level phase over 240 min'
    Check ((Get-EvalTimeoutReason $t0.AddMinutes(151) $t0 $true '2.00' 240 150) -match 'level f2\.00 exceeded') 'level over its limit'
    Check ($null -eq (Get-EvalTimeoutReason $t0.AddMinutes(149) $t0 $true '2.00' 240 150)) 'level within its limit'

    # Final-review fix 1: a stress level above 1.20 gets 8 x the f1.00 time, at least 360 min;
    # levels up to 1.20 keep 3 x, at least 90 min.
    Check ((Get-EvalLevelLimitMinutes 50.0 $null 10 '2.00') -eq 400) 'f2.00: 8 x measured f1.00'
    Check ((Get-EvalLevelLimitMinutes 30.0 $null 10 '2.00') -eq 360) 'f2.00: at least 360 min'
    Check ((Get-EvalLevelLimitMinutes 50.0 $null 10 2.0) -eq 400) 'f2.00 given as a number'
    Check ((Get-EvalLevelLimitMinutes $null ([double[]]@(300, 600, 900)) 10 '2.00') -eq 800) 'f2.00 estimate: 8 x 10 seeds x median 600 s'
    Check ((Get-EvalLevelLimitMinutes $null $null 10 '2.00') -eq 360) 'f2.00 with no data: 360 min'
    foreach ($lvl in '1.00', '1.06', '1.10', '1.13', '1.20') {
        Check ((Get-EvalLevelLimitMinutes 50.0 $null 10 $lvl) -eq 150) "f$lvl keeps 3 x measured f1.00"
        Check ((Get-EvalLevelLimitMinutes 10.0 $null 10 $lvl) -eq 90) "f$lvl keeps the 90 min floor"
    }
    Check ((Get-EvalLevelLimitMinutes 50.0 $null 10 1.2) -eq 150) '1.2 as a number is not a stress level'
    Check ((Test-StressLevel '1.21') -and -not (Test-StressLevel '1.20') -and -not (Test-StressLevel $null) -and -not (Test-StressLevel 'x')) 'Test-StressLevel'
    $saved = [Threading.Thread]::CurrentThread.CurrentCulture
    try {
        [Threading.Thread]::CurrentThread.CurrentCulture = [cultureinfo]'de-DE'
        Check ((Get-EvalLevelLimitMinutes 50.0 $null 10 2.0) -eq 400 -and (Get-EvalLevelLimitMinutes 50.0 $null 10 '1.20') -eq 150) 'stress level under de-DE'
    } finally { [Threading.Thread]::CurrentThread.CurrentCulture = $saved }
    $sj = [pscustomobject]@{ CurrentLevel = $null; LevelSeen = $false; DoneLevels = 0; Levels = @(); Cfg = [pscustomobject]@{ EvalLevels = [double[]]@(1.0, 1.06, 1.1, 1.13, 1.2, 2.0) } }
    Check ($null -eq (Get-JobLevelForLimit $sj)) 'no level before the first START'
    $sj.LevelSeen = $true; $sj.CurrentLevel = '1.20'; $sj.DoneLevels = 4
    Check ((Get-JobLevelForLimit $sj) -eq '1.20') 'the level in progress'
    $sj.CurrentLevel = $null; $sj.DoneLevels = 5
    Check ((Get-JobLevelForLimit $sj) -eq '2.00') 'after f1.20 DONE the next level (f2.00) sets the limit'
    $sj.DoneLevels = 6
    Check ($null -eq (Get-JobLevelForLimit $sj)) 'after the last level: none'
    $sj.Levels = @('1.10', '2.00'); $sj.DoneLevels = 1
    Check ((Get-JobLevelForLimit $sj) -eq '2.00') 'a --levels retry: the next of its levels'

    # Final-review fix 3: the gap limit is a parameter (arms.<arm>.checkpoint_gap_min).
    $w = Get-TrainWatchdogReason -Now $t0.AddMinutes(520) -Started $t0 -LastCheckpoint $t0.AddMinutes(420) -FirstLimitMin 240 -GapMin 150 -SaveErrorAt $null
    Check ($null -eq $w) 'a 150 min gap limit allows 100 min'
    $w = Get-TrainWatchdogReason -Now $t0.AddMinutes(420.1) -Started $t0 -LastCheckpoint $t0.AddMinutes(420) -FirstLimitMin 240 -GapMin 0.05 -SaveErrorAt $null
    Check ($w -match 'for 0\.1 min \(limit 0\.05 min\)') "a fractional gap limit is printed: $w"

    # Refusals (fix round 1): exit 2 or 4 AND main.py's refusal text at the end of the log.
    $abortTail = "2026-11-16 10:00:00,000 [INFO] step 100`r`nFatal Python error: Aborted`r`n`r`nCurrent thread 0x00001234 (most recent call first):`r`n  File `"main.py`", line 1 in <module>`r`n"
    Check (-not (Test-RefusalExit 3 $abortTail)) 'exit 3 from a native abort is not a refusal'
    Check (-not (Test-RefusalExit 3 'refusing to write into RQ1 store Malta/signal_route/ without --rq1')) 'exit 3 is never a refusal'
    Check (Test-RefusalExit 4 "x`r`nresume refused for Malta/runs/fog_base/: config differs: ENV_CONFIG.seed`r`n") 'exit 4 with the resume refusal'
    Check (Test-RefusalExit 4 'evaluate refused for Malta/runs/fog_base/: neighbour hash differs: stored=a current=b') 'exit 4 with the evaluate refusal'
    Check (-not (Test-RefusalExit 4 "Traceback (most recent call last):`r`nOSError: disk`r`n")) 'exit 4 without the refusal text'
    Check (Test-RefusalExit 2 "invalid run name: 'X'`r`n") 'exit 2: invalid run name'
    Check (Test-RefusalExit 2 "refusing: invalid eval_levels '1,x'") 'exit 2: refusing'
    Check (Test-RefusalExit 2 'main.py: error: argument --option: invalid choice') 'exit 2: argparse error'
    Check (-not (Test-RefusalExit 2 "Traceback (most recent call last):`r`nValueError: something odd`r`n")) 'exit 2 without the refusal text'
    Check (-not (Test-RefusalExit 2 ("refusing: x`r`n" + (((1..6) | ForEach-Object { "later line $_" }) -join "`r`n")))) 'refusal text far above the end of the log does not count'
    Check (-not (Test-RefusalExit 6 'refusing: another main.py --run is running (pid 5)')) 'exit 6 is not a refusal'
    Check (-not (Test-RefusalExit $null 'refusing: x')) 'no exit code'
    Check (Test-MemoryErrorText 'OSError: [WinError 1455] The paging file is too small for this operation to complete') 'paging file text'
    Check (Test-MemoryErrorText 'numpy.core._exceptions._ArrayMemoryError: MemoryError: Unable to allocate 1.2 GiB') 'MemoryError text'
    Check (Test-MemoryErrorText 'ResourceExhaustedError: OOM ... out of memory') 'out of memory text'
    Check (-not (Test-MemoryErrorText "ModuleNotFoundError: No module named 'tensorflow'")) 'an import error is not a memory error'
    Check (-not (Test-MemoryErrorText $null)) 'no text'

    $smokeCfg = [pscustomobject]@{ TotalStep = 288; T = 144; Seed = 12; TestSeeds = [int[]]@(10000, 20000); EvalLevels = [double[]]@(1.0, 1.1) }
    $ta = Get-TrafficCheckArgs $smokeCfg
    Check ($ta.Train -ceq 'tools/gen_traffic.py --seeds 12-13 --factors 1.0 --check-only') "smoke train traffic args: $($ta.Train)"
    Check ($ta.Test -ceq 'tools/gen_traffic.py --seeds 10000,20000 --factors 1.00,1.10 --check-only') "smoke test traffic args: $($ta.Test)"
    $realCfg = [pscustomobject]@{ TotalStep = 72000; T = 720; Seed = 12; TestSeeds = [int[]]@(10000, 20000, 30000); EvalLevels = [double[]]@(1.0, 1.06, 1.1, 1.13, 1.2, 2.0) }
    $ta = Get-TrafficCheckArgs $realCfg
    Check ($ta.Train -ceq 'tools/gen_traffic.py --seeds 12-111 --factors 1.0 --check-only') "real train traffic args: $($ta.Train)"
    Check ($ta.Test -ceq 'tools/gen_traffic.py --seeds 10000,20000,30000 --factors 1.00,1.06,1.10,1.13,1.20,2.00 --check-only') "real test traffic args: $($ta.Test)"

    $state = [ordered]@{ version = 1; runs = [ordered]@{}; arms = [ordered]@{ cloud = [ordered]@{ peak_commit_gb = 71.5; first_checkpoint_min = 'x' } } }
    Check ((Get-ArmSetting $state 'cloud' 'peak_commit_gb' 60) -eq 71.5) 'arms.cloud.peak_commit_gb'
    Check ((Get-ArmSetting $state 'cloud' 'first_checkpoint_min' 240) -eq 240) 'a non-number falls back'
    Check ((Get-ArmSetting $state 'fog' 'peak_commit_gb' 60) -eq 60) 'a missing arm falls back'
    Check ((Get-ArmSetting ([ordered]@{ runs = [ordered]@{} }) 'fog' 'peak_commit_gb' 60) -eq 60) 'no arms key falls back'

    # -----------------------------------------------------------------------------------------
    # Machine free, process trees, cleanup targets
    # -----------------------------------------------------------------------------------------
    $procs = @(
        (New-Proc 100 1 'python.exe' '"C:\Program Files\Python311\python.exe" -X faulthandler main.py --run fog_base --option train' $t0),
        (New-Proc 101 1 'python.exe' $null $t0),
        (New-Proc 102 1 'python.exe' 'python tools\check_env.py --arm fog --port 20' $t0),
        (New-Proc 103 1 'pythonw.exe' 'pythonw -m ipykernel_launcher -f x.json' $t0),
        (New-Proc 104 1 'python.exe' 'python "E:\T (2)\tools\gen_traffic.py" --seeds test' $t0),
        (New-Proc 105 1 'python.exe' 'python mymain.py --x' $t0),
        (New-Proc 106 1 'sumo.exe' 'sumo -c x.sumocfg --remote-port 27520' $t0),
        (New-Proc 107 1 'notepad.exe' 'notepad' $t0),
        (New-Proc 108 1 'NETCONVERT.exe' 'netconvert' $t0),
        (New-Proc 109 1 'python.exe' 'python baselines.py actuated_rerouting' $t0))
    $busy = @(Get-BusyReasons $procs $null)
    Check ($busy.Count -eq 7) "7 busy processes expected, got $($busy.Count): $($busy -join '; ')"
    Check (($busy -join ';') -match 'pid 100 runs main\.py' -and ($busy -join ';') -match 'pid 101 \(command line not readable\)' -and ($busy -join ';') -match 'pid 103 runs ipykernel' -and
           ($busy -join ';') -match 'pid 104 runs gen_traffic\.py' -and ($busy -join ';') -match 'sumo\.exe pid 106' -and ($busy -join ';') -match 'netconvert\.exe pid 108' -and ($busy -join ';') -match 'pid 109 runs baselines\.py') "busy reasons: $($busy -join '; ')"
    $ex = New-Object 'System.Collections.Generic.HashSet[int]'
    foreach ($i in 100, 101, 103, 104, 106, 108, 109) { [void]$ex.Add($i) }
    Check (@(Get-BusyReasons $procs $ex).Count -eq 0) 'excluded ids are not busy'

    $tree = @(
        (New-Proc 10 1 'cmd.exe' 'cmd' $t0 1GB),
        (New-Proc 11 10 'python.exe' 'python main.py' $t0.AddSeconds(1) 40GB),
        (New-Proc 12 11 'sumo.exe' 'sumo' $t0.AddSeconds(2) 2GB),
        (New-Proc 13 10 'python.exe' 'old' $t0.AddSeconds(-5) 9GB),
        (New-Proc 14 99 'x.exe' 'x' $t0 9GB))
    $ids = Get-TreeIds $tree 10
    Check ($ids -is [System.Collections.Generic.HashSet[int]] -and $ids.Count -eq 3 -and $ids.Contains(12) -and -not $ids.Contains(13)) 'tree: root, child, grandchild; an older process with a reused parent id is excluded'
    Check ((Get-TreeIds $tree 55).Count -eq 0) 'tree of a missing root is empty'
    $fakeJob = [pscustomobject]@{ CmdPid = 10; PyPid = 11 }
    Check ([Math]::Abs((Get-JobCommitGB $fakeJob $tree) - 43.0) -lt 1e-9) 'job commit = sum of the tree'

    $ct = @(
        (New-Proc 20 1 'python.exe' 'python main.py --run fog_base --option train' $t0),
        (New-Proc 21 20 'python.exe' 'python "C:\sumo\tools\randomTrips.py" -o Malta/data/_tmp_x/bg.trips.xml' $t0.AddSeconds(1)),
        (New-Proc 22 21 'duarouter.exe' 'duarouter -o Malta\data\_tmp_x\bg.rou.xml' $t0.AddSeconds(2)),
        (New-Proc 23 777 'duarouter.exe' 'duarouter -o Malta\data\_tmp_y\bg.rou.xml' $t0),
        (New-Proc 30 1 'python.exe' 'python tools/gen_traffic.py --seeds 12-111' $t0),
        (New-Proc 31 30 'python.exe' 'python randomTrips.py -o Malta/data/_tmp_z/x.xml' $t0.AddSeconds(1)),
        (New-Proc 32 31 'duarouter.exe' 'duarouter -o Malta/data/_tmp_z/x.rou.xml' $t0.AddSeconds(2)),
        (New-Proc 33 1 'duarouter.exe' 'duarouter -o C:\elsewhere\x.rou.xml' $t0.AddSeconds(3)),
        (New-Proc 34 35 'duarouter.exe' 'duarouter -o Malta\data\a.rou.xml' $t0),
        (New-Proc 35 1 'explorer.exe' 'explorer' $t0.AddSeconds(9)))
    $byId = @{}
    foreach ($p in $ct) { $byId[[int]$p.ProcessId] = $p }
    $targets = @($ct | Where-Object { Test-CleanupTarget $_ $byId } | ForEach-Object { [int]$_.ProcessId })
    Check (($targets -join ',') -eq '21,23,34') "cleanup targets: a main.py child, an orphan, a reused parent id; not gen_traffic's children: got $($targets -join ',')"

    # -----------------------------------------------------------------------------------------
    # Fake project: stores, code hashes, decisions, adoption, log reading
    # -----------------------------------------------------------------------------------------
    $proj = New-FakeProject 'p1 (x)'
    Initialize-RunnerContext -ProjectRoot $proj -QueueFile 'config/queue_smoke.txt' -PythonExe $py -DryRun $true -ChecksDir (Join-Path $proj 'Malta\runs\_queue\checks')
    Check ($Q.QueueFile -eq (Join-Path $proj 'config\queue_smoke.txt')) 'context resolves the queue file'
    Check ((Assert-QueueStore '_smoke_fog') -eq (Join-Path $proj 'Malta\runs\_smoke_fog')) 'store of a smoke run'
    foreach ($bad in '..\signal_route', 'signal_route\x', 'Fog') {
        $threw = $false
        try { [void](Assert-QueueStore $bad) } catch { $threw = $true }
        Check $threw "Assert-QueueStore should refuse '$bad'"
    }
    $list = Get-QueueRunList
    Check ($null -eq $list.Problem -and @($list.Runs).Count -eq 1 -and $list.Runs[0] -eq '_smoke_fog') 'queue run list'
    Put (Join-Path $proj 'config\queue_bad.txt') "_smoke_fog`n../signal_route`n"
    Initialize-RunnerContext -ProjectRoot $proj -QueueFile 'config\queue_bad.txt' -PythonExe $py -DryRun $true
    Check ((Get-QueueRunList).Problem -match 'invalid run name') 'a bad queue entry is reported, not used'
    Initialize-RunnerContext -ProjectRoot $proj -QueueFile 'config/queue_smoke.txt' -PythonExe $py -DryRun $true -ChecksDir (Join-Path $proj 'Malta\runs\_queue\checks')

    $ini = Get-RunIniPath '_smoke_fog'
    Check ($ini -eq (Join-Path $proj 'config\runs\_smoke_fog.ini')) 'ini falls back to config\runs before the first launch'
    Check ((Get-RunArm $ini) -eq 'fog') 'arm from comm_arch'
    Check ($null -eq (Get-RunIniPath 'nothing_here')) 'no ini -> $null'
    # Final-review fix 3: the job's limits come from the run's arm in state.json.
    $lj0 = New-JobRecord -RunName '_smoke_fog' -Cfg $smokeCfg -Kind 'train' -N 1 -Started $t0 -CmdPid $null -PyPid 1 -GoodBefore -1 -Levels $null
    Set-JobLimits $lj0 ([ordered]@{ version = 1; runs = [ordered]@{} })
    Check ($lj0.FirstLimitMin -eq 240 -and $lj0.GapMin -eq 90) 'default limits: 240 min first checkpoint, 90 min gap'
    Set-JobLimits $lj0 ([ordered]@{ version = 1; runs = [ordered]@{}; arms = [ordered]@{ fog = [ordered]@{ first_checkpoint_min = 300; checkpoint_gap_min = 125.5 }; cloud = [ordered]@{ checkpoint_gap_min = 10 } } })
    Check ($lj0.FirstLimitMin -eq 300 -and $lj0.GapMin -eq 125.5) "arms.fog limits are honoured: $($lj0.FirstLimitMin) / $($lj0.GapMin)"
    Set-JobLimits $lj0 ([ordered]@{ version = 1; runs = [ordered]@{}; arms = [ordered]@{ fog = [ordered]@{ checkpoint_gap_min = 0 } } })
    Check ($lj0.GapMin -eq 90) 'a zero gap falls back to 90 min'

    $st = Read-QueueState $Q.QueueDir
    $d = Get-RunDecision '_smoke_fog' $st
    Check ($d.Stage -eq 'TRAIN' -and $d.Text -eq 'TRAIN (no checkpoint)' -and $d.Actionable) "decision for a new run: $($d.Text)"
    Check (-not [IO.Directory]::Exists((Join-Path $proj 'Malta\runs\_smoke_fog'))) 'the decision creates no store'

    $c1 = Test-CodeHashes '_smoke_fog'
    Check ($c1.Ok -and $c1.Detail -match 'first launch') 'first launch passes'
    Check (@($c1.Lines).Count -eq 9) "9 hashed files expected (5 + agents 2 + envs 1 + config), got $(@($c1.Lines).Count)"
    Check ((@($c1.Lines) -join "`n") -match 'agents\\models\.py' -and (@($c1.Lines) -join "`n") -notmatch 'notes\.txt' -and (@($c1.Lines) -join "`n") -match '  config\.ini') 'hash list content'
    $logs = Join-Path $proj 'Malta\runs\_smoke_fog\logs'
    Put (Join-Path $logs 'launch_r1.code.txt') ("# header`r`n" + (@($c1.Lines) -join "`r`n") + "`r`n")
    # main.py copies the ini byte for byte into the store on the first launch.
    Put (Join-Path $proj 'Malta\runs\_smoke_fog\config.ini') $SmokeIni
    $c2 = Test-CodeHashes '_smoke_fog'
    Check ($c2.Ok -and $c2.Detail -match 'same as launch_r1\.code\.txt') "store config.ini equals the source ini: $($c2.Detail)"
    Put (Join-Path $proj 'Malta\runs\_smoke_fog\config.ini') ($SmokeIni + "`n; edited`n")
    $cc = Test-CodeHashes '_smoke_fog'
    Check (-not $cc.Ok -and $cc.Detail -match 'config\.ini') "a changed store config.ini is a code change: $($cc.Detail)"
    Put (Join-Path $proj 'Malta\runs\_smoke_fog\config.ini') $SmokeIni
    Put (Join-Path $logs 'launch_r3.code.txt') "# later, ignored for the comparison`r`nxx  main.py`r`n"
    Put (Join-Path $proj 'utils.py') "# changed`n"
    Put (Join-Path $proj 'envs\new_env.py') "# new`n"
    $c3 = Test-CodeHashes '_smoke_fog'
    Check (-not $c3.Ok -and $c3.Detail -match 'launch_r1\.code\.txt' -and $c3.Detail -match 'utils\.py' -and $c3.Detail -match 'envs\\new_env\.py \(new\)' -and $c3.Detail -notmatch 'main\.py') "code change detected: $($c3.Detail)"
    Check (@(Compare-CodeHashLines @('aa  x', 'bb  y') @('aa  x', 'bb  y')).Count -eq 0) 'equal hash lists'
    Check ((@(Compare-CodeHashLines @('aa  x', 'bb  y') @('aa  x')) -join ',') -eq 'y (gone)') 'a removed file'

    # The NaN check on the last train_reward.csv row.
    $store = Join-Path $proj 'Malta\runs\_smoke_fog'
    $hdr = ',agent,step,test_id,reward,total arrived vehicle number,total departured vehicle number,total waiting time,total travel time,average waiting time,average travel time,seed'
    Put (Join-Path $store 'train_reward.csv') ($hdr + "`r`n0,ma2c,144,-1,-12.5,320,330,4000.0,90000.0,12.5,272.7,12`r`n")
    $nan = Test-LastTrainRow $store
    Check ($nan.Checked -and $nan.Finite) 'finite last row'
    Put (Join-Path $store 'train_reward.csv') ($hdr + "`r`n0,ma2c,144,-1,-12.5,320,330,4000.0,90000.0,12.5,272.7,12`r`n1,ma2c,288,-1,,320,330,4000.0,90000.0,12.5,272.7,13`r`n")
    $nan = Test-LastTrainRow $store
    Check ($nan.Checked -and -not $nan.Finite -and $nan.Detail -match "row 2: reward ''") "empty (NaN) reward: $($nan.Detail)"
    Put (Join-Path $store 'train_reward.csv') ($hdr + "`r`n0,ma2c,144,-1,-12.5,320,330,4000.0,90000.0,inf,272.7,12`r`n")
    Check (-not (Test-LastTrainRow $store).Finite) 'inf wait'
    Remove-Item -LiteralPath (Join-Path $store 'train_reward.csv')
    Check (-not (Test-LastTrainRow $store).Checked) 'a missing file is not checked'

    # Incremental log reading: complete lines only, partial lines and UTF-8 kept for later.
    $job = New-JobRecord -RunName '_smoke_fog' -Cfg $smokeCfg -Kind 'train' -N 2 -Started $t0 -CmdPid 10 -PyPid 11 -GoodBefore 144 -Levels $null
    Check ($job.Log -eq (Join-Path $logs 'launch_r2.log') -and $job.Exit -eq (Join-Path $logs 'launch_r2.exit') -and $job.Label -eq 'launch_r2') 'job paths'
    Check (@(Read-NewLogLines $job).Count -eq 0) 'no log yet'
    $fs = [IO.FileStream]::new($job.Log, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::ReadWrite)
    try {
        $write = { param([string]$s) $b = $Utf8.GetBytes($s); $fs.Write($b, 0, $b.Length); $fs.Flush() }
        & $write "line one`r`nline tw"
        $got = @(Read-NewLogLines $job)
        Check ($got.Count -eq 1 -and $got[0] -eq 'line one' -and $job.Offset -eq 10) "first read: $($got -join '|') offset $($job.Offset)"
        & $write "o`r`n"
        $b = $Utf8.GetBytes([string][char]0x017C)    # z with dot above, 2 bytes
        $fs.Write($b, 0, 1); $fs.Flush()
        $got = @(Read-NewLogLines $job)
        Check ($got.Count -eq 1 -and $got[0] -eq 'line two') "partial line completed: $($got -join '|')"
        $fs.Write($b, 1, 1); & $write "ebbug`n"
        $got = @(Read-NewLogLines $job)
        Check ($got.Count -eq 1 -and $got[0] -eq ([string][char]0x017C + 'ebbug')) 'a multi-byte character split across reads'
        Check (@(Read-NewLogLines $job).Count -eq 0) 'nothing new'
    } finally { $fs.Dispose() }

    # Training lines: this run's checkpoints, not another run's; save errors via a traceback.
    $own = '2026-11-16 10:30:00,001 [INFO] Checkpoint saved at step 288 -> Malta/runs/_smoke_fog/model/'
    $other = '2026-11-16 10:31:00,001 [INFO] Checkpoint saved at step 432 -> Malta/runs/_smoke_fogx/model/'
    Update-JobFromLine $job $other
    Check ($job.CkptCount -eq 0) "another run's checkpoint line is ignored"
    Update-JobFromLine $job $own
    Check ($job.CkptCount -eq 1 -and $job.LastCkptStep -eq 288 -and $job.LastCkptTime -eq (Get-Date '2026-11-16 10:30:00') -and [Math]::Abs($job.FirstCkptMin - 30) -lt 1e-9) 'own checkpoint line'
    Update-JobFromLine $job '2026-11-16 10:55:00,001 [INFO] Checkpoint saved at step 432 -> Malta/runs/_smoke_fog/model/'
    Check ($job.CkptCount -eq 2 -and [Math]::Abs($job.MaxGapMin - 25) -lt 1e-9) 'gap between checkpoints'
    Update-JobFromLine $job 'Traceback (most recent call last):'
    Update-JobFromLine $job '  File "E:\x\main.py", line 220, in train'
    Check ($null -eq $job.SaveErrorAt) 'a traceback alone is not a save error'
    Update-JobFromLine $job '  File "E:\x\utils.py", line 466, in _persist_episode'
    Check ($null -ne $job.SaveErrorAt) 'a traceback through _persist_episode is a save error'
    $job2 = New-JobRecord -RunName '_smoke_fog' -Cfg $smokeCfg -Kind 'train' -N 3 -Started $t0 -CmdPid $null -PyPid 11 -GoodBefore -1 -Levels $null
    Update-JobFromLine $job2 '2026-11-16 10:29:00,001 [INFO] csv write retried in _persist_episode'
    Check ($null -eq $job2.SaveErrorAt) 'a line naming _persist_episode outside a traceback is not a save error'
    Update-JobFromLine $job2 '2026-11-16 10:30:00,001 [WARNING] Per-episode checkpoint save failed: disk full'
    Check ($null -ne $job2.SaveErrorAt) 'the non-strict save warning is a save error'

    # Evaluation lines: level timing, f1.00 measured.
    $ej = New-JobRecord -RunName '_smoke_fog' -Cfg $smokeCfg -Kind 'evaluate' -N 1 -Started $t0 -CmdPid 10 -PyPid 11 -GoodBefore -1 -Levels @('1.10')
    Check ($ej.Log -eq (Join-Path $logs 'eval_r1.log') -and $ej.Levels.Count -eq 1) 'eval job paths and levels'
    Update-JobFromLine $ej '2026-11-16 10:10:00,000 [INFO] EVAL_LEVEL_START f1.00 bg_tag=p1.2 corridor_total=100 corridor_expected=100'
    Check ($ej.LevelSeen -and $ej.CurrentLevel -eq '1.00' -and $ej.PhaseStart -eq (Get-Date '2026-11-16 10:10:00')) 'level start'
    Update-JobFromLine $ej '2026-11-16 10:52:00,000 [INFO] EVAL_LEVEL_DONE f1.00 seeds=2'
    Check ($null -eq $ej.CurrentLevel -and [Math]::Abs($ej.F100Min - 42) -lt 1e-9 -and $ej.DoneLevels -eq 1) 'f1.00 measured at 42 min'
    Update-JobFromLine $ej '2026-11-16 10:53:00,000 [INFO] EVAL_LEVEL_START f1.10 bg_tag=p1.091 corridor_total=110 corridor_expected=110'
    Update-JobFromLine $ej '2026-11-16 11:20:00,000 [INFO] EVAL_LEVEL_DONE f1.10 seeds=2'
    Check ([Math]::Abs($ej.F100Min - 42) -lt 1e-9 -and $ej.DoneLevels -eq 2) 'other levels do not change f1.00'

    # Adoption from a process table: the wrapper's command line names the launch.
    $storeLogs = Join-Path $proj 'Malta\runs\_smoke_fog\logs'
    $wrapTrain = 'cmd.exe /v:on /c "set PYTHONIOENCODING=utf-8&& ... -X faulthandler main.py --run _smoke_fog --option train > "' + $storeLogs + '\launch_r4.log" 2>&1 & echo ..."'
    $tbl = @(
        (New-Proc 500 1 'cmd.exe' $wrapTrain $t0),
        (New-Proc 501 500 'python.exe' '"C:\Program Files\Python311\python.exe" -X faulthandler main.py --run _smoke_fog --option train' $t0.AddSeconds(1)),
        (New-Proc 502 1 'python.exe' 'python main.py --run other_run --option train' $t0))
    $st = Read-QueueState $Q.QueueDir
    $aj = Find-LiveJob @('_smoke_fog') $st $tbl
    Check ($null -ne $aj -and -not $aj.Foreign -and $aj.Kind -eq 'train' -and $aj.N -eq 4 -and $aj.CmdPid -eq 500 -and $aj.PyPid -eq 501 -and $aj.GoodBefore -eq -1) 'adopt a train launch through its wrapper'
    Check (Test-JobAlive $aj $tbl) 'adopted job alive'
    Check (-not (Test-JobAlive $aj @($tbl[2]))) 'job dead when python and wrapper are gone'
    (Get-RunCounters $st '_smoke_fog')['current_launch'] = [ordered]@{ kind = 'evaluate'; n = 2; good_before = -1; started = $t0.AddMinutes(-5).ToString('o'); cmd_pid = $null; py_pid = 601; levels = @('1.10') }
    $tbl2 = @((New-Proc 601 1 'python.exe' 'python main.py --run _smoke_fog --option evaluate --levels 1.10' $t0))
    $aj = Find-LiveJob @() $st $tbl2
    Check ($null -ne $aj -and -not $aj.Foreign -and $aj.Kind -eq 'evaluate' -and $aj.N -eq 2 -and $aj.Started -eq $t0.AddMinutes(-5) -and ($aj.Levels -join ',') -eq '1.10') 'adopt an evaluation from state.json (run not in the queue file)'
    $tbl3 = @((New-Proc 701 1 'python.exe' 'python main.py --run _smoke_fog --option train' $t0))
    $aj = Find-LiveJob @('_smoke_fog') (Read-QueueState $Q.QueueDir) $tbl3
    Check ($null -ne $aj -and $aj.Foreign -and $aj.PyPid -eq 701) 'a python the queue did not start is foreign'
    Check ($null -eq (Find-LiveJob @('_smoke_fog') (Read-QueueState $Q.QueueDir) @($tbl[2]))) 'another run is not adopted'

    # A launch in flight in state.json becomes a job record to complete.
    $cur = [ordered]@{ kind = 'train'; n = 5; good_before = 288; started = $t0.ToString('o'); cmd_pid = 900; py_pid = 901; levels = @(); last_ckpt_step = -1; last_ckpt_time = $null }
    $fj = ConvertFrom-CurrentLaunch '_smoke_fog' $cur
    Check ($fj.Label -eq 'launch_r5' -and $fj.GoodBefore -eq 288 -and $fj.EndReason -eq 'EXITED' -and $fj.Started -eq $t0) 'launch in flight from state.json'

    # Scheduled tasks of this queue (smoke vs real).
    $smokeTask = [pscustomobject]@{ TaskName = 'TORRS smoke queue'; Actions = @([pscustomobject]@{ Arguments = '-NoProfile -File "E:\T (2)\tools\run_queue.ps1" -QueueFile "config/queue_smoke.txt" -ProjectRoot "E:\T (2)"'; WorkingDirectory = 'E:\T (2)' }) }
    $realTask = [pscustomobject]@{ TaskName = 'TORRS training queue'; Actions = @([pscustomobject]@{ Arguments = '-NoProfile -File "E:\T (2)\tools\run_queue.ps1" -QueueFile "config/queue.txt" -ProjectRoot "E:\T (2)"'; WorkingDirectory = 'E:\T (2)' }) }
    $defaultTask = [pscustomobject]@{ TaskName = 'x'; Actions = @([pscustomobject]@{ Arguments = '-File "E:\T (2)\tools\run_queue.ps1"'; WorkingDirectory = 'E:\T (2)' }) }
    $otherTask = [pscustomobject]@{ TaskName = 'y'; Actions = @([pscustomobject]@{ Arguments = '-File other.ps1 -QueueFile "config/queue.txt"'; WorkingDirectory = 'E:\T (2)' }) }
    Check (Test-TaskRunsQueue $smokeTask 'E:\T (2)\config\queue_smoke.txt' 'E:\T (2)') 'smoke task runs the smoke queue'
    Check (-not (Test-TaskRunsQueue $smokeTask 'E:\T (2)\config\queue.txt' 'E:\T (2)')) 'smoke task does not run the real queue'
    Check (Test-TaskRunsQueue $realTask 'E:\T (2)\config\queue.txt' 'E:\T (2)') 'real task runs the real queue'
    Check (-not (Test-TaskRunsQueue $realTask 'E:\T (2)\config\queue_smoke.txt' 'E:\T (2)')) 'real task does not run the smoke queue'
    Check (Test-TaskRunsQueue $defaultTask 'E:\T (2)\config\queue.txt' 'E:\T (2)') 'no -QueueFile means config/queue.txt'
    Check (-not (Test-TaskRunsQueue $otherTask 'E:\T (2)\config\queue.txt' 'E:\T (2)')) 'a task that does not run run_queue.ps1'
    $all = @($smokeTask, $realTask, $defaultTask, $otherTask)
    $sel = @(Select-QueueTasks $all 'TORRS training queue' $false 'E:\T (2)\config\queue_smoke.txt' 'E:\T (2)' | ForEach-Object { $_.TaskName })
    Check (($sel -join ',') -eq 'TORRS smoke queue') "the smoke runner disables only the smoke task: $($sel -join ',')"
    $sel = @(Select-QueueTasks $all 'TORRS training queue' $false 'E:\T (2)\config\queue.txt' 'E:\T (2)' | ForEach-Object { $_.TaskName })
    Check (($sel -join ',') -eq 'TORRS training queue,x') "the real runner disables the tasks of the real queue: $($sel -join ',')"
    $sel = @(Select-QueueTasks $all 'y' $true 'E:\T (2)\config\queue_smoke.txt' 'E:\T (2)' | ForEach-Object { $_.TaskName })
    Check (($sel -join ',') -eq 'TORRS smoke queue,y') "an explicit -TaskName is disabled too: $($sel -join ',')"

    # Markers.
    $r = Set-RunFailed '_smoke_fog' 'STUCK' 'three launches without progress'
    $fp = Join-Path $proj 'Malta\runs\_smoke_fog\FAILED'
    Check ($r.Action -eq 'TERMINAL' -and $r.Status -eq 'STUCK' -and (Get-MarkerFirstLine $fp) -eq 'STUCK') 'FAILED (STUCK) marker and result'
    $bytes = [IO.File]::ReadAllBytes($fp)
    Check (-not ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF)) 'FAILED marker has no BOM'
    $r = Set-RunFailed '_smoke_fog' 'NAN' 'second reason'
    Check ((Get-MarkerFirstLine $fp) -eq 'STUCK' -and $r.Status -eq 'STUCK') 'an existing FAILED marker is not overwritten'
    $d = Get-RunDecision '_smoke_fog' (Read-QueueState $Q.QueueDir)
    Check ($d.Stage -eq 'FAILED' -and $d.Text -match 'FAILED \(STUCK\)' -and -not $d.Actionable) "decision with a FAILED marker: $($d.Text)"

    # -----------------------------------------------------------------------------------------
    # Final-review fix 4: cross-run warnings (queue.log only, never a failure)
    # -----------------------------------------------------------------------------------------
    $xp = New-FakeProject 'p3 (xrun) x'
    foreach ($arm in 'cloud', 'fog', 'hybrid') { Put (Join-Path $xp ('config\runs\xr_' + $arm + '.ini')) ($SmokeIni -replace 'comm_arch = fog', ('comm_arch = ' + $arm)) }
    Put (Join-Path $xp 'config\queue_x.txt') "xr_cloud`nxr_fog`nxr_hybrid`n"
    Initialize-RunnerContext -ProjectRoot $xp -QueueFile 'config/queue_x.txt' -PythonExe $py -DryRun $false
    $xlog = Join-Path $xp 'Malta\runs\_queue\queue.log'
    # Code: the previous DONE run's first launch against this run's first launch.
    Check ($null -eq (Compare-CrossRunCode (Get-CodeHashLines 'xr_fog') $null)) 'no previous DONE run: nothing to compare'
    Check ($null -eq (Compare-CrossRunCode (Get-CodeHashLines 'xr_fog') 'xr_cloud')) 'a previous run without a launch: nothing to compare'
    Put (Join-Path $xp 'Malta\runs\xr_cloud\logs\launch_r1.code.txt') ("# code hashes for launch_r1 of xr_cloud`r`n" + (@(Get-CodeHashLines 'xr_cloud') -join "`r`n") + "`r`n")
    Put (Join-Path $xp 'Malta\runs\xr_cloud\logs\launch_r2.code.txt') "# a later launch, ignored`r`nxx  utils.py`r`n"
    $fogLines = Get-CodeHashLines 'xr_fog'
    Check ((@($fogLines) -join "`n") -match '  config\.ini' -and (@(Get-CodeHashLines 'xr_cloud') -join "`n") -ne (@($fogLines) -join "`n")) 'precondition: the two runs differ in config.ini'
    $x = Compare-CrossRunCode $fogLines 'xr_cloud'
    Check ($null -ne $x -and $x.Same -and $x.Detail -match 'launch_r1\.code\.txt') "same code apart from config.ini: $($x.Detail)"
    Write-CrossRunCodeCheck 'xr_fog' $fogLines 'xr_cloud'
    Check ((Read-SharedText $xlog) -match 'xr_fog: first launch: same code as the first launch of xr_cloud' -and (Read-SharedText $xlog) -notmatch 'WARNING') 'a match is logged without a warning'
    Put (Join-Path $xp 'utils.py') "# changed between the runs`n"
    Put (Join-Path $xp 'agents\new_policy.py') "# new`n"
    $fogLines = Get-CodeHashLines 'xr_fog'
    $x = Compare-CrossRunCode $fogLines 'xr_cloud'
    Check ($null -ne $x -and -not $x.Same -and $x.Detail -match 'utils\.py' -and $x.Detail -match 'agents\\new_policy\.py \(new\)' -and $x.Detail -notmatch 'config\.ini' -and $x.Detail -notmatch 'main\.py') "changed code is listed, config.ini is not: $($x.Detail)"
    Write-CrossRunCodeCheck 'xr_fog' $fogLines 'xr_cloud'
    Check ((Read-SharedText $xlog) -match 'xr_fog: WARNING: first launch: code differs from the first launch of xr_cloud \(xr_cloud\\logs\\launch_r1\.code\.txt\): .*utils\.py.*\(not a failure\)') "the code difference is a WARNING in queue.log: $(Read-SharedText $xlog)"
    $nWarn = ([regex]::Matches((Read-SharedText $xlog), 'xr_fog: WARNING: first launch: code differs')).Count
    Put (Join-Path $xp 'Malta\runs\xr_cloud\logs\launch_r1.code.txt') "not a hash line`r`n"
    Write-CrossRunCodeCheck 'xr_fog' $fogLines 'xr_cloud'
    Check (([regex]::Matches((Read-SharedText $xlog), 'xr_fog: WARNING: first launch: code differs')).Count -eq $nWarn + 1) 'a garbled code file of the other run still only warns'
    # Neighbour maps: hybrid against fog (spec 5: hybrid uses the fog maps).
    Check (@(Compare-HybridNeighbourHash 'xr_hybrid').Count -eq 0) 'no hybrid neighbours.sha256 yet: nothing to compare'
    Put (Join-Path $xp 'Malta\runs\xr_hybrid\neighbours.sha256') "abc123`n"
    Check (@(Compare-HybridNeighbourHash 'xr_hybrid').Count -eq 0) 'no fog neighbours.sha256: nothing to compare'
    Put (Join-Path $xp 'Malta\runs\xr_fog\neighbours.sha256') "abc123`n"
    Put (Join-Path $xp 'Malta\runs\xr_cloud\neighbours.sha256') "fff999`n"
    $nl = @(Compare-HybridNeighbourHash 'xr_hybrid')
    Check ($nl.Count -eq 1 -and $nl[0] -match 'xr_hybrid: neighbours\.sha256 matches fog run xr_fog' -and $nl[0] -notmatch 'WARNING') "equal hashes match (the cloud run is not compared): $($nl -join ' | ')"
    Check (@(Compare-HybridNeighbourHash 'xr_fog').Count -eq 0 -and @(Compare-HybridNeighbourHash 'xr_cloud').Count -eq 0) 'only a hybrid run is checked'
    Put (Join-Path $xp 'Malta\runs\xr_fog\neighbours.sha256') "abd124`n"
    $nl = @(Compare-HybridNeighbourHash 'xr_hybrid')
    Check ($nl.Count -eq 1 -and $nl[0] -match 'xr_hybrid: WARNING: neighbours\.sha256 abc123 differs from fog run xr_fog \(abd124\)') "different hashes warn: $($nl -join ' | ')"
    Put (Join-Path $xp 'Malta\runs\xr_fog\DONE') 'DONE'
    Check (@(Compare-HybridNeighbourHash 'xr_hybrid').Count -eq 1) 'a DONE fog run is compared'
    Write-HybridNeighbourCheck 'xr_hybrid'
    Write-HybridNeighbourCheck 'xr_hybrid'
    Check (([regex]::Matches((Read-SharedText $xlog), 'xr_hybrid: WARNING: neighbours\.sha256')).Count -eq 1) 'the neighbour check is logged once per runner process'
    Put (Join-Path $xp 'Malta\runs\xr_fog\FAILED') 'STUCK'
    Check (@(Compare-HybridNeighbourHash 'xr_hybrid').Count -eq 0) 'a FAILED fog run is not compared'
    Initialize-RunnerContext -ProjectRoot $proj -QueueFile 'config/queue_smoke.txt' -PythonExe $py -DryRun $true -ChecksDir (Join-Path $proj 'Malta\runs\_queue\checks')

    # -----------------------------------------------------------------------------------------
    # Single instance: a held mutex is reported; a dead owner's (abandoned) mutex is taken.
    # A test-only mutex name, so a real runner is never affected.
    # -----------------------------------------------------------------------------------------
    $RunnerMutexName = 'Global\TORRS_queue_runner_test_' + [guid]::NewGuid().ToString('N').Substring(0, 8)
    $keep = New-Object System.Threading.Mutex($false, $RunnerMutexName)   # keeps the object alive
    try {
        Check ((Get-MutexProbe) -match '^free') 'an unowned mutex is free'
        $holder = Start-Process -FilePath 'powershell.exe' -WindowStyle Hidden -PassThru -ArgumentList @('-NoProfile', '-Command',
            ('$m = New-Object System.Threading.Mutex($false, ''{0}''); [void]$m.WaitOne(); Start-Sleep -Seconds 8' -f $RunnerMutexName))
        $deadline = (Get-Date).AddSeconds(20)
        $probe = ''
        while ((Get-Date) -lt $deadline) { $probe = Get-MutexProbe; if ($probe -match '^held') { break }; Start-Sleep -Milliseconds 300 }
        Check ($probe -match '^held') "a mutex held by another process is reported held: $probe"
        Check (-not (Enter-RunnerMutex $keep 0)) 'a second copy does not get a held mutex'
        # Final-review fix 2: -ResetRun is refused while a runner holds the mutex.
        $fpText = [IO.File]::ReadAllText($fp)
        $out = @(Invoke-ResetRun '_smoke_fog')
        Check ($out.Count -eq 1 -and $out[0] -eq 1 -and [IO.File]::Exists($fp) -and [IO.File]::ReadAllText($fp) -eq $fpText) "ResetRun refused while a runner holds the mutex: $($out -join ' | ')"
        $holder.WaitForExit(30000) | Out-Null
        Check $holder.HasExited 'the holder process exited'
        Check (Enter-RunnerMutex $keep 0) 'an abandoned mutex counts as ownership'
        $keep.ReleaseMutex()
        Check ((Get-MutexProbe) -match '^free') 'free again after release'

        # -------------------------------------------------------------------------------------
        # Final-review fix 2: -ResetRun <name> (still on the test mutex name)
        # -------------------------------------------------------------------------------------
        Initialize-RunnerContext -ProjectRoot $proj -QueueFile 'config/queue_smoke.txt' -PythonExe $py -DryRun $false
        $stR = Read-QueueState $Q.QueueDir
        $rcR = Get-RunCounters $stR '_smoke_fog'
        $rcR['launches'] = 12; $rcR['no_progress'] = 3; $rcR['interrupted'] = 2; $rcR['good_at_launch'] = @(-1, 144)
        $rcR['eval_attempts'] = 2; $rcR['eval_launches'] = 6; $rcR['eval_levels_next'] = @('1.10')
        Save-QueueState $Q.QueueDir $stR
        $resetLogs = Join-Path $proj 'Malta\runs\_smoke_fog\logs'
        $logsBefore = @(Get-ChildItem -LiteralPath $resetLogs | ForEach-Object { $_.Name } | Sort-Object)
        $origTable = ${function:Get-ProcessTable}
        try {
            ${function:Get-ProcessTable} = { @((New-Proc 900 1 'python.exe' '"C:\Program Files\Python311\python.exe" -X faulthandler main.py --run cloud_base --option train' (Get-Date))) }
            $out = @(Invoke-ResetRun '_smoke_fog')
            Check ($out.Count -eq 1 -and $out[0] -eq 1 -and [IO.File]::Exists($fp)) "ResetRun refused while a queue python is alive: $($out -join ' | ')"
            Check ((Read-QueueState $Q.QueueDir)['runs']['_smoke_fog']['launches'] -eq 12) 'a refused reset changes no counter'
            ${function:Get-ProcessTable} = { @((New-Proc 901 1 'python.exe' 'python tools\check_env.py --arm fog --port 20' (Get-Date))) }
            Check (@(Get-QueuePythonList @(Get-ProcessTable)).Count -eq 0) 'a python that is not a queue launch does not block a reset'
        } finally { ${function:Get-ProcessTable} = $origTable }
        foreach ($badName in 'Fog', '..\signal_route') {
            $out = @(Invoke-ResetRun $badName)
            Check ($out.Count -eq 1 -and $out[0] -eq 1) "ResetRun refuses the run name '$badName'"
        }
        $out = @(Invoke-ResetRun 'nothing_here')
        Check ($out.Count -eq 1 -and $out[0] -eq 1) 'ResetRun refuses a run without a store'
        $out = @(Invoke-ResetRun '_smoke_fog')
        Check ($out.Count -eq 1 -and $out[0] -eq 0) "ResetRun returns 0: $($out -join ' | ')"
        Check (-not [IO.File]::Exists($fp)) 'the FAILED marker was moved away'
        $kept = @(Get-ChildItem -LiteralPath $resetLogs -Filter 'FAILED_*.txt')
        Check ($kept.Count -eq 1 -and $kept[0].Name -match '^FAILED_\d{8}_\d{6}\.txt$' -and [IO.File]::ReadAllText($kept[0].FullName) -eq $fpText) 'the FAILED marker is kept, unchanged, as logs\FAILED_<stamp>.txt'
        $logsAfter = @(Get-ChildItem -LiteralPath $resetLogs | Where-Object { $_.Name -notlike 'FAILED_*' } | ForEach-Object { $_.Name } | Sort-Object)
        Check (($logsAfter -join '|') -eq ($logsBefore -join '|')) 'nothing launched (no new log files)'
        $rcR = (Read-QueueState $Q.QueueDir)['runs']['_smoke_fog']
        Check ($rcR['launches'] -eq 0 -and $rcR['no_progress'] -eq 0 -and $rcR['eval_attempts'] -eq 0 -and $rcR['eval_launches'] -eq 0 -and @($rcR['eval_levels_next']).Count -eq 0) "counters reset: $(ConvertTo-Json $rcR -Compress -Depth 6)"
        Check ($rcR['interrupted'] -eq 2 -and @($rcR['good_at_launch']).Count -eq 2) 'interrupted and good_at_launch are kept'
        $rsR = @($rcR['resets'])
        Check ($rsR.Count -eq 1 -and $rsR[0]['failed'] -eq 'STUCK' -and $rsR[0]['launches'] -eq 12 -and $rsR[0]['no_progress'] -eq 3 -and $rsR[0]['eval_attempts'] -eq 2 -and $rsR[0]['eval_launches'] -eq 6 -and (@($rsR[0]['eval_levels_next']) -join ',') -eq '1.10') "the old values are kept in resets: $(ConvertTo-Json $rsR -Compress -Depth 6)"
        Check ((Read-SharedText $Q.LogFile) -match '_smoke_fog: ResetRun \(pid \d+\): FAILED \(STUCK\) moved to .*\\logs\\FAILED_\d{8}_\d{6}\.txt; state\.json counters reset: launches 12 -> 0, no_progress 3 -> 0, eval_attempts 2 -> 0, eval_launches 6 -> 0, eval_levels_next \[1\.10\] -> \[\]') "the reset is logged in queue.log: $(Read-SharedText $Q.LogFile)"
        $d = Get-RunDecision '_smoke_fog' (Read-QueueState $Q.QueueDir)
        Check ($d.Stage -eq 'TRAIN' -and $d.Actionable) "after the reset the run trains again: $($d.Text)"
        $out = @(Invoke-ResetRun '_smoke_fog')
        Check ($out.Count -eq 1 -and $out[0] -eq 0 -and @((Read-QueueState $Q.QueueDir)['runs']['_smoke_fog']['resets']).Count -eq 2) 'a second reset (no FAILED marker) only resets the counters again'
        Check ((Read-SharedText $Q.LogFile) -match '_smoke_fog: ResetRun \(pid \d+\): no FAILED marker was present') 'the second reset is logged'
        # A STUCK verdict after the reset starts from fresh counters again.
        Check ((Update-LaunchCounters (Read-QueueState $Q.QueueDir) '_smoke_fog' 144 144 $false) -eq 'OK') 'the counters allow new launches after a reset'
        Initialize-RunnerContext -ProjectRoot $proj -QueueFile 'config/queue_smoke.txt' -PythonExe $py -DryRun $true -ChecksDir (Join-Path $proj 'Malta\runs\_queue\checks')
    } finally { $keep.Dispose() }

    # The parts below start python processes; they are skipped while a real runner is active,
    # so a real queue never sees (or waits for) a test's 'main.py --run' python.
    $RunnerMutexName = 'Global\TORRS_queue_runner'
    $realRunnerActive = ((Get-MutexProbe) -notmatch '^free')
    if ($realRunnerActive) { Write-Output 'SKIPPED the launch test and the main-loop simulation: a real queue runner holds the mutex' }

    # -----------------------------------------------------------------------------------------
    # End to end: the launch wrapper started through Win32_Process.Create, in a project folder
    # with spaces and parentheses, running a FAKE main.py that echoes its arguments.
    # -----------------------------------------------------------------------------------------
    if (-not [IO.File]::Exists($py)) { throw "python not found at $py" }
    if (-not $realRunnerActive) {
    $proj2 = New-FakeProject 'p2 (launch) x'
    Put (Join-Path $proj2 'main.py') "import os, sys`nprint('ARGS ' + ' '.join(sys.argv[1:]))`nprint('CWD ' + os.getcwd())`nprint('ENV %s %s' % (os.environ.get('PYTHONIOENCODING'), os.environ.get('PYTHONUNBUFFERED')))`nprint('UTF8 ' + chr(0x17C))`nsys.exit(7)`n"
    Initialize-RunnerContext -ProjectRoot $proj2 -QueueFile 'config/queue_smoke.txt' -PythonExe $py -DryRun $true -ChecksDir (Join-Path $proj2 'Malta\runs\_queue\checks (2)')
    $res = Invoke-DetachedPython (New-LaunchArguments '_smoke_echo' 'evaluate' @('1.10')) 'fake launch' 120
    Check (-not $res.TimedOut -and $res.Code -eq 7) "fake main.py exit code: $($res.Code), output: $($res.Output)"
    Check ($res.Output -match '(?m)^ARGS --run _smoke_echo --option evaluate --levels 1\.10\s*$') "arguments reached python: $($res.Output)"
    Check ($res.Output -match ('(?m)^CWD ' + [regex]::Escape($proj2) + '\s*$')) "working directory is the project root: $($res.Output)"
    Check ($res.Output -match '(?m)^ENV utf-8 1\s*$') "PYTHONIOENCODING and PYTHONUNBUFFERED set by the wrapper: $($res.Output)"
    Check ($res.Output.Contains('UTF8 ' + [string][char]0x017C)) 'UTF-8 output'
    $exitText = [IO.File]::ReadAllText((Join-Path $proj2 'Malta\runs\_queue\checks (2)\fake launch.exit'))
    Check ($exitText -match '^EXIT=7 at ') "exit file: $exitText"

    # Final-review fix 2, end to end: 'run_queue.ps1 -ResetRun <name> -ProjectRoot <fake>' as the
    # author runs it (the real mutex name, held only for the reset; skipped while a real runner
    # is active).
    $proj4 = New-FakeProject 'p4 (reset) x'
    Put (Join-Path $proj4 'Malta\runs\_smoke_fog\FAILED') "EVAL_FAILED`r`neval_r2: level f2.00 exceeded its time limit`r`nwritten by a test`r`n"
    Put (Join-Path $proj4 'Malta\runs\_queue\state.json') '{ "version": 1, "runs": { "_smoke_fog": { "launches": 1, "no_progress": 0, "interrupted": 0, "good_at_launch": [ -1 ], "eval_attempts": 2, "eval_launches": 2, "eval_levels_next": [ "2.00" ], "current_launch": null } }, "arms": { "fog": { "checkpoint_gap_min": 120 } } }'
    $script4 = Join-Path $RealRoot 'tools\run_queue.ps1'
    $o = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script4 -ResetRun '_smoke_fog' -ProjectRoot $proj4 -QueueFile 'config/queue_smoke.txt'
    $code4 = $LASTEXITCODE
    $oText = ($o | ForEach-Object { [string]$_ }) -join "`n"
    Check ($code4 -eq 0 -and $oText -match 'ResetRun _smoke_fog \(nothing launched\)' -and $oText -match 'FAILED \(EVAL_FAILED\) moved to' -and $oText -match 'eval_attempts 2 -> 0, eval_launches 2 -> 0, eval_levels_next \[2\.00\] -> \[\]') "the -ResetRun command: exit $code4`n$oText"
    $st4 = Read-QueueState (Join-Path $proj4 'Malta\runs\_queue')
    Check (-not [IO.File]::Exists((Join-Path $proj4 'Malta\runs\_smoke_fog\FAILED')) -and @(Get-ChildItem -LiteralPath (Join-Path $proj4 'Malta\runs\_smoke_fog\logs') -Filter 'FAILED_*.txt').Count -eq 1) 'the command moved FAILED into logs'
    Check ($st4['runs']['_smoke_fog']['eval_attempts'] -eq 0 -and $st4['arms']['fog']['checkpoint_gap_min'] -eq 120) 'the command reset the counters and kept the rest of state.json'
    $o = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script4 -ResetRun 'Fog_x' -ProjectRoot $proj4
    Check ($LASTEXITCODE -eq 1) "-ResetRun with a bad run name exits 1: $(($o | ForEach-Object { [string]$_ }) -join ' ')"
    $o = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script4 -ResetRun '_smoke_fog' -DryRun -ProjectRoot $proj4
    Check ($LASTEXITCODE -eq 1 -and (($o | ForEach-Object { [string]$_ }) -join ' ') -match 'cannot be combined with -DryRun') '-ResetRun with -DryRun is refused'
    }

    # -----------------------------------------------------------------------------------------
    # Simulation of the main loop: Invoke-RunnerIteration driven against a fake project whose
    # FAKE main.py emulates training (a crash after checkpoint 144, then a resume to 288) and
    # evaluation (f1.10 fails the first attempt, the --levels retry passes), a run that hangs
    # (watchdog -> STUCK after 3 launches) and a run with a NaN reward (FAILED NAN). Real WMI
    # launches, real environment check, real preflight (fake gen_traffic.py, small arm peaks).
    # The TraCI port is a test value (27599) and orphan cleanup is switched off, so nothing
    # outside the fake project can be touched. Skipped while a real queue runner is active.
    # -----------------------------------------------------------------------------------------
    if (-not $realRunnerActive) {
        $RunnerPollSeconds = 1
        $RunnerStopGraceSec = 1
        $QueueTrainPort = 27599
        function Test-CleanupTarget { $false }
        $sim = New-FakeProject 'sim (loop) x'
        $mainPy = @'
import logging, os, shutil, sys, time
a = sys.argv[1:]
run = a[a.index('--run') + 1]
opt = a[a.index('--option') + 1]
levels = a[a.index('--levels') + 1].split(',') if '--levels' in a else ['1.00', '1.10']
store = 'Malta/runs/%s/' % run
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)
print('ARGS ' + ' '.join(a), flush=True)
os.makedirs(store + 'model', exist_ok=True)
if not os.path.exists(store + 'config.ini'):
    shutil.copyfile('config/runs/%s.ini' % run, store + 'config.ini')
HDR = ',agent,step,test_id,reward,total arrived vehicle number,total departured vehicle number,total waiting time,total travel time,average waiting time,average travel time,seed'
COST = 'episode,step,seed,wall_s,rollout_s,update_s,save_s,peak_private_gb,peak_working_set_gb,ckpt_data_mb,ckpt_meta_mb,lost'
def append(path, line, header=None):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        if new and header:
            f.write(header + '\r\n')
        f.write(line + '\r\n')
def episode(step, seed, reward):
    p = store + 'train_reward.csv'
    n = len([l for l in open(p) if l.strip()]) - 1 if os.path.exists(p) else 0
    append(p, '%d,ma2c,%d,-1,%s,320,330,4000.0,90000.0,12.5,272.7,%d' % (n, step, reward, seed), HDR)
    append(store + 'agent_reward.csv', '1.0,2.0,3.0')
    for ext in ('.data-00000-of-00001', '.index', '.meta'):
        open(store + 'model/checkpoint-%d%s' % (step, ext), 'w').write('x')
    append(store + 'episode_cost.csv', '%d,%d,%d,1.0,600.0,0.5,0.1,1.0,1.0,1.0,1.0,0' % (step // 144, step, seed), COST)
    logging.info('Checkpoint saved at step %d -> %smodel/' % (step, store))
if opt == 'train':
    if os.path.exists(store + 'model/checkpoint-288.index'):
        print('>>> TRAINING COMPLETE step 288', flush=True)
        sys.exit(0)
    if 'refuse' in run:
        print('resume refused for %s: config differs: ENV_CONFIG.seed' % store, file=sys.stderr, flush=True)
        sys.exit(4)
    if 'abort' in run and not os.path.exists(store + 'died_once'):
        open(store + 'died_once', 'w').write('x')
        sys.stderr.write('Fatal Python error: Aborted\n\nCurrent thread 0x00001234 (most recent call first):\n  File "main.py", line 1 in <module>\n')
        sys.stderr.flush()
        os._exit(3)
    if 'exittwo' in run and not os.path.exists(store + 'died_once'):
        open(store + 'died_once', 'w').write('x')
        print('ValueError: something odd in the env', file=sys.stderr, flush=True)
        sys.exit(2)
    if 'late' in run:
        episode(144, 12, '-12.5')
        time.sleep(6)
        episode(288, 13, '-11.0')
        time.sleep(60)
        sys.exit(0)
    if 'hang' in run:
        time.sleep(120)
        sys.exit(0)
    if 'nan' in run:
        episode(144, 12, '')
        time.sleep(120)
        sys.exit(0)
    if 'gap' in run and not os.path.exists(store + 'model/checkpoint-144.index'):
        episode(144, 12, '-12.5')
        time.sleep(120)
        sys.exit(0)
    fresh = not os.path.exists(store + 'model/checkpoint-144.index')
    if 'stop' in run and fresh:
        open('Malta/runs/_queue/STOP', 'w').write('stop')
        time.sleep(3)
        episode(144, 12, '-12.5')
        time.sleep(60)
        sys.exit(0)
    if fresh:
        episode(144, 12, '-12.5')
        if 'fake' in run:
            raise RuntimeError('simulated crash after the first checkpoint')
        if 'adopt' in run:
            time.sleep(4)
    episode(288, 13, '-11.0')
    print('>>> TRAINING COMPLETE step 288', flush=True)
    sys.exit(0)
if 'slow' in run:
    time.sleep(60)
    sys.exit(0)
if 'stop' in run and not os.path.exists(store + 'eva_data/stop_done'):
    os.makedirs(store + 'eva_data', exist_ok=True)
    open(store + 'eva_data/stop_done', 'w').write('x')
    open('Malta/runs/_queue/STOP', 'w').write('stop')
    time.sleep(60)
    sys.exit(0)
first = not os.path.exists(store + 'eva_data/attempt1_done')
logging.info('Restoring parameters from %smodel/checkpoint-288' % store)
for lv in levels:
    d = store + 'eva_data/f%s/' % lv
    os.makedirs(d, exist_ok=True)
    total = 100 if lv == '1.00' else 110
    expected = 999 if (lv == '1.10' and first) else total
    logging.info('EVAL_LEVEL_START f%s bg_tag=p%s corridor_total=%g corridor_expected=%g' % (lv, round(1.2 / float(lv), 3), total, expected))
    for i, s in enumerate([10000, 20000]):
        logging.info('test %d, avg reward -1.0' % i)
        open(d + 'edgedata_%d.xml' % s, 'w').write('<meandata>\n</meandata>\n')
        open(d + 'summary_%d.xml' % s, 'w').write('<summary>\n</summary>\n')
    for kind in ('control', 'traffic', 'trip'):
        k = (3 if lv == '1.00' else 4) if kind == 'trip' else 1
        rows = ['episode,x'] + ['%d,%d' % (ep, j) for ep in (1, 2) for j in range(k)]
        open(d + 'Malta_signal_route_%s.csv' % kind, 'w').write('\n'.join(rows) + '\n')
    logging.info('EVAL_LEVEL_DONE f%s seeds=2' % lv)
open(store + 'eva_data/attempt1_done', 'w').write('x')
print('>>> EVALUATE COMPLETE levels=%d' % len(levels), flush=True)
'@
        Put (Join-Path $sim 'main.py') $mainPy
        Put (Join-Path $sim 'tools\gen_traffic.py') "import sys`nprint(' '.join(sys.argv[1:]))`nprint('2 files, 2 valid, 0 not valid')`n"
        Put (Join-Path $sim 'config\runs\_smoke_fake.ini') $SmokeIni
        Put (Join-Path $sim 'config\runs\_smoke_hang.ini') ($SmokeIni -replace 'comm_arch = fog', 'comm_arch = cloud')
        Put (Join-Path $sim 'config\runs\_smoke_nan.ini') ($SmokeIni -replace 'comm_arch = fog', 'comm_arch = hybrid')
        Put (Join-Path $sim 'config\queue_a.txt') "_smoke_fake`n"
        Put (Join-Path $sim 'config\queue_b.txt') "_smoke_fake`n_smoke_hang`n"
        Put (Join-Path $sim 'config\queue_c.txt') "_smoke_nan`n"
        foreach ($r in '_smoke_stop', '_smoke_adopt') { Put (Join-Path $sim ('config\runs\' + $r + '.ini')) $SmokeIni }
        Put (Join-Path $sim 'config\runs\_smoke_slow.ini') ($SmokeIni -replace 'comm_arch = fog', 'comm_arch = local')
        Put (Join-Path $sim 'config\queue_d.txt') "_smoke_stop`n"
        Put (Join-Path $sim 'config\queue_e.txt') "_smoke_slow`n"
        Put (Join-Path $sim 'config\queue_f.txt') "_smoke_adopt`n"
        foreach ($r in '_smoke_abort', '_smoke_exittwo', '_smoke_refuse', '_smoke_late', '_smoke_envmem') { Put (Join-Path $sim ('config\runs\' + $r + '.ini')) $SmokeIni }
        Put (Join-Path $sim 'config\queue_i.txt') "_smoke_abort`n_smoke_exittwo`n_smoke_refuse`n"
        Put (Join-Path $sim 'config\queue_j.txt') "_smoke_late`n"
        Put (Join-Path $sim 'config\queue_k.txt') "_smoke_envmem`n"
        Put (Join-Path $sim 'Malta\runs\_queue\state.json') '{ "version": 1, "runs": {}, "arms": { "fog": { "peak_commit_gb": 0.5 }, "cloud": { "peak_commit_gb": 0.5, "first_checkpoint_min": 0.05 }, "hybrid": { "peak_commit_gb": 0.5 }, "local": { "peak_commit_gb": 0.5, "first_checkpoint_min": 0.05 } } }'

        $drive = {
            param([string]$QueueName)
            Initialize-RunnerContext -ProjectRoot $sim -QueueFile ('config/' + $QueueName) -PythonExe $py -DryRun $false
            $Q.EnvOk = $script:SimEnvOk
            $res = $null
            for ($k = 0; $k -lt 20; $k++) {
                $out = @(Invoke-RunnerIteration)
                if ($out.Count -ne 1) { throw ('a loop pass returned {0} objects instead of 1: {1}' -f $out.Count, (($out | ForEach-Object { [string]$_ }) -join ' | ')) }
                $res = $out[0]
                if ($res.Action -eq 'WAIT') { throw ('simulation waited: ' + $res.Status + ' ' + $res.Detail) }
                if ($res.Action -ne 'CONTINUE') { break }
            }
            $script:SimEnvOk = $Q.EnvOk
            $res
        }
        $script:SimEnvOk = $false

        $res = & $drive 'queue_a.txt'
        Check ($res.Action -eq 'TERMINAL' -and $res.Status -eq 'QUEUE_DONE') "queue A ends QUEUE_DONE: $($res.Status) $($res.Detail)"
        $fs = Join-Path $sim 'Malta\runs\_smoke_fake'
        Check ([IO.File]::Exists((Join-Path $fs 'DONE')) -and [IO.File]::Exists((Join-Path $fs 'eva_data\EVAL_OK'))) 'DONE and EVAL_OK written'
        foreach ($f in 'launch_r1.log', 'launch_r1.exit', 'launch_r1.code.txt', 'launch_r2.log', 'launch_r2.exit', 'eval_r1.log', 'eval_r1.exit', 'eval_r1.code.txt', 'eval_r2.log', 'eval_r2.exit') {
            Check ([IO.File]::Exists((Join-Path $fs ('logs\' + $f)))) "$f exists"
        }
        Check (-not [IO.File]::Exists((Join-Path $fs 'logs\launch_r3.log'))) 'no third training launch'
        Check ((Read-ExitCode (Join-Path $fs 'logs\launch_r1.exit')) -eq 1 -and (Read-ExitCode (Join-Path $fs 'logs\launch_r2.exit')) -eq 0) 'launch exit codes 1 then 0'
        Check ((Read-SharedText (Join-Path $fs 'logs\eval_r1.log')) -match 'ARGS --run _smoke_fake --option evaluate\s') 'first evaluation runs every level'
        Check ((Read-SharedText (Join-Path $fs 'logs\eval_r2.log')) -match 'ARGS --run _smoke_fake --option evaluate --levels 1\.10\s') 'the retry runs only the failed level'
        $sst = Read-QueueState (Join-Path $sim 'Malta\runs\_queue')
        $rc = $sst['runs']['_smoke_fake']
        Check ($rc['launches'] -eq 1 -and $rc['no_progress'] -eq 0 -and $rc['eval_attempts'] -eq 2 -and $rc['eval_launches'] -eq 2 -and $null -eq $rc['current_launch']) "state after queue A: $(ConvertTo-Json $rc -Compress -Depth 5)"
        Check ([double]$rc['measured']['first_checkpoint_min'] -ge 0) 'first-checkpoint time recorded'
        Check ($sst['arms']['cloud']['first_checkpoint_min'] -eq 0.05) 'the arms settings are kept'
        $ql = Read-SharedText (Join-Path $sim 'Malta\runs\_queue\queue.log')
        Check ($ql -match 'training complete' -and $ql -match 'recovery RESUME' -and $ql -match 'evaluation passed' -and $ql -match 'failed levels 1\.10') 'queue.log events'
        foreach ($f in 'Malta\runs\_queue\queue.log', 'Malta\runs\_queue\status.txt', 'Malta\runs\_queue\state.json', 'Malta\runs\_smoke_fake\DONE', 'Malta\runs\_smoke_fake\eva_data\EVAL_OK') {
            $b = [IO.File]::ReadAllBytes((Join-Path $sim $f))
            Check (-not ($b.Length -ge 3 -and $b[0] -eq 0xEF -and $b[1] -eq 0xBB -and $b[2] -eq 0xBF)) "$f has no BOM"
        }
        Check ((Read-SharedText (Join-Path $sim 'Malta\runs\_queue\status.txt')) -match '\| heartbeat \d{4}-\d\d-\d\d \d\d:\d\d:\d\d \| runner pid \d+') 'status line with a heartbeat'

        $res = & $drive 'queue_b.txt'
        Check ($res.Action -eq 'TERMINAL' -and $res.Status -eq 'STUCK') "queue B ends STUCK: $($res.Status) $($res.Detail)"
        $hs = Join-Path $sim 'Malta\runs\_smoke_hang'
        Check ((Get-MarkerFirstLine (Join-Path $hs 'FAILED')) -eq 'STUCK') 'FAILED (STUCK) after the watchdog kills'
        $rc = (Read-QueueState (Join-Path $sim 'Malta\runs\_queue'))['runs']['_smoke_hang']
        Check ($rc['launches'] -eq 3 -and $rc['no_progress'] -eq 3) "three launches without progress: $(ConvertTo-Json $rc -Compress -Depth 5)"
        Check ((Read-SharedText (Join-Path $sim 'Malta\runs\_queue\queue.log')) -match 'watchdog: no Checkpoint saved line') 'the watchdog reason is logged'
        $res = & $drive 'queue_b.txt'
        Check ($res.Status -eq 'STUCK' -and $res.Detail -match '_smoke_hang \(STUCK\)') 'a FAILED run stops the queue at once on a restart'

        $res = & $drive 'queue_c.txt'
        Check ($res.Action -eq 'TERMINAL' -and $res.Status -eq 'FAILED' -and $res.Detail -match 'NAN') "queue C ends FAILED (NAN): $($res.Status) $($res.Detail)"
        Check ((Get-MarkerFirstLine (Join-Path $sim 'Malta\runs\_smoke_nan\FAILED')) -eq 'NAN') 'FAILED (NAN) marker'

        # STOP during training (the fake main.py creates STOP): wait for the next checkpoint, kill,
        # recover, stop. STOP is never deleted by the runner.
        $stopFile = Join-Path $sim 'Malta\runs\_queue\STOP'
        $res = & $drive 'queue_d.txt'
        Check ($res.Action -eq 'STOPPED') "STOP during training stops the runner: $($res.Action) $($res.Status) $($res.Detail)"
        Check ([IO.File]::Exists($stopFile)) 'the runner does not delete STOP'
        $ps = Join-Path $sim 'Malta\runs\_smoke_stop'
        Check ([IO.File]::Exists((Join-Path $ps 'model\checkpoint-144.index')) -and (Get-GoodCheckpoint $sim '_smoke_stop') -eq 144) 'killed after the checkpoint it waited for'
        $rc = (Read-QueueState (Join-Path $sim 'Malta\runs\_queue'))['runs']['_smoke_stop']
        Check ($rc['launches'] -eq 1 -and $rc['no_progress'] -eq 0 -and $rc['interrupted'] -eq 1 -and $null -eq $rc['current_launch']) "a STOP kill counts as an interruption: $(ConvertTo-Json $rc -Compress -Depth 5)"
        $res = & $drive 'queue_d.txt'
        Check ($res.Action -eq 'STOPPED' -and -not [IO.File]::Exists((Join-Path $ps 'logs\launch_r2.log'))) 'with STOP present nothing is launched'
        # STOP during evaluation (the fake creates STOP again): kill at once, attempt not counted.
        Remove-Item -LiteralPath $stopFile
        $res = & $drive 'queue_d.txt'
        Check ($res.Action -eq 'STOPPED' -and [IO.File]::Exists((Join-Path $ps 'logs\eval_r1.log'))) "STOP during evaluation: $($res.Action) $($res.Detail)"
        $rc = (Read-QueueState (Join-Path $sim 'Malta\runs\_queue'))['runs']['_smoke_stop']
        Check ($rc['eval_attempts'] -eq 0 -and $rc['eval_launches'] -eq 1) "the stopped evaluation is not counted: $(ConvertTo-Json $rc -Compress -Depth 5)"
        Remove-Item -LiteralPath $stopFile
        $res = & $drive 'queue_d.txt'
        Check ($res.Status -eq 'QUEUE_DONE' -and [IO.File]::Exists((Join-Path $ps 'DONE'))) "after STOP is removed the run finishes: $($res.Status) $($res.Detail)"
        $rc = (Read-QueueState (Join-Path $sim 'Malta\runs\_queue'))['runs']['_smoke_stop']
        Check ($rc['eval_attempts'] -eq 2 -and $rc['eval_launches'] -eq 3 -and $rc['launches'] -eq 1) "counters after the stops: $(ConvertTo-Json $rc -Compress -Depth 5)"

        # The evaluation time limit (arm 'local': 0.05 min before the first level): two timed-out
        # attempts -> FAILED (EVAL_FAILED); the retry re-runs every level, none having passed.
        $res = & $drive 'queue_e.txt'
        Check ($res.Status -eq 'FAILED' -and $res.Detail -match 'EVAL_FAILED') "evaluation time limit: $($res.Status) $($res.Detail)"
        $ls = Join-Path $sim 'Malta\runs\_smoke_slow\logs'
        Check ((Read-SharedText (Join-Path $ls 'eval_r2.log')) -match '--levels 1\.00,1\.10\s') 'the retry runs the failed levels'
        Check ((Read-SharedText (Join-Path $sim 'Malta\runs\_queue\queue.log')) -match 'no EVAL_LEVEL_START line') 'the time-limit reason is logged'

        # Adoption: a launch left running by a runner that died is watched, not relaunched.
        Initialize-RunnerContext -ProjectRoot $sim -QueueFile 'config/queue_f.txt' -PythonExe $py -DryRun $false
        $Q.EnvOk = $true
        $ast = Read-QueueState $Q.QueueDir
        $ajob = Start-RunnerLaunch '_smoke_adopt' (Read-RunConfig (Get-RunIniPath '_smoke_adopt')) 'TRAIN' $null $ast (Get-CodeHashLines '_smoke_adopt')
        Check ($null -ne $ajob.PyPid) 'the launch to adopt is running'
        $res = & $drive 'queue_f.txt'
        Check ($res.Status -eq 'QUEUE_DONE') "adopted run finishes: $($res.Status) $($res.Detail)"
        $as = Join-Path $sim 'Malta\runs\_smoke_adopt'
        Check (-not [IO.File]::Exists((Join-Path $as 'logs\launch_r2.log'))) 'the adopted launch was not relaunched'
        $rc = (Read-QueueState (Join-Path $sim 'Malta\runs\_queue'))['runs']['_smoke_adopt']
        Check ($rc['launches'] -eq 0 -and [double]$rc['measured']['first_checkpoint_min'] -ge 0) "adopted launch completed training: $(ConvertTo-Json $rc -Compress -Depth 5)"
        Check ((Read-SharedText (Join-Path $sim 'Malta\runs\_queue\queue.log')) -match '_smoke_adopt: watching launch_r1') 'adoption logged'

        # WAIT results: a missing run config, and too little commit headroom; nothing is launched.
        Put (Join-Path $sim 'config\queue_g.txt') "_smoke_nocfg`n"
        Initialize-RunnerContext -ProjectRoot $sim -QueueFile 'config/queue_g.txt' -PythonExe $py -DryRun $false
        $Q.EnvOk = $true
        $out = @(Invoke-RunnerIteration)
        Check ($out.Count -eq 1 -and $out[0].Action -eq 'WAIT' -and $out[0].Status -eq 'WAITING_CONFIG') "missing config waits: $($out[0].Status)"
        Put (Join-Path $sim 'config\runs\_smoke_big.ini') ($SmokeIni -replace 'comm_arch = fog', 'comm_arch = big')
        Put (Join-Path $sim 'config\queue_h.txt') "_smoke_big`n"
        $stp = Join-Path $sim 'Malta\runs\_queue\state.json'
        Put $stp ((Read-SharedText $stp) -replace '"arms":\s*\{', '"arms": { "big": { "peak_commit_gb": 100000 },')
        Initialize-RunnerContext -ProjectRoot $sim -QueueFile 'config/queue_h.txt' -PythonExe $py -DryRun $false
        $Q.EnvOk = $true
        $out = @(Invoke-RunnerIteration)
        Check ($out.Count -eq 1 -and $out[0].Action -eq 'WAIT' -and $out[0].Status -eq 'WAITING_MEMORY' -and $out[0].Detail -match 'peak 100000') "not enough commit waits: $($out[0].Status) $($out[0].Detail)"
        Check (-not [IO.Directory]::Exists((Join-Path $sim 'Malta\runs\_smoke_big'))) 'a waiting run gets no store'

        # One real pass of Invoke-RunnerMain (test mutex name): a queue whose runs are all DONE
        # ends QUEUE_DONE; the terminal step only lists scheduled tasks (none runs this queue).
        $RunnerMutexName = 'Global\TORRS_queue_runner_test_' + [guid]::NewGuid().ToString('N').Substring(0, 8)
        Initialize-RunnerContext -ProjectRoot $sim -QueueFile 'config/queue_f.txt' -PythonExe $py -DryRun $false
        $mainOut = @(Invoke-RunnerMain)
        Check ($mainOut.Count -eq 0) "Invoke-RunnerMain writes nothing to the pipeline: $($mainOut -join ' | ')"
        Check ((Read-SharedText (Join-Path $sim 'Malta\runs\_queue\status.txt')) -match '^QUEUE_DONE ') 'final status QUEUE_DONE'
        $ql = Read-SharedText (Join-Path $sim 'Malta\runs\_queue\queue.log')
        Check ($ql -match 'runner exits: QUEUE_DONE' -and ($ql -match 'no scheduled task of this queue found to disable' -or $ql -match 'cannot list scheduled tasks')) 'runner exit and the task step are logged'
        Check ((Get-MutexProbe) -match '^free') 'the runner released its mutex'

        # --- Fix round 1 -------------------------------------------------------------------
        # 1. Exit codes: a native abort (exit 3) and an exit 2 without main.py's refusal text are
        #    counted deaths and relaunched; exit 4 with the refusal text is FAILED (REFUSED).
        $res = & $drive 'queue_i.txt'
        Check ($res.Action -eq 'TERMINAL' -and $res.Status -eq 'FAILED' -and $res.Detail -match '_smoke_refuse \(REFUSED\)') "queue I ends FAILED (REFUSED) at _smoke_refuse: $($res.Status) $($res.Detail)"
        foreach ($pair in @(@('_smoke_abort', 3), @('_smoke_exittwo', 2))) {
            $rs = Join-Path $sim ('Malta\runs\' + $pair[0])
            Check ((Read-ExitCode (Join-Path $rs 'logs\launch_r1.exit')) -eq $pair[1]) "$($pair[0]) launch_r1 exited $($pair[1])"
            Check ([IO.File]::Exists((Join-Path $rs 'logs\launch_r2.log')) -and [IO.File]::Exists((Join-Path $rs 'DONE')) -and -not [IO.File]::Exists((Join-Path $rs 'FAILED'))) "$($pair[0]) was relaunched and finished"
            $rc = (Read-QueueState (Join-Path $sim 'Malta\runs\_queue'))['runs'][$pair[0]]
            Check ($rc['launches'] -eq 1 -and $rc['no_progress'] -eq 1 -and $rc['interrupted'] -eq 0) "$($pair[0]) counted one death: $(ConvertTo-Json $rc -Compress -Depth 5)"
        }
        $rf = Join-Path $sim 'Malta\runs\_smoke_refuse'
        Check ((Get-MarkerFirstLine (Join-Path $rf 'FAILED')) -eq 'REFUSED' -and -not [IO.File]::Exists((Join-Path $rf 'logs\launch_r2.log'))) 'exit 4 with the refusal text: FAILED (REFUSED), no relaunch'
        Check ((Get-MarkerLine (Join-Path $rf 'FAILED') 1) -match 'resume refused for') 'the FAILED detail quotes the refusal'

        # 2. STOP present when a live launch is adopted: wait for its NEXT checkpoint (288), not
        #    the ones already in its log (144).
        Initialize-RunnerContext -ProjectRoot $sim -QueueFile 'config/queue_j.txt' -PythonExe $py -DryRun $false
        $Q.EnvOk = $true
        $lj = Start-RunnerLaunch '_smoke_late' (Read-RunConfig (Get-RunIniPath '_smoke_late')) 'TRAIN' $null (Read-QueueState $Q.QueueDir) (Get-CodeHashLines '_smoke_late')
        $deadline = (Get-Date).AddSeconds(20)
        while ((Get-Date) -lt $deadline -and -not ((Get-SharedTextOrEmpty $lj.Log) -match 'step 144 ->')) { Start-Sleep -Milliseconds 200 }
        Check ((Get-SharedTextOrEmpty $lj.Log) -match 'Checkpoint saved at step 144 ->' -and (Get-SharedTextOrEmpty $lj.Log) -notmatch 'step 288 ->') 'the launch to adopt has saved checkpoint 144 only'
        Put $stopFile 'stop'
        $res = & $drive 'queue_j.txt'
        Check ($res.Action -eq 'STOPPED') "STOP with an adopted launch stops the runner: $($res.Action) $($res.Detail)"
        Check ((Get-GoodCheckpoint $sim '_smoke_late') -eq 288 -and (Read-SharedText $lj.Log) -match 'Checkpoint saved at step 288 ->') 'the adopted launch was killed after its next checkpoint (288), not at 144'
        Remove-Item -LiteralPath $stopFile
        $res = & $drive 'queue_j.txt'
        Check ($res.Status -eq 'QUEUE_DONE' -and [IO.File]::Exists((Join-Path $sim 'Malta\runs\_smoke_late\logs\launch_r2.log'))) "after STOP is removed the adopted run finishes: $($res.Status)"

        # 3. Heartbeats: during a long check, in a failed watch poll, while waiting for the port.
        Initialize-RunnerContext -ProjectRoot $sim -QueueFile 'config/queue_k.txt' -PythonExe $py -DryRun $false
        $statusFile = Join-Path $sim 'Malta\runs\_queue\status.txt'
        $RunnerCheckStatusSec = 1
        $r = Invoke-DetachedPython '-c "import time; time.sleep(2.5)"' 'slow_check' 60 'TESTCHECK'
        Check ($r.Code -eq 0 -and (Read-SharedText $statusFile) -match '^TESTCHECK slow_check: running for \d+ s \(limit 60 s\) \| heartbeat ') "status during a long check: $(Read-SharedText $statusFile)"
        $RunnerCheckStatusSec = 30
        $wj = New-JobRecord -RunName '_smoke_envmem' -Cfg $smokeCfg -Kind 'train' -N 7 -Started (Get-Date) -CmdPid $null -PyPid 1 -GoodBefore -1 -Levels $null
        try { throw 'simulated poll failure' } catch { Write-WatchError $wj $_ }
        Check ((Read-SharedText $statusFile) -match '^TRAINING _smoke_envmem launch_r7: poll failed, retrying in 1 s: simulated poll failure.* \| heartbeat ') "status after a failed poll: $(Read-SharedText $statusFile)"
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
        $listener.Start()
        try {
            $lport = ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port
            Check (-not (Wait-PortFree $lport 2)) 'a listening port is not free after the wait'
            Check ((Read-SharedText $statusFile) -match ('^CLEANUP waiting up to 2 s for TCP {0} to be free' -f $lport)) "status while waiting for the port: $(Read-SharedText $statusFile)"
        } finally { $listener.Stop() }
        Check (Wait-PortFree $lport 2) 'the port is free once the listener stops'

        # 4. The environment check runs only after the preflight passed; a memory error in it
        #    waits (WAITING_MEMORY), any other failure is ENV_ERROR.
        $origEnvArgs = $RunnerEnvCheckArgs
        try {
            $RunnerEnvCheckArgs = '-c "import sys; print(''ModuleNotFoundError: No module named tensorflow''); sys.exit(1)"'
            $logBefore = ([regex]::Matches((Read-SharedText (Join-Path $sim 'Malta\runs\_queue\queue.log')), 'environment check failed')).Count
            Initialize-RunnerContext -ProjectRoot $sim -QueueFile 'config/queue_h.txt' -PythonExe $py -DryRun $false
            $out = @(Invoke-RunnerIteration)
            Check ($out.Count -eq 1 -and $out[0].Status -eq 'WAITING_MEMORY' -and $out[0].Detail -match 'peak 100000' -and -not $Q.EnvOk) "a failed preflight waits before any environment check: $($out[0].Status) $($out[0].Detail)"
            Check (([regex]::Matches((Read-SharedText (Join-Path $sim 'Malta\runs\_queue\queue.log')), 'environment check failed')).Count -eq $logBefore) 'the environment check did not run'
            foreach ($memText in 'MemoryError: Unable to allocate 1.2 GiB', 'OSError: [WinError 1455] The paging file is too small for this operation to complete') {
                $RunnerEnvCheckArgs = '-c "import sys; print(''' + $memText + '''); sys.exit(1)"'
                Initialize-RunnerContext -ProjectRoot $sim -QueueFile 'config/queue_k.txt' -PythonExe $py -DryRun $false
                $out = @(Invoke-RunnerIteration)
                Check ($out.Count -eq 1 -and $out[0].Action -eq 'WAIT' -and $out[0].Status -eq 'WAITING_MEMORY' -and $out[0].Detail -match 'environment check hit a memory error' -and -not $Q.EnvOk) "a memory error in the environment check waits: $($out[0].Status) $($out[0].Detail)"
            }
            $RunnerEnvCheckArgs = '-c "import sys; print(''ModuleNotFoundError: No module named tensorflow''); sys.exit(1)"'
            Initialize-RunnerContext -ProjectRoot $sim -QueueFile 'config/queue_k.txt' -PythonExe $py -DryRun $false
            $out = @(Invoke-RunnerIteration)
            Check ($out.Count -eq 1 -and $out[0].Action -eq 'TERMINAL' -and $out[0].Status -eq 'ENV_ERROR' -and $out[0].Detail -match 'No module named tensorflow') "any other environment failure is ENV_ERROR: $($out[0].Status) $($out[0].Detail)"
            Check (-not [IO.Directory]::Exists((Join-Path $sim 'Malta\runs\_smoke_envmem\logs'))) 'nothing was launched for _smoke_envmem'
        } finally { $RunnerEnvCheckArgs = $origEnvArgs }

        # 5. TCP 27500 still busy after the cleanup wait -> WAITING_PORT_BUSY, nothing launched.
        $origCleanup = ${function:Invoke-RunnerCleanup}
        try {
            ${function:Invoke-RunnerCleanup} = { $false }
            Initialize-RunnerContext -ProjectRoot $sim -QueueFile 'config/queue_k.txt' -PythonExe $py -DryRun $false
            $Q.EnvOk = $true
            $out = @(Invoke-RunnerIteration)
            Check ($out.Count -eq 1 -and $out[0].Action -eq 'WAIT' -and $out[0].Status -eq 'WAITING_PORT_BUSY') "a busy port after cleanup waits: $($out[0].Status) $($out[0].Detail)"
            Check (-not [IO.Directory]::Exists((Join-Path $sim 'Malta\runs\_smoke_envmem\logs'))) 'nothing was launched while the port was busy'
        } finally { ${function:Invoke-RunnerCleanup} = $origCleanup }

        # --- Final-review fixes -----------------------------------------------------------
        # 3. arms.<arm>.checkpoint_gap_min is honoured (arm 'gap': 0.05 min): the watchdog kills a
        #    launch that saved checkpoint 144 and then hangs; the relaunch finishes the run.
        # 4. utils.py changes after _smoke_fake (the previous DONE run) finished: _smoke_gap's
        #    first launch logs a WARNING and goes ahead; its later launches do not warn again.
        Put (Join-Path $sim 'config\runs\_smoke_gap.ini') ($SmokeIni -replace 'comm_arch = fog', 'comm_arch = gap')
        Put (Join-Path $sim 'config\queue_l.txt') "_smoke_fake`n_smoke_gap`n"
        $stp = Join-Path $sim 'Malta\runs\_queue\state.json'
        Put $stp ((Read-SharedText $stp) -replace '"arms":\s*\{', '"arms": { "gap": { "peak_commit_gb": 0.5, "checkpoint_gap_min": 0.05 },')
        Put (Join-Path $sim 'utils.py') "# changed after _smoke_fake finished`n"
        $res = & $drive 'queue_l.txt'
        Check ($res.Status -eq 'QUEUE_DONE' -and [IO.File]::Exists((Join-Path $sim 'Malta\runs\_smoke_gap\DONE'))) "the gap run finishes: $($res.Status) $($res.Detail)"
        $ql = Read-SharedText (Join-Path $sim 'Malta\runs\_queue\queue.log')
        Check ($ql -match '_smoke_gap: watching launch_r1 \(.*checkpoint-gap limit 0\.05 min\)') 'the configured gap limit is used for the launch'
        Check ($ql -match '_smoke_gap: killing launch_r1: watchdog: no Checkpoint saved line for [\d.]+ min \(limit 0\.05 min\)') 'the watchdog killed launch_r1 at the configured gap'
        $gs = Join-Path $sim 'Malta\runs\_smoke_gap'
        Check ((Read-SharedText (Join-Path $gs 'logs\launch_r1.log')) -match 'step 144 ->' -and [IO.File]::Exists((Join-Path $gs 'logs\launch_r2.log')) -and -not [IO.File]::Exists((Join-Path $gs 'logs\launch_r3.log'))) 'killed after checkpoint 144, relaunched once'
        $rc = (Read-QueueState (Join-Path $sim 'Malta\runs\_queue'))['runs']['_smoke_gap']
        Check ($rc['launches'] -eq 1 -and $rc['no_progress'] -eq 0) "the killed launch made progress (144): $(ConvertTo-Json $rc -Compress -Depth 5)"
        $xw = [regex]::Matches($ql, '_smoke_gap: WARNING: first launch: code differs from the first launch of _smoke_fake \(_smoke_fake\\logs\\launch_r1\.code\.txt\): ([^;]*);')
        Check ($xw.Count -eq 1 -and $xw[0].Groups[1].Value -eq 'utils.py') "one cross-run code WARNING naming utils.py only: $($xw.Count) $(if ($xw.Count) { $xw[0].Value })"
        Check ($ql -notmatch '_smoke_gap: FAILED') 'the code warning does not fail the run'

        foreach ($r in '_smoke_fake', '_smoke_hang', '_smoke_nan', '_smoke_stop', '_smoke_slow', '_smoke_adopt', '_smoke_abort', '_smoke_exittwo', '_smoke_refuse', '_smoke_late', '_smoke_envmem', '_smoke_gap') { Check (-not (Test-RunProcessAlive $r)) "no python of $r is left alive" }
    }

    Write-Output ("ALL RUN_QUEUE TESTS PASSED ({0} checks)" -f $script:Asserts)
    $exitCode = 0
} catch {
    Write-Output ('TEST FAILED: ' + $_.Exception.Message)
    Write-Output $_.ScriptStackTrace
    $exitCode = 1
} finally {
    if ([IO.Directory]::Exists($TestRoot)) { Remove-Item -LiteralPath $TestRoot -Recurse -Force -ErrorAction SilentlyContinue }
    if ([IO.Directory]::Exists($TestRoot)) { Write-Output "warning: could not remove $TestRoot" }
}
exit $exitCode
