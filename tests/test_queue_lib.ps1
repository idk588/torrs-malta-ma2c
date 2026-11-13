# Plain-assert tests for tools/queue_lib.ps1, the queue runner's decision library (Task 12).
#
# Run:  powershell -NoProfile -ExecutionPolicy Bypass -File tests/test_queue_lib.ps1
#
# Every case builds its own fake project root under $env:TEMP\torrs_qlib_test_<random>\
# (Malta\runs\<name>\... and a fake Malta\runs\_queue\), so nothing in the real project is read
# or written. The temp folder is removed at the end. Success prints ALL QUEUE_LIB TESTS PASSED
# and exits 0; the first failed assert prints its message and exits 1.

$ErrorActionPreference = 'Stop'
$TestRoot = Join-Path $env:TEMP ('torrs_qlib_test_' + [guid]::NewGuid().ToString('N').Substring(0, 12))
$Utf8 = New-Object System.Text.UTF8Encoding $false
$Held = New-Object System.Collections.Generic.List[System.IDisposable]
$exitCode = 1

# ---------------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------------
$SmokeIni = @'
[MODEL_CONFIG]
adam_beta1=0.9
; route
save_optimizer = true

[TRAIN_CONFIG]
total_step = 288
test_interval = 1440
log_interval = 720

[ENV_CONFIG]
control_interval_sec = 5
# smoke: two 720 s episodes
episode_length_sec=720
seed = 12
test_seeds = 10000,20000
eval_levels = 1.00,1.10
'@

$TrainHeader = ',agent,step,test_id,reward,total arrived vehicle number,total departured vehicle number,total waiting time,total travel time,average waiting time,average travel time,seed'
$CostHeader = 'episode,step,seed,wall_s,rollout_s,update_s,save_s,peak_private_gb,peak_working_set_gb,ckpt_data_mb,ckpt_meta_mb,lost'
$ExitOk = "EXIT=0 at Fri 11/13/2026 14:00:00.00 `r`n"

function Put([string]$Path, [string]$Text) {
    $dir = [IO.Path]::GetDirectoryName($Path)
    if (-not [IO.Directory]::Exists($dir)) { [void][IO.Directory]::CreateDirectory($dir) }
    [IO.File]::WriteAllText($Path, $Text, $Utf8)
}

function New-Project {
    $p = Join-Path $TestRoot ('p' + [guid]::NewGuid().ToString('N').Substring(0, 8))
    [void][IO.Directory]::CreateDirectory((Join-Path $p 'Malta\runs\_queue'))
    $p
}

function New-Store([string]$Project, [string]$Name, [string]$Ini = $SmokeIni) {
    $s = Join-Path $Project ('Malta\runs\' + $Name)
    foreach ($d in 'model', 'logs') { [void][IO.Directory]::CreateDirectory((Join-Path $s $d)) }
    Put (Join-Path $s 'config.ini') $Ini
    $s
}

function Add-Ckpt([string]$Store, [int]$Step, [string[]]$Suffixes = @('.data-00000-of-00001', '.index', '.meta')) {
    foreach ($x in $Suffixes) { Put (Join-Path $Store ('model\checkpoint-' + $Step + $x)) 'x' }
}

function Get-SavedLine([string]$Name, [int]$Step) {
    "2026-11-13 10:00:00,000 [INFO] Checkpoint saved at step $Step -> Malta/runs/$Name/model/"
}

function Get-Lines([string[]]$Lines) { ($Lines -join "`r`n") + "`r`n" }

# Rows are 'step,seed,reward,average waiting time' strings, so '' or 'inf' can be written.
function Get-TrainCsv([string[]]$Rows) {
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.Append($TrainHeader + "`r`n")
    $i = 0
    foreach ($r in $Rows) {
        $f = $r.Split(',')
        [void]$sb.Append(('{0},ma2c,{1},-1,{2},320,330,4000.0,90000.0,{3},272.7,{4}' -f $i, $f[0], $f[2], $f[3], $f[1]) + "`r`n")
        $i++
    }
    $sb.ToString()
}

function Get-AgentCsv([int]$Count, [string]$Value = '-1.000000000000000000e+02') {
    $sb = New-Object System.Text.StringBuilder
    for ($i = 0; $i -lt $Count; $i++) { [void]$sb.Append("$Value,0.000000000000000000e+00,-3.000000000000000000e+01`n") }
    $sb.ToString()
}

# Rows are 'episode,step,seed' strings.
function Get-CostCsv([string[]]$Rows) {
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.Append($CostHeader + "`r`n")
    foreach ($r in $Rows) { [void]$sb.Append($r + ",100.0000,80.0000,10.0000,5.0000,40.1000,30.2000,150.0000,740.0000,0`r`n") }
    $sb.ToString()
}

function Get-DataLines([string]$Path) {
    @([IO.File]::ReadAllLines($Path) | Where-Object { $_.Trim() -ne '' })
}

function Get-Dirs([string]$Parent, [string]$Filter) {
    @(Get-ChildItem -LiteralPath $Parent -Directory -Filter $Filter)
}

function Test-NoBom([string]$Path) {
    $b = [IO.File]::ReadAllBytes($Path)
    -not ($b.Length -ge 3 -and $b[0] -eq 0xEF -and $b[1] -eq 0xBB -and $b[2] -eq 0xBF)
}

# A store whose training finished: checkpoints 144 and 288, both logged, TRAINING COMPLETE,
# exit 0, two reward rows with seeds 12 and 13.
function New-CompleteStore([string]$Project, [string]$Name) {
    $s = New-Store $Project $Name
    Add-Ckpt $s 144
    Add-Ckpt $s 288
    Put (Join-Path $s 'logs\launch_r1.log') (Get-Lines @((Get-SavedLine $Name 144), (Get-SavedLine $Name 288), '>>> TRAINING COMPLETE step 288'))
    Put (Join-Path $s 'logs\launch_r1.exit') $ExitOk
    Put (Join-Path $s 'train_reward.csv') (Get-TrainCsv @('144,12,-1234.5,12.1', '288,13,-1100.25,11.4'))
    Put (Join-Path $s 'agent_reward.csv') (Get-AgentCsv 2)
    Put (Join-Path $s 'episode_cost.csv') (Get-CostCsv @('1,144,12', '2,288,13'))
    $s
}

# One demand level's evaluation output: the three CSVs (episodes 1..2; the control CSV has a
# quoted action list with commas, as pandas writes it) and the per-seed XML files.
function New-LevelFiles([string]$Store, [string]$Tag, [int]$TripsPerEpisode) {
    $dir = Join-Path $Store ('eva_data\f' + $Tag)
    $ctl = New-Object System.Text.StringBuilder
    $trf = New-Object System.Text.StringBuilder
    $trp = New-Object System.Text.StringBuilder
    [void]$ctl.Append(",episode,time_sec,step,action,reward`r`n")
    [void]$trf.Append(",episode,time_sec,number_total_car,avg_wait_sec`r`n")
    [void]$trp.Append(",episode,id,depart_sec,arrival_sec,duration_sec,wait_step,wait_sec`r`n")
    $i = 0; $t = 0
    foreach ($e in 1, 2) {
        foreach ($s in 1, 2) {
            [void]$ctl.Append(('{0},{1},{2},{3}.0,"1,0,2,1",-3.5' -f $i, $e, (5 * $s), $s) + "`r`n")
            [void]$trf.Append(('{0},{1},{2},8,0.5' -f $i, $e, (5 * $s)) + "`r`n")
            $i++
        }
        for ($v = 0; $v -lt $TripsPerEpisode; $v++) {
            [void]$trp.Append(('{0},{1},flow_{2}.0,0.00,54.00,54.00,0,0.00' -f $t, $e, $v) + "`r`n")
            $t++
        }
    }
    Put (Join-Path $dir 'Malta_signal_route_control.csv') $ctl.ToString()
    Put (Join-Path $dir 'Malta_signal_route_traffic.csv') $trf.ToString()
    Put (Join-Path $dir 'Malta_signal_route_trip.csv') $trp.ToString()
    foreach ($seed in 10000, 20000) {
        Put (Join-Path $dir "edgedata_$seed.xml") "<meandata>`r`n    <interval begin=`"0.00`" end=`"720.00`"/>`r`n</meandata>`r`n"
        Put (Join-Path $dir "summary_$seed.xml") "<summary>`r`n    <step time=`"0.00`"/>`r`n</summary>`r`n"
    }
}

function Get-Block([string]$Tag, [string]$BgTag, [string]$Total, [string]$Expected,
                   [int[]]$Tests = @(0, 1), [bool]$Done = $true, [string[]]$Extra = @()) {
    @{ Tag = $Tag; BgTag = $BgTag; Total = $Total; Expected = $Expected; Tests = $Tests; Done = $Done; Extra = $Extra }
}

function New-EvalLog([string]$Name, [object[]]$Blocks, [string]$Complete, [bool]$Restore, [string]$RestoreRun) {
    $L = New-Object System.Collections.Generic.List[string]
    $L.Add('>>> Initializing env (launches SUMO for dimensions)...')
    if ($Restore) {
        $L.Add("INFO:tensorflow:Restoring parameters from Malta/runs/$RestoreRun/model/checkpoint-288")
        $L.Add("2026-11-13 12:00:01,000 [INFO] Restoring parameters from Malta/runs/$RestoreRun/model/checkpoint-288")
        $L.Add('2026-11-13 12:00:02,000 [INFO] Checkpoint loaded: checkpoint-288')
    }
    foreach ($b in $Blocks) {
        $L.Add(('2026-11-13 12:01:00,000 [INFO] EVAL_LEVEL_START f{0} bg_tag={1} corridor_total={2} corridor_expected={3}' -f $b.Tag, $b.BgTag, $b.Total, $b.Expected))
        foreach ($t in $b.Tests) {
            $L.Add("Malta/runs/$Name/eva_data/f$($b.Tag)/Malta_signal_route_trip.xml")
            $L.Add("2026-11-13 12:02:00,000 [INFO] test $t, avg reward -12.34")
        }
        foreach ($x in $b.Extra) { $L.Add($x) }
        if ($b.Done) { $L.Add(('2026-11-13 12:03:00,000 [INFO] EVAL_LEVEL_DONE f{0} seeds=2' -f $b.Tag)) }
    }
    if ($Complete) { $L.Add($Complete) }
    Get-Lines $L.ToArray()
}

# A fresh store with level folders (trips per episode by level tag), eval_r1.log and its exit
# file. The launch time is 10 minutes before the files were written.
function New-EvalCase([object[]]$Blocks, [string]$Complete = '>>> EVALUATE COMPLETE levels=2',
                      [string]$Exit = $ExitOk, [bool]$Restore = $true, [hashtable]$Trips = $null,
                      [string]$RestoreRun = $null) {
    if ($null -eq $Trips) { $Trips = @{ '1.00' = 3; '1.10' = 4 } }
    $proj = New-Project
    $name = 'ev_' + [guid]::NewGuid().ToString('N').Substring(0, 6)
    if (-not $RestoreRun) { $RestoreRun = $name }
    $store = New-Store $proj $name
    $launch = (Get-Date).AddMinutes(-10)
    foreach ($tag in $Trips.Keys) { New-LevelFiles $store $tag $Trips[$tag] }
    $log = Join-Path $store 'logs\eval_r1.log'
    Put $log (New-EvalLog $name $Blocks $Complete $Restore $RestoreRun)
    if ($Exit) { Put (Join-Path $store 'logs\eval_r1.exit') $Exit }
    @{ Project = $proj; Name = $name; Store = $store; Log = $log; Launch = $launch
       Cfg = (Read-RunConfig (Join-Path $store 'config.ini')) }
}

function Invoke-Eval([hashtable]$C, [double[]]$Levels = $null) {
    if ($null -eq $Levels) { return Test-EvalAttempt $C.Project $C.Name $C.Cfg $C.Log $C.Launch }
    Test-EvalAttempt $C.Project $C.Name $C.Cfg $C.Log $C.Launch -Levels $Levels
}

try {
    [void][IO.Directory]::CreateDirectory($TestRoot)
    . (Join-Path $PSScriptRoot '..\tools\queue_lib.ps1')

    # -----------------------------------------------------------------------------------------
    # Write-Utf8NoBom / Add-Utf8NoBomLine: no BOM, absolute paths only
    # -----------------------------------------------------------------------------------------
    $p = Join-Path $TestRoot 'nobom\status.txt'
    Write-Utf8NoBom $p "caf$([char]0xE9) RUNNING`r`n"
    if (-not (Test-NoBom $p)) { throw 'Write-Utf8NoBom wrote a BOM (EF BB BF)' }
    if (-not ([IO.File]::ReadAllText($p, $Utf8) -eq "caf$([char]0xE9) RUNNING`r`n")) { throw 'Write-Utf8NoBom text did not round-trip' }
    $p2 = Join-Path $TestRoot 'nobom\queue.log'
    Add-Utf8NoBomLine $p2 'line one'
    Add-Utf8NoBomLine $p2 'line two'
    if (-not (Test-NoBom $p2)) { throw 'Add-Utf8NoBomLine wrote a BOM' }
    if (-not ([IO.File]::ReadAllText($p2) -eq "line one`r`nline two`r`n")) { throw 'Add-Utf8NoBomLine did not append two CRLF lines' }
    $threw = $false
    try { Write-Utf8NoBom 'relative\x.txt' 'x' } catch { $threw = $true }
    if (-not $threw) { throw 'Write-Utf8NoBom accepted a relative path' }

    # -----------------------------------------------------------------------------------------
    # Read-SharedText: a log another handle is writing (FileShare.ReadWrite)
    # -----------------------------------------------------------------------------------------
    $heldPath = Join-Path $TestRoot 'held.log'
    $w = [IO.FileStream]::new($heldPath, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::ReadWrite)
    $Held.Add($w)
    $bytes = $Utf8.GetBytes("first line`r`n")
    $w.Write($bytes, 0, $bytes.Length); $w.Flush()
    $threw = $false
    try { [void][IO.File]::ReadAllText($heldPath) } catch { $threw = $true }
    if (-not $threw) { throw 'precondition: ReadAllText should fail on a file another handle is writing' }
    if (-not ((Read-SharedText $heldPath) -eq "first line`r`n")) { throw 'Read-SharedText did not read the held file' }
    $bytes = $Utf8.GetBytes("second line`r`n")
    $w.Write($bytes, 0, $bytes.Length); $w.Flush()
    if (-not ((Read-SharedText $heldPath) -eq "first line`r`nsecond line`r`n")) { throw 'Read-SharedText missed text written after the first read' }
    $w.Dispose()

    # -----------------------------------------------------------------------------------------
    # Read-RunConfig, Get-QueueRuns, level folders and background tags
    # -----------------------------------------------------------------------------------------
    $proj = New-Project
    $ini = Join-Path $proj 'smoke.ini'
    Put $ini $SmokeIni
    $cfg = Read-RunConfig $ini
    if (-not ($cfg.TotalStep -eq 288)) { throw "Read-RunConfig TotalStep: expected 288, got $($cfg.TotalStep)" }
    if (-not ($cfg.T -eq 144)) { throw "Read-RunConfig T: expected 144, got $($cfg.T)" }
    if (-not ($cfg.EvalLevels.Count -eq 2 -and $cfg.EvalLevels[0] -eq 1.0 -and $cfg.EvalLevels[1] -eq 1.1)) { throw "Read-RunConfig EvalLevels: expected 1.0,1.1, got $($cfg.EvalLevels -join ' ')" }
    if (-not ($cfg.TestSeeds.Count -eq 2 -and $cfg.TestSeeds[0] -eq 10000 -and $cfg.TestSeeds[1] -eq 20000)) { throw 'Read-RunConfig TestSeeds: expected 10000,20000' }
    if (-not ($cfg.Seed -eq 12)) { throw "Read-RunConfig Seed: expected 12, got $($cfg.Seed)" }

    $ini2 = Join-Path $proj 'rq2.ini'
    Put $ini2 ("`r`n[TRAIN_CONFIG]`r`ntotal_step=72000.0`r`n[ENV_CONFIG]`r`ncontrol_interval_sec=5`r`nepisode_length_sec = 3600`r`nseed=12`r`ntest_seeds = 10000, 20000,30000`r`n")
    $cfg2 = Read-RunConfig $ini2
    if (-not ($cfg2.TotalStep -eq 72000 -and $cfg2.T -eq 720)) { throw "Read-RunConfig RQ2: expected 72000/720, got $($cfg2.TotalStep)/$($cfg2.T)" }
    if (-not ($cfg2.EvalLevels.Count -eq 1 -and $cfg2.EvalLevels[0] -eq 1.0)) { throw 'Read-RunConfig: eval_levels fallback should be 1.0' }
    if (-not ($cfg2.TestSeeds.Count -eq 3 -and $cfg2.TestSeeds[2] -eq 30000)) { throw 'Read-RunConfig: test_seeds with spaces' }

    $qf = Join-Path $proj 'queue.txt'
    Put $qf "# RQ2 queue`r`ncloud_base`r`n`r`n  fog_base  `r`n# hybrid_base`r`nhybrid_base`r`n"
    $runs = @(Get-QueueRuns $qf)
    if (-not (($runs -join '|') -eq 'cloud_base|fog_base|hybrid_base')) { throw "Get-QueueRuns: got '$($runs -join '|')'" }

    $expectTags = @{ 1.00 = 'p1.2'; 1.06 = 'p1.132'; 1.10 = 'p1.091'; 1.13 = 'p1.062'; 1.20 = 'p1.0'; 2.00 = 'p0.6' }
    foreach ($lv in $expectTags.Keys) {
        if (-not ((Format-BgTag $lv) -eq $expectTags[$lv])) { throw "Format-BgTag $lv gave $(Format-BgTag $lv), expected $($expectTags[$lv])" }
    }
    $saved = [Threading.Thread]::CurrentThread.CurrentCulture
    try {
        [Threading.Thread]::CurrentThread.CurrentCulture = [cultureinfo]'de-DE'
        if (-not ((Format-Level 1.1) -eq '1.10')) { throw "Format-Level 1.1 under de-DE gave '$(Format-Level 1.1)'" }
        if (-not ((Get-LevelDir $proj 'arm' 1.1) -eq (Join-Path $proj 'Malta\runs\arm\eva_data\f1.10'))) { throw "Get-LevelDir under de-DE gave $(Get-LevelDir $proj 'arm' 1.1)" }
        $cfgDe = Read-RunConfig $ini
        if (-not ($cfgDe.EvalLevels[1] -eq 1.1)) { throw 'Read-RunConfig under de-DE misread 1.10' }
    } finally { [Threading.Thread]::CurrentThread.CurrentCulture = $saved }

    # -----------------------------------------------------------------------------------------
    # Get-GoodCheckpoint
    # -----------------------------------------------------------------------------------------
    $proj = New-Project
    $name = 'arm_a'
    $store = New-Store $proj $name
    $cfg = Read-RunConfig (Join-Path $store 'config.ini')
    Add-Ckpt $store 144
    Add-Ckpt $store 288 @('.index', '.data-00000-of-00001.tempstate123')
    Add-Ckpt $store 432    # complete, but logged only for ANOTHER run's store
    Add-Ckpt $store 576 @('.data-00000-of-00001', '.index')   # logged, but no .meta
    Put (Join-Path $store 'model\checkpoint') 'model_checkpoint_path: "checkpoint-288"'
    Put (Join-Path $store 'logs\launch_r1.log') (Get-Lines @('>>> Starting trainer.run()', (Get-SavedLine $name 144),
        (Get-SavedLine 'other' 432), (Get-SavedLine $name 576), 'Traceback (most recent call last):'))
    Put (Join-Path $store 'logs\launch_r1.exit') "EXIT=1 at Fri 11/13/2026 10:00:00.00 `r`n"
    $good = Get-GoodCheckpoint $proj $name
    if (-not ($good -is [int])) { throw "Get-GoodCheckpoint should return an int, got $($good.GetType().Name)" }
    if (-not ($good -eq 144)) { throw "Get-GoodCheckpoint: expected 144, got $good" }
    if (-not ((Get-GoodCheckpoint $proj 'no_such_run') -eq -1)) { throw 'Get-GoodCheckpoint without a store should be -1' }

    # A launch log that python is still writing is read through FileShare.ReadWrite.
    $live = [IO.FileStream]::new((Join-Path $store 'logs\launch_r2.log'), [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::ReadWrite)
    $Held.Add($live)
    Add-Ckpt $store 720
    $bytes = $Utf8.GetBytes((Get-SavedLine $name 720) + "`r`n")
    $live.Write($bytes, 0, $bytes.Length); $live.Flush()
    if (-not ((Get-GoodCheckpoint $proj $name) -eq 720)) { throw 'Get-GoodCheckpoint did not read a launch log held open by its writer' }
    $live.Dispose()
    Move-Item -LiteralPath (Join-Path $store 'logs\launch_r2.log') -Destination (Join-Path $TestRoot 'arm_a_launch_r2.log')
    foreach ($x in '.data-00000-of-00001', '.index', '.meta') { Move-Item -LiteralPath (Join-Path $store "model\checkpoint-720$x") -Destination $TestRoot }

    # -----------------------------------------------------------------------------------------
    # Invoke-Recovery: RESUME (quarantine, trim, lost cost row)
    # -----------------------------------------------------------------------------------------
    Put (Join-Path $store 'train_reward.csv') (Get-TrainCsv @('144,12,-1234.5,12.1', '288,13,-1100.2,11.4'))
    Put (Join-Path $store 'agent_reward.csv') (Get-AgentCsv 2)
    Put (Join-Path $store 'episode_cost.csv') (Get-CostCsv @())
    $queue = Join-Path $proj 'Malta\runs\_queue'
    $recLog = New-Object System.Collections.Generic.List[string]
    $r = Invoke-Recovery $proj $name $cfg $queue -ProcessAlive $false -Log { param($m) $recLog.Add($m) }
    if (-not ($r -is [string])) { throw 'Invoke-Recovery should return one string' }
    if (-not ($r -eq 'RESUME')) { throw "Invoke-Recovery: expected RESUME, got $r" }
    $qd = Get-Dirs $queue 'quarantine_*'
    if (-not ($qd.Count -eq 1)) { throw "expected one quarantine folder, got $($qd.Count)" }
    $qnames = @(Get-ChildItem -LiteralPath $qd[0].FullName | ForEach-Object { $_.Name } | Sort-Object)
    $expectQ = @('checkpoint-288.data-00000-of-00001.tempstate123', 'checkpoint-288.index', 'checkpoint-432.data-00000-of-00001',
                 'checkpoint-432.index', 'checkpoint-432.meta', 'checkpoint-576.data-00000-of-00001', 'checkpoint-576.index') | Sort-Object
    if (-not (($qnames -join '|') -eq ($expectQ -join '|'))) { throw "quarantine holds '$($qnames -join ', ')'" }
    $left = @(Get-ChildItem -LiteralPath (Join-Path $store 'model') | ForEach-Object { $_.Name } | Sort-Object)
    if (-not (($left -join '|') -eq 'checkpoint|checkpoint-144.data-00000-of-00001|checkpoint-144.index|checkpoint-144.meta')) { throw "model\ keeps '$($left -join ', ')'" }
    $tl = Get-DataLines (Join-Path $store 'train_reward.csv')
    if (-not ($tl.Count -eq 2 -and $tl[1] -like '0,ma2c,144,*')) { throw "train_reward.csv not trimmed to the step-144 row: $($tl -join ' / ')" }
    if (-not ((Get-DataLines (Join-Path $store 'agent_reward.csv')).Count -eq 1)) { throw 'agent_reward.csv not trimmed to 1 row' }
    $cl = Get-DataLines (Join-Path $store 'episode_cost.csv')
    if (-not ($cl.Count -eq 2 -and $cl[0] -eq $CostHeader)) { throw "episode_cost.csv: expected header + 1 row, got $($cl -join ' / ')" }
    if (-not ($cl[1] -eq '1,144,12,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,1')) { throw "lost cost row is '$($cl[1])'" }
    foreach ($f in 'train_reward.csv', 'agent_reward.csv', 'episode_cost.csv') {
        if (-not (Test-NoBom (Join-Path $store $f))) { throw "$f was rewritten with a BOM" }
    }
    $rd = Get-Dirs $queue 'recovery_*'
    if (-not ($rd.Count -eq 1)) { throw "expected one recovery folder, got $($rd.Count)" }
    if (-not ((Get-DataLines (Join-Path $rd[0].FullName 'train_reward.csv')).Count -eq 3)) { throw 'recovery folder lacks the original 2-row train_reward.csv' }
    if (-not ((Get-DataLines (Join-Path $rd[0].FullName 'agent_reward.csv')).Count -eq 2)) { throw 'recovery folder lacks the original agent_reward.csv' }
    if (-not ([IO.File]::Exists((Join-Path $rd[0].FullName 'episode_cost.csv')))) { throw 'recovery folder lacks the original episode_cost.csv' }
    if (-not ($recLog.Count -ge 2)) { throw 'Invoke-Recovery logged no actions through -Log' }
    # A second recovery of the now-consistent store changes nothing.
    $r = Invoke-Recovery $proj $name $cfg $queue -ProcessAlive $false
    if (-not ($r -eq 'RESUME')) { throw "second Invoke-Recovery: expected RESUME, got $r" }
    if (-not ((Get-Dirs $queue 'quarantine_*').Count -eq 1 -and (Get-Dirs $queue 'recovery_*').Count -eq 1)) { throw 'a no-op recovery created folders' }
    # Recovery refuses while a process of the run is alive.
    $threw = $false
    try { [void](Invoke-Recovery $proj $name $cfg $queue -ProcessAlive $true) } catch { $threw = $true }
    if (-not $threw) { throw 'Invoke-Recovery ran while a process of the run was alive' }

    # -----------------------------------------------------------------------------------------
    # Invoke-Recovery: SCRATCH
    # -----------------------------------------------------------------------------------------
    $proj = New-Project; $queue = Join-Path $proj 'Malta\runs\_queue'
    $store = New-Store $proj 'arm_b'
    Put (Join-Path $store 'train_reward.csv') ''
    $r = Invoke-Recovery $proj 'arm_b' $cfg $queue -ProcessAlive $false
    if (-not ($r -eq 'SCRATCH')) { throw "empty model\ + empty train_reward.csv: expected SCRATCH, got $r" }
    if ([IO.File]::Exists((Join-Path $store 'train_reward.csv'))) { throw 'SCRATCH left the empty train_reward.csv in the store' }
    $rd = Get-Dirs $queue 'recovery_*'
    if (-not ($rd.Count -eq 1 -and [IO.File]::Exists((Join-Path $rd[0].FullName 'train_reward.csv')))) { throw 'SCRATCH did not move train_reward.csv to recovery_<stamp>' }

    $store = New-Store $proj 'arm_b2'
    Put (Join-Path $store 'train_reward.csv') ($TrainHeader + "`r`n")
    Put (Join-Path $store 'episode_cost.csv') ($CostHeader + "`r`n")
    $r = Invoke-Recovery $proj 'arm_b2' $cfg $queue -ProcessAlive $false
    if (-not ($r -eq 'SCRATCH')) { throw "header-only train_reward.csv: expected SCRATCH, got $r" }
    if ([IO.File]::Exists((Join-Path $store 'episode_cost.csv'))) { throw 'SCRATCH left the partial episode_cost.csv' }
    $r = Invoke-Recovery $proj 'never_launched' $cfg $queue -ProcessAlive $false
    if (-not ($r -eq 'SCRATCH')) { throw "a run without a store: expected SCRATCH, got $r" }

    # -----------------------------------------------------------------------------------------
    # Invoke-Recovery: NEEDS_REVIEW (nothing is moved)
    # -----------------------------------------------------------------------------------------
    $proj = New-Project; $queue = Join-Path $proj 'Malta\runs\_queue'
    $store = New-Store $proj 'arm_c'
    Add-Ckpt $store 144
    Put (Join-Path $store 'train_reward.csv') (Get-TrainCsv @('144,12,-1234.5,12.1'))
    $r = Invoke-Recovery $proj 'arm_c' $cfg $queue -ProcessAlive $false
    if (-not ($r -like 'NEEDS_REVIEW:?*')) { throw "checkpoint files without a log line: expected NEEDS_REVIEW:<reason>, got $r" }
    if (-not ((@(Get-ChildItem -LiteralPath (Join-Path $store 'model')).Count -eq 3))) { throw 'NEEDS_REVIEW moved checkpoint files' }

    $store = New-Store $proj 'arm_c2'      # no checkpoint at all, but a training row
    Put (Join-Path $store 'train_reward.csv') (Get-TrainCsv @('144,12,-1234.5,12.1'))
    $r = Invoke-Recovery $proj 'arm_c2' $cfg $queue -ProcessAlive $false
    if (-not ($r -like 'NEEDS_REVIEW:?*')) { throw "no checkpoint but a train row: expected NEEDS_REVIEW, got $r" }

    $store = New-Store $proj 'arm_c3'      # reward CSV SHORT: checkpoint-288 logged, one train row
    Add-Ckpt $store 288
    Put (Join-Path $store 'logs\launch_r1.log') (Get-Lines @((Get-SavedLine 'arm_c3' 288)))
    Put (Join-Path $store 'train_reward.csv') (Get-TrainCsv @('144,12,-1234.5,12.1'))
    Put (Join-Path $store 'agent_reward.csv') (Get-AgentCsv 2)
    Add-Ckpt $store 432 @('.index')
    $r = Invoke-Recovery $proj 'arm_c3' $cfg $queue -ProcessAlive $false
    if (-not ($r -like 'NEEDS_REVIEW:*train_reward.csv*')) { throw "short train_reward.csv: expected NEEDS_REVIEW, got $r" }
    if (-not ([IO.File]::Exists((Join-Path $store 'model\checkpoint-432.index')))) { throw 'NEEDS_REVIEW (short CSV) still quarantined files' }

    $store = New-Store $proj 'arm_c4'      # agent_reward.csv SHORT
    Add-Ckpt $store 288
    Put (Join-Path $store 'logs\launch_r1.log') (Get-Lines @((Get-SavedLine 'arm_c4' 288)))
    Put (Join-Path $store 'train_reward.csv') (Get-TrainCsv @('144,12,-1234.5,12.1', '288,13,-1100.2,11.4'))
    Put (Join-Path $store 'agent_reward.csv') (Get-AgentCsv 1)
    $r = Invoke-Recovery $proj 'arm_c4' $cfg $queue -ProcessAlive $false
    if (-not ($r -like 'NEEDS_REVIEW:*agent_reward.csv*')) { throw "short agent_reward.csv: expected NEEDS_REVIEW, got $r" }

    $store = New-Store $proj 'arm_c5'      # train_reward.csv fails to parse
    Add-Ckpt $store 144
    Put (Join-Path $store 'logs\launch_r1.log') (Get-Lines @((Get-SavedLine 'arm_c5' 144)))
    Put (Join-Path $store 'train_reward.csv') ((Get-TrainCsv @('144,12,-1234.5,12.1')) + "1,ma2c,288,broken`r`n")
    Put (Join-Path $store 'agent_reward.csv') (Get-AgentCsv 2)
    $r = Invoke-Recovery $proj 'arm_c5' $cfg $queue -ProcessAlive $false
    if (-not ($r -like 'NEEDS_REVIEW:*train_reward.csv*')) { throw "unparseable train_reward.csv: expected NEEDS_REVIEW, got $r" }
    if (-not ((Get-Dirs $queue 'quarantine_*').Count -eq 0 -and (Get-Dirs $queue 'recovery_*').Count -eq 0)) { throw 'NEEDS_REVIEW cases moved files' }

    # -----------------------------------------------------------------------------------------
    # Invoke-Recovery: a restore error quarantines that checkpoint and falls back one
    # -----------------------------------------------------------------------------------------
    $proj = New-Project; $queue = Join-Path $proj 'Malta\runs\_queue'
    $store = New-Store $proj 'arm_d'
    Add-Ckpt $store 144
    Add-Ckpt $store 288
    Put (Join-Path $store 'logs\launch_r1.log') (Get-Lines @((Get-SavedLine 'arm_d' 144), (Get-SavedLine 'arm_d' 288)))
    Put (Join-Path $store 'logs\launch_r1.exit') "EXIT=-1073741819 at Fri 11/13/2026 10:00:00.00 `r`n"
    Put (Join-Path $store 'logs\launch_r2.log') (Get-Lines @('INFO:tensorflow:Restoring parameters from Malta/runs/arm_d/model/checkpoint-288',
        '2026-11-13 11:00:00,000 [INFO] Restoring parameters from Malta/runs/arm_d/model/checkpoint-288',
        'Traceback (most recent call last):',
        'tensorflow.python.framework.errors_impl.DataLossError: Checksum does not match: stored 1 vs. calculated on the restored bytes 2'))
    Put (Join-Path $store 'logs\launch_r2.exit') "EXIT=1 at Fri 11/13/2026 11:00:30.00 `r`n"
    Put (Join-Path $store 'train_reward.csv') (Get-TrainCsv @('144,12,-1234.5,12.1', '288,13,-1100.2,11.4'))
    Put (Join-Path $store 'agent_reward.csv') (Get-AgentCsv 2)
    Put (Join-Path $store 'episode_cost.csv') (Get-CostCsv @('1,144,12', '2,288,13'))
    $r = Invoke-Recovery $proj 'arm_d' $cfg $queue -ProcessAlive $false
    if (-not ($r -eq 'RESUME')) { throw "restore error: expected RESUME, got $r" }
    if ([IO.File]::Exists((Join-Path $store 'model\checkpoint-288.index'))) { throw 'the checkpoint that failed to restore was not quarantined' }
    if (-not ((Get-GoodCheckpoint $proj 'arm_d') -eq 144)) { throw 'after a restore error the good checkpoint should fall back to 144' }
    if (-not ((Get-DataLines (Join-Path $store 'train_reward.csv')).Count -eq 2)) { throw 'restore fallback did not trim train_reward.csv to 1 row' }
    $cl = Get-DataLines (Join-Path $store 'episode_cost.csv')
    if (-not ($cl.Count -eq 2 -and $cl[1] -like '1,144,12,*,0')) { throw "restore fallback cost CSV: $($cl -join ' / ')" }

    # The same error in an OLDER launch log (the newest launch saved checkpoints) is ignored.
    $store = New-Store $proj 'arm_d2'
    Add-Ckpt $store 144
    Add-Ckpt $store 288
    Put (Join-Path $store 'logs\launch_r1.log') (Get-Lines @('DataLossError: Unable to open table file Malta/runs/arm_d2/model/checkpoint-144.index'))
    Put (Join-Path $store 'logs\launch_r2.log') (Get-Lines @((Get-SavedLine 'arm_d2' 144), (Get-SavedLine 'arm_d2' 288)))
    Put (Join-Path $store 'train_reward.csv') (Get-TrainCsv @('144,12,-1234.5,12.1', '288,13,-1100.2,11.4'))
    Put (Join-Path $store 'agent_reward.csv') (Get-AgentCsv 2)
    Put (Join-Path $store 'episode_cost.csv') (Get-CostCsv @('1,144,12', '2,288,13'))
    $r = Invoke-Recovery $proj 'arm_d2' $cfg $queue -ProcessAlive $false
    if (-not ($r -eq 'RESUME' -and (Get-GoodCheckpoint $proj 'arm_d2') -eq 288)) { throw "an old restore error changed recovery: $r" }
    if (-not ([IO.File]::Exists((Join-Path $store 'model\checkpoint-144.index')))) { throw 'an old restore error quarantined checkpoint-144' }

    # A quarantine move that fails (file held open without delete sharing) stops recovery with
    # an error, even when the caller runs with ErrorActionPreference Continue.
    $store = New-Store $proj 'arm_d3'
    Add-Ckpt $store 144
    Add-Ckpt $store 288 @('.index')
    Put (Join-Path $store 'logs\launch_r1.log') (Get-Lines @((Get-SavedLine 'arm_d3' 144)))
    Put (Join-Path $store 'train_reward.csv') (Get-TrainCsv @('144,12,-1234.5,12.1'))
    Put (Join-Path $store 'agent_reward.csv') (Get-AgentCsv 1)
    Put (Join-Path $store 'episode_cost.csv') (Get-CostCsv @('1,144,12'))
    $lock = [IO.FileStream]::new((Join-Path $store 'model\checkpoint-288.index'), [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::None)
    $Held.Add($lock)
    $outcome = & {
        $ErrorActionPreference = 'Continue'
        try { $null = Invoke-Recovery $proj 'arm_d3' $cfg $queue -ProcessAlive $false 2>$null; 'returned' } catch { 'threw' }
    }
    $lock.Dispose()
    if (-not ($outcome -eq 'threw')) { throw "a failed quarantine move did not stop Invoke-Recovery (got '$outcome')" }

    # -----------------------------------------------------------------------------------------
    # Test-TrainingComplete and Get-RunStage
    # -----------------------------------------------------------------------------------------
    $proj = New-Project
    $store = New-CompleteStore $proj 'arm_e'
    $tc = Test-TrainingComplete $proj 'arm_e' $cfg -ProcessAlive $false
    if (-not $tc.Ok) { throw "complete training judged incomplete: $($tc.Reason)" }
    $tc = Test-TrainingComplete $proj 'arm_e' $cfg -ProcessAlive $true
    if ($tc.Ok -or $tc.Reason -notlike '*alive*') { throw 'a live process of the run should make training incomplete' }
    $tc = Test-TrainingComplete $proj 'arm_e' $cfg    # real process check: nothing runs this run name
    if (-not ($tc.Ok -eq (-not (Test-RunProcessAlive 'arm_e')))) { throw 'Test-TrainingComplete default process check disagrees with Test-RunProcessAlive' }
    if (-not ((Get-RunStage $proj 'arm_e' $cfg -ProcessAlive $false) -eq 'EVALUATE')) { throw 'complete training without EVAL_OK should be EVALUATE' }
    Put (Join-Path $store 'eva_data\EVAL_OK') ''
    if (-not ((Get-RunStage $proj 'arm_e' $cfg -ProcessAlive $false) -eq 'DONE')) { throw 'complete training with EVAL_OK should be DONE' }
    Put (Join-Path $store 'DONE') ''
    if (-not ((Get-RunStage $proj 'arm_e' $cfg -ProcessAlive $false) -eq 'DONE')) { throw 'a DONE marker should give DONE' }
    Put (Join-Path $store 'FAILED') 'STUCK'
    if (-not ((Get-RunStage $proj 'arm_e' $cfg -ProcessAlive $false) -eq 'FAILED')) { throw 'a FAILED marker should give FAILED' }
    if (-not ((Get-RunStage $proj 'never_launched' $cfg -ProcessAlive $false) -eq 'TRAIN')) { throw 'a run without a store should be TRAIN' }

    $bad = @{
        'exit 1'          = { param($s) Put (Join-Path $s 'logs\launch_r1.exit') "EXIT=1 at Fri 11/13/2026 14:00:00.00 `r`n" }
        'no exit file'    = { param($s) Remove-Item -LiteralPath (Join-Path $s 'logs\launch_r1.exit') }
        'complete 144'    = { param($s) Put (Join-Path $s 'logs\launch_r1.log') (Get-Lines @('>>> TRAINING COMPLETE step 144')) }
        'newer launch'    = { param($s) Put (Join-Path $s 'logs\launch_r2.log') (Get-Lines @('>>> Starting trainer.run()')) }
        'no meta'         = { param($s) Remove-Item -LiteralPath (Join-Path $s 'model\checkpoint-288.meta') }
        'one row'         = { param($s) Put (Join-Path $s 'train_reward.csv') (Get-TrainCsv @('144,12,-1234.5,12.1')) }
        'NaN reward'      = { param($s) Put (Join-Path $s 'train_reward.csv') (Get-TrainCsv @('144,12,-1234.5,12.1', '288,13,,11.4')) }
        'inf wait'        = { param($s) Put (Join-Path $s 'train_reward.csv') (Get-TrainCsv @('144,12,-1234.5,12.1', '288,13,-1100.2,inf')) }
        'repeated seed'   = { param($s) Put (Join-Path $s 'train_reward.csv') (Get-TrainCsv @('144,12,-1234.5,12.1', '288,12,-1100.2,11.4')) }
        'agent nan'       = { param($s) Put (Join-Path $s 'agent_reward.csv') ((Get-AgentCsv 1) + (Get-AgentCsv 1 'nan')) }
        'agent one row'   = { param($s) Put (Join-Path $s 'agent_reward.csv') (Get-AgentCsv 1) }
    }
    $k = 0
    foreach ($case in $bad.Keys) {
        $k++
        $n = 'arm_f' + $k
        $s = New-CompleteStore $proj $n
        & $bad[$case] $s
        $tc = Test-TrainingComplete $proj $n $cfg -ProcessAlive $false
        if ($tc.Ok) { throw "Test-TrainingComplete accepted the '$case' store" }
        if (-not ((Get-RunStage $proj $n $cfg -ProcessAlive $false) -eq 'TRAIN')) { throw "Get-RunStage of the '$case' store should be TRAIN" }
    }

    # -----------------------------------------------------------------------------------------
    # Exit files and interruptions
    # -----------------------------------------------------------------------------------------
    $proj = New-Project
    $store = New-Store $proj 'arm_g'
    $launch = (Get-Date).AddHours(-2)
    Put (Join-Path $store 'logs\launch_r1.exit') "EXIT=1073807364 at Fri 11/13/2026 14:00:00.00 `r`n"
    Put (Join-Path $store 'logs\launch_r2.exit') "EXIT=-1073741819 at Fri 11/13/2026 14:00:00.00 `r`n"
    $o = Get-LaunchOutcome $proj 'arm_g' 1 $launch -BootTime $launch.AddDays(-1)
    if (-not ($o.Interrupted -and $o.ExitCode -eq 1073807364)) { throw 'EXIT=1073807364 (0x40010004) should mark an interruption' }
    $o = Get-LaunchOutcome $proj 'arm_g' 2 $launch -BootTime $launch.AddDays(-1)
    if (-not ((-not $o.Interrupted) -and $o.ExitCode -eq -1073741819 -and $o.ExitCode -is [long])) { throw 'a negative NTSTATUS exit should be a death, read as Int64' }
    $o = Get-LaunchOutcome $proj 'arm_g' 3 $launch -BootTime $launch.AddMinutes(30)
    if (-not ($o.Interrupted -and $null -eq $o.ExitCode)) { throw 'a missing exit file with a boot after the launch should be an interruption' }
    $o = Get-LaunchOutcome $proj 'arm_g' 4 $launch -BootTime $launch.AddMinutes(-30)
    if ($o.Interrupted) { throw 'a missing exit file without a later boot should be a death' }
    foreach ($i in 1..10) { Put (Join-Path $store "logs\launch_r$i.log") 'x' }
    if (-not ((Get-NextLaunchNumber $proj 'arm_g' 'launch') -eq 11)) { throw 'Get-NextLaunchNumber should sort launch numbers numerically' }
    if (-not ((Get-NextLaunchNumber $proj 'arm_g' 'eval') -eq 1)) { throw 'Get-NextLaunchNumber eval should start at 1' }

    # -----------------------------------------------------------------------------------------
    # Process command lines of a run
    # -----------------------------------------------------------------------------------------
    $py = '"C:\Program Files\Python311\python.exe" -X faulthandler main.py --run arm_h --option train'
    if (-not (Test-RunCommandLine 'arm_h' 'python.exe' $py)) { throw 'the launch command line of arm_h was not matched' }
    if (Test-RunCommandLine 'arm' 'python.exe' $py) { throw 'run name arm matched arm_h' }
    if (Test-RunCommandLine 'arm_h2' 'python.exe' $py) { throw 'run name arm_h2 matched arm_h' }
    if (-not (Test-RunCommandLine 'arm_h' 'python.exe' 'python main.py --option evaluate --run arm_h')) { throw '--option before --run was not matched' }
    if (-not (Test-RunCommandLine 'arm_h' 'python.exe' 'python main.py --run=arm_h')) { throw '--run=<name> was not matched' }
    if (Test-RunCommandLine 'arm_h' 'python.exe' 'python tools/gen_traffic.py --run arm_h') { throw 'a python without main.py matched' }
    if (Test-RunCommandLine 'arm_h' 'python.exe' 'python notmain.py --run arm_h') { throw 'notmain.py matched as main.py' }
    if (Test-RunCommandLine 'arm_h' 'python.exe' $null) { throw 'a null command line matched' }
    $sumo = 'sumo -c Malta/data/exp_0.sumocfg --tripinfo-output Malta/runs/arm_h/eva_data/f1.00/Malta_signal_route_trip.xml --remote-port 27520'
    if (-not (Test-RunCommandLine 'arm_h' 'sumo.exe' $sumo)) { throw 'a SUMO writing into the store was not matched' }
    if (Test-RunCommandLine 'arm_i' 'sumo.exe' $sumo) { throw 'a SUMO of another store matched' }
    if (-not (Test-RunCommandLine 'arm_i' 'sumo.exe' 'sumo -c Malta/data/exp_0.sumocfg --remote-port 27500')) { throw 'a SUMO on the queue port 27500 was not matched' }
    if (-not ((Test-RunProcessAlive ('qlib_' + [guid]::NewGuid().ToString('N'))) -is [bool])) { throw 'Test-RunProcessAlive should return a bool' }

    # -----------------------------------------------------------------------------------------
    # Test-EvalAttempt
    # -----------------------------------------------------------------------------------------
    $good2 = @((Get-Block '1.00' 'p1.2' '9762' '9762'), (Get-Block '1.10' 'p1.091' '10738' '10738'))
    $c = New-EvalCase $good2
    $e = Invoke-Eval $c
    if (-not $e.Ok) { throw "a good evaluation attempt failed: $($e.Reason)" }
    if (-not ($e.FailedLevels.Count -eq 0)) { throw 'a good evaluation attempt has failed levels' }

    $c = New-EvalCase @((Get-Block '1.00' 'p1.2' '9762' '9762'), (Get-Block '1.10' 'p1.091' '10000' '10738'))
    $e = Invoke-Eval $c
    if ($e.Ok -or -not (($e.FailedLevels -join ',') -eq '1.10')) { throw "corridor_total 10000 vs 10738: expected FailedLevels 1.10, got '$($e.FailedLevels -join ',')' ($($e.Reason))" }

    $c = New-EvalCase @((Get-Block '1.00' 'p1.0' '9762' '9762'), (Get-Block '1.10' 'p1.091' '10738' '10738'))
    $e = Invoke-Eval $c
    if (-not (($e.FailedLevels -join ',') -eq '1.00')) { throw "wrong bg_tag at f1.00: got '$($e.FailedLevels -join ',')'" }

    $c = New-EvalCase @((Get-Block '1.00' 'p1.2' '9762' '9762' @(0)), (Get-Block '1.10' 'p1.091' '10738' '10738'))
    $e = Invoke-Eval $c
    if (-not (($e.FailedLevels -join ',') -eq '1.00')) { throw "f1.00 without test 1: got '$($e.FailedLevels -join ',')'" }

    $c = New-EvalCase @((Get-Block '1.00' 'p1.2' '9762' '9762' @(0, 0)), (Get-Block '1.10' 'p1.091' '10738' '10738'))
    $e = Invoke-Eval $c
    if (-not (($e.FailedLevels -join ',') -eq '1.00')) { throw "f1.00 with test 0 twice: got '$($e.FailedLevels -join ',')'" }

    $c = New-EvalCase $good2
    Put (Join-Path $c.Store 'eva_data\f1.10\edgedata_20000.xml') "<meandata>`r`n    <interval begin=`"0.00`" end=`"720.00`">`r`n"
    $e = Invoke-Eval $c
    if (-not (($e.FailedLevels -join ',') -eq '1.10')) { throw "unclosed edgedata XML at f1.10: got '$($e.FailedLevels -join ',')'" }

    $c = New-EvalCase $good2
    Remove-Item -LiteralPath (Join-Path $c.Store 'eva_data\f1.10\summary_10000.xml')
    $e = Invoke-Eval $c
    if (-not (($e.FailedLevels -join ',') -eq '1.10')) { throw "missing summary XML at f1.10: got '$($e.FailedLevels -join ',')'" }

    $c = New-EvalCase $good2
    (Get-Item -LiteralPath (Join-Path $c.Store 'eva_data\f1.00\summary_20000.xml')).LastWriteTime = $c.Launch.AddMinutes(-5)
    $e = Invoke-Eval $c
    if (-not (($e.FailedLevels -join ',') -eq '1.00')) { throw "summary XML older than the launch at f1.00: got '$($e.FailedLevels -join ',')'" }

    $c = New-EvalCase $good2
    Put (Join-Path $c.Store 'eva_data\f1.00\Malta_signal_route_traffic.csv') ",episode,time_sec`r`n0,1,5`r`n1,1,10`r`n"
    $e = Invoke-Eval $c
    if (-not (($e.FailedLevels -join ',') -eq '1.00')) { throw "traffic CSV covering only episode 1: got '$($e.FailedLevels -join ',')'" }

    $c = New-EvalCase $good2 -Trips @{ '1.00' = 5; '1.10' = 4 }
    $e = Invoke-Eval $c
    if (-not (($e.FailedLevels -join ',') -eq '1.10')) { throw "fewer trips at f1.10 than f1.00: got '$($e.FailedLevels -join ',')'" }

    # A crash in f1.10: exit 1, a traceback, no DONE and no COMPLETE line. f1.00 still passes.
    $crash = @((Get-Block '1.00' 'p1.2' '9762' '9762'),
               (Get-Block '1.10' 'p1.091' '10738' '10738' @(0) $false @('Traceback (most recent call last):', 'RuntimeError: boom')))
    $c = New-EvalCase $crash -Complete '' -Exit "EXIT=1 at Fri 11/13/2026 14:00:00.00 `r`n"
    $e = Invoke-Eval $c
    if ($e.Ok -or -not (($e.FailedLevels -join ',') -eq '1.10')) { throw "crash in f1.10: expected FailedLevels 1.10, got '$($e.FailedLevels -join ',')'" }

    # A traceback after the last DONE fails the attempt; with no failed level, every level is retried.
    $c = New-EvalCase $good2 -Exit "EXIT=0 at Fri 11/13/2026 14:00:00.00 `r`n"
    Add-Content -LiteralPath $c.Log -Value 'Exception in thread Thread-3:'
    $e = Invoke-Eval $c
    if ($e.Ok -or -not (($e.FailedLevels -join ',') -eq '1.00,1.10')) { throw "late traceback: expected every level failed, got '$($e.FailedLevels -join ',')'" }

    $c = New-EvalCase $good2 -Exit $null
    $e = Invoke-Eval $c
    if ($e.Ok) { throw 'an attempt without an exit file passed' }

    $c = New-EvalCase $good2 -Restore $false
    $e = Invoke-Eval $c
    if ($e.Ok -or -not (($e.FailedLevels -join ',') -eq '1.00,1.10')) { throw "no Restoring line: expected every level failed, got '$($e.FailedLevels -join ',')'" }

    $c = New-EvalCase $good2 -RestoreRun 'other'
    $e = Invoke-Eval $c
    if ($e.Ok) { throw "another run's Restoring line was accepted" }

    $c = New-EvalCase $good2
    Add-Content -LiteralPath $c.Log -Value '2026-11-13 12:09:00,000 [ERROR] Can not find old checkpoint for Malta/runs/x/model/'
    $e = Invoke-Eval $c
    if ($e.Ok) { throw "'Can not find old checkpoint' was accepted" }

    # Retry of f1.10 only (--levels 1.10): f1.00 is from the previous attempt (older files).
    $c = New-EvalCase @((Get-Block '1.10' 'p1.091' '10738' '10738')) -Complete '>>> EVALUATE COMPLETE levels=1'
    Get-ChildItem -LiteralPath (Join-Path $c.Store 'eva_data\f1.00') | ForEach-Object { $_.LastWriteTime = $c.Launch.AddHours(-3) }
    $e = Invoke-Eval $c @(1.10)
    if (-not $e.Ok) { throw "a good retry of f1.10 failed: $($e.Reason)" }
    $c = New-EvalCase @((Get-Block '1.10' 'p1.091' '10738' '10738')) -Complete '>>> EVALUATE COMPLETE levels=1' -Trips @{ '1.00' = 5; '1.10' = 4 }
    $e = Invoke-Eval $c @(1.10)
    if ($e.Ok -or -not (($e.FailedLevels -join ',') -eq '1.10')) { throw "retry of f1.10 with fewer trips than the earlier f1.00: got '$($e.FailedLevels -join ',')'" }

    # -----------------------------------------------------------------------------------------
    # state.json and the launch counters
    # -----------------------------------------------------------------------------------------
    $proj = New-Project
    $queue = Join-Path $proj 'Malta\runs\_queue'
    $state = Read-QueueState $queue
    if (-not ($state['runs'].Count -eq 0)) { throw 'a fresh state should have no runs' }
    if (-not ((Update-LaunchCounters $state 'arm_x' 144 144 $false) -eq 'OK')) { throw 'first no-progress launch should be OK' }
    if (-not ((Update-LaunchCounters $state 'arm_x' 144 144 $false) -eq 'OK')) { throw 'second no-progress launch should be OK' }
    if (-not ((Update-LaunchCounters $state 'arm_x' 144 144 $false) -eq 'STUCK')) { throw 'three consecutive no-progress launches should be STUCK' }

    $rs = Get-RunCounters $state 'arm_y'
    $rs['current_launch'] = @{ kind = 'train'; n = 1; good_before = -1; started = (Get-Date).ToString('o') }
    [void](Update-LaunchCounters $state 'arm_y' -1 -1 $false)
    if ($null -ne $rs['current_launch']) { throw 'Update-LaunchCounters should clear the finished train launch' }
    [void](Update-LaunchCounters $state 'arm_y' -1 -1 $false)
    if (-not ((Update-LaunchCounters $state 'arm_y' -1 144 $false) -eq 'OK')) { throw 'a launch with progress should be OK' }
    if (-not ($rs['no_progress'] -eq 0)) { throw 'progress should reset the no-progress count' }
    [void](Update-LaunchCounters $state 'arm_y' 144 144 $false)
    [void](Update-LaunchCounters $state 'arm_y' 144 144 $false)
    if (-not ((Update-LaunchCounters $state 'arm_y' 144 144 $true) -eq 'OK')) { throw 'an interrupted launch should not count toward the 3' }
    if (-not ($rs['no_progress'] -eq 2)) { throw "an interruption changed the no-progress count to $($rs['no_progress'])" }
    if (-not ((Update-LaunchCounters $state 'arm_y' 144 144 $false) -eq 'STUCK')) { throw 'the third real no-progress launch should be STUCK' }

    for ($i = 1; $i -le 11; $i++) {
        if (-not ((Update-LaunchCounters $state 'arm_z' 144 144 $true) -eq 'OK')) { throw "interrupted launch $i of 11 should be OK" }
    }
    $rz = Get-RunCounters $state 'arm_z'
    if (-not ($rz['no_progress'] -eq 0 -and $rz['interrupted'] -eq 11 -and $rz['launches'] -eq 11)) { throw 'interrupted launches should count only toward the total' }
    if (-not ((Update-LaunchCounters $state 'arm_z' 144 144 $true) -eq 'STUCK')) { throw 'the 12th launch in total should be STUCK' }
    $rw = Get-RunCounters $state 'arm_w'
    for ($i = 1; $i -le 11; $i++) { [void](Update-LaunchCounters $state 'arm_w' ($i * 144) (($i + 1) * 144) $false) }
    if (-not ((Update-LaunchCounters $state 'arm_w' 1728 1872 $false) -eq 'STUCK')) { throw '12 launches with progress should still be STUCK' }
    $rv = Get-RunCounters $state 'arm_v'
    [void](Update-LaunchCounters $state 'arm_v' -1 144 $false)
    $rv['eval_attempts'] = 1

    Save-QueueState $queue $state
    $sp = Join-Path $queue 'state.json'
    if (-not ([IO.File]::Exists($sp) -and -not [IO.File]::Exists($sp + '.tmp'))) { throw 'Save-QueueState should leave state.json and no temp file' }
    if (-not (Test-NoBom $sp)) { throw 'state.json has a BOM' }
    $back = Read-QueueState $queue
    if (-not ($back['runs']['arm_x']['launches'] -eq 3 -and $back['runs']['arm_x']['no_progress'] -eq 3)) { throw 'state.json did not round-trip arm_x' }
    if (-not (@($back['runs']['arm_x']['good_at_launch']).Count -eq 3)) { throw 'good_at_launch did not round-trip' }
    $one = $back['runs']['arm_v']['good_at_launch']
    if (-not ($one -is [array] -and $one.Count -eq 1 -and $one[0] -eq -1)) { throw 'a one-element good_at_launch did not stay an array' }
    if (-not ($back['runs']['arm_v']['eval_attempts'] -eq 1)) { throw 'eval_attempts did not round-trip' }
    if (-not ((Update-LaunchCounters $back 'arm_v' 144 288 $false) -eq 'OK' -and $back['runs']['arm_v']['launches'] -eq 2)) { throw 'counters do not continue after a reload' }
    Save-QueueState $queue $back
    if (-not ((Read-QueueState $queue)['runs']['arm_v']['launches'] -eq 2)) { throw 'second save did not replace state.json' }
    # A crash between removing state.json and moving the temp file over it.
    Move-Item -LiteralPath $sp -Destination ($sp + '.tmp')
    if (-not ((Read-QueueState $queue)['runs']['arm_v']['launches'] -eq 2)) { throw 'Read-QueueState should fall back to state.json.tmp' }
    Move-Item -LiteralPath ($sp + '.tmp') -Destination $sp
    Put ($sp + '.tmp') '{ "runs": { "arm_v": '
    if (-not ((Read-QueueState $queue)['runs']['arm_v']['launches'] -eq 2)) { throw 'a half-written temp file should not hide state.json' }
    Put $sp '{ "runs": '
    Remove-Item -LiteralPath ($sp + '.tmp')
    $threw = $false
    try { [void](Read-QueueState $queue) } catch { $threw = $true }
    if (-not $threw) { throw 'an unreadable state.json should throw, not reset the counters' }

    Write-Output 'ALL QUEUE_LIB TESTS PASSED'
    $exitCode = 0
} catch {
    Write-Output ('TEST FAILED: ' + $_.Exception.Message)
    Write-Output $_.ScriptStackTrace
    $exitCode = 1
} finally {
    foreach ($h in $Held) { try { $h.Dispose() } catch { } }
    if ([IO.Directory]::Exists($TestRoot)) { Remove-Item -LiteralPath $TestRoot -Recurse -Force -ErrorAction SilentlyContinue }
    if ([IO.Directory]::Exists($TestRoot)) { Write-Output "warning: could not remove $TestRoot" }
}
exit $exitCode
