<#
tools/queue_lib.ps1 - the decision library of the RQ2 training queue runner (tools/run_queue.ps1).

Spec: docs/superpowers/specs/2026-11-08-rq2-training-queue-design.md, sections 3, 8 and 9.
Tests: tests/test_queue_lib.ps1 (powershell -NoProfile -ExecutionPolicy Bypass -File tests/test_queue_lib.ps1).
Windows PowerShell 5.1. Dot-source it:  . (Join-Path $PSScriptRoot 'queue_lib.ps1')

These functions decide; they do not launch processes and never delete a file. The only side
effects are the writes a caller asks for (Write-Utf8NoBom, Add-Utf8NoBomLine, Save-QueueState)
and Invoke-Recovery, which only moves files into the queue folder and rewrites the three run
CSVs after copying the originals there.

Paths
  Functions take the absolute project root and the run name. Files are reached through absolute
  paths built with Join-Path ([IO.File] resolves relative paths against the process directory,
  not the PowerShell location). Log lines are matched against the literal store string the
  Python side writes: 'Malta/runs/<name>/' (relative, forward slashes, trailing '/').
  Logs are read with FileShare.ReadWrite, because python may still be writing them.
  Numbers are parsed and formatted with the invariant culture (level folders are f1.10, never f1,10).

Functions
  Write-Utf8NoBom $Path $Text                 Add-Utf8NoBomLine $Path $Line
  Read-SharedText $Path -> string              Read-ExitCode $ExitFile -> Int64 or $null
  Read-RunConfig $IniPath -> { IniPath; TotalStep; T; Seed; TestSeeds; EvalLevels; Scenario }
  Get-QueueRuns $QueueFile -> run names (blank and # lines skipped; wrap the call in @())
  Get-StoreDir / Get-StoreRel / Get-LevelDir / Format-Level / Format-BgTag
  Get-LaunchFiles $ProjectRoot $RunName [launch|eval] -> { N; Log; Exit } sorted by N
  Get-NextLaunchNumber $ProjectRoot $RunName [launch|eval] -> int
  Get-LaunchOutcome $ProjectRoot $RunName $N $LaunchTime [-BootTime] [-Kind] -> { N; ExitFile; ExitCode; Interrupted }
  Get-GoodCheckpoint $ProjectRoot $RunName -> int (-1 if none)
  Test-RunCommandLine $RunName $ProcessName $CommandLine -> bool
  Test-RunProcessAlive $RunName -> bool
  Test-TrainingComplete $ProjectRoot $RunName $Cfg [-ProcessAlive] -> { Ok; Reason }
  Get-RunStage $ProjectRoot $RunName $Cfg [-ProcessAlive] -> 'DONE' | 'FAILED' | 'EVALUATE' | 'TRAIN'
  Invoke-Recovery $ProjectRoot $RunName $Cfg $QueueDir [-ProcessAlive] [-Log {param($m)}]
                                              -> 'RESUME' | 'SCRATCH' | 'NEEDS_REVIEW:<reason>'
  Test-EvalAttempt $ProjectRoot $RunName $Cfg $LogPath $LaunchTime [-Levels] -> { Ok; FailedLevels; Reason }
  Read-QueueState $QueueDir -> state          Save-QueueState $QueueDir $State
  Get-RunCounters $State $Run -> the run's entry (created with defaults if missing)
  Update-LaunchCounters $State $Run $GoodBefore $GoodAfter $Interrupted -> 'OK' | 'STUCK'

-ProcessAlive: $null (the default) asks Test-RunProcessAlive; $true or $false skips the query.

state.json (<QueueDir>\state.json, UTF-8 without BOM; Save-QueueState writes state.json.tmp
and then Move-Item -Force onto state.json; Read-QueueState falls back to state.json.tmp when
state.json is missing or unreadable). Read-QueueState returns nested ordered hashtables; JSON
arrays stay arrays. Schema version 1:

  {
    "version": 1,
    "updated": "<ISO 8601>",            set by Save-QueueState
    "runs": {
      "<run name>": {
        "launches":       0,     training launches that have ended, interrupted ones included.
                                 Update-LaunchCounters returns STUCK when it reaches 12.
        "no_progress":    0,     consecutive ended launches whose good checkpoint did not rise.
                                 Interrupted launches leave it unchanged; a launch with progress
                                 resets it to 0. STUCK when it reaches 3.
        "interrupted":    0,     launches ended by a reboot or logoff: exit code 0x40010004
                                 (1073807364), or no exit file and a boot after the launch.
        "good_at_launch": [],    the good checkpoint step at the start of each ended launch (-1 = none)
        "eval_attempts":  0,     evaluation attempts started; the runner increments it
        "current_launch": null   the launch in flight, set by the runner so a restarted runner can
                                 adopt it: { "kind": "train"|"evaluate", "n": <N of launch_rN /
                                 eval_rN>, "good_before": <step>, "started": "<ISO 8601>" }.
                                 Update-LaunchCounters clears a "train" entry.
      }
    }
  }
  The runner may add keys of its own (for example a smoke-test peak commit); they are kept.

Call Update-LaunchCounters once for each training launch that ended without completing
training (a death or an interruption), with the good checkpoint at its start and after
Invoke-Recovery. Call Get-RunStage and Invoke-Recovery only when no process of the run is alive
(the runner adopts a live one first): a live evaluation python makes training "incomplete",
and Invoke-Recovery throws rather than move files under a live process.
#>

# ---------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------
$QueueInterruptExitCode = [long]1073807364      # 0x40010004: the process ended at logoff/shutdown
$QueueMaxNoProgress = 3
$QueueMaxLaunches = 12
$QueueTrainPort = 27500                          # TraCI port 0, reserved for the queue
$QueueCostHeader = 'episode,step,seed,wall_s,rollout_s,update_s,save_s,peak_private_gb,peak_working_set_gb,ckpt_data_mb,ckpt_meta_mb,lost'
$QueueRestoreErrorPattern = '(?i)DataLossError|Unable to open table file|corrupted'
$QueueUtf8NoBom = New-Object System.Text.UTF8Encoding $false
$QueueInvariant = [System.Globalization.CultureInfo]::InvariantCulture
$QueueFloatStyle = [System.Globalization.NumberStyles]::Float

# ---------------------------------------------------------------------------------------------
# Paths and text files
# ---------------------------------------------------------------------------------------------
function Assert-AbsolutePath([string]$Path, [string]$What) {
    if ([string]::IsNullOrWhiteSpace($Path) -or -not ($Path -match '^[A-Za-z]:[\\/]' -or $Path -match '^\\\\')) {
        throw "$What must be an absolute path, got '$Path'"
    }
}

function Get-StoreRel([string]$RunName) {
    # The store string in every Python log line: relative, forward slashes, trailing '/'.
    'Malta/runs/' + $RunName + '/'
}

function Get-StoreDir([string]$ProjectRoot, [string]$RunName) {
    Assert-AbsolutePath $ProjectRoot 'ProjectRoot'
    if ($RunName -notmatch '^[A-Za-z0-9_.-]+$' -or $RunName -match '^\.+$') { throw "invalid run name '$RunName'" }
    Join-Path $ProjectRoot ('Malta\runs\' + $RunName)
}

function Format-Level([double]$Level) {
    # '%.2f' % level, as main.py names the level folder: 1.1 -> '1.10' (invariant culture).
    [string]::Format($QueueInvariant, '{0:0.00}', $Level)
}

function Get-LevelDir([string]$ProjectRoot, [string]$RunName, [double]$Level) {
    Join-Path (Join-Path (Get-StoreDir $ProjectRoot $RunName) 'eva_data') ('f' + (Format-Level $Level))
}

function Format-PyFloat([double]$Value) {
    # Python's str(float) for the short values used here: 1.0 -> '1.0', 1.132 -> '1.132'.
    $s = $Value.ToString('R', $QueueInvariant)
    if ($s -match '^-?\d+$') { $s += '.0' }
    $s
}

function Format-BgTag([double]$Level) {
    # 'p' + str(round(1.2 / level, 3)): p1.2, p1.132, p1.091, p1.062, p1.0, p0.6.
    'p' + (Format-PyFloat ([Math]::Round(1.2 / $Level, 3)))
}

function Write-Utf8NoBom {
    param([Parameter(Mandatory = $true)][string]$Path, [AllowEmptyString()][string]$Text = '')
    $ErrorActionPreference = 'Stop'
    Assert-AbsolutePath $Path 'Write-Utf8NoBom path'
    $dir = [IO.Path]::GetDirectoryName($Path)
    if ($dir -and -not [IO.Directory]::Exists($dir)) { [void][IO.Directory]::CreateDirectory($dir) }
    [IO.File]::WriteAllText($Path, $Text, $QueueUtf8NoBom)
}

function Add-Utf8NoBomLine {
    param([Parameter(Mandatory = $true)][string]$Path, [AllowEmptyString()][string]$Line = '')
    $ErrorActionPreference = 'Stop'
    Assert-AbsolutePath $Path 'Add-Utf8NoBomLine path'
    $dir = [IO.Path]::GetDirectoryName($Path)
    if ($dir -and -not [IO.Directory]::Exists($dir)) { [void][IO.Directory]::CreateDirectory($dir) }
    [IO.File]::AppendAllText($Path, $Line + "`r`n", $QueueUtf8NoBom)
}

function Open-SharedRead([string]$Path) {
    # Read access that tolerates a writer (python's log handle) and a rename by it.
    [IO.FileStream]::new($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]'ReadWrite, Delete')
}

function Read-SharedText {
    param([Parameter(Mandatory = $true)][string]$Path)
    $ErrorActionPreference = 'Stop'
    Assert-AbsolutePath $Path 'Read-SharedText path'
    $fs = Open-SharedRead $Path
    try {
        $sr = [IO.StreamReader]::new($fs, $QueueUtf8NoBom, $true)
        try { return $sr.ReadToEnd() } finally { $sr.Dispose() }
    } finally { $fs.Dispose() }
}

function Get-FileTail([string]$Path, [int]$Bytes = 1024) {
    $fs = Open-SharedRead $Path
    try {
        $len = $fs.Length
        $start = [Math]::Max([long]0, $len - $Bytes)
        [void]$fs.Seek($start, [IO.SeekOrigin]::Begin)
        $buf = New-Object byte[] ([int]($len - $start))
        $read = 0
        while ($read -lt $buf.Length) {
            $k = $fs.Read($buf, $read, $buf.Length - $read)
            if ($k -le 0) { break }
            $read += $k
        }
        return $QueueUtf8NoBom.GetString($buf, 0, $read)
    } finally { $fs.Dispose() }
}

function Test-EndsWithNewline([string]$Path) {
    $fs = Open-SharedRead $Path
    try {
        if ($fs.Length -eq 0) { return $false }
        [void]$fs.Seek(-1, [IO.SeekOrigin]::End)
        return ($fs.ReadByte() -eq 10)
    } finally { $fs.Dispose() }
}

function Test-XmlClosed([string]$Path, [string]$Tag) {
    (Get-FileTail $Path).TrimEnd().EndsWith('</' + $Tag + '>')
}

function Move-ToFolder([string]$Path, [string]$Folder) {
    if (-not [IO.Directory]::Exists($Folder)) { [void][IO.Directory]::CreateDirectory($Folder) }
    Move-Item -LiteralPath $Path -Destination (Join-Path $Folder ([IO.Path]::GetFileName($Path))) -ErrorAction Stop
}

function New-QueueStamp([string]$QueueDir) {
    # yyyyMMdd_HHmmss, with _2, _3 ... when a folder of that stamp already exists.
    $base = Get-Date -Format 'yyyyMMdd_HHmmss'
    $stamp = $base
    $k = 1
    while ([IO.Directory]::Exists((Join-Path $QueueDir ('quarantine_' + $stamp))) -or
           [IO.Directory]::Exists((Join-Path $QueueDir ('recovery_' + $stamp)))) {
        $k++
        $stamp = $base + '_' + $k
    }
    $stamp
}

# ---------------------------------------------------------------------------------------------
# Numbers and CSV
# ---------------------------------------------------------------------------------------------
function Test-FiniteNumber([string]$Text) {
    $d = 0.0
    if ($null -eq $Text) { return $false }
    if (-not [double]::TryParse($Text.Trim(), $QueueFloatStyle, $QueueInvariant, [ref]$d)) { return $false }
    -not ([double]::IsNaN($d) -or [double]::IsInfinity($d))
}

function Test-NumberToken([string]$Text) {
    # A value numpy or pandas may write: a number, or nan / inf (finite or not).
    if ($null -eq $Text) { return $false }
    $t = $Text.Trim()
    if ($t -match '^[+-]?(nan|inf|infinity)$') { return $true }
    $d = 0.0
    [double]::TryParse($t, $QueueFloatStyle, $QueueInvariant, [ref]$d)
}

function ConvertTo-WholeNumber([string]$Text) {
    # '144' or '144.0' -> [long] 144; anything else -> $null.
    $d = 0.0
    if ($null -eq $Text -or -not [double]::TryParse($Text.Trim(), $QueueFloatStyle, $QueueInvariant, [ref]$d)) { return $null }
    if ([double]::IsNaN($d) -or [double]::IsInfinity($d) -or $d -ne [Math]::Floor($d)) { return $null }
    [long]$d
}

function ConvertFrom-CsvLine([string]$Line) {
    # Split one CSV line; fields in double quotes may hold commas ("" is a literal quote).
    if ($Line.IndexOf([char]34) -lt 0) {
        $plain = $Line.Split(',')
        return , $plain
    }
    $fields = New-Object System.Collections.Generic.List[string]
    $sb = New-Object System.Text.StringBuilder
    $inQuotes = $false
    for ($i = 0; $i -lt $Line.Length; $i++) {
        $c = $Line[$i]
        if ($inQuotes) {
            if ($c -eq [char]34) {
                if ($i + 1 -lt $Line.Length -and $Line[$i + 1] -eq [char]34) { [void]$sb.Append([char]34); $i++ }
                else { $inQuotes = $false }
            } else { [void]$sb.Append($c) }
        } elseif ($c -eq [char]34) { $inQuotes = $true }
        elseif ($c -eq ',') { $fields.Add($sb.ToString()); [void]$sb.Clear() }
        else { [void]$sb.Append($c) }
    }
    $fields.Add($sb.ToString())
    $arr = $fields.ToArray()
    return , $arr
}

function Read-CsvFile([string]$Path, [switch]$NoHeader) {
    # Rows keep their original text (Line, with its line end) for trimming, and their Fields.
    # Blank lines are skipped. Ok is false when a row's field count differs from the header's
    # (or, with -NoHeader, from the first row's).
    $res = [pscustomobject]@{
        Exists = $false; Ok = $true; Error = $null; Text = $null; NewLine = "`r`n"
        HeaderLine = $null; Header = $null; Rows = (New-Object System.Collections.Generic.List[object])
    }
    if (-not [IO.File]::Exists($Path)) { return $res }
    $res.Exists = $true
    $text = Read-SharedText $Path
    $res.Text = $text
    if ($text.IndexOf("`r`n") -lt 0 -and $text.IndexOf("`n") -ge 0) { $res.NewLine = "`n" }
    $width = -1
    $lineNo = 0
    foreach ($part in [regex]::Split($text, '(?<=\n)')) {
        if ($part.Length -eq 0) { continue }
        $lineNo++
        $body = $part.TrimEnd([char[]]"`r`n")
        if ($body.Trim().Length -eq 0) { continue }
        $fields = ConvertFrom-CsvLine $body
        if (-not $NoHeader -and $null -eq $res.HeaderLine) {
            $res.HeaderLine = $part
            $res.Header = $fields
            $width = $fields.Length
            continue
        }
        if ($width -lt 0) { $width = $fields.Length }
        if ($fields.Length -ne $width) {
            $res.Ok = $false
            $res.Error = "line $lineNo has $($fields.Length) fields, expected $width"
            return $res
        }
        $line = $part
        if (-not $line.EndsWith("`n")) { $line += $res.NewLine }
        $res.Rows.Add([pscustomobject]@{ Line = $line; Fields = $fields })
    }
    $res
}

function Get-HeaderIndex($Csv, [string]$Column) {
    if ($null -eq $Csv.Header) { return -1 }
    [array]::IndexOf([string[]]$Csv.Header, $Column)
}

function Get-ColumnWholeNumbers($Csv, [string]$Column, [switch]$Increasing) {
    # -> { Error; Values (long[]) }. Every value must be a whole number; with -Increasing,
    # strictly increasing.
    $idx = Get-HeaderIndex $Csv $Column
    if ($idx -lt 0) { return [pscustomobject]@{ Error = "no $Column column"; Values = @() } }
    $values = New-Object System.Collections.Generic.List[long]
    for ($i = 0; $i -lt $Csv.Rows.Count; $i++) {
        $raw = $Csv.Rows[$i].Fields[$idx]
        $v = ConvertTo-WholeNumber $raw
        if ($null -eq $v) { return [pscustomobject]@{ Error = "row $($i + 1) has $Column '$raw'"; Values = @() } }
        if ($Increasing -and $i -gt 0 -and $v -le $values[$i - 1]) {
            return [pscustomobject]@{ Error = "$Column does not increase at row $($i + 1)"; Values = @() }
        }
        $values.Add($v)
    }
    [pscustomobject]@{ Error = $null; Values = $values.ToArray() }
}

function Get-CsvEpisodeCounts([string]$Path) {
    # Rows per value of the 'episode' column of an evaluation CSV -> { Error; Counts; Rows }.
    $counts = @{}
    $rows = 0
    $fs = Open-SharedRead $Path
    $sr = [IO.StreamReader]::new($fs, $QueueUtf8NoBom, $true)
    try {
        $header = $sr.ReadLine()
        if ($null -eq $header) { return [pscustomobject]@{ Error = 'empty file'; Counts = $counts; Rows = 0 } }
        $idx = [array]::IndexOf([string[]](ConvertFrom-CsvLine $header), 'episode')
        if ($idx -lt 0) { return [pscustomobject]@{ Error = 'no episode column'; Counts = $counts; Rows = 0 } }
        $sep = [char[]]@(',')
        $lineNo = 1
        while ($null -ne ($line = $sr.ReadLine())) {
            $lineNo++
            if ($line.Length -eq 0) { continue }
            $parts = $line.Split($sep, $idx + 2)
            for ($k = 0; $k -le $idx -and $k -lt $parts.Length; $k++) {
                if ($parts[$k].IndexOf([char]34) -ge 0) { $parts = ConvertFrom-CsvLine $line; break }
            }
            if ($parts.Length -le $idx) { return [pscustomobject]@{ Error = "line $lineNo has no episode field"; Counts = $counts; Rows = $rows } }
            $ep = ConvertTo-WholeNumber $parts[$idx]
            if ($null -eq $ep) { return [pscustomobject]@{ Error = "line $lineNo has episode '$($parts[$idx])'"; Counts = $counts; Rows = $rows } }
            $key = [int]$ep
            if ($counts.ContainsKey($key)) { $counts[$key] = $counts[$key] + 1 } else { $counts[$key] = 1 }
            $rows++
        }
    } finally { $sr.Dispose(); $fs.Dispose() }
    [pscustomobject]@{ Error = $null; Counts = $counts; Rows = $rows }
}

# ---------------------------------------------------------------------------------------------
# Config and queue file
# ---------------------------------------------------------------------------------------------
function Read-RunConfig {
    param([Parameter(Mandatory = $true)][string]$IniPath)
    $ErrorActionPreference = 'Stop'
    Assert-AbsolutePath $IniPath 'Read-RunConfig path'
    if (-not [IO.File]::Exists($IniPath)) { throw "config $IniPath not found" }
    $sections = @{}
    $current = $null
    foreach ($raw in ((Read-SharedText $IniPath) -split "`r?`n")) {
        $line = $raw.Trim()
        if ($line.Length -eq 0 -or $line.StartsWith(';') -or $line.StartsWith('#')) { continue }
        if ($line -match '^\[(.+)\]$') {
            $current = $Matches[1].Trim()
            if (-not $sections.ContainsKey($current)) { $sections[$current] = @{} }
            continue
        }
        $eq = $line.IndexOf('=')
        if ($eq -lt 1 -or $null -eq $current) { continue }
        $sections[$current][$line.Substring(0, $eq).Trim()] = $line.Substring($eq + 1).Trim()
    }
    $get = {
        param([string]$Section, [string]$Key, [switch]$Optional)
        if ($sections.ContainsKey($Section) -and $sections[$Section].ContainsKey($Key)) { return $sections[$Section][$Key] }
        if ($Optional) { return $null }
        throw "config $IniPath has no [$Section] $Key"
    }
    $number = {
        param([string]$Section, [string]$Key)
        $raw = & $get $Section $Key
        $d = 0.0
        if (-not [double]::TryParse($raw, $QueueFloatStyle, $QueueInvariant, [ref]$d) -or [double]::IsNaN($d) -or [double]::IsInfinity($d)) {
            throw "config $IniPath [$Section] $Key = '$raw' is not a number"
        }
        $d
    }
    $totalStep = [int][Math]::Truncate((& $number 'TRAIN_CONFIG' 'total_step'))   # int(getfloat()), as main.py
    $episode = & $number 'ENV_CONFIG' 'episode_length_sec'
    $interval = & $number 'ENV_CONFIG' 'control_interval_sec'
    if ($interval -le 0) { throw "config $IniPath control_interval_sec must be > 0" }
    $seed = [int](& $number 'ENV_CONFIG' 'seed')
    $seeds = New-Object System.Collections.Generic.List[int]
    foreach ($s in ((& $get 'ENV_CONFIG' 'test_seeds') -split ',')) {
        if ($s.Trim().Length -eq 0) { continue }
        $v = ConvertTo-WholeNumber $s
        if ($null -eq $v) { throw "config $IniPath test_seeds has '$s'" }
        $seeds.Add([int]$v)
    }
    if ($seeds.Count -eq 0) { throw "config $IniPath has no test_seeds" }
    $levels = New-Object System.Collections.Generic.List[double]
    $rawLevels = & $get 'ENV_CONFIG' 'eval_levels' -Optional
    if ($null -ne $rawLevels) {
        foreach ($s in ($rawLevels -split ',')) {
            if ($s.Trim().Length -eq 0) { continue }
            $d = 0.0
            if (-not [double]::TryParse($s.Trim(), $QueueFloatStyle, $QueueInvariant, [ref]$d) -or -not ($d -gt 0) -or [double]::IsInfinity($d)) {
                throw "config $IniPath eval_levels has '$s'"
            }
            $levels.Add($d)
        }
    }
    if ($levels.Count -eq 0) { $levels.Add(1.0) }
    $scenario = & $get 'ENV_CONFIG' 'scenario' -Optional
    if (-not $scenario) { $scenario = 'Malta' }
    [pscustomobject]@{
        IniPath    = $IniPath
        TotalStep  = $totalStep
        T          = [int][Math]::Ceiling($episode / $interval)
        Seed       = $seed
        TestSeeds  = [int[]]$seeds.ToArray()
        EvalLevels = [double[]]$levels.ToArray()
        Scenario   = $scenario
    }
}

function Get-QueueRuns {
    param([Parameter(Mandatory = $true)][string]$QueueFile)
    $ErrorActionPreference = 'Stop'
    Assert-AbsolutePath $QueueFile 'Get-QueueRuns path'
    foreach ($raw in ((Read-SharedText $QueueFile) -split "`r?`n")) {
        $line = $raw.Trim()
        if ($line.Length -eq 0 -or $line.StartsWith('#')) { continue }
        $line
    }
}

# ---------------------------------------------------------------------------------------------
# Launch files, exit codes, checkpoints
# ---------------------------------------------------------------------------------------------
function Get-LaunchFiles {
    param([string]$ProjectRoot, [string]$RunName, [ValidateSet('launch', 'eval')][string]$Kind = 'launch')
    $ErrorActionPreference = 'Stop'
    # <store>logs\<kind>_rN.log / .exit, by N ascending (numerically: r10 after r9).
    $logs = Join-Path (Get-StoreDir $ProjectRoot $RunName) 'logs'
    if (-not [IO.Directory]::Exists($logs)) { return }
    $re = New-Object regex ('^' + $Kind + '_r(\d{1,9})\.(log|exit)$'), 'IgnoreCase'
    $ns = @{}
    foreach ($f in [IO.Directory]::GetFiles($logs)) {
        $m = $re.Match([IO.Path]::GetFileName($f))
        if ($m.Success) { $ns[[int]$m.Groups[1].Value] = $true }
    }
    foreach ($n in ($ns.Keys | Sort-Object)) {
        [pscustomobject]@{
            N    = $n
            Log  = Join-Path $logs ('{0}_r{1}.log' -f $Kind, $n)
            Exit = Join-Path $logs ('{0}_r{1}.exit' -f $Kind, $n)
        }
    }
}

function Get-NextLaunchNumber {
    param([string]$ProjectRoot, [string]$RunName, [ValidateSet('launch', 'eval')][string]$Kind = 'launch')
    $ErrorActionPreference = 'Stop'
    $all = @(Get-LaunchFiles $ProjectRoot $RunName $Kind)
    if ($all.Count -eq 0) { return 1 }
    $all[-1].N + 1
}

function Read-ExitCode([string]$Path) {
    # The code in an exit file 'EXIT=<code> at <date> <time>' as Int64, or $null.
    if (-not [IO.File]::Exists($Path)) { return $null }
    $m = [regex]::Match((Read-SharedText $Path), 'EXIT=(-?\d{1,19})')
    if (-not $m.Success) { return $null }
    $v = [long]0
    if (-not [long]::TryParse($m.Groups[1].Value, [Globalization.NumberStyles]::AllowLeadingSign, $QueueInvariant, [ref]$v)) { return $null }
    $v
}

function Get-LaunchOutcome {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [Parameter(Mandatory = $true)][string]$RunName,
        [Parameter(Mandatory = $true)][int]$N,
        [Parameter(Mandatory = $true)][datetime]$LaunchTime,
        [object]$BootTime = $null,
        [ValidateSet('launch', 'eval')][string]$Kind = 'launch'
    )
    $ErrorActionPreference = 'Stop'
    # Interrupted: exit code 0x40010004, or no exit code and the machine booted after the launch.
    $exitPath = Join-Path (Join-Path (Get-StoreDir $ProjectRoot $RunName) 'logs') ('{0}_r{1}.exit' -f $Kind, $N)
    $code = Read-ExitCode $exitPath
    if ($null -ne $code) {
        $interrupted = ($code -eq $QueueInterruptExitCode)
    } else {
        if ($null -eq $BootTime) { $BootTime = (Get-CimInstance -ClassName Win32_OperatingSystem).LastBootUpTime }
        $interrupted = (([datetime]$BootTime).ToUniversalTime() -gt $LaunchTime.ToUniversalTime())
    }
    [pscustomobject]@{ N = $N; ExitFile = $exitPath; ExitCode = $code; Interrupted = $interrupted }
}

function Get-SavedSteps([string]$ProjectRoot, [string]$RunName) {
    # Steps with an exact 'Checkpoint saved at step <N> -> Malta/runs/<name>/model/' line in any
    # launch_rN.log of the run. main.py writes it only after the save and the cost row.
    $pattern = '(?m)(?:^|\s)Checkpoint saved at step (\d{1,9}) -> ' + [regex]::Escape((Get-StoreRel $RunName) + 'model/') + '\r?$'
    $steps = @{}
    foreach ($l in @(Get-LaunchFiles $ProjectRoot $RunName 'launch')) {
        if (-not [IO.File]::Exists($l.Log)) { continue }
        foreach ($m in [regex]::Matches((Read-SharedText $l.Log), $pattern)) { $steps[[int]$m.Groups[1].Value] = $true }
    }
    $steps.Keys
}

function Test-CheckpointFiles([string]$ModelDir, [long]$Step) {
    $p = Join-Path $ModelDir ('checkpoint-' + $Step)
    [IO.File]::Exists($p + '.data-00000-of-00001') -and [IO.File]::Exists($p + '.index') -and [IO.File]::Exists($p + '.meta')
}

function Get-GoodCheckpoint {
    param([Parameter(Mandatory = $true)][string]$ProjectRoot, [Parameter(Mandatory = $true)][string]$RunName)
    $ErrorActionPreference = 'Stop'
    # The newest step with all three checkpoint files and its saved line; -1 if none.
    $modelDir = Join-Path (Get-StoreDir $ProjectRoot $RunName) 'model'
    foreach ($s in (@(Get-SavedSteps $ProjectRoot $RunName) | Sort-Object -Descending)) {
        if (Test-CheckpointFiles $modelDir $s) { return [int]$s }
    }
    return -1
}

function Get-CheckpointEntries([string]$ModelDir) {
    # Files and folders named checkpoint-* (not the TF state file 'checkpoint').
    if (-not [IO.Directory]::Exists($ModelDir)) { return }
    Get-ChildItem -LiteralPath $ModelDir -Force -ErrorAction Stop | Where-Object { $_.Name -like 'checkpoint-*' }
}

function Test-UnsafeCheckpointName([string]$Name, [long]$GoodStep) {
    # Unsafe: anything main.py's loader could pick over the good checkpoint or crash on - a step
    # above it (complete or not, .tempstate*/.tmp* included) or a step that is not an integer.
    $m = [regex]::Match($Name, '^checkpoint-(\d{1,18})\.[^\\/]+$')
    if (-not $m.Success) { return $true }
    [long]$m.Groups[1].Value -gt $GoodStep
}

function Get-RestoreError([string]$ProjectRoot, [string]$RunName) {
    # A checkpoint restore error in the newest launch log -> { Log; Step } (Step -1 = unknown),
    # or $null. Only a launch that saved nothing counts: a restore error stops main.py before
    # its first save, and older logs are ignored so a re-saved checkpoint is never blamed.
    $launches = @(Get-LaunchFiles $ProjectRoot $RunName 'launch')
    if ($launches.Count -eq 0) { return $null }
    $log = $launches[-1].Log
    if (-not [IO.File]::Exists($log)) { return $null }
    $text = Read-SharedText $log
    $relModel = (Get-StoreRel $RunName) + 'model/'
    if ([regex]::IsMatch($text, '(?m)(?:^|\s)Checkpoint saved at step \d+ -> ' + [regex]::Escape($relModel) + '\r?$')) { return $null }
    $errors = @(($text -split "`r?`n") | Where-Object { $_ -match $QueueRestoreErrorPattern })
    if ($errors.Count -eq 0) { return $null }
    $step = -1
    foreach ($e in $errors) {
        $m = [regex]::Match($e, [regex]::Escape($relModel + 'checkpoint-') + '(\d{1,9})')
        if (-not $m.Success) { $m = [regex]::Match($e, '(?:^|[\s\\/])checkpoint-(\d{1,9})') }
        if ($m.Success) { $step = [int]$m.Groups[1].Value; break }
    }
    if ($step -lt 0) {
        $ms = [regex]::Matches($text, 'Restoring parameters from ' + [regex]::Escape($relModel + 'checkpoint-') + '(\d{1,9})')
        if ($ms.Count -gt 0) { $step = [int]$ms[$ms.Count - 1].Groups[1].Value }
    }
    [pscustomobject]@{ Log = $log; Step = $step }
}

# ---------------------------------------------------------------------------------------------
# Processes of a run
# ---------------------------------------------------------------------------------------------
function Test-RunCommandLine([string]$RunName, [string]$ProcessName, [string]$CommandLine) {
    # python running 'main.py ... --run <name>', or SUMO writing into the run's store or on the
    # queue's TraCI port. A null command line (another user's or an elevated process) is not
    # the run's: the runner reads the command lines of the processes it started.
    if ([string]::IsNullOrEmpty($CommandLine)) { return $false }
    $p = $ProcessName.ToLowerInvariant()
    if ($p -eq 'python.exe' -or $p -eq 'pythonw.exe') {
        $re = '(?i)(?:^|[\s"''\\/])main\.py["'']?\s.*?--run(?:\s+|=)["'']?' + [regex]::Escape($RunName) + '["'']?(?:\s|$)'
        return [regex]::IsMatch($CommandLine, $re)
    }
    if ($p -eq 'sumo.exe' -or $p -eq 'sumo-gui.exe') {
        foreach ($s in ('Malta/runs/' + $RunName + '/'), ('Malta\runs\' + $RunName + '\')) {
            if ($CommandLine.IndexOf($s, [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $true }
        }
        return [regex]::IsMatch($CommandLine, '--remote-port\s+' + $QueueTrainPort + '(?:\s|$)')
    }
    $false
}

function Test-RunProcessAlive {
    param([Parameter(Mandatory = $true)][string]$RunName)
    $ErrorActionPreference = 'Stop'
    $filter = "Name='python.exe' OR Name='pythonw.exe' OR Name='sumo.exe' OR Name='sumo-gui.exe'"
    foreach ($p in @(Get-CimInstance -ClassName Win32_Process -Filter $filter)) {
        if (Test-RunCommandLine $RunName $p.Name $p.CommandLine) { return $true }
    }
    $false
}

# ---------------------------------------------------------------------------------------------
# Training complete, stage
# ---------------------------------------------------------------------------------------------
function Add-RewardCsvProblems([string]$Store, $Cfg, $Problems) {
    # Spec 9: both reward CSVs have exactly total_step / T rows, every reward and average wait
    # is finite, and the seed column runs seed, seed + 1, ... with no repeats.
    $total = [int]$Cfg.TotalStep
    $T = [int]$Cfg.T
    if ($T -le 0 -or $total % $T -ne 0) { $Problems.Add("total_step $total is not a whole number of $T-step episodes"); return }
    $n = $total / $T
    $tr = Read-CsvFile (Join-Path $Store 'train_reward.csv')
    if (-not $tr.Exists) { $Problems.Add('train_reward.csv is missing') }
    elseif (-not $tr.Ok) { $Problems.Add("train_reward.csv does not parse ($($tr.Error))") }
    elseif ($null -eq $tr.Header) { $Problems.Add('train_reward.csv is empty') }
    else {
        if ($tr.Rows.Count -ne $n) { $Problems.Add("train_reward.csv has $($tr.Rows.Count) rows, expected $n") }
        $ri = Get-HeaderIndex $tr 'reward'
        $wi = Get-HeaderIndex $tr 'average waiting time'
        $si = Get-HeaderIndex $tr 'seed'
        if ($ri -lt 0 -or $wi -lt 0 -or $si -lt 0) { $Problems.Add('train_reward.csv lacks a reward, average waiting time or seed column') }
        else {
            for ($i = 0; $i -lt $tr.Rows.Count; $i++) {
                $f = $tr.Rows[$i].Fields
                if (-not (Test-FiniteNumber $f[$ri])) { $Problems.Add("train_reward.csv row $($i + 1) reward '$($f[$ri])' is not finite"); break }
                if (-not (Test-FiniteNumber $f[$wi])) { $Problems.Add("train_reward.csv row $($i + 1) average waiting time '$($f[$wi])' is not finite"); break }
            }
            for ($i = 0; $i -lt $tr.Rows.Count; $i++) {
                $raw = $tr.Rows[$i].Fields[$si]
                $seed = ConvertTo-WholeNumber $raw
                if ($null -eq $seed -or $seed -ne ([long]$Cfg.Seed + $i)) {
                    $Problems.Add("train_reward.csv row $($i + 1) seed '$raw', expected $([long]$Cfg.Seed + $i)")
                    break
                }
            }
        }
    }
    $ag = Read-CsvFile (Join-Path $Store 'agent_reward.csv') -NoHeader
    if (-not $ag.Exists) { $Problems.Add('agent_reward.csv is missing') }
    elseif (-not $ag.Ok) { $Problems.Add("agent_reward.csv does not parse ($($ag.Error))") }
    else {
        if ($ag.Rows.Count -ne $n) { $Problems.Add("agent_reward.csv has $($ag.Rows.Count) rows, expected $n") }
        for ($i = 0; $i -lt $ag.Rows.Count; $i++) {
            $bad = @($ag.Rows[$i].Fields | Where-Object { -not (Test-FiniteNumber $_) })
            if ($bad.Count -gt 0) { $Problems.Add("agent_reward.csv row $($i + 1) has a non-finite value '$($bad[0])'"); break }
        }
    }
}

function Test-TrainingComplete {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [Parameter(Mandatory = $true)][string]$RunName,
        [Parameter(Mandatory = $true)]$Cfg,
        [object]$ProcessAlive = $null
    )
    $ErrorActionPreference = 'Stop'
    # The five conditions of spec 9 -> { Ok; Reason } (Reason lists every failed condition).
    $store = Get-StoreDir $ProjectRoot $RunName
    $total = [int]$Cfg.TotalStep
    $problems = New-Object System.Collections.Generic.List[string]
    # 1. The newest training launch printed TRAINING COMPLETE step <total_step> and exited 0.
    $launches = @(Get-LaunchFiles $ProjectRoot $RunName 'launch')
    if ($launches.Count -eq 0) { $problems.Add('no training launch yet') }
    else {
        $last = $launches[-1]
        $code = Read-ExitCode $last.Exit
        if ($null -eq $code) { $problems.Add("launch_r$($last.N).exit has no exit code") }
        elseif ($code -ne 0) { $problems.Add("launch_r$($last.N) exited $code") }
        $text = ''
        if ([IO.File]::Exists($last.Log)) { $text = Read-SharedText $last.Log }
        if (-not [regex]::IsMatch($text, '(?m)>>> TRAINING COMPLETE step ' + $total + '\r?$')) {
            $problems.Add("launch_r$($last.N).log has no '>>> TRAINING COMPLETE step $total' line")
        }
    }
    # 2. No python or SUMO of the run is alive.
    if ($null -eq $ProcessAlive) { $ProcessAlive = Test-RunProcessAlive $RunName }
    if ($ProcessAlive) { $problems.Add('a python or SUMO process of the run is alive') }
    # 3. checkpoint-<total_step> .data, .index and .meta exist.
    if (-not (Test-CheckpointFiles (Join-Path $store 'model') $total)) { $problems.Add("checkpoint-$total lacks a .data, .index or .meta file") }
    # 4-5. Reward CSVs: row counts, finite values, seed column.
    Add-RewardCsvProblems $store $Cfg $problems
    $ok = ($problems.Count -eq 0)
    $reason = 'training complete'
    if (-not $ok) { $reason = $problems -join '; ' }
    [pscustomobject]@{ Ok = $ok; Reason = $reason }
}

function Get-RunStage {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [Parameter(Mandatory = $true)][string]$RunName,
        [Parameter(Mandatory = $true)]$Cfg,
        [object]$ProcessAlive = $null
    )
    $ErrorActionPreference = 'Stop'
    # FAILED is checked before DONE: with both markers present the queue stops for review.
    # Training complete with eva_data\EVAL_OK but no DONE marker (a stop between the two
    # writes) is DONE too; the runner then writes the missing DONE marker.
    $store = Get-StoreDir $ProjectRoot $RunName
    if ([IO.File]::Exists((Join-Path $store 'FAILED'))) { return 'FAILED' }
    if ([IO.File]::Exists((Join-Path $store 'DONE'))) { return 'DONE' }
    $tc = Test-TrainingComplete $ProjectRoot $RunName $Cfg -ProcessAlive $ProcessAlive
    if (-not $tc.Ok) { return 'TRAIN' }
    if ([IO.File]::Exists((Join-Path $store 'eva_data\EVAL_OK'))) { return 'DONE' }
    'EVALUATE'
}

# ---------------------------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------------------------
function New-RepairFile([string]$Name, [string]$Path, $Csv, [string]$NewText) {
    $changed = (-not $Csv.Exists) -or ($Csv.Text -cne $NewText)
    [pscustomobject]@{ Name = $Name; Path = $Path; Exists = $Csv.Exists; New = $NewText; Changed = $changed }
}

function Get-CsvRepairPlan([string]$Store, [int]$Good, $Cfg) {
    # The three run CSVs trimmed to the good checkpoint, validated before anything moves.
    # -> { Error; Files } - Error set means NEEDS_REVIEW; Files hold the new texts.
    $fail = { param([string]$m) [pscustomobject]@{ Error = $m; Files = @() } }
    $T = [int]$Cfg.T
    if ($T -le 0) { return (& $fail "T is $T") }
    if ($Good % $T -ne 0) { return (& $fail "checkpoint-$Good is not a whole number of $T-step episodes") }
    $n = [int]($Good / $T)

    # train_reward.csv: rows with step <= good, which must be exactly the n episodes.
    $trainPath = Join-Path $Store 'train_reward.csv'
    $tr = Read-CsvFile $trainPath
    if (-not $tr.Exists) { return (& $fail "train_reward.csv is missing (checkpoint-$Good)") }
    if (-not $tr.Ok) { return (& $fail "train_reward.csv does not parse ($($tr.Error))") }
    if ($tr.Rows.Count -lt $n) { return (& $fail "train_reward.csv has $($tr.Rows.Count) row(s), fewer than the $n episode(s) in checkpoint-$Good") }
    $steps = Get-ColumnWholeNumbers $tr 'step' -Increasing
    if ($null -ne $steps.Error) { return (& $fail "train_reward.csv does not parse ($($steps.Error))") }
    $kept = 0
    while ($kept -lt $steps.Values.Count -and $steps.Values[$kept] -le $Good) { $kept++ }
    if ($kept -lt $n) { return (& $fail "train_reward.csv has $kept row(s) up to step $Good, fewer than the $n episode(s) in checkpoint-$Good") }
    if ($kept -gt $n) { return (& $fail "train_reward.csv has $kept row(s) up to step $Good, more than the $n episode(s) in checkpoint-$Good") }
    if ($n -gt 0 -and $steps.Values[$n - 1] -ne $Good) { return (& $fail "train_reward.csv row $n has step $($steps.Values[$n - 1]), expected $Good") }
    $sb = New-Object System.Text.StringBuilder
    $head = $tr.HeaderLine
    if (-not $head.EndsWith("`n")) { $head += $tr.NewLine }
    [void]$sb.Append($head)
    for ($i = 0; $i -lt $n; $i++) { [void]$sb.Append($tr.Rows[$i].Line) }
    $trainNew = $sb.ToString()

    # agent_reward.csv: one row per episode, no step column - keep the first n rows.
    $agentPath = Join-Path $Store 'agent_reward.csv'
    $ag = Read-CsvFile $agentPath -NoHeader
    if (-not $ag.Exists) { return (& $fail "agent_reward.csv is missing (checkpoint-$Good)") }
    if (-not $ag.Ok) { return (& $fail "agent_reward.csv does not parse ($($ag.Error))") }
    if ($ag.Rows.Count -lt $n) { return (& $fail "agent_reward.csv has $($ag.Rows.Count) row(s), fewer than the $n episode(s) in checkpoint-$Good") }
    for ($i = 0; $i -lt $ag.Rows.Count; $i++) {
        $bad = @($ag.Rows[$i].Fields | Where-Object { -not (Test-NumberToken $_) })
        if ($bad.Count -gt 0) { return (& $fail "agent_reward.csv does not parse (row $($i + 1) has '$($bad[0])')") }
    }
    $sb = New-Object System.Text.StringBuilder
    for ($i = 0; $i -lt $n; $i++) { [void]$sb.Append($ag.Rows[$i].Line) }
    $agentNew = $sb.ToString()

    # episode_cost.csv: rows with step <= good matching the kept train steps; exactly one
    # missing row is replaced by a NaN row with lost = 1 (episode, step and seed from train).
    $costPath = Join-Path $Store 'episode_cost.csv'
    $co = Read-CsvFile $costPath
    if (-not $co.Ok) { return (& $fail "episode_cost.csv does not parse ($($co.Error))") }
    $nl = $co.NewLine
    $costHead = $QueueCostHeader + $nl
    $costSteps = @()
    if ($null -ne $co.HeaderLine) {
        if ($co.HeaderLine.Trim() -ne $QueueCostHeader) { return (& $fail 'episode_cost.csv does not start with the cost header') }
        $costHead = $co.HeaderLine
        if (-not $costHead.EndsWith("`n")) { $costHead += $nl }
        $cs = Get-ColumnWholeNumbers $co 'step' -Increasing
        if ($null -ne $cs.Error) { return (& $fail "episode_cost.csv does not parse ($($cs.Error))") }
        $costSteps = $cs.Values
    }
    $ck = 0
    while ($ck -lt $costSteps.Count -and $costSteps[$ck] -le $Good) { $ck++ }
    $seedIdx = Get-HeaderIndex $tr 'seed'
    $rows = New-Object System.Collections.Generic.List[string]
    $lost = $null
    if ($ck -eq $n) {
        for ($i = 0; $i -lt $n; $i++) {
            if ($costSteps[$i] -ne $steps.Values[$i]) { return (& $fail "episode_cost.csv row $($i + 1) has step $($costSteps[$i]), train_reward.csv has $($steps.Values[$i])") }
            $rows.Add($co.Rows[$i].Line)
        }
    } elseif ($ck -eq $n - 1) {
        $j = 0
        for ($i = 0; $i -lt $n; $i++) {
            $step = $steps.Values[$i]
            if ($j -lt $ck -and $costSteps[$j] -eq $step) { $rows.Add($co.Rows[$j].Line); $j++; continue }
            if ($null -ne $lost) { return (& $fail "episode_cost.csv steps do not match train_reward.csv (step $step)") }
            $seedText = 'NaN'
            if ($seedIdx -ge 0) {
                $sv = ConvertTo-WholeNumber $tr.Rows[$i].Fields[$seedIdx]
                if ($null -ne $sv) { $seedText = [string]::Format($QueueInvariant, '{0}', $sv) }
            }
            $lost = [string]::Format($QueueInvariant, '{0},{1},{2},NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,1', [long]($step / $T), $step, $seedText)
            $rows.Add($lost + $nl)
        }
        if ($j -ne $ck) { return (& $fail 'episode_cost.csv steps do not match train_reward.csv') }
    } else {
        return (& $fail "episode_cost.csv has $ck row(s) up to step $Good for $n training episode(s)")
    }
    $costNew = $costHead + ($rows -join '')

    $files = @(
        (New-RepairFile 'train_reward.csv' $trainPath $tr $trainNew),
        (New-RepairFile 'agent_reward.csv' $agentPath $ag $agentNew),
        (New-RepairFile 'episode_cost.csv' $costPath $co $costNew)
    )
    $notes = New-Object System.Collections.Generic.List[string]
    if ($tr.Rows.Count -ne $n) { $notes.Add("train_reward.csv $($tr.Rows.Count) -> $n rows") }
    if ($ag.Rows.Count -ne $n) { $notes.Add("agent_reward.csv $($ag.Rows.Count) -> $n rows") }
    if ($co.Rows.Count -ne $n) { $notes.Add("episode_cost.csv $($co.Rows.Count) -> $n rows") }
    if ($null -ne $lost) { $notes.Add("lost cost row added: $lost") }
    [pscustomobject]@{ Error = $null; Files = $files; Notes = ($notes -join ', ') }
}

function Invoke-Recovery {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [Parameter(Mandatory = $true)][string]$RunName,
        [Parameter(Mandatory = $true)]$Cfg,
        [Parameter(Mandatory = $true)][string]$QueueDir,
        [object]$ProcessAlive = $null,
        [scriptblock]$Log = $null
    )
    $ErrorActionPreference = 'Stop'
    # Spec 9 Recovery: make model\ and the run CSVs consistent with the good checkpoint. Only
    # moves files - into <QueueDir>\quarantine_<stamp>\ (checkpoint files) and
    # <QueueDir>\recovery_<stamp>\ (CSV originals, partial CSVs before a scratch start).
    #   RESUME         the good checkpoint and the CSVs agree; main.py resumes from it.
    #   SCRATCH        model\ has no checkpoint-* and train_reward.csv is absent or has no rows.
    #   NEEDS_REVIEW:  the run must be marked FAILED for the author. Nothing was moved, except a
    #                  checkpoint that failed to restore in the newest launch.
    Assert-AbsolutePath $ProjectRoot 'ProjectRoot'
    Assert-AbsolutePath $QueueDir 'QueueDir'
    if ($null -eq $ProcessAlive) { $ProcessAlive = Test-RunProcessAlive $RunName }
    if ($ProcessAlive) { throw "Invoke-Recovery refused: a python or SUMO process of run $RunName is alive" }
    $store = Get-StoreDir $ProjectRoot $RunName
    $modelDir = Join-Path $store 'model'
    $stamp = New-QueueStamp $QueueDir
    $quarantine = Join-Path $QueueDir ('quarantine_' + $stamp)
    $recovery = Join-Path $QueueDir ('recovery_' + $stamp)
    $say = { param([string]$m) if ($null -ne $Log) { $null = & $Log ('recovery ' + $RunName + ': ' + $m) } }

    # 1. A checkpoint that failed to restore in the newest launch is quarantined, so the good
    #    checkpoint falls back to the next older one.
    $restore = Get-RestoreError $ProjectRoot $RunName
    if ($null -ne $restore) {
        $bad = $restore.Step
        if ($bad -lt 0) { $bad = Get-GoodCheckpoint $ProjectRoot $RunName }
        if ($bad -ge 0) {
            $prefix = 'checkpoint-' + $bad + '.'
            $hits = @(Get-CheckpointEntries $modelDir | Where-Object { $_.Name.StartsWith($prefix) })
            foreach ($e in $hits) { Move-ToFolder $e.FullName $quarantine }
            if ($hits.Count -gt 0) {
                & $say ("restore error in {0}: moved checkpoint-{1} ({2} file(s)) to {3}" -f [IO.Path]::GetFileName($restore.Log), $bad, $hits.Count, $quarantine)
            }
        }
    }

    # 2. No good checkpoint: a scratch start, or review.
    $good = Get-GoodCheckpoint $ProjectRoot $RunName
    $entries = @(Get-CheckpointEntries $modelDir)
    if ($good -lt 0) {
        if ($entries.Count -gt 0) { return "NEEDS_REVIEW:model\ has $($entries.Count) checkpoint-* file(s) but no checkpoint is complete and logged as saved" }
        $tr = Read-CsvFile (Join-Path $store 'train_reward.csv')
        if (-not $tr.Ok) { return "NEEDS_REVIEW:train_reward.csv does not parse ($($tr.Error))" }
        if ($tr.Rows.Count -gt 0) { return "NEEDS_REVIEW:no checkpoint in model\ but train_reward.csv has $($tr.Rows.Count) row(s)" }
        $partial = @('train_reward.csv', 'agent_reward.csv', 'episode_cost.csv' |
            ForEach-Object { Join-Path $store $_ } | Where-Object { [IO.File]::Exists($_) })
        foreach ($p in $partial) { Move-ToFolder $p $recovery }
        if ($partial.Count -gt 0) { & $say ("no checkpoint; moved {0} partial CSV(s) to {1}" -f $partial.Count, $recovery) }
        return 'SCRATCH'
    }

    # 3. Validate and plan the CSV trim before moving anything.
    $plan = Get-CsvRepairPlan $store $good $Cfg
    if ($null -ne $plan.Error) { return 'NEEDS_REVIEW:' + $plan.Error }

    # 4. Quarantine what the loader could pick over the good checkpoint.
    $unsafe = @($entries | Where-Object { Test-UnsafeCheckpointName $_.Name $good })
    foreach ($e in $unsafe) { Move-ToFolder $e.FullName $quarantine }
    if ($unsafe.Count -gt 0) {
        & $say ("good checkpoint-{0}; moved {1} to {2}" -f $good, (($unsafe | ForEach-Object { $_.Name }) -join ', '), $quarantine)
    }

    # 5. Copy the originals, then write the trimmed CSVs (temp file, then move over the original).
    $changed = @($plan.Files | Where-Object { $_.Changed })
    if ($changed.Count -gt 0) {
        foreach ($f in $changed) {
            if ($f.Exists) {
                if (-not [IO.Directory]::Exists($recovery)) { [void][IO.Directory]::CreateDirectory($recovery) }
                Copy-Item -LiteralPath $f.Path -Destination (Join-Path $recovery $f.Name)
            }
        }
        foreach ($f in $changed) {
            $tmp = $f.Path + '.qtmp'
            Write-Utf8NoBom $tmp $f.New
            Move-Item -LiteralPath $tmp -Destination $f.Path -Force
        }
        & $say ("trimmed CSVs to checkpoint-{0} ({1}); originals in {2}" -f $good, $plan.Notes, $recovery)
    }
    'RESUME'
}

# ---------------------------------------------------------------------------------------------
# Evaluation attempt
# ---------------------------------------------------------------------------------------------
function Get-LevelFileProblems([string]$Dir, $Cfg, [datetime]$SinceUtc, $Problems) {
    # One level folder: the three CSVs and per seed edgedata_<seed>.xml and summary_<seed>.xml,
    # all newer than the launch and closed properly; the CSVs cover episodes 1..n.
    # -> mean trips per episode, or $null when the trip CSV is not usable.
    $n = @($Cfg.TestSeeds).Count
    if (-not [IO.Directory]::Exists($Dir)) { $Problems.Add('no level folder'); return $null }
    $mean = $null
    $want = (1..$n) -join ','
    foreach ($kind in 'control', 'traffic', 'trip') {
        $name = '{0}_signal_route_{1}.csv' -f $Cfg.Scenario, $kind
        $path = Join-Path $Dir $name
        if (-not [IO.File]::Exists($path)) { $Problems.Add("$name missing"); continue }
        $before = $Problems.Count
        if ([IO.File]::GetLastWriteTimeUtc($path) -le $SinceUtc) { $Problems.Add("$name is older than the launch") }
        if (-not (Test-EndsWithNewline $path)) { $Problems.Add("$name does not end with a line end") }
        $c = Get-CsvEpisodeCounts $path
        if ($null -ne $c.Error) { $Problems.Add("${name}: $($c.Error)") }
        else {
            $got = ($c.Counts.Keys | Sort-Object) -join ','
            if ($got -ne $want) { $Problems.Add("$name covers episodes '$got', expected $want") }
        }
        if ($kind -eq 'trip' -and $Problems.Count -eq $before) { $mean = $c.Rows / [double]$n }
    }
    foreach ($seed in $Cfg.TestSeeds) {
        foreach ($x in @(@('edgedata', 'meandata'), @('summary', 'summary'))) {
            $name = '{0}_{1}.xml' -f $x[0], $seed
            $path = Join-Path $Dir $name
            if (-not [IO.File]::Exists($path)) { $Problems.Add("$name missing"); continue }
            if ([IO.File]::GetLastWriteTimeUtc($path) -le $SinceUtc) { $Problems.Add("$name is older than the launch") }
            if (-not (Test-XmlClosed $path $x[1])) { $Problems.Add("$name does not end with </$($x[1])>") }
        }
    }
    $mean
}

function Get-TripMean([string]$Dir, $Cfg) {
    # Mean trips per episode of a level folder from an earlier attempt, or $null.
    $n = @($Cfg.TestSeeds).Count
    $path = Join-Path $Dir ('{0}_signal_route_trip.csv' -f $Cfg.Scenario)
    if (-not [IO.File]::Exists($path)) { return $null }
    $c = Get-CsvEpisodeCounts $path
    if ($null -ne $c.Error -or (($c.Counts.Keys | Sort-Object) -join ',') -ne ((1..$n) -join ',')) { return $null }
    $c.Rows / [double]$n
}

function Test-EvalAttempt {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [Parameter(Mandatory = $true)][string]$RunName,
        [Parameter(Mandatory = $true)]$Cfg,
        [Parameter(Mandatory = $true)][string]$LogPath,
        [Parameter(Mandatory = $true)][datetime]$LaunchTime,
        [double[]]$Levels = $null
    )
    $ErrorActionPreference = 'Stop'
    # Spec 8 checks of one evaluation attempt (its eval_rN.log, the matching .exit file and the
    # level folders it wrote) -> { Ok; FailedLevels ('1.10' strings, for --levels); Reason }.
    # $Levels: the levels this attempt ran (default: the config's eval_levels). The trips
    # check also reads the trip CSVs of the config's other levels (earlier attempts).
    # When the attempt fails but no single level can be blamed, every level is failed.
    Assert-AbsolutePath $LogPath 'Test-EvalAttempt log path'
    if ($null -eq $Levels -or $Levels.Count -eq 0) { $Levels = [double[]]$Cfg.EvalLevels }
    $rel = Get-StoreRel $RunName
    $n = @($Cfg.TestSeeds).Count
    $since = $LaunchTime.ToUniversalTime()
    $reasons = New-Object System.Collections.Generic.List[string]
    $failed = @{}
    $modelBad = $false

    $text = ''
    if ([IO.File]::Exists($LogPath)) { $text = Read-SharedText $LogPath } else { $reasons.Add('no evaluation log') }
    $lines = $text -split "`r?`n"

    # Attempt: exit 0, COMPLETE, restore of checkpoint-<total_step> before the first level,
    # no traceback, no 'Can not find old checkpoint'.
    $code = Read-ExitCode ([IO.Path]::ChangeExtension($LogPath, '.exit'))
    if ($null -eq $code) { $reasons.Add('no exit code') }
    elseif ($code -ne 0) { $reasons.Add("evaluation exited $code") }
    $completeRe = '>>> EVALUATE COMPLETE levels=' + $Levels.Count + '\s*$'
    if (@($lines | Where-Object { $_ -match $completeRe }).Count -eq 0) { $reasons.Add("no '>>> EVALUATE COMPLETE levels=$($Levels.Count)' line") }
    $restoreRe = 'Restoring parameters from ' + [regex]::Escape($rel + 'model/checkpoint-' + $Cfg.TotalStep) + '\s*$'
    $firstStart = -1
    $restoreAt = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($firstStart -lt 0 -and $lines[$i].Contains('EVAL_LEVEL_START ')) { $firstStart = $i }
        if ($restoreAt -lt 0 -and $lines[$i] -match $restoreRe) { $restoreAt = $i }
    }
    if ($restoreAt -lt 0 -or ($firstStart -ge 0 -and $restoreAt -gt $firstStart)) {
        $reasons.Add("no 'Restoring parameters from ${rel}model/checkpoint-$($Cfg.TotalStep)' before the first level")
        $modelBad = $true
    }
    if (@($lines | Where-Object { $_.Contains('Can not find old checkpoint') }).Count -gt 0) {
        $reasons.Add("'Can not find old checkpoint' in the log")
        $modelBad = $true
    }
    $tracebacks = @($lines | Where-Object { $_ -match '^(Traceback|Exception in thread)' }).Count
    if ($tracebacks -gt 0) { $reasons.Add("$tracebacks Traceback / Exception in thread line(s)") }

    # Level blocks: EVAL_LEVEL_START ... EVAL_LEVEL_DONE with the test lines between.
    $blocks = @{}
    $open = $null
    foreach ($l in $lines) {
        $m = [regex]::Match($l, 'EVAL_LEVEL_START f(\S+) bg_tag=(\S+) corridor_total=(\S+) corridor_expected=(\S+)\s*$')
        if ($m.Success) {
            $open = [pscustomobject]@{
                Tag = $m.Groups[1].Value; BgTag = $m.Groups[2].Value; Total = $m.Groups[3].Value; Expected = $m.Groups[4].Value
                Tests = (New-Object System.Collections.Generic.List[int]); Done = $null; Errors = 0
            }
            if (-not $blocks.ContainsKey($open.Tag)) { $blocks[$open.Tag] = New-Object System.Collections.Generic.List[object] }
            $blocks[$open.Tag].Add($open)
            continue
        }
        $m = [regex]::Match($l, 'EVAL_LEVEL_DONE f(\S+) seeds=(\d+)\s*$')
        if ($m.Success) {
            if ($null -ne $open -and $open.Tag -eq $m.Groups[1].Value) { $open.Done = [int]$m.Groups[2].Value }
            $open = $null
            continue
        }
        if ($null -eq $open) { continue }
        $m = [regex]::Match($l, '(?:^|\s)test (\d+), avg reward ')
        if ($m.Success) { $open.Tests.Add([int]$m.Groups[1].Value); continue }
        if ($l -match '^(Traceback|Exception in thread)') { $open.Errors++ }
    }

    # Each level of the attempt.
    $tags = New-Object System.Collections.Generic.List[string]
    $means = @{}
    $wantTests = (0..($n - 1)) -join ','
    foreach ($lv in $Levels) {
        $tag = Format-Level $lv
        $tags.Add($tag)
        $lp = New-Object System.Collections.Generic.List[string]
        $found = @()
        if ($blocks.ContainsKey($tag)) { $found = $blocks[$tag].ToArray() }   # not @(): PS 5.1 throws on @(List[object])
        if ($found.Count -eq 0) { $lp.Add('no EVAL_LEVEL_START') }
        elseif ($found.Count -gt 1) { $lp.Add("$($found.Count) EVAL_LEVEL_START lines") }
        else {
            $b = $found[0]
            if ($null -eq $b.Done) { $lp.Add('no EVAL_LEVEL_DONE') }
            elseif ($b.Done -ne $n) { $lp.Add("EVAL_LEVEL_DONE seeds=$($b.Done), expected $n") }
            $got = $b.Tests -join ','
            if ($got -ne $wantTests) { $lp.Add("test lines '$got', expected $wantTests") }
            if ($b.Errors -gt 0) { $lp.Add('a traceback inside the level') }
            $wantTag = Format-BgTag $lv
            if ($b.BgTag -ne $wantTag) { $lp.Add("bg_tag=$($b.BgTag), expected $wantTag") }
            $t = 0.0; $x = 0.0
            if (-not ([double]::TryParse($b.Total, $QueueFloatStyle, $QueueInvariant, [ref]$t) -and
                      [double]::TryParse($b.Expected, $QueueFloatStyle, $QueueInvariant, [ref]$x) -and $t -eq $x)) {
                $lp.Add("corridor_total=$($b.Total) differs from corridor_expected=$($b.Expected)")
            }
        }
        $mean = Get-LevelFileProblems (Get-LevelDir $ProjectRoot $RunName $lv) $Cfg $since $lp
        if ($lp.Count -gt 0) {
            $failed[$tag] = $true
            $reasons.Add("f${tag}: " + ($lp -join ', '))
        } elseif ($null -ne $mean) {
            $means[$tag] = [pscustomobject]@{ Tag = $tag; Level = $lv; Mean = $mean; InAttempt = $true }
        }
    }

    # Mean trips per episode must not fall as the level rises (with earlier attempts' levels).
    foreach ($lv in $Cfg.EvalLevels) {
        $tag = Format-Level $lv
        if ($tags.Contains($tag)) { continue }
        $mean = Get-TripMean (Get-LevelDir $ProjectRoot $RunName $lv) $Cfg
        if ($null -ne $mean) { $means[$tag] = [pscustomobject]@{ Tag = $tag; Level = $lv; Mean = $mean; InAttempt = $false } }
    }
    $chain = @($means.Values | Sort-Object Level)
    for ($i = 1; $i -lt $chain.Count; $i++) {
        $a = $chain[$i - 1]
        $b = $chain[$i]
        if ($b.Mean -lt $a.Mean) {
            $reasons.Add([string]::Format($QueueInvariant, 'mean trips per episode fall from {0} at f{1} to {2} at f{3}', $a.Mean, $a.Tag, $b.Mean, $b.Tag))
            # Blame the higher level (the one that fell); the lower one only when the higher
            # level is from an earlier attempt.
            if ($b.InAttempt) { $failed[$b.Tag] = $true }
            elseif ($a.InAttempt) { $failed[$a.Tag] = $true }
        }
    }

    if ($modelBad) { foreach ($tag in $tags) { $failed[$tag] = $true } }
    $ok = ($reasons.Count -eq 0)
    if (-not $ok -and $failed.Count -eq 0) { foreach ($tag in $tags) { $failed[$tag] = $true } }
    $failedLevels = @($tags | Where-Object { $failed.ContainsKey($_) })
    $reason = 'ok'
    if (-not $ok) { $reason = $reasons -join '; ' }
    [pscustomobject]@{ Ok = $ok; FailedLevels = [string[]]$failedLevels; Reason = $reason }
}

# ---------------------------------------------------------------------------------------------
# state.json and launch counters
# ---------------------------------------------------------------------------------------------
function New-QueueState {
    [ordered]@{ version = 1; updated = $null; runs = [ordered]@{} }
}

function ConvertTo-QueueHashtable($Value) {
    # ConvertFrom-Json output -> ordered hashtables; arrays stay arrays (also with one element).
    if ($null -eq $Value) { return $null }
    if ($Value -is [System.Management.Automation.PSCustomObject]) {
        $h = [ordered]@{}
        foreach ($p in $Value.PSObject.Properties) { $h[$p.Name] = ConvertTo-QueueHashtable $p.Value }
        return $h
    }
    if ($Value -is [array]) {
        $list = @()
        foreach ($e in $Value) { $list += , (ConvertTo-QueueHashtable $e) }
        return , $list
    }
    $Value
}

function Read-QueueState {
    param([Parameter(Mandatory = $true)][string]$QueueDir)
    $ErrorActionPreference = 'Stop'
    # state.json, else state.json.tmp (a stop between the two steps of Save-QueueState), else a
    # new state. Throws when state.json exists and neither file reads, rather than reset the
    # death counters.
    Assert-AbsolutePath $QueueDir 'QueueDir'
    $path = Join-Path $QueueDir 'state.json'
    $errors = @()
    foreach ($candidate in $path, ($path + '.tmp')) {
        if (-not [IO.File]::Exists($candidate)) { continue }
        try {
            $obj = (Read-SharedText $candidate) | ConvertFrom-Json
            if ($obj -isnot [System.Management.Automation.PSCustomObject]) { throw 'not a JSON object' }
            $state = ConvertTo-QueueHashtable $obj
            if (-not $state.Contains('runs') -or $null -eq $state['runs']) { $state['runs'] = [ordered]@{} }
            return $state
        } catch {
            $errors += ([IO.Path]::GetFileName($candidate) + ': ' + $_.Exception.Message)
        }
    }
    if ([IO.File]::Exists($path)) { throw ('state.json is unreadable: ' + ($errors -join '; ')) }
    New-QueueState
}

function Save-QueueState {
    param([Parameter(Mandatory = $true)][string]$QueueDir, [Parameter(Mandatory = $true)]$State)
    $ErrorActionPreference = 'Stop'
    Assert-AbsolutePath $QueueDir 'QueueDir'
    $path = Join-Path $QueueDir 'state.json'
    $State['updated'] = (Get-Date).ToString('o')
    Write-Utf8NoBom ($path + '.tmp') (ConvertTo-Json -InputObject $State -Depth 10)
    Move-Item -LiteralPath ($path + '.tmp') -Destination $path -Force
}

function Get-RunCounters {
    param([Parameter(Mandatory = $true)]$State, [Parameter(Mandatory = $true)][string]$Run)
    $ErrorActionPreference = 'Stop'
    # The run's entry in $State (created, and missing keys filled, with the schema defaults).
    if (-not $State.Contains('runs') -or $null -eq $State['runs']) { $State['runs'] = [ordered]@{} }
    $runs = $State['runs']
    if (-not $runs.Contains($Run)) { $runs[$Run] = [ordered]@{} }
    $r = $runs[$Run]
    foreach ($k in 'launches', 'no_progress', 'interrupted', 'eval_attempts') {
        if (-not $r.Contains($k) -or $null -eq $r[$k]) { $r[$k] = 0 }
    }
    if (-not $r.Contains('good_at_launch') -or $null -eq $r['good_at_launch']) { $r['good_at_launch'] = @() }
    if (-not $r.Contains('current_launch')) { $r['current_launch'] = $null }
    $r
}

function Update-LaunchCounters {
    param(
        [Parameter(Mandatory = $true)]$State,
        [Parameter(Mandatory = $true)][string]$Run,
        [Parameter(Mandatory = $true)][int]$GoodBefore,
        [Parameter(Mandatory = $true)][int]$GoodAfter,
        [Parameter(Mandatory = $true)][bool]$Interrupted
    )
    $ErrorActionPreference = 'Stop'
    # Count one ended training launch (spec 9 death counters). Progress = the good checkpoint
    # rose. STUCK after 3 consecutive launches without progress (interruptions excluded) or
    # 12 launches in total (interruptions included). The caller saves the state.
    $r = Get-RunCounters $State $Run
    $r['launches'] = [int]$r['launches'] + 1
    $r['good_at_launch'] = @($r['good_at_launch']) + @($GoodBefore)
    if ($GoodAfter -gt $GoodBefore) { $r['no_progress'] = 0 }
    elseif (-not $Interrupted) { $r['no_progress'] = [int]$r['no_progress'] + 1 }
    if ($Interrupted) { $r['interrupted'] = [int]$r['interrupted'] + 1 }
    $cur = $r['current_launch']
    if ($null -ne $cur -and $cur.kind -eq 'train') { $r['current_launch'] = $null }
    if ([int]$r['no_progress'] -ge $QueueMaxNoProgress -or [int]$r['launches'] -ge $QueueMaxLaunches) { return 'STUCK' }
    'OK'
}
