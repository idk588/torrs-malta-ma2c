<#
Registers (or unregisters) the Windows scheduled task that runs the RQ2 training queue.
It only registers; it never starts the task. Start it with Start-ScheduledTask.
Use -DryRun to print the task definition without registering anything.
#>
param(
    [string]$TaskName = 'TORRS training queue',
    [string]$QueueFile = 'config/queue.txt',
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [switch]$Unregister,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$elevationMsg = 'Registration needs elevation: run this script in an elevated PowerShell (Run as administrator).'

if ($Unregister) {
    if ($DryRun) { Write-Host "DryRun: would unregister task '$TaskName'"; exit 0 }
    try {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    } catch {
        if ($_.Exception -is [System.UnauthorizedAccessException] -or $_.Exception.Message -match 'Access is denied|0x80070005') {
            Write-Host $elevationMsg; exit 1
        }
        throw
    }
    Write-Host "Unregistered task '$TaskName'"
    exit 0
}

$user = "$env:USERDOMAIN\$env:USERNAME"
$script = Join-Path $ProjectRoot 'tools\run_queue.ps1'

$startup = New-ScheduledTaskTrigger -AtStartup
$startup.Delay = 'PT2M'
$logon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$logon.Delay = 'PT2M'
$triggers = @($startup, $logon)

$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -RestartCount 99 -RestartInterval (New-TimeSpan -Minutes 5)
$arg = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $script +
    '" -QueueFile "' + $QueueFile + '" -ProjectRoot "' + $ProjectRoot + '"'
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arg -WorkingDirectory $ProjectRoot

if ($DryRun) {
    Write-Host "DryRun: task '$TaskName' would be registered as follows (nothing registered)"
    Write-Host "Principal : $user (S4U, RunLevel Limited)"
    Write-Host "Triggers  : AtStartup (Delay $($startup.Delay)); AtLogOn $env:USERNAME (Delay $($logon.Delay))"
    Write-Host "Settings  : ExecutionTimeLimit=$($settings.ExecutionTimeLimit) MultipleInstances=$($settings.MultipleInstances) StartWhenAvailable=$($settings.StartWhenAvailable) DisallowStartIfOnBatteries=$($settings.DisallowStartIfOnBatteries) StopIfGoingOnBatteries=$($settings.StopIfGoingOnBatteries) RestartCount=$($settings.RestartCount) RestartInterval=$($settings.RestartInterval)"
    Write-Host "Execute   : $($action.Execute)"
    Write-Host "Arguments : $($action.Arguments)"
    Write-Host "WorkingDir: $($action.WorkingDirectory)"
    exit 0
}

try {
    Register-ScheduledTask -TaskName $TaskName -Trigger $triggers -Principal $principal `
        -Settings $settings -Action $action -Force | Out-Null
} catch {
    if ($_.Exception -is [System.UnauthorizedAccessException] -or $_.Exception.Message -match 'Access is denied|0x80070005') {
        Write-Host $elevationMsg; exit 1
    }
    throw
}
Write-Host "Registered task '$TaskName' (not started). Start it with: Start-ScheduledTask -TaskName '$TaskName'"
