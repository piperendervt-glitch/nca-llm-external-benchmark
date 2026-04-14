<#
.SYNOPSIS
    Launch and manage experiments via Windows Task Scheduler.
    Tasks continue running after SSH disconnect.

.USAGE
    # Start task
    .\run_background.ps1 -Action start -JobName "hgnn_n200" -Script "run_nca_hgnn.py" -Args "--benchmark alphanli --n_samples 200"

    # List tasks
    .\run_background.ps1 -Action list

    # Check progress (recent log + progress)
    .\run_background.ps1 -Action tail -JobName "hgnn_n200"

    # View full log
    .\run_background.ps1 -Action result -JobName "hgnn_n200"

    # Stop task
    .\run_background.ps1 -Action stop -JobName "hgnn_n200"

    # Stop all tasks
    .\run_background.ps1 -Action stopall
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("start", "list", "tail", "result", "stop", "stopall")]
    [string]$Action,

    [string]$JobName = "",
    [string]$Script  = "",
    [string]$Args    = ""
)

$REPO_ROOT  = "C:\Users\pipe_render\nca-llm-external-benchmark"
$SCRIPT_DIR = "$REPO_ROOT\experiments\nca_llm"
$LOG_DIR    = "$REPO_ROOT\logs"

# Create log directory
if (-not (Test-Path $LOG_DIR)) {
    New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
}

switch ($Action) {

    "start" {
        if (-not $JobName) { Write-Error "JobName is required"; exit 1 }
        if (-not $Script)  { Write-Error "Script is required";  exit 1 }

        $logFile    = "$LOG_DIR\${JobName}.log"
        $scriptPath = "$SCRIPT_DIR\$Script"

        if (-not (Test-Path $scriptPath)) {
            Write-Error "Script not found: $scriptPath"
            exit 1
        }

        # Check and remove existing task with same name
        $existing = Get-ScheduledTask -TaskName $JobName -ErrorAction SilentlyContinue
        if ($existing) {
            Write-Warning "Task already exists: $JobName (State: $($existing.State))"
            Stop-ScheduledTask -TaskName $JobName -ErrorAction SilentlyContinue
            Unregister-ScheduledTask -TaskName $JobName -Confirm:$false
        }

        # Start marker
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START: python $Script $Args" |
            Out-File -FilePath $logFile -Encoding utf8

        # Generate wrapper script (executed by Task Scheduler)
        $wrapperPath = "$LOG_DIR\${JobName}_wrapper.ps1"
        $wrapperContent = @'
Set-Location '{0}'
python '{1}' {2} 2>&1 | Tee-Object -FilePath '{3}' -Append
$doneTime = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Add-Content -Path '{3}' -Value "[$doneTime] DONE"
'@ -f $SCRIPT_DIR, $scriptPath, $Args, $logFile
        $wrapperContent | Out-File -FilePath $wrapperPath -Encoding utf8

        # Run via Task Scheduler (persists after SSH disconnect)
        $taskAction = New-ScheduledTaskAction `
            -Execute "powershell.exe" `
            -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$wrapperPath`"" `
            -WorkingDirectory $SCRIPT_DIR

        $principal = New-ScheduledTaskPrincipal `
            -UserId $env:USERNAME `
            -LogonType S4U `
            -RunLevel Limited

        Register-ScheduledTask `
            -TaskName $JobName `
            -Action $taskAction `
            -Principal $principal `
            -Force | Out-Null

        Start-ScheduledTask -TaskName $JobName

        Write-Host ""
        Write-Host "Job started (Task Scheduler)" -ForegroundColor Green
        Write-Host "   JobName : $JobName"
        Write-Host "   Script  : $Script $Args"
        Write-Host "   Log     : $logFile"
        Write-Host ""
        Write-Host "Check progress:"
        Write-Host "   .\run_background.ps1 -Action tail -JobName `"$JobName`""
        Write-Host "Check result:"
        Write-Host "   .\run_background.ps1 -Action result -JobName `"$JobName`""
    }

    "list" {
        $tasks = Get-ScheduledTask | Where-Object {
            $_.TaskPath -eq "\" -and
            (Test-Path "$LOG_DIR\$($_.TaskName).log")
        }

        if (-not $tasks) {
            Write-Host "No experiment tasks found."
        } else {
            Write-Host ""
            Write-Host "Experiment task list:" -ForegroundColor Cyan
            Write-Host ("-" * 60)
            foreach ($t in $tasks) {
                $logFile  = "$LOG_DIR\$($t.TaskName).log"
                $lastLine = ""
                if (Test-Path $logFile) {
                    $lastLine = Get-Content $logFile -Tail 1
                }
                $info = Get-ScheduledTaskInfo -TaskName $t.TaskName `
                    -ErrorAction SilentlyContinue
                Write-Host "[$($t.TaskName)]" -ForegroundColor Yellow
                Write-Host "  State    : $($t.State)"
                Write-Host "  LastRun  : $($info.LastRunTime)"
                Write-Host "  LastLog  : $lastLine"
                Write-Host ""
            }
        }
    }

    "tail" {
        if (-not $JobName) { Write-Error "JobName is required"; exit 1 }

        $logFile = "$LOG_DIR\${JobName}.log"

        if (-not (Test-Path $logFile)) {
            Write-Error "Log file not found: $logFile"
            exit 1
        }

        $task  = Get-ScheduledTask -TaskName $JobName -ErrorAction SilentlyContinue
        $state = if ($task) { $task.State } else { "Unknown" }

        Write-Host ""
        Write-Host "[$JobName] State: $state" -ForegroundColor Cyan
        Write-Host ("-" * 60)
        Write-Host "Last 20 lines:" -ForegroundColor Yellow
        Get-Content $logFile -Tail 20
        Write-Host ""

        $progress = Get-Content $logFile |
            Where-Object { $_ -match "\[\d+/\d+\]" } |
            Select-Object -Last 1
        if ($progress) {
            Write-Host "Latest progress: $progress" -ForegroundColor Green
        }

        $done = Get-Content $logFile |
            Where-Object { $_ -match "DONE" } |
            Select-Object -Last 1
        if ($done) {
            Write-Host "Experiment done: $done" -ForegroundColor Green
        }
    }

    "result" {
        if (-not $JobName) { Write-Error "JobName is required"; exit 1 }

        $logFile = "$LOG_DIR\${JobName}.log"

        $task  = Get-ScheduledTask -TaskName $JobName -ErrorAction SilentlyContinue
        $state = if ($task) { $task.State } else { "Unknown" }

        Write-Host ""
        Write-Host "Task: $JobName (State: $state)" -ForegroundColor Cyan
        Write-Host ("-" * 60)

        if (Test-Path $logFile) {
            Write-Host "Log file: $logFile"
            Write-Host ""
            Get-Content $logFile
        } else {
            Write-Host "Log file not found: $logFile"
        }
    }

    "stop" {
        if (-not $JobName) { Write-Error "JobName is required"; exit 1 }

        $task = Get-ScheduledTask -TaskName $JobName -ErrorAction SilentlyContinue
        if (-not $task) {
            Write-Error "Task not found: $JobName"
            exit 1
        }

        Stop-ScheduledTask -TaskName $JobName -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $JobName -Confirm:$false

        # Remove wrapper script
        $wrapperPath = "$LOG_DIR\${JobName}_wrapper.ps1"
        if (Test-Path $wrapperPath) { Remove-Item $wrapperPath }

        Write-Host "Job stopped: $JobName" -ForegroundColor Yellow
    }

    "stopall" {
        $tasks = Get-ScheduledTask | Where-Object {
            $_.TaskPath -eq "\" -and
            (Test-Path "$LOG_DIR\$($_.TaskName).log")
        }

        if (-not $tasks) {
            Write-Host "No tasks to stop."
        } else {
            foreach ($t in $tasks) {
                Stop-ScheduledTask -TaskName $t.TaskName -ErrorAction SilentlyContinue
                Unregister-ScheduledTask -TaskName $t.TaskName -Confirm:$false

                $wrapperPath = "$LOG_DIR\$($t.TaskName)_wrapper.ps1"
                if (Test-Path $wrapperPath) { Remove-Item $wrapperPath }
            }
            Write-Host "All tasks stopped ($($tasks.Count))" -ForegroundColor Yellow
        }
    }
}
