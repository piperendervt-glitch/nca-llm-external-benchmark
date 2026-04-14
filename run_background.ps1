<#
.SYNOPSIS
    実験をWindows タスクスケジューラで起動・管理するスクリプト。
    SSH切断後もタスクが継続して動作する。

.USAGE
    # タスク起動
    .\run_background.ps1 -Action start -JobName "hgnn_n200" -Script "run_nca_hgnn.py" -Args "--benchmark alphanli --n_samples 200"

    # タスク一覧
    .\run_background.ps1 -Action list

    # 途中経過確認（直近ログ + 進捗）
    .\run_background.ps1 -Action tail -JobName "hgnn_n200"

    # 全ログ確認
    .\run_background.ps1 -Action result -JobName "hgnn_n200"

    # タスク停止
    .\run_background.ps1 -Action stop -JobName "hgnn_n200"

    # 全タスク停止
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

# ログディレクトリ作成
if (-not (Test-Path $LOG_DIR)) {
    New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
}

switch ($Action) {

    "start" {
        if (-not $JobName) { Write-Error "JobName が必要です"; exit 1 }
        if (-not $Script)  { Write-Error "Script が必要です";  exit 1 }

        $logFile    = "$LOG_DIR\${JobName}.log"
        $scriptPath = "$SCRIPT_DIR\$Script"

        if (-not (Test-Path $scriptPath)) {
            Write-Error "スクリプトが見つかりません: $scriptPath"
            exit 1
        }

        # 既存の同名タスクを確認・削除
        $existing = Get-ScheduledTask -TaskName $JobName -ErrorAction SilentlyContinue
        if ($existing) {
            Write-Warning "同名のタスクが既に存在します: $JobName (State: $($existing.State))"
            Stop-ScheduledTask -TaskName $JobName -ErrorAction SilentlyContinue
            Unregister-ScheduledTask -TaskName $JobName -Confirm:$false
        }

        # 開始マーカー
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START: python $Script $Args" |
            Out-File -FilePath $logFile -Encoding utf8

        # ラッパースクリプトを生成（タスクスケジューラから実行）
        $wrapperPath = "$LOG_DIR\${JobName}_wrapper.ps1"
        $wrapperContent = @'
Set-Location '{0}'
python '{1}' {2} 2>&1 | Tee-Object -FilePath '{3}' -Append
$doneTime = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Add-Content -Path '{3}' -Value "[$doneTime] DONE"
'@ -f $SCRIPT_DIR, $scriptPath, $Args, $logFile
        $wrapperContent | Out-File -FilePath $wrapperPath -Encoding utf8

        # タスクスケジューラで実行（SSH切断後も継続）
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
        Write-Host "✅ タスクを起動しました" -ForegroundColor Green
        Write-Host "   JobName : $JobName"
        Write-Host "   Script  : $Script $Args"
        Write-Host "   Log     : $logFile"
        Write-Host ""
        Write-Host "途中経過の確認:"
        Write-Host "   .\run_background.ps1 -Action tail -JobName `"$JobName`""
        Write-Host "結果確認:"
        Write-Host "   .\run_background.ps1 -Action result -JobName `"$JobName`""
    }

    "list" {
        $tasks = Get-ScheduledTask | Where-Object {
            $_.TaskPath -eq "\" -and
            (Test-Path "$LOG_DIR\$($_.TaskName).log")
        }

        if (-not $tasks) {
            Write-Host "実験タスクが見つかりません。"
        } else {
            Write-Host ""
            Write-Host "実験タスク一覧:" -ForegroundColor Cyan
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
        if (-not $JobName) { Write-Error "JobName が必要です"; exit 1 }

        $logFile = "$LOG_DIR\${JobName}.log"

        if (-not (Test-Path $logFile)) {
            Write-Error "ログファイルが見つかりません: $logFile"
            exit 1
        }

        $task  = Get-ScheduledTask -TaskName $JobName -ErrorAction SilentlyContinue
        $state = if ($task) { $task.State } else { "Unknown" }

        Write-Host ""
        Write-Host "[$JobName] State: $state" -ForegroundColor Cyan
        Write-Host ("-" * 60)
        Write-Host "直近20行:" -ForegroundColor Yellow
        Get-Content $logFile -Tail 20
        Write-Host ""

        $progress = Get-Content $logFile |
            Where-Object { $_ -match "\[\d+/\d+\]" } |
            Select-Object -Last 1
        if ($progress) {
            Write-Host "最新進捗: $progress" -ForegroundColor Green
        }

        $done = Get-Content $logFile |
            Where-Object { $_ -match "DONE" } |
            Select-Object -Last 1
        if ($done) {
            Write-Host "✅ 完了: $done" -ForegroundColor Green
        }
    }

    "result" {
        if (-not $JobName) { Write-Error "JobName が必要です"; exit 1 }

        $logFile = "$LOG_DIR\${JobName}.log"

        $task  = Get-ScheduledTask -TaskName $JobName -ErrorAction SilentlyContinue
        $state = if ($task) { $task.State } else { "Unknown" }

        Write-Host ""
        Write-Host "タスク: $JobName (State: $state)" -ForegroundColor Cyan
        Write-Host ("-" * 60)

        if (Test-Path $logFile) {
            Write-Host "ログファイル: $logFile"
            Write-Host ""
            Get-Content $logFile
        } else {
            Write-Host "ログファイルが見つかりません: $logFile"
        }
    }

    "stop" {
        if (-not $JobName) { Write-Error "JobName が必要です"; exit 1 }

        $task = Get-ScheduledTask -TaskName $JobName -ErrorAction SilentlyContinue
        if (-not $task) {
            Write-Error "タスクが見つかりません: $JobName"
            exit 1
        }

        Stop-ScheduledTask -TaskName $JobName -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $JobName -Confirm:$false

        # ラッパースクリプト削除
        $wrapperPath = "$LOG_DIR\${JobName}_wrapper.ps1"
        if (Test-Path $wrapperPath) { Remove-Item $wrapperPath }

        Write-Host "✅ タスクを停止しました: $JobName" -ForegroundColor Yellow
    }

    "stopall" {
        $tasks = Get-ScheduledTask | Where-Object {
            $_.TaskPath -eq "\" -and
            (Test-Path "$LOG_DIR\$($_.TaskName).log")
        }

        if (-not $tasks) {
            Write-Host "停止するタスクはありません。"
        } else {
            foreach ($t in $tasks) {
                Stop-ScheduledTask -TaskName $t.TaskName -ErrorAction SilentlyContinue
                Unregister-ScheduledTask -TaskName $t.TaskName -Confirm:$false

                $wrapperPath = "$LOG_DIR\$($t.TaskName)_wrapper.ps1"
                if (Test-Path $wrapperPath) { Remove-Item $wrapperPath }
            }
            Write-Host "✅ 全タスクを停止しました ($($tasks.Count)件)" -ForegroundColor Yellow
        }
    }
}
