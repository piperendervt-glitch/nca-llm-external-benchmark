<#
.SYNOPSIS
    実験をPowerShellバックグラウンドジョブとして起動・管理するスクリプト。
    SSH切断後もジョブが継続して動作する。

.USAGE
    # ジョブ起動
    .\run_background.ps1 -Action start -JobName "hgnn_n200" -Script "run_nca_hgnn.py" -Args "--benchmark alphanli --n_samples 200"

    # ジョブ一覧
    .\run_background.ps1 -Action list

    # ジョブ結果確認
    .\run_background.ps1 -Action result -JobName "hgnn_n200"

    # ジョブ停止
    .\run_background.ps1 -Action stop -JobName "hgnn_n200"

    # 全ジョブ停止
    .\run_background.ps1 -Action stopall
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("start", "list", "result", "stop", "stopall")]
    [string]$Action,

    [string]$JobName = "",
    [string]$Script  = "",
    [string]$Args    = ""
)

$REPO_ROOT = "C:\Users\pipe_render\nca-llm-external-benchmark"
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

        $logFile = "$LOG_DIR\${JobName}.log"
        $scriptPath = "$SCRIPT_DIR\$Script"

        if (-not (Test-Path $scriptPath)) {
            Write-Error "スクリプトが見つかりません: $scriptPath"
            exit 1
        }

        # 既存の同名ジョブを確認
        $existing = Get-Job -Name $JobName -ErrorAction SilentlyContinue
        if ($existing) {
            Write-Warning "同名のジョブが既に存在します: $JobName (State: $($existing.State))"
            $confirm = Read-Host "上書きしますか？ (y/n)"
            if ($confirm -ne "y") { exit 0 }
            Stop-Job -Name $JobName -ErrorAction SilentlyContinue
            Remove-Job -Name $JobName -ErrorAction SilentlyContinue
        }

        $jobScript = {
            param($scriptDir, $scriptPath, $scriptArgs, $logFile)
            Set-Location $scriptDir
            $cmd = "python `"$scriptPath`" $scriptArgs"
            Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START: $cmd" |
                Tee-Object -FilePath $logFile
            Invoke-Expression $cmd 2>&1 |
                Tee-Object -FilePath $logFile -Append
            Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DONE" |
                Tee-Object -FilePath $logFile -Append
        }

        $job = Start-Job `
            -Name $JobName `
            -ScriptBlock $jobScript `
            -ArgumentList $SCRIPT_DIR, $scriptPath, $Args, $logFile

        Write-Host ""
        Write-Host "✅ ジョブを起動しました" -ForegroundColor Green
        Write-Host "   JobName : $JobName"
        Write-Host "   JobId   : $($job.Id)"
        Write-Host "   Script  : $Script $Args"
        Write-Host "   Log     : $logFile"
        Write-Host ""
        Write-Host "結果確認:"
        Write-Host "   .\run_background.ps1 -Action result -JobName `"$JobName`""
        Write-Host "ログ確認:"
        Write-Host "   Get-Content `"$logFile`" -Wait"
    }

    "list" {
        $jobs = Get-Job | Where-Object { $_.Name -notlike "Job*" -or $true }
        if (-not $jobs) {
            Write-Host "実行中のジョブはありません。"
        } else {
            Write-Host ""
            Write-Host "ジョブ一覧:" -ForegroundColor Cyan
            Write-Host ("-" * 60)
            $jobs | Format-Table -AutoSize `
                @{L="JobName"; E={$_.Name}},
                @{L="State";   E={$_.State}},
                @{L="JobId";   E={$_.Id}},
                @{L="Started"; E={$_.PSBeginTime.ToString("HH:mm:ss")}}
            Write-Host ""
            # ログファイルの存在確認
            foreach ($j in $jobs) {
                $logFile = "$LOG_DIR\$($j.Name).log"
                if (Test-Path $logFile) {
                    $lastLine = Get-Content $logFile -Tail 1
                    Write-Host "[$($j.Name)] 最終ログ: $lastLine"
                }
            }
        }
    }

    "result" {
        if (-not $JobName) { Write-Error "JobName が必要です"; exit 1 }

        $job = Get-Job -Name $JobName -ErrorAction SilentlyContinue
        if (-not $job) {
            Write-Error "ジョブが見つかりません: $JobName"
            exit 1
        }

        Write-Host ""
        Write-Host "ジョブ: $JobName (State: $($job.State))" -ForegroundColor Cyan
        Write-Host ("-" * 60)

        # ログファイルがあればそちらを表示
        $logFile = "$LOG_DIR\${JobName}.log"
        if (Test-Path $logFile) {
            Write-Host "ログファイル: $logFile"
            Write-Host ""
            Get-Content $logFile
        } else {
            # ログがなければReceive-Jobで取得
            Receive-Job -Name $JobName -Keep
        }
    }

    "stop" {
        if (-not $JobName) { Write-Error "JobName が必要です"; exit 1 }

        $job = Get-Job -Name $JobName -ErrorAction SilentlyContinue
        if (-not $job) {
            Write-Error "ジョブが見つかりません: $JobName"
            exit 1
        }

        Stop-Job   -Name $JobName
        Remove-Job -Name $JobName
        Write-Host "✅ ジョブを停止しました: $JobName" -ForegroundColor Yellow
    }

    "stopall" {
        $jobs = Get-Job
        if (-not $jobs) {
            Write-Host "停止するジョブはありません。"
        } else {
            $jobs | Stop-Job
            $jobs | Remove-Job
            Write-Host "✅ 全ジョブを停止しました ($($jobs.Count)件)" -ForegroundColor Yellow
        }
    }
}
