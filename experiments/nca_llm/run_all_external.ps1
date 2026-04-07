# MVE-20260405-02 全8実験をバックグラウンドで順次実行
# 実行: Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File run_all_external.ps1" -WindowStyle Minimized

$ScriptDir = $PSScriptRoot
$RepoRoot = "$ScriptDir\..\.."
$LogFile = "$RepoRoot\results\nca_llm\run_all_external.log"

New-Item -ItemType Directory -Force -Path "$RepoRoot\results\nca_llm\alphanli" | Out-Null
New-Item -ItemType Directory -Force -Path "$RepoRoot\results\nca_llm\ruletaker_d1" | Out-Null

$StartTime = Get-Date
"MVE-20260405-02: Starting 8 experiments at $StartTime" | Tee-Object -FilePath $LogFile

$experiments = @(
    @{benchmark="alphanli";    condition="homo_nca";       label="[1/8]"},
    @{benchmark="alphanli";    condition="het_nca_v1";     label="[2/8]"},
    @{benchmark="alphanli";    condition="het_nca_v2";     label="[3/8]"},
    @{benchmark="alphanli";    condition="majority_vote";  label="[4/8]"; script="run_majority_vote.py"},
    @{benchmark="ruletaker_d1";condition="homo_nca";       label="[5/8]"},
    @{benchmark="ruletaker_d1";condition="het_nca_v1";     label="[6/8]"},
    @{benchmark="ruletaker_d1";condition="het_nca_v2";     label="[7/8]"},
    @{benchmark="ruletaker_d1";condition="majority_vote";  label="[8/8]"; script="run_majority_vote.py"}
)

foreach ($exp in $experiments) {
    $label     = $exp.label
    $benchmark = $exp.benchmark
    $condition = $exp.condition
    $script    = if ($exp.script) { $exp.script } else { "run_nca_external.py" }
    $start     = Get-Date

    "$label $benchmark x $condition - started at $start" |
        Tee-Object -FilePath $LogFile -Append

    if ($condition -eq "majority_vote") {
        & python "$ScriptDir\$script" --benchmark $benchmark
    } else {
        & python "$ScriptDir\$script" --benchmark $benchmark --condition $condition
    }

    $end     = Get-Date
    $elapsed = [math]::Round(($end - $start).TotalMinutes, 1)
    "$label $benchmark x $condition - completed in ${elapsed}min" |
        Tee-Object -FilePath $LogFile -Append
}

# 分析
"Running analysis..." | Tee-Object -FilePath $LogFile -Append
& python "$ScriptDir\analyze_external.py"

# コミット
Set-Location $RepoRoot
git add results/nca_llm/
git commit -m "data: run MVE-20260405-02 external benchmark verification (8 experiments)"
git push
"Committed and pushed." | Tee-Object -FilePath $LogFile -Append

$EndTime  = Get-Date
$TotalMin = [math]::Round(($EndTime - $StartTime).TotalMinutes, 1)
"All done at $EndTime (total: ${TotalMin}min)" |
    Tee-Object -FilePath $LogFile -Append
