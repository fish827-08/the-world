# run_d2x_parallel.ps1 - D2.1x local rerun (Python path, 20-core parallel, 15x60k)
# usage: powershell -File experiments\run_d2x_parallel.ps1
# R4-compliant config: use_sim_core=False AND info-asymmetry ON (radius=4/noise=0.05/tau=0.15)
#   (R4-2026-09-12: previous batch silently disabled info-asymmetry; this batch enables it explicitly)
# completion marker writes to _rerun_logs\done when all exit.
$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot            # _gitee_review
$py   = Join-Path $repo '.venv\Scripts\python.exe'
$scr  = Join-Path $PSScriptRoot 'run_d2_experiment.py'
$logD = Join-Path $repo '_rerun_logs'

# kill stale run_d2_experiment processes (avoid conflicts with this batch)
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'run_d2_experiment' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

New-Item -ItemType Directory -Force -Path $logD | Out-Null
# only remove loose files at logD root; keep subdirs (e.g. ref_sym_off reference data)
Get-ChildItem $logD -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

# ---- R4 precheck: 1-tick smoke run + read manifest to verify final config ----
# ensure use_sim_core=False and info-asym ON (perception_radius=4 / noise=0.05 / tau=0.15)
$preTag = 'r4_precheck'
$preArgs = @($scr,'--ticks','120','--tag',$preTag,
             '--learning-bottleneck','--steels-alignment','--info-asymmetry',
             '--learning-rate','0.05','--seed','42')
Start-Process -FilePath $py -ArgumentList $preArgs -WorkingDirectory $repo -Wait -NoNewWindow -PassThru | Out-Null
$pre = Join-Path $repo ("experiments/manifest_$preTag`_s42.json")
if (Test-Path $pre) {
    $j = Get-Content $pre -Raw | ConvertFrom-Json
    $d = $j.config.simulation
    $i = $j.config.info_structure
    Write-Host ("precheck: use_sim_core={0} radius={1} noise={2} tau={3}" -f $d.use_sim_core,$i.perception_radius,$i.perception_noise,$i.softmax_tau)
    if ($d.use_sim_core -ne $false -or $i.perception_radius -ne 4 -or $i.perception_noise -ne 0.05 -or $i.softmax_tau -ne 0.15) {
        throw "R4 config precheck FAILED: use_sim_core=$($d.use_sim_core) radius=$($i.perception_radius)"
    }
    Remove-Item (Join-Path $repo "experiments/long_${preTag}_s42.csv"),$pre -Force -ErrorAction SilentlyContinue
    Write-Host "R4 precheck OK"
} else {
    throw "R4 precheck FAILED: manifest not produced"
}

$lrTag = @{ '0.05' = 'd2x_lr050'; '0.1' = 'd2x_lr100'; '0.3' = 'd2x_lr300' }
$jobs = @()
foreach ($lr in 0.05, 0.10, 0.30) {
    foreach ($s in 42..46) {
        $tag = $lrTag["$lr"]
        $lout = Join-Path $logD "run_${tag}_s${s}.log"
        $lerr = Join-Path $logD "run_${tag}_s${s}.err.log"
        # R4-compliant: explicitly enable info-asymmetry + fixed perception args (no defaults reliance)
        $argsL = @($scr,'--ticks','60000','--tag',"$tag",
                   '--learning-bottleneck','--steels-alignment',
                   '--info-asymmetry',
                   '--perception-radius','4',
                   '--perception-noise','0.05',
                   '--softmax-tau','0.15',
                   '--learning-rate',("$lr"),'--seed',("$s"))
        $p = Start-Process -FilePath $py -ArgumentList $argsL -WorkingDirectory $repo `
             -RedirectStandardOutput $lout -RedirectStandardError $lerr `
             -PassThru -WindowStyle Hidden
        $jobs += [pscustomobject]@{ Pid=$p.Id; Tag=$tag; Seed=$s }
        Write-Host ("started pid={0} {1} s{2}" -f $p.Id,$tag,$s)
    }
}
$jobs | ConvertTo-Json | Set-Content (Join-Path $logD 'started.json')
Write-Host ("all {0} experiments launched (parallel). marker -> {1}\done after completion" -f $jobs.Count,$logD)