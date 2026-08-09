param(
    [string]$RunRoot = "logs/local_hora_pipeline",
    [string[]]$Scales = @("0.8", "1.0", "1.2"),
    [switch]$CacheOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$UvCache = Join-Path $env:TEMP "unilab-uv-cache"
$CachePrefix = "robots/leap_hand/caches/ball_grasp_hora_sharpa_style_50k"
$CacheDiskPrefix = Join-Path $RepoRoot "src/unilab/assets/$CachePrefix"
$StatusFile = Join-Path $RepoRoot "$RunRoot/pipeline_status.txt"

New-Item -ItemType Directory -Force (Split-Path -Parent $StatusFile) | Out-Null

function Write-Phase([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    $line | Tee-Object -FilePath $StatusFile -Append
}

function Invoke-Uv([string[]]$UvArgs) {
    & uv run --cache-dir $UvCache @UvArgs
    if ($LASTEXITCODE -ne 0) {
        throw "uv command failed with exit code $LASTEXITCODE`: uv run $($UvArgs -join ' ')"
    }
}

function Test-Cache([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    $verifyCode = @'
import sys
import numpy as np

rows = np.load(sys.argv[1], mmap_mode='r')
if rows.ndim != 2 or rows.shape[1] != 23 or rows.shape[0] < 50_000:
    raise SystemExit(rows.shape)
print(sys.argv[1], rows.shape)
'@
    & uv run --cache-dir $UvCache python -c $verifyCode $Path
    return $LASTEXITCODE -eq 0
}

try {
    Write-Phase "pipeline_started"

    foreach ($scale in $Scales) {
        $scaleTag = if ($scale -eq "1.0") { "1" } else { $scale }
        $cachePath = "${CacheDiskPrefix}_${scaleTag}.npy"
        $logicalPath = "${CachePrefix}_${scaleTag}.npy"

        if (Test-Cache $cachePath) {
            Write-Phase "cache_skip scale=$scale path=$cachePath"
            continue
        }

        Write-Phase "cache_start scale=$scale path=$cachePath"
        Invoke-Uv @(
            "train",
            "--algo", "appo",
            "--task", "leap_inhand_ball_grasp_allegro",
            "--sim", "mujoco",
            "training.no_play=true",
            "training.device=cpu",
            "env.object_scale=$scale",
            "env.grasp_cache_path=$logicalPath"
        )
        if (-not (Test-Cache $cachePath)) {
            throw "cache validation failed for scale=$scale path=$cachePath"
        }
        Write-Phase "cache_complete scale=$scale path=$cachePath"
    }

    if ($CacheOnly) {
        Write-Phase "cache_only_complete scales=$($Scales -join ',')"
        return
    }

    Write-Phase "tests_start"
    Invoke-Uv @("pytest", "tests/envs/test_leap_inhand_hora_appo.py", "-q")
    Write-Phase "tests_complete"

    Write-Phase "smoke_start"
    Invoke-Uv @(
        "train",
        "--algo", "appo",
        "--task", "leap_inhand_ball_0730",
        "--sim", "mujoco",
        "--profile", "hora",
        "algo.seed=1",
        "algo.num_envs=64",
        "algo.steps_per_env=8",
        "algo.max_iterations=2",
        "algo.save_interval=1",
        "training.no_play=true",
        "training.device=cpu",
        "training.collector_device=cpu",
        "training.log_root=$RunRoot/smoke"
    )
    Write-Phase "smoke_complete"

    Write-Phase "long_training_start envs=4096 steps_per_env=32 iterations=5000"
    Invoke-Uv @(
        "train",
        "--algo", "appo",
        "--task", "leap_inhand_ball_0730",
        "--sim", "mujoco",
        "--profile", "hora",
        "algo.seed=1",
        "algo.num_envs=4096",
        "algo.steps_per_env=32",
        "algo.max_iterations=5000",
        "algo.save_interval=25",
        "algo.actor.distribution_cfg.init_std=1.0",
        "training.no_play=true",
        "training.device=cpu",
        "training.collector_device=cpu",
        "training.log_root=$RunRoot/training"
    )
    Write-Phase "long_training_complete"
}
catch {
    Write-Phase "pipeline_failed error=$($_.Exception.Message)"
    throw
}
