#requires -Version 5.1
<#
.SYNOPSIS
    小恐龙 WBC 一键训练（Windows PowerShell 5.1+）。

.DESCRIPTION
    默认按课程顺序训练 stand -> lift -> turn -> walk：
      * stand 从零训练；
      * 其余阶段自动用本次新训练出的 stand/model.zip 热启动；
      * 若 -Stages 里不含 stand，则自动用最新的 checkpoints/stand_* / stand 目录。
    训练产物、TensorBoard、registry 更新都由 train.py 自己完成，本脚本只负责
    串流程、解析 checkpoint、记录日志和失败即停。

.EXAMPLE
    # 默认全流程（stand -> lift -> turn -> walk）
    .\code\run_train.bat

.EXAMPLE
    # 只训 walk，用最新的 stand 模型热启动
    .\code\run_train.bat -Stages walk

.EXAMPLE
    # 自定义阶段顺序/步数/并行环境
    .\code\run_train.bat -Stages stand,lift -Envs 8 -Steps 2000000

.EXAMPLE
    # 先看会执行什么，不真正训练
    .\code\run_train.bat -DryRun
#>
[CmdletBinding()]
param(
    # 要训练的阶段（逗号分隔）；默认 stand,lift,turn,walk
    [string[]]$Stages = @("stand", "lift", "turn", "walk"),

    # 并行环境数
    [int]$Envs = 16,

    # 所有阶段统一覆盖的步数；0 = 用每阶段默认值
    [int]$Steps = 0,

    # auto / cpu / cuda
    [string]$Device = "auto",

    # 随机种子
    [int]$Seed = 0,

    # 额外加上 standup 阶段（默认不加；它最难且较慢）
    [switch]$WithStandup,

    # 训练结束后不自动更新 code/registry.json
    [switch]$NoRegister,

    # 跳过训练前的环境自检
    [switch]$NoCheck,

    # 只做环境自检，不训练
    [switch]$CheckOnly,

    # 只打印将要执行的命令，不训练也不自检
    [switch]$DryRun,

    # 指定 stand 的 model.zip（或包含 model.zip 的目录）；不给则自动找最新
    [string]$StandCheckpoint = "",

    # 可选：stand 阶段本身的 --init-from（默认从零训练 stand）
    [string]$StandInitFrom = "",

    # 可选：MuJoCo 模型路径（默认 models/dino_scene.xml）
    [string]$Model = "",

    # Python 解释器；默认项目 conda 环境
    [string]$Python = "E:\conda_envs\desktop-robot\python.exe"
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

# ----------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------
$CodeDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $CodeDir
$TrainPy = Join-Path $CodeDir "train\train.py"
$CheckpointRoot = Join-Path $CodeDir "checkpoints"

Set-Location -LiteralPath $RepoRoot

if (-not (Test-Path -LiteralPath $TrainPy)) {
    throw "找不到训练入口：$TrainPy（脚本应放在 code/ 目录下）"
}
if (-not (Test-Path -LiteralPath $Python)) {
    $candidate = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $candidate) {
        $Python = $candidate.Source
    } else {
        throw "找不到 Python：$Python。请确认 conda 环境，或用 -Python 指定解释器。"
    }
}
$Python = (Resolve-Path -LiteralPath $Python).Path

# ----------------------------------------------------------------------
# 参数校验
# ----------------------------------------------------------------------
$ValidStages = @("stand", "lift", "standup", "walk", "turn")
# 兼容三种写法：-Stages stand,lift / -Stages "stand,lift" / -Stages stand
$Stages = @(
    $Stages |
        ForEach-Object { ([string]$_).Split(",") } |
        ForEach-Object { $_.Trim().ToLower() } |
        Where-Object { $_ -ne "" } |
        Select-Object -Unique
)
foreach ($stage in $Stages) {
    if ($ValidStages -notcontains $stage) {
        throw "未知阶段 '$stage'；可选：$($ValidStages -join ', ')"
    }
}
if ($WithStandup -and ($Stages -notcontains "standup")) {
    $Stages += "standup"
}
if ($Stages.Count -eq 0) {
    throw "没有要训练的阶段（-Stages 为空）。"
}
if ($Envs -lt 1) { throw "-Envs 必须 >= 1" }
if ($Steps -lt 0) { throw "-Steps 不能为负" }

# 每阶段默认步数（与 README 的推荐量一致）
$DefaultSteps = @{
    stand   = 6000000
    lift    = 8000000
    turn    = 8000000
    standup = 8000000
    walk    = 8000000
}

# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------
function Get-RegistryStageModel {
    param([string]$Stage)
    $registry = Join-Path $CodeDir "registry.json"
    if (-not (Test-Path -LiteralPath $registry)) { return $null }
    try {
        $payload = Get-Content -LiteralPath $registry -Raw | ConvertFrom-Json
    } catch {
        return $null
    }
    $prop = $payload.PSObject.Properties[$Stage]
    if ($null -eq $prop) { return $null }
    $checkpoint = [string]$prop.Value.checkpoint
    if ([string]::IsNullOrWhiteSpace($checkpoint)) { return $null }
    $checkpointPath = Join-Path $CodeDir $checkpoint
    if (-not (Test-Path -LiteralPath $checkpointPath)) { return $null }
    $model = Join-Path (Split-Path -Parent $checkpointPath) "model.zip"
    if (-not (Test-Path -LiteralPath $model)) { return $null }
    return (Resolve-Path -LiteralPath $model).Path
}

function Get-LatestStageModel {
    param([string]$Stage)
    if (-not (Test-Path -LiteralPath $CheckpointRoot)) { return $null }
    $dirs = @(Get-ChildItem -LiteralPath $CheckpointRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq $Stage -or $_.Name -like ($Stage + "_*") } |
        Sort-Object LastWriteTime -Descending)
    foreach ($dir in $dirs) {
        $model = Join-Path $dir.FullName "model.zip"
        if (Test-Path -LiteralPath $model) { return $model }
    }
    return $null
}

function Resolve-StandModel {
    param([string]$Path)
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $item = Get-Item -LiteralPath $resolved
    if ($item.PSIsContainer) {
        $model = Join-Path $item.FullName "model.zip"
    } else {
        $model = $item.FullName
    }
    if (-not (Test-Path -LiteralPath $model)) {
        throw "找不到 stand 的 model.zip：$model"
    }
    return (Resolve-Path -LiteralPath $model).Path
}

function Format-Command {
    param([string]$Exe, [string[]]$Arguments)
    $parts = @($Exe) + @($Arguments)
    $quoted = @()
    foreach ($part in $parts) {
        if ($part -match '[\s"]') {
            $quoted += ('"' + ($part -replace '"', '\"') + '"')
        } else {
            $quoted += $part
        }
    }
    return ($quoted -join " ")
}

function Invoke-Python {
    param([string[]]$Arguments)
    # 关键：PowerShell 5.1 下 `2>&1` 会把 Python 的 stderr（例如 SB3 的
    # "PPO on GPU" UserWarning）变成 NativeCommandError；全局 Stop 会把它
    # 当成致命错误直接中断训练。这里临时改成 Continue，让 stderr 正常进日志，
    # 真正的失败仍由 $LASTEXITCODE 检查。
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python @Arguments 2>&1 | ForEach-Object {
            $line = [string]$_
            $line | Out-File -FilePath $Log -Append -Encoding utf8
            Write-Host $line
        }
    } finally {
        $ErrorActionPreference = $previous
    }
    return $LASTEXITCODE
}

# ----------------------------------------------------------------------
# 日志
# ----------------------------------------------------------------------
$RunsDir = Join-Path $CodeDir "runs"
if ($DryRun) {
    $Log = "(dry-run, no log)"
} else {
    New-Item -ItemType Directory -Force -Path $RunsDir | Out-Null
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $Log = Join-Path $RunsDir "run_train_$Stamp.log"
}

Write-Host ""
Write-Host "===== dino WBC 一键训练 =====" -ForegroundColor Cyan
Write-Host "仓库      : $RepoRoot"
Write-Host "Python    : $Python"
Write-Host "阶段      : $($Stages -join ' -> ')"
Write-Host "环境数    : $Envs"
Write-Host "设备      : $Device"
Write-Host "日志      : $Log"
Write-Host ""

# ----------------------------------------------------------------------
# 环境自检
# ----------------------------------------------------------------------
if ($DryRun) {
    Write-Host "[DryRun] 不执行训练，只打印计划命令。" -ForegroundColor Yellow
} elseif ($CheckOnly -or (-not $NoCheck)) {
    Write-Host "===== 环境自检 =====" -ForegroundColor Cyan
    foreach ($stage in $Stages) {
        $checkArgs = @($TrainPy, "--stage", $stage, "--check")
        if ($Model) { $checkArgs += @("--model", $Model) }
        Write-Host "  [check] $stage"
        $rc = Invoke-Python -Arguments $checkArgs
        if ($rc -ne 0) {
            throw "环境自检失败：$stage（exit $rc）"
        }
    }
    Write-Host ""
    if ($CheckOnly) {
        Write-Host "[done] 环境自检全部通过（-CheckOnly）。" -ForegroundColor Green
        exit 0
    }
}

# ----------------------------------------------------------------------
# 解析 stand 热启动模型
# ----------------------------------------------------------------------
$StandModel = $null
if ($StandCheckpoint) {
    $StandModel = Resolve-StandModel -Path $StandCheckpoint
    Write-Host "stand 热启动 : $StandModel"
}
if ($StandInitFrom) {
    $StandInitFrom = (Resolve-Path -LiteralPath $StandInitFrom).Path
    Write-Host "stand 初始化 : $StandInitFrom"
}
if (($Stages -notcontains "stand") -and (-not $StandModel)) {
    # 优先用 registry 里正在运行的那个 stand（和 run.py --controller stand 一致），
    # 找不到再退回"最新的 stand 训练目录"。
    $StandModel = Get-RegistryStageModel -Stage "stand"
    $source = "registry"
    if (-not $StandModel) {
        $StandModel = Get-LatestStageModel -Stage "stand"
        $source = "最新"
    }
    if (-not $StandModel) {
        throw "没有找到 stand 的 model.zip。请先训 stand，或用 -StandCheckpoint 指定。"
    }
    Write-Host "stand 热启动 : $StandModel（$source）"
}
Write-Host ""

# ----------------------------------------------------------------------
# 逐阶段训练
# ----------------------------------------------------------------------
$TotalStart = Get-Date
$Summary = @()

foreach ($stage in $Stages) {
    $stageSteps = if ($Steps -gt 0) { $Steps } else { [int]$DefaultSteps[$stage] }
    $trainArgs = @(
        $TrainPy,
        "--stage", $stage,
        "--envs", "$Envs",
        "--steps", "$stageSteps",
        "--device", $Device,
        "--seed", "$Seed"
    )
    if ($Model) { $trainArgs += @("--model", $Model) }
    if ($stage -eq "stand") {
        if ($StandInitFrom) { $trainArgs += @("--init-from", $StandInitFrom) }
    } else {
        if (-not $StandModel) {
            throw "训练 $stage 需要 stand 的 model.zip；请先训 stand，或用 -StandCheckpoint 指定。"
        }
        $trainArgs += @("--init-from", $StandModel)
    }
    if ($NoRegister) { $trainArgs += "--no-register" }

    Write-Host "===== 训练 $stage（$stageSteps 步）=====" -ForegroundColor Cyan
    Write-Host (Format-Command -Exe $Python -Arguments $trainArgs) -ForegroundColor DarkGray

    if ($DryRun) {
        if ($stage -eq "stand") {
            # 后续阶段的 dry-run 只能标记"本次新训练的 stand"
            $StandModel = "<本次新训练的 stand>/model.zip"
        }
        $Summary += [pscustomobject]@{ Stage = $stage; Model = "<dry-run>"; Seconds = 0 }
        Write-Host ""
        continue
    }

    $started = Get-Date
    $rc = Invoke-Python -Arguments $trainArgs
    $elapsed = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
    if ($rc -ne 0) {
        throw "训练 $stage 失败（exit $rc）。日志：$Log"
    }

    # 本次训练产物
    $stageModel = Get-LatestStageModel -Stage $stage
    Write-Host "[done] $stage（$elapsed s） -> $stageModel" -ForegroundColor Green
    if ($stage -eq "stand") {
        if ($stageModel) { $StandModel = $stageModel }
        if (-not $StandModel) {
            throw "stand 训练结束但找不到 model.zip（预期在 checkpoints/stand_* 下）。"
        }
    }
    $Summary += [pscustomobject]@{ Stage = $stage; Model = $stageModel; Seconds = $elapsed }
    Write-Host ""
}

# ----------------------------------------------------------------------
# 汇总
# ----------------------------------------------------------------------
$TotalSeconds = [math]::Round(((Get-Date) - $TotalStart).TotalSeconds, 1)
Write-Host "===== 全部完成（$TotalSeconds s）=====" -ForegroundColor Green
if (-not $DryRun) {
    foreach ($row in $Summary) {
        Write-Host ("  {0,-8} {1,8} s  {2}" -f $row.Stage, $row.Seconds, $row.Model)
    }
    Write-Host ""
    Write-Host "registry : $(Join-Path $CodeDir 'registry.json')"
    if ($NoRegister) {
        Write-Host "（本次使用了 -NoRegister，registry 未更新）" -ForegroundColor Yellow
    }
}
Write-Host "日志     : $Log"
Write-Host ""
Write-Host "运行示例：" -ForegroundColor Cyan
Write-Host "  & `"$Python`" code/run/run.py --controller stand"
Write-Host "  & `"$Python`" code/run/run.py --controller walk --target-x 0.10 --target-y 0.00"
exit 0
