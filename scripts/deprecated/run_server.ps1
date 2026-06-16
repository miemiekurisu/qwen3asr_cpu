<#
.SYNOPSIS
    Qwen3-ASR Server 一键启动脚本 (Windows)
    模仿 tools/run_linux_server.sh

.DESCRIPTION
    支持前台/后台启动、停止、健康检查、HTTPS 反代。

.ENVIRONMENT
    QASR_MODEL_DIR              [必填] 模型目录 (含 model.safetensors)
    QASR_REALTIME_MODEL_DIR     [可选] realtime 模型 (默认共享内存)
    QASR_HOST                   [可选] 监听地址 (默认 0.0.0.0)
    QASR_PORT                   [可选] 监听端口 (默认 19991)
    QASR_HTTPS_PORT             [可选] HTTPS 反代端口 (默认 19992)
    QASR_THREADS                [可选] 推理线程数 (默认 0 = 自动)
    QASR_VERBOSITY              [可选] 日志级别 0-3 (默认 0)
    QASR_VAD_MODEL              [可选] Silero VAD ONNX 模型路径
    QASR_UI_DIR                 [可选] UI 目录 (默认 $PROJECT_ROOT/ui)
    QASR_BUILD_DIR              [可选] 编译输出 (默认 $PROJECT_ROOT/build/Release)
    QASR_PID_FILE               [可选] PID 文件 (默认 %TEMP%/qasr_server.pid)
    QASR_LOG_FILE               [可选] 日志文件 (默认 %TEMP%/qasr_server.log)
    QASR_PROXY_PID              [可选] proxy PID 文件 (默认 %TEMP%/qasr_proxy.pid)
    QASR_PROXY_LOG              [可选] proxy 日志 (默认 %TEMP%/qasr_proxy.log)
    QASR_TLS_CERT_DIR           [可选] 持久 cert 目录 (默认 mktemp, 退出时删)

.PARAMETER Detach
    后台启动

.PARAMETER Stop
    停止后台服务器

.PARAMETER Status
    健康检查

.PARAMETER Https
    同时启动 HTTPS 反代 (浏览器 mic 权限需要 https)

.PARAMETER Verbose
    覆盖 QASR_VERBOSITY=3

.EXAMPLE
    $env:QASR_MODEL_DIR="D:\models\Qwen3-ASR-0___6B"
    .\tools\run_server.bat --detach --https
    .\tools\run_server.bat --status
    .\tools\run_server.bat --stop
#>
param(
    [switch]$Detach,
    [switch]$Stop,
    [switch]$DoHealthCheck,
    [switch]$UseHttps,
    [switch]$HttpsInfo,
    [switch]$Verb,
    [switch]$Help
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# -- 颜色 --
if ($host.UI.SupportsVirtualTerminal) {
    $C_RED = "`e[31m"; $C_GRN = "`e[32m"; $C_YEL = "`e[33m"
    $C_BLU = "`e[34m"; $C_BLD = "`e[1m";   $C_RST = "`e[0m"
} else {
    $C_RED = ""; $C_GRN = ""; $C_YEL = ""
    $C_BLU = ""; $C_BLD = ""; $C_RST = ""
}

function Log-Info  { Write-Host "${C_BLU}[INFO]${C_RST}  $args" -ForegroundColor Blue }
function Log-Ok    { Write-Host "${C_GRN}[OK]${C_RST}    $args" -ForegroundColor Green }
function Log-Warn  { Write-Host "${C_YEL}[WARN]${C_RST}  $args" -ForegroundColor Yellow }
function Log-Err   { Write-Host "${C_RED}[ERROR]${C_RST} $args" -ForegroundColor Red }
function Log-Step  { Write-Host "`n${C_BLD}${C_BLU}-- $args --${C_RST}" }

# -- 路径 --
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = if ($env:QASR_PROJECT_ROOT) {
    $env:QASR_PROJECT_ROOT
} else {
    Split-Path $scriptRoot -Parent
}
if (-not (Test-Path $projectRoot)) {
    Log-Err "项目根不存在: $projectRoot"
    exit 1
}

# -- 默认值 --
$modelDir        = if ($env:QASR_MODEL_DIR) { $env:QASR_MODEL_DIR.Trim() } else { "" }
$realtimeModelDir = if ($env:QASR_REALTIME_MODEL_DIR) { $env:QASR_REALTIME_MODEL_DIR } else { "" }
$hostAddr        = if ($env:QASR_HOST) { $env:QASR_HOST } else { "0.0.0.0" }
$port            = if ($env:QASR_PORT) { [int]$env:QASR_PORT } else { 19991 }
$httpsPort       = if ($env:QASR_HTTPS_PORT) { [int]$env:QASR_HTTPS_PORT } else { 19992 }
$uiDir           = if ($env:QASR_UI_DIR) { $env:QASR_UI_DIR } else { Join-Path $projectRoot "ui" }
$threads         = if ($env:QASR_THREADS) { $env:QASR_THREADS } else { "0" }
$verbosity       = if ($Verb) { 3 } elseif ($env:QASR_VERBOSITY) { [int]$env:QASR_VERBOSITY } else { 0 }
$vadModel        = if ($env:QASR_VAD_MODEL) { $env:QASR_VAD_MODEL } else { Join-Path $projectRoot "models/silero_vad/silero_vad.onnx" }
$buildDir        = if ($env:QASR_BUILD_DIR) { $env:QASR_BUILD_DIR } else { Join-Path $projectRoot "build\Release" }
$pidFile         = if ($env:QASR_PID_FILE) { $env:QASR_PID_FILE } else { Join-Path $env:TEMP "qasr_server.pid" }
$logFile         = if ($env:QASR_LOG_FILE) { $env:QASR_LOG_FILE } else { Join-Path $env:TEMP "qasr_server.log" }
$proxyPidFile    = if ($env:QASR_PROXY_PID) { $env:QASR_PROXY_PID } else { Join-Path $env:TEMP "qasr_proxy.pid" }
$proxyLogFile    = if ($env:QASR_PROXY_LOG) { $env:QASR_PROXY_LOG } else { Join-Path $env:TEMP "qasr_proxy.log" }
$proxyScript     = Join-Path $scriptRoot "https_proxy.py"
$tlsCertDir      = if ($env:QASR_TLS_CERT_DIR) { $env:QASR_TLS_CERT_DIR } else { "" }

# -- Help --
if ($Help) {
    Write-Host 'Usage: run_server.bat [flags]

Flags:
  --detach            后台启动 qasr_server, 写 PID 到 PID_FILE
  --https             同时启动 HTTPS 反代 (推荐, 浏览器 mic 权限需要 https)
                      默认临时 cert (退出时删), 想持久: 设 QASR_TLS_CERT_DIR
  --stop              停掉 --detach 起的 server (和 proxy, 若有)
  --status            查 /api/health (HTTP + HTTPS, 若后者在跑)
  --https-info        显示 cert 目录 / proxy 状态
  --verbose           覆盖 QASR_VERBOSITY=3
  -h, --help          打印本帮助

Env vars (必填):
  QASR_MODEL_DIR      Qwen3-ASR 模型目录
                      例: set QASR_MODEL_DIR=D:\models\Qwen3-ASR-0___6B

Env vars (可选):
  QASR_REALTIME_MODEL_DIR  realtime 模型 (默认共享内存)
                           推荐: 1.7B batch + 0.6B realtime
  QASR_HOST=0.0.0.0   QASR_PORT=19991   QASR_HTTPS_PORT=19992
  QASR_THREADS=0      (0=auto)
  QASR_VERBOSITY=0    (0=silent, 1=commit, 2=per-poll, 3=raw)
  QASR_VAD_MODEL=...  (默认 $PROJECT_ROOT/models/silero_vad/silero_vad.onnx)
  QASR_UI_DIR=...     (默认 $PROJECT_ROOT/ui)
  QASR_LOG_FILE=...   (默认 %TEMP%/qasr_server.log)
  QASR_PID_FILE=...   (默认 %TEMP%/qasr_server.pid)
  QASR_PROXY_LOG=...  (默认 %TEMP%/qasr_proxy.log)
  QASR_PROXY_PID=...  (默认 %TEMP%/qasr_proxy.pid)
  QASR_TLS_CERT_DIR=... 持久 cert 目录 (默认临时目录, 退出时删)

HTTPS 方案对比:
  --https (本工具)   Python 反代 + 自签 cert, 0 配置
  A. qasr_server 自带 TLS        待 mbedTLS 集成
  C. Caddy / nginx 反代         需系统装, 适合生产'
    exit 0
}

# -- 查找 OpenBLAS DLL --
function Find-OpenBlasBin {
    if ($env:OPENBLAS_DIR) {
        $parent = Split-Path (Split-Path (Split-Path $env:OPENBLAS_DIR -Parent) -Parent) -Parent
        $bin = Join-Path $parent "bin"
        if (Test-Path (Join-Path $bin "libopenblas.dll")) { return $bin }
    }

    $candidates = @(
        "D:\dev\OpenBLAS\bin",
        "C:\OpenBLAS\bin",
        "$env:ProgramFiles\OpenBLAS\bin"
    )
    foreach ($dir in $candidates) {
        if (Test-Path (Join-Path $dir "libopenblas.dll")) { return $dir }
    }

    throw "未找到 OpenBLAS DLL, 请安装 OpenBLAS 或设置 OPENBLAS_DIR"
}

# -- 自动检测模型 --
function Find-ModelDir {
    param([string]$Hint)

    if ($Hint -and (Test-Path $Hint)) {
        return (Resolve-Path $Hint).Path
    }

    $candidates = @(
        "Qwen3-ASR-0___6B", "Qwen3-ASR-0.6B",
        "Qwen3-ASR-1___7B", "Qwen3-ASR-1.7B"
    )
    foreach ($name in $candidates) {
        $path = Join-Path $projectRoot $name
        if (Test-Path $path) { return $path }
    }

    foreach ($cacheRoot in @(
        (Join-Path $env:USERPROFILE ".cache\modelscope\hub\models\Qwen"),
        (Join-Path $env:USERPROFILE ".cache\huggingface\hub\models\Qwen")
    )) {
        if (Test-Path $cacheRoot) {
            foreach ($name in $candidates) {
                $path = Join-Path $cacheRoot $name
                if (Test-Path $path) { return $path }
            }
        }
    }

    throw "未找到模型目录, 请设置 QASR_MODEL_DIR"
}

# -- 校验模型目录 --
function Test-ModelDir {
    param([string]$Path)
    if (Test-Path (Join-Path $Path "model.safetensors")) { return $true }
    if (Test-Path (Join-Path $Path "model-00001-of-00002.safetensors")) { return $true }
    if (Test-Path (Join-Path $Path "model-00001-of-00003.safetensors")) { return $true }
    return $false
}

# -- 查找可执行文件 --
function Find-ServerExe {
    $exe = Join-Path $buildDir "qasr_server.exe"
    if (Test-Path $exe) { return $exe }

    foreach ($dir in @("build\Release", "build\windows-openblas")) {
        $exe = Join-Path $projectRoot "$dir\qasr_server.exe"
        if (Test-Path $exe) { return $exe }
    }

    throw "未找到 qasr_server.exe, 请先运行 start.bat 编译"
}

# -- 端口检查 --
function Check-PortFree {
    param([int]$Port, [string]$Label)

    $tcp = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($tcp) {
        $procName = (Get-Process -Id $tcp.OwningProcess -ErrorAction SilentlyContinue).Name
        Log-Err "$Label 端口 $Port 已被占用: $procName"
        Log-Err "  解决: 运行 .\tools\run_server.bat --stop 或换端口 QASR_PORT=19993"
        return $false
    }
    return $true
}

# -- 停止 --
function Stop-PidFile {
    param([string]$Label, [string]$PidFile)

    if (-not (Test-Path $PidFile)) { return }

    $procId = (Get-Content $PidFile 2>$null | ForEach-Object { [int]($_.Trim()) })
    if (-not $procId) {
        Remove-Item $PidFile -ErrorAction SilentlyContinue
        return
    }

    $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
    if (-not $proc) {
        Log-Info "进程 $procId 不存在"
        Remove-Item $PidFile -ErrorAction SilentlyContinue
        return
    }

    Log-Info "停 $Label PID $procId"
    Stop-Process -Id $procId -ErrorAction SilentlyContinue

    for ($i = 1; $i -le 10; $i++) {
        Start-Sleep -Milliseconds 500
        $alive = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if (-not $alive) { break }
    }

    if ($alive) {
        Log-Warn "强杀 PID $procId"
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }

    Remove-Item $PidFile -ErrorAction SilentlyContinue
    Log-Ok "  $Label 已停"
}

function Do-Stop {
    Log-Step "停止"
    Stop-PidFile "proxy"  $proxyPidFile
    Stop-PidFile "server" $pidFile

    # 兜底: 强制杀掉残留进程
    Get-Process -Name "qasr_server" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}

# -- 状态 --
function Do-Status {
    $url = "http://127.0.0.1:${port}/api/health"
    $ok = $false
    for ($i = 1; $i -le 3; $i++) {
        try {
            $r = Invoke-RestMethod -Uri $url -TimeoutSec 3 -ErrorAction Stop
            Log-Ok "HTTP  $url  status=$($r.status)"
            $ok = $true
            break
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    if (-not $ok) {
        Log-Err "HTTP  $url  server 没起?"
    }

    # HTTPS 状态 (如果 proxy 在跑)
    if (Test-Path $proxyPidFile) {
        $proxyPid = (Get-Content $proxyPidFile 2>$null | ForEach-Object { [int]($_.Trim()) })
        if ($proxyPid -and (Get-Process -Id $proxyPid -ErrorAction SilentlyContinue)) {
            $httpsUrl = "https://127.0.0.1:${httpsPort}/api/health"
            $httpsOk = $false
            [System.Net.ServicePointManager]::ServerCertificateValidationCallback = { $true }
            for ($i = 1; $i -le 3; $i++) {
                try {
                    $req = [System.Net.HttpWebRequest]::Create($httpsUrl)
                    $req.Timeout = 3000
                    $resp = $req.GetResponse()
                    $sr = New-Object System.IO.StreamReader($resp.GetResponseStream())
                    $body = $sr.ReadToEnd()
                    $sr.Close()
                    $resp.Close()
                    $json = $body | ConvertFrom-Json
                    Log-Ok "HTTPS $httpsUrl  status=$($json.status)"
                    $httpsOk = $true
                    break
                } catch {
                    Start-Sleep -Seconds 1
                }
            }
            if (-not $httpsOk) {
                Log-Err "HTTPS $httpsUrl  proxy 没起来?"
            }
        }
    }
}

# -- HTTPS info --
function Do-HttpsInfo {
    if (-not (Test-Path $proxyPidFile)) {
        Log-Info "proxy 没在跑 (PID 文件不存在: $proxyPidFile)"
        return
    }

    $procId = (Get-Content $proxyPidFile 2>$null | ForEach-Object { [int]($_.Trim()) })
    if (-not $procId -or -not (Get-Process -Id $procId -ErrorAction SilentlyContinue)) {
        Log-Warn "proxy PID 文件过期"
        return
    }

    Log-Ok "proxy PID $procId, log=$proxyLogFile"

    # 查找 cert 目录
    $certDir = ""
    if (Test-Path $proxyLogFile) {
        $logContent = Get-Content $proxyLogFile -Raw
        if ($logContent -match 'ephemeral cert dir:\s*(\S+)') {
            $certDir = $Matches[1]
        }
    }

    if ($certDir -and (Test-Path $certDir)) {
        Log-Ok "cert 目录 (ephemeral): $certDir"
        Get-ChildItem $certDir | ForEach-Object { Log-Info "  $($_.Name)" }
    } elseif ($tlsCertDir -and (Test-Path $tlsCertDir)) {
        Log-Ok "cert 目录 (persistent): $tlsCertDir"
        Get-ChildItem $tlsCertDir | ForEach-Object { Log-Info "  $($_.Name)" }
    } else {
        Log-Info "cert 目录没找到"
        Log-Info "  看 log: Get-Content $proxyLogFile"
    }
}

# -- 启动 --
function Do-Start {
    Log-Step "参数校验"

    # Model
    if (-not $modelDir) {
        Log-Err "未设置 QASR_MODEL_DIR"
        Log-Err ""
        Log-Err "请先下载模型, 然后:"
        Log-Err "  set QASR_MODEL_DIR=D:\path\to\Qwen3-ASR-0___6B"
        exit 1
    }
    if (-not (Test-ModelDir -Path $modelDir)) {
        Log-Err "QASR_MODEL_DIR=$modelDir 不含 model.safetensors"
        exit 1
    }
    Log-Ok "QASR_MODEL_DIR=$modelDir"

    # Realtime model
    if ($realtimeModelDir) {
        if (-not (Test-Path $realtimeModelDir)) {
            Log-Err "QASR_REALTIME_MODEL_DIR=$realtimeModelDir 目录不存在"
            exit 1
        }
        if (-not (Test-ModelDir -Path $realtimeModelDir)) {
            Log-Err "QASR_REALTIME_MODEL_DIR=$realtimeModelDir 不含 model.safetensors"
            exit 1
        }
        Log-Ok "QASR_REALTIME_MODEL_DIR=$realtimeModelDir"
    } else {
        Log-Ok "QASR_REALTIME_MODEL_DIR=<unset>  (realtime 与 batch 共享内存)"
    }

    $serverExe = Find-ServerExe
    Log-Ok "binary: $serverExe"

    $openBlasBin = Find-OpenBlasBin
    Log-Ok "OpenBLAS: $openBlasBin"

    # VAD
    if (-not (Test-Path $vadModel)) {
        Log-Warn "Silero VAD 模型不存在: $vadModel"
        Log-Warn "  实时 VAD 段式将退化为 40s 强制切段, 但仍可用"
        $vadModel = ""
    } else {
        Log-Ok "VAD: $vadModel"
    }

    # HTTPS proxy
    if ($UseHttps) {
        if (-not (Test-Path $proxyScript)) {
            Log-Err "找不到 $proxyScript (--https 需要)"
            exit 1
        }
        $pythonCmd = Get-Command "python" -ErrorAction SilentlyContinue
        if (-not $pythonCmd) {
            $pythonCmd = Get-Command "python3" -ErrorAction SilentlyContinue
        }
        if (-not $pythonCmd) {
            Log-Err "--https 需要 python, 但 PATH 里没找到"
            exit 1
        }
        Log-Ok "HTTPS: enabled ($($pythonCmd.Name) proxy=$proxyScript, port=$httpsPort)"
    }

    Log-Step "启动参数"
    Log-Info "binary:  $serverExe"
    Log-Info "model:   $modelDir"
    if ($realtimeModelDir) {
        Log-Info "realtime-model: $realtimeModelDir  (2 个独立实例, 内存吃紧)"
    } else {
        Log-Info "realtime-model: <shared with batch>     (0 额外内存)"
    }
    Log-Info "ui:      $uiDir"
    Log-Info "host:    $hostAddr"
    Log-Info "port:    $port  (HTTP)"
    if ($UseHttps) { Log-Info "         $httpsPort  (HTTPS, --https)" }
    Log-Info "threads: $threads (0=auto)"
    Log-Info "verbose: $verbosity"

    if (-not (Check-PortFree -Port $port -Label "server")) {
        exit 1
    }
    if ($UseHttps -and -not (Check-PortFree -Port $httpsPort -Label "proxy")) {
        exit 1
    }

    # 构建启动参数
    $startArgs = @(
        "--model-dir", "`"$modelDir`""
    )
    if ($realtimeModelDir) {
        $startArgs += "--realtime-model-dir", "`"$realtimeModelDir`""
    }
    $startArgs += "--ui-dir", "`"$uiDir`""
    $startArgs += "--host", $hostAddr
    $startArgs += "--port", $port
    $startArgs += "--threads", $threads
    $startArgs += "--verbosity", $verbosity

    if ($Detach) {
        Log-Step "后台启动 server (日志 $logFile, PID $pidFile)"

        $env:PATH = "$openBlasBin;$env:PATH"
        if ($vadModel) { $env:QWEN_SILERO_VAD_MODEL = $vadModel }

        # Use cmd.exe to redirect stdout+stderr to log file (mirrors nohup on Linux)
        $serverCmd = '"' + $serverExe + '"' + ' ' + (($startArgs | ForEach-Object { '"' + $_ + '"' }) -join " ")
        $batFile = [System.IO.Path]::GetTempFileName() + ".bat"
        Set-Content -Path $batFile -Value ('@echo off' + "`r`n" + "$serverCmd >`"$logFile`" 2>&1") -Encoding ASCII
        $startInfo = New-Object System.Diagnostics.ProcessStartInfo
        $startInfo.FileName = $batFile
        $startInfo.WorkingDirectory = $projectRoot
        $startInfo.CreateNoWindow = $true
        $startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
        $startInfo.UseShellExecute = $false
        $serverProc = [System.Diagnostics.Process]::Start($startInfo)

        # Wait a moment then find the actual server process by name
        Start-Sleep -Milliseconds 500
        $actualProc = Get-Process -Name "qasr_server" -ErrorAction SilentlyContinue | Sort-Object StartTime -Descending | Select-Object -First 1
        if ($actualProc) {
            $actualProc.Id | Out-File $pidFile
            Log-Ok "已起, PID $($actualProc.Id)"
        } else {
            Log-Err "qasr_server 未启动, 看日志: $logFile"
            Get-Content $logFile -Tail 20 2>$null | ForEach-Object { Write-Host $_ -ForegroundColor Red }
            exit 1
        }

        # HTTPS proxy
        if ($UseHttps) {
            Log-Step "后台启动 HTTPS proxy (日志 $proxyLogFile, PID $proxyPidFile)"

            $pythonCmd = Get-Command "python" -ErrorAction SilentlyContinue
            if (-not $pythonCmd) {
                $pythonCmd = Get-Command "python3" -ErrorAction SilentlyContinue
            }
            $proxyArgs = @(
                "--bind-host", $hostAddr,
                "--bind-port", $httpsPort,
                "--upstream", "http://127.0.0.1:$port"
            )
            if ($tlsCertDir) {
                if (-not (Test-Path $tlsCertDir)) {
                    New-Item -ItemType Directory -Path $tlsCertDir -Force | Out-Null
                }
                $proxyArgs += "--cert-dir", "`"$tlsCertDir`"", "--reuse-cert"
            }

            $proxyCmd = "$($pythonCmd.Source) $proxyScript $($proxyArgs -join " ")"
            $proxyBat = [System.IO.Path]::GetTempFileName() + ".bat"
            Set-Content -Path $proxyBat -Value ('@echo off' + "`r`n" + "$proxyCmd >`"$proxyLogFile`" 2>&1") -Encoding ASCII
            $proxyStart = New-Object System.Diagnostics.ProcessStartInfo
            $proxyStart.FileName = $proxyBat
            $proxyStart.WorkingDirectory = $projectRoot
            $proxyStart.CreateNoWindow = $true
            $proxyStart.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
            $proxyStart.UseShellExecute = $false
            [System.Diagnostics.Process]::Start($proxyStart)

            # Wait and find the python proxy process
            Start-Sleep -Milliseconds 500
            $proxyProcs = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match "https_proxy" } | Sort-Object StartTime -Descending | Select-Object -First 1
            if ($proxyProcs) {
                $proxyProcs.Id | Out-File $proxyPidFile
                Log-Ok "proxy PID $($proxyProcs.Id)"
            } else {
                Log-Warn "proxy 进程未找到, 但可能仍在启动中"
            }

            Start-Sleep -Seconds 2
            $proxyAlive = $proxyProcs | Get-Process -ErrorAction SilentlyContinue
            if (-not $proxyAlive) {
                Log-Err "proxy 启动后挂掉, 看日志: $proxyLogFile"
                Get-Content $proxyLogFile -Tail 10 2>$null | ForEach-Object { Write-Host $_ -ForegroundColor Red }
                Stop-Process -Id $actualProc.Id -ErrorAction SilentlyContinue
                Remove-Item $pidFile, $proxyPidFile -ErrorAction SilentlyContinue
                exit 1
            }
        }

        Log-Info "等几秒让模型加载..."
        Start-Sleep -Seconds 3
        Do-Status
        Do-Status

        Write-Host ""
        Log-Info "停止: .\tools\run_server.bat --stop"
        Log-Info "状态: .\tools\run_server.bat --status"
        if ($UseHttps) {
            Log-Info "HTTPS URL: https://<lan-ip>:$httpsPort/"
            Log-Info "  浏览器首次警告选'高级→继续'即可"
        } else {
            Log-Info "访问: http://<ip>:$port/"
        }
    } else {
        # 前台 + (可选) proxy 后台
        if ($UseHttps) {
            Log-Step "后台启动 HTTPS proxy (前台 Ctrl+C 时它继续跑, 用 --stop 杀)"

            $pythonCmd = Get-Command "python" -ErrorAction SilentlyContinue
            if (-not $pythonCmd) {
                $pythonCmd = Get-Command "python3" -ErrorAction SilentlyContinue
            }
            $proxyArgs = @(
                "--bind-host", $hostAddr,
                "--bind-port", $httpsPort,
                "--upstream", "http://127.0.0.1:$port"
            )
            if ($tlsCertDir) {
                if (-not (Test-Path $tlsCertDir)) {
                    New-Item -ItemType Directory -Path $tlsCertDir -Force | Out-Null
                }
                $proxyArgs += "--cert-dir", "`"$tlsCertDir`"", "--reuse-cert"
            }

            $proxyCmd = "$($pythonCmd.Source) $proxyScript $($proxyArgs -join " ")"
            $proxyStart = New-Object System.Diagnostics.ProcessStartInfo
            $proxyStart.FileName = "cmd.exe"
            $proxyStart.Arguments = "/c start /min cmd /c $proxyCmd >`"$proxyLogFile`" 2>&1"
            $proxyStart.WorkingDirectory = $projectRoot
            $proxyStart.CreateNoWindow = $true
            $proxyStart.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
            $proxyStart.UseShellExecute = $false
            [System.Diagnostics.Process]::Start($proxyStart)

            Start-Sleep -Milliseconds 500
            $proxyProcs = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match "https_proxy" } | Sort-Object StartTime -Descending | Select-Object -First 1
            if ($proxyProcs) {
                $proxyProcs.Id | Out-File $proxyPidFile
                Log-Ok "proxy PID $($proxyProcs.Id), 端口 $httpsPort"
            }
            Start-Sleep -Seconds 2
        }

        Log-Step "前台启动 server (Ctrl+C 退出)"
        Write-Host ""

        $env:PATH = "$openBlasBin;$env:PATH"
        if ($vadModel) { $env:QWEN_SILERO_VAD_MODEL = $vadModel }
        Push-Location $projectRoot
        try {
            & $serverExe @startArgs
            exit $LASTEXITCODE
        } catch {
            Log-Err "服务器异常退出: $_"
            exit 1
        } finally {
            Pop-Location
        }
    }
}

# -- 主流程 --
if ($Stop) {
    Do-Stop
    exit 0
}

if ($DoHealthCheck) {
    Do-Status
    exit 0
}

if ($HttpsInfo) {
    Do-HttpsInfo
    exit 0
}

Do-Start
