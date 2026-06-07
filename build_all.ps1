<#
.SYNOPSIS
    Qwen3-ASR one-key build script (Windows + MSVC + OpenBLAS)

.DESCRIPTION
    Workflow (mirrors tools/build_linux.sh):
      1. Check OS (Windows 10/11)
      2. Check toolchain (MSVC, cmake, git)
      3. Detect OpenBLAS (env var / known paths / prompt download)
      4. Detect model (HF cache / $QASR_MODEL_DIR / ./models/...)
      5. Detect test audio (testfile/*.wav)
      6. Clean -> cmake -> build -> ctest (optional)
      7. Build summary + launch commands

.PARAMETER Incremental
    Incremental build, do not delete build/

.PARAMETER CleanOnly
    Only clean build/, do not compile

.PARAMETER NoTest
    Skip ctest

.PARAMETER NoDep
    Skip OpenBLAS dependency check

.PARAMETER NoModel
    Skip model detection

.PARAMETER NoAudio
    Skip audio detection

.PARAMETER ModelDir
    Override model directory (overrides $QASR_MODEL_DIR)

.PARAMETER DepsDir
    OpenBLAS root directory

.PARAMETER BuildDir
    Build output directory (default: build)

.PARAMETER Jobs
    Parallel jobs (default: $env:NUMBER_OF_PROCESSORS)

.PARAMETER Help
    Show help

.EXAMPLE
    .\build_all.ps1
    .\build_all.ps1 -Incremental -NoTest
    .\build_all.ps1 -ModelDir "D:\models\Qwen3-ASR-0.6B"
#>
param(
    [switch]$Incremental,
    [switch]$CleanOnly,
    [switch]$NoTest,
    [switch]$NoDep,
    [switch]$NoModel,
    [switch]$NoAudio,
    [string]$ModelDir,
    [string]$DepsDir,
    [string]$BuildDir,
    [int]$Jobs,
    [switch]$Help
)

# ============================================================
# Logging helpers
# ============================================================
function Log-Info  { Write-Host "[INFO]  " -NoNewline -ForegroundColor Cyan;   Write-Host $args }
function Log-Ok    { Write-Host "[OK]    " -NoNewline -ForegroundColor Green;  Write-Host $args }
function Log-Warn  { Write-Host "[WARN]  " -NoNewline -ForegroundColor Yellow; Write-Host $args -ForegroundColor Yellow }
function Log-Err   { Write-Host "[ERROR] " -NoNewline -ForegroundColor Red;    Write-Host $args -ForegroundColor Red }
function Log-Step  { Write-Host ""; Write-Host ("-- " + ($args -join " ") + " --") -ForegroundColor Cyan }
function Log-Cmd   { Write-Host ("  " + ($args -join " ")) -ForegroundColor DarkGray }

if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path -Full
    exit 0
}

# ============================================================
# Paths and defaults
# ============================================================
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ScriptRoot
try {
    $ProjectRoot = (Get-Item $ScriptRoot).FullName

    # Environment variable overrides
    if ($env:QASR_DEPS_DIR)        { $DepsDir = $env:QASR_DEPS_DIR }
    if ($env:QASR_BUILD_DIR)       { $BuildDir = $env:QASR_BUILD_DIR }
    if ($env:QASR_MODEL_DIR -and (-not $ModelDir)) { $ModelDir = $env:QASR_MODEL_DIR }
    if ($env:QASR_JOBS)            { $Jobs = [int]$env:QASR_JOBS }

    $HfRepo  = if ($env:QASR_HF_REPO)  { $env:QASR_HF_REPO } else { "Qwen/Qwen3-ASR-0.6B" }
    $HfCache = if ($env:QASR_HF_CACHE) { $env:QASR_HF_CACHE } else { "$env:USERPROFILE\.cache\huggingface" }

    # Defaults
    if (-not $BuildDir) { $BuildDir = "build" }
    if (-not $Jobs)     { $Jobs = [int]$env:NUMBER_OF_PROCESSORS }
    if (-not $DepsDir)  { $DepsDir = "D:\dev\OpenBLAS" }

    # Relative -> absolute
    if (-not ([System.IO.Path]::IsPathRooted($BuildDir))) {
        $BuildDir = Join-Path $ProjectRoot $BuildDir
    }
    if (-not ([System.IO.Path]::IsPathRooted($DepsDir))) {
        $DepsDir = Join-Path $ProjectRoot $DepsDir
    }

    # Detection state
    $DetectedOpenBlasDir = $null
    $DetectedModelDir    = $null
    $DetectedAudio       = $null
    $DetectedMsvcVer     = $null

    $RepoBase = [System.IO.Path]::GetFileName($HfRepo)

    Write-Host "======================================" -ForegroundColor Cyan
    Write-Host "  Qwen3-ASR Windows One-Key Build" -ForegroundColor Cyan
    Write-Host "======================================" -ForegroundColor Cyan
    Write-Host ""
    Log-Info "Project: $ProjectRoot"
    Log-Info "Build:   $BuildDir   Jobs: $Jobs"
    Log-Info "OpenBLAS search: $DepsDir"
    Write-Host ""

    # ============================================================
    # 1. OS check
    # ============================================================
    Log-Step "System check"
    $WinVer = [System.Environment]::OSVersion.Version
    if ($WinVer.Major -lt 10) {
        Log-Err "Windows 10+ required. Current: $([System.Environment]::OSVersion.VersionString)"
        exit 1
    }
    $Arch = [System.Environment]::GetEnvironmentVariable("PROCESSOR_ARCHITECTURE", "Machine")
    Log-Ok "OS: $([System.Environment]::OSVersion.VersionString)  Arch: $Arch"

    # ============================================================
    # 2. Toolchain check
    # ============================================================
    Log-Step "Toolchain check"

    function Find-Exe {
        param([string]$Name)
        $cmd = Get-Command $Name -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
        return $null
    }

    # Find MSVC via vswhere
    $VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    $MsvcOk = $false
    $ClExe  = $null

    if (Test-Path $VsWhere) {
        $VsPath = (& $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
        $VsVer  = (& $VsWhere -latest -products * -property installationVersion).Trim()

        if ($VsPath) {
            # Inject MSVC env via temp bat file
            $VsDevCmd = Join-Path $VsPath "Common7\Tools\VsDevCmd.bat"
            $TmpBat = [System.IO.Path]::GetTempFileName() + ".bat"
            $BatContent = @()
            $BatContent += "@echo off"
            $BatContent += ('"' + $VsDevCmd + '" -no_logo -arch=amd64 -host_arch=amd64 >nul')
            $BatContent += "set"
            Set-Content -Path $TmpBat -Value $BatContent -Encoding ASCII
            $EnvLines = & cmd.exe /c $TmpBat
            Remove-Item $TmpBat -ErrorAction SilentlyContinue

            foreach ($el in $EnvLines) {
                $idx = $el.IndexOf("=")
                if ($idx -gt 0) {
                    $n = $el.Substring(0, $idx)
                    $v = $el.Substring($idx + 1)
                    [System.Environment]::SetEnvironmentVariable($n, $v, "Process")
                }
            }

            # Add VS cmake/ninja to PATH
            $CmkDir = Join-Path $VsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
            $NjnDir = Join-Path $VsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
            if (Test-Path $CmkDir) { $env:PATH = $CmkDir + ";" + $env:PATH }
            if (Test-Path $NjnDir) { $env:PATH = $NjnDir + ";" + $env:PATH }

            $ClExe = Find-Exe "cl.exe"
            if ($ClExe) {
                $MsvcOk = $true
                $DetectedMsvcVer = $VsVer
                Log-Ok ("MSVC " + $VsVer + " (cl.exe: " + $ClExe + ")")
            }
        }
    }

    # Required tools
    $Missing = @()
    $NeedCmake = Find-Exe "cmake.exe"
    $NeedGit   = Find-Exe "git.exe"
    if ($NeedCmake) { Log-Ok ("cmake: " + $NeedCmake) } else { $Missing += "cmake" }
    if ($NeedGit)   { Log-Ok ("git:   " + $NeedGit) }   else { $Missing += "git" }

    # Optional: ninja
    $NinjaExe = Find-Exe "ninja.exe"
    if ($NinjaExe) { Log-Ok ("ninja: " + $NinjaExe + " (faster builds)") }
    else           { Log-Warn "ninja not found (optional, MSBuild will be used)" }

    # Optional: ffmpeg
    $FfmpegExe = Find-Exe "ffmpeg.exe"
    if ($FfmpegExe) { Log-Ok ("ffmpeg: " + $FfmpegExe) }
    else            { Log-Warn "ffmpeg not found (optional, needed for some audio formats)" }

    if (-not $MsvcOk) {
        Log-Err "MSVC C++ toolchain not found"
        Log-Info "Install Visual Studio 2022 + Desktop development with C++ workload"
        Log-Info "Or Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/"
        exit 1
    }
    if ($Missing.Count -gt 0) {
        Log-Err ("Missing: " + ($Missing -join ", "))
        Log-Info "Install with: winget install <package>"
        exit 1
    }
    Log-Ok "Toolchain OK"

    # ============================================================
    # 3. OpenBLAS detection
    # ============================================================
    Log-Step "OpenBLAS detection"

    $SearchDirs = @(
        $DepsDir,
        $env:OPENBLAS_DIR,
        "D:\dev\OpenBLAS",
        "C:\OpenBLAS",
        "${env:ProgramFiles}\OpenBLAS",
        (Join-Path $ProjectRoot "vendor\OpenBLAS"),
        (Join-Path $ProjectRoot "third_party\OpenBLAS")
    )

    $FoundBlas = $null
    foreach ($sd in $SearchDirs) {
        if (-not $sd) { continue }
        # As cmake config dir
        $c1 = Join-Path $sd "OpenBLASConfig.cmake"
        if (Test-Path $c1) {
            $FoundBlas = @{ Root = (Split-Path (Split-Path (Split-Path $sd -Parent) -Parent) -Parent); Cmake = $sd }
            break
        }
        # As root dir
        $c2 = Join-Path $sd "lib\cmake\openblas\OpenBLASConfig.cmake"
        if (Test-Path $c2) {
            $FoundBlas = @{ Root = $sd; Cmake = (Join-Path $sd "lib\cmake\openblas") }
            break
        }
    }

    # Fallback: CMakeCache.txt
    if (-not $FoundBlas) {
        $CkFile = Join-Path $ProjectRoot "build\CMakeCache.txt"
        if (Test-Path $CkFile) {
            $CkText = Get-Content $CkFile -Raw
            if ($CkText -match 'OpenBLAS_DIR:PATH=(.+?)\r?\n') {
                $CachedDir = $Matches[1].Trim()
                $c3 = Join-Path $CachedDir "OpenBLASConfig.cmake"
                if (Test-Path $c3) {
                    $CachedRoot = Split-Path (Split-Path (Split-Path $CachedDir -Parent) -Parent) -Parent
                    $FoundBlas = @{ Root = $CachedRoot; Cmake = $CachedDir }
                }
            }
        }
    }

    if ($FoundBlas) {
        $DetectedOpenBlasDir = $FoundBlas.Cmake
        $Dll1 = Join-Path $FoundBlas.Root "bin\libopenblas.dll"
        $Dll2 = Join-Path $FoundBlas.Root "win64\bin\libopenblas.dll"
        if (Test-Path $Dll1) {
            Log-Ok ("OpenBLAS DLL: " + $Dll1)
        } elseif (Test-Path $Dll2) {
            Log-Ok ("OpenBLAS DLL: " + $Dll2)
        } else {
            Log-Warn "OpenBLAS config found but libopenblas.dll not in bin/ or win64/bin/"
        }
        Log-Ok ("OpenBLAS_DIR: " + $DetectedOpenBlasDir)
    } else {
        Log-Warn "OpenBLAS not found"
        if ($NoDep) {
            Log-Err "OpenBLAS missing and -NoDep prevents auto-install"
            Log-Err "Manual install options:"
            Log-Err "  1) winget install OpenBLAS"
            Log-Err "  2) Build from source: https://github.com/OpenBLAS/OpenBLAS"
            Log-Err "  3) Prebuilt: https://sourceforge.net/projects/openblas/files/"
            exit 1
        }
        Log-Info "Install OpenBLAS (pick one):"
        Log-Info "  1) winget: winget install OpenBLAS"
        Log-Info "  2) Prebuilt: https://sourceforge.net/projects/openblas/files/"
        Log-Info "  3) From source:"
        Log-Info ("     git clone --depth 1 https://github.com/OpenBLAS/OpenBLAS")
        Log-Info ("     cd OpenBLAS && cmake -B build -DCMAKE_INSTALL_PREFIX=" + $DepsDir)
        Log-Info ("     cmake --build build --config Release && cmake --install build")
        Log-Warn "Build will continue (cmake may use oneDNN fallback)"
    }

    # ============================================================
    # 4. Model detection
    # ============================================================
    if (-not $NoModel) {
        Log-Step "Model detection"

        $FoundModel = $null

        if ($ModelDir) {
            if (Test-Path (Join-Path $ModelDir "model.safetensors")) {
                $FoundModel = $ModelDir
            } else {
                Log-Warn ("--ModelDir=" + $ModelDir + " has no model.safetensors")
            }
        }

        if (-not $FoundModel -and $HfCache -and (Test-Path $HfCache)) {
            $HfF = Get-ChildItem -Path $HfCache -Recurse -Filter "model.safetensors" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($HfF) { $FoundModel = $HfF.DirectoryName }
        }

        if (-not $FoundModel) {
            $LocalDirs = @(
                (Join-Path $ProjectRoot ("models\" + $RepoBase)),
                (Join-Path $ProjectRoot $RepoBase),
                (Join-Path $ProjectRoot "Qwen3-ASR-0___6B"),
                (Join-Path $ProjectRoot "Qwen3-ASR-1.7B")
            )
            foreach ($ld in $LocalDirs) {
                if (Test-Path (Join-Path $ld "model.safetensors")) {
                    $FoundModel = $ld
                    break
                }
            }
        }

        if ($FoundModel) {
            $DetectedModelDir = $FoundModel
            Log-Ok ("Model: " + $DetectedModelDir)
        } else {
            Log-Warn ("Model not found (" + $HfRepo + ")")
            Log-Info "Download options:"
            Log-Info "  1) pip install -U huggingface_hub"
            Log-Info ("     python -c " + '"' + "from huggingface_hub import snapshot_download; snapshot_download('" + $HfRepo + "')" + '"')
            Log-Info ("  2) git lfs install; git clone https://huggingface.co/" + $HfRepo)
            Log-Info ("  3) Manual: https://huggingface.co/" + $HfRepo)
        }
    }

    # ============================================================
    # 5. Audio detection
    # ============================================================
    if (-not $NoAudio) {
        Log-Step "Test audio detection"
        $TfDir = Join-Path $ProjectRoot "testfile"
        $AFiles = $null
        if (Test-Path $TfDir) {
            $AFiles = Get-ChildItem -Path $TfDir -Include "*.wav","*.mp3","*.flac" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        }
        if ($AFiles) {
            $DetectedAudio = $AFiles.FullName
            Log-Ok ("Audio: " + $DetectedAudio)
        } else {
            Log-Warn "No .wav/.mp3/.flac in testfile/ (ignore with -NoAudio)"
            Log-Info "Fetch sample audio:"
            Log-Info "  python tools/aishell_fetch.py --speaker S0002 --clips 18"
        }
    }

    # ============================================================
    # 6. Clean
    # ============================================================
    if ($CleanOnly) {
        Log-Step ("Clean " + $BuildDir)
        if (Test-Path $BuildDir) {
            Remove-Item -Recurse -Force $BuildDir
            Log-Ok "Deleted"
        } else {
            Log-Info "build/ does not exist"
        }
        Log-Ok "clean-only done"
        exit 0
    }

    if (-not $Incremental) {
        Log-Step ("Clean " + $BuildDir)
        if (Test-Path $BuildDir) {
            Remove-Item -Recurse -Force $BuildDir
            Log-Ok "Deleted"
        }
    } else {
        Log-Info "Skipping clean (incremental)"
    }

    # ============================================================
    # 7. CMake configure
    # ============================================================
    Log-Step "CMake configure"
    $CmkArgs = @(
        "-S", $ProjectRoot,
        "-B", $BuildDir,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DQASR_ENABLE_TESTS=ON",
        "-DQASR_ENABLE_CPU_BACKEND=ON"
    )
    if ($DetectedOpenBlasDir) {
        $CmkArgs += ("-DOpenBLAS_DIR=" + $DetectedOpenBlasDir)
    }

    Log-Cmd ("cmake " + ($CmkArgs -join " "))
    $R = Start-Process -FilePath "cmake" -ArgumentList $CmkArgs -NoNewWindow -Wait -PassThru
    if ($R.ExitCode -ne 0) {
        Log-Err ("CMake configure failed (exit code: " + $R.ExitCode + ")")
        Log-Err "Common causes:"
        Log-Err "  - OpenBLAS not installed (try: winget install OpenBLAS)"
        Log-Err "  - MSVC incomplete (verify C++ desktop workload)"
        Log-Err "  - cmake too old (>= 3.21 recommended)"
        exit 1
    }
    Log-Ok "Configure OK"

    # ============================================================
    # 8. Build
    # ============================================================
    Log-Step ("Build (jobs=" + $Jobs + ")")
    $BArgs = @(
        "--build", $BuildDir,
        "--config", "Release",
        "--", "/m:" + $Jobs
    )
    Log-Cmd ("cmake " + ($BArgs -join " "))
    $R = Start-Process -FilePath "cmake" -ArgumentList $BArgs -NoNewWindow -Wait -PassThru
    if ($R.ExitCode -ne 0) {
        Log-Err ("Build failed (exit code: " + $R.ExitCode + ")")
        exit 1
    }
    Log-Ok "Build OK"

    # ============================================================
    # 9. Tests
    # ============================================================
    if (-not $NoTest) {
        Log-Step "Unit tests (qasr_unit_tests)"
        $TestExe = Join-Path $BuildDir "Release\qasr_unit_tests.exe"
        if (Test-Path $TestExe) {
            # Ensure OpenBLAS DLL in PATH
            if ($DetectedOpenBlasDir) {
                $BlasRoot = Split-Path (Split-Path (Split-Path $DetectedOpenBlasDir -Parent) -Parent) -Parent
                $BlasBin  = Join-Path $BlasRoot "bin"
                if (Test-Path $BlasBin) {
                    $env:PATH = $BlasBin + ";" + $env:PATH
                }
            }
            Log-Cmd ("ctest --test-dir " + $BuildDir + " -C Release -R qasr_unit_tests --output-on-failure")
            $R = Start-Process -FilePath "ctest" -ArgumentList @(
                "--test-dir", $BuildDir,
                "-C", "Release",
                "-R", "qasr_unit_tests",
                "--output-on-failure"
            ) -NoNewWindow -Wait -PassThru
            if ($R.ExitCode -ne 0) {
                Log-Err ("Tests failed (exit code: " + $R.ExitCode + ")")
                exit 1
            }
            Log-Ok "Tests PASSED"
        } else {
            Log-Warn ("Cannot find " + $TestExe + ", skipping tests")
        }
    } else {
        Log-Info "Skipping tests (-NoTest)"
    }

    # ============================================================
    # 10. Summary
    # ============================================================
    Log-Step "Build Summary"
    Log-Ok ("Build dir: " + $BuildDir)
    if ($DetectedOpenBlasDir) { Log-Ok ("OpenBLAS:  " + $DetectedOpenBlasDir) }
    if ($DetectedModelDir)    { Log-Ok ("Model:     " + $DetectedModelDir) }
    if ($DetectedAudio)       { Log-Ok ("Audio:     " + $DetectedAudio) }
    if ($DetectedMsvcVer)     { Log-Ok ("MSVC:      " + $DetectedMsvcVer) }
    Write-Host ""

    $RelDir = Join-Path $BuildDir "Release"
    foreach ($ex in @("qasr_server.exe", "qasr_cli.exe", "qasr_cpu_bench.exe", "qasr_unit_tests.exe")) {
        $p = Join-Path $RelDir $ex
        if (Test-Path $p) { Log-Ok $ex }
    }
    Write-Host ""

    if (Test-Path (Join-Path $RelDir "qasr_server.exe")) {
        $Md = $DetectedModelDir
        if (-not $Md) { $Md = "(set QASR_MODEL_DIR)" }
        Log-Info ("Start server:  " + $RelDir + "\qasr_server.exe --model-dir """ + $Md + """")
        Log-Info ("Or use:        .\tools\run_server.bat --detach")
    }
    if (Test-Path (Join-Path $RelDir "qasr_cli.exe")) {
        $Md = $DetectedModelDir
        if (-not $Md) { $Md = "(set QASR_MODEL_DIR)" }
        $Ad = $DetectedAudio
        if (-not $Ad) { $Ad = "(testfile/*.wav)" }
        Log-Info ("E2E transcribe: " + $RelDir + "\qasr_cli.exe --model-dir """ + $Md + """ --audio """ + $Ad + """ --language Chinese")
    }
    Write-Host ""
    Log-Ok "Build complete"
    Write-Host "======================================" -ForegroundColor Cyan
} catch {
    Write-Host ""
    Log-Err $_.Exception.Message
    Write-Host $_.ScriptStackTrace -ForegroundColor DarkGray
    exit 1
} finally {
    Pop-Location
}
