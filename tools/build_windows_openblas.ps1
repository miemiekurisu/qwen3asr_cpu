[CmdletBinding()]
param(
    [string]$Preset = "windows-openblas",
    [string]$OpenBlasDir,
    [switch]$SkipTests,
    [switch]$ConfigureOnly,
    [switch]$Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Add-ToPath {
    param([string]$Directory)

    if (-not $Directory) {
        return
    }
    if (-not (Test-Path $Directory)) {
        return
    }

    $resolved = (Resolve-Path $Directory).Path
    $parts = ($env:PATH -split ';') | Where-Object { $_ }
    if ($parts -notcontains $resolved) {
        $env:PATH = "$resolved;$env:PATH"
    }
}

function Get-VsWherePath {
    $cmd = Get-Command "vswhere.exe" -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $installerRoot = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer"
    $candidate = Join-Path $installerRoot "vswhere.exe"
    if (Test-Path $candidate) {
        return $candidate
    }

    return $null
}

function Import-VsEnvironment {
    if (Get-Command "cl.exe" -ErrorAction SilentlyContinue) {
        return
    }

    $vswhere = Get-VsWherePath
    if (-not $vswhere) {
        throw "MSVC toolchain not found. Install Visual Studio C++ tools or run from a Developer PowerShell."
    }

    $installationPath = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
    if (-not $installationPath) {
        throw "Visual Studio installation with C++ workload not found."
    }

    $vsDevCmd = Join-Path $installationPath "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path $vsDevCmd)) {
        throw "VsDevCmd.bat not found under $installationPath"
    }

    $envDump = & cmd.exe /s /c "`"$vsDevCmd`" -no_logo -arch=amd64 -host_arch=amd64 >nul && set"
    foreach ($line in $envDump) {
        $index = $line.IndexOf('=')
        if ($index -gt 0) {
            $name = $line.Substring(0, $index)
            $value = $line.Substring($index + 1)
            Set-Item -Path "Env:$name" -Value $value
        }
    }

    Add-ToPath (Join-Path $installationPath "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin")
    Add-ToPath (Join-Path $installationPath "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja")
}

function Normalize-OpenBlasDir {
    param([string]$PathHint)

    if (-not $PathHint) {
        return $null
    }

    try {
        $resolved = (Resolve-Path $PathHint -ErrorAction Stop).Path
    } catch {
        return $null
    }

    $candidates = @(
        $resolved,
        (Join-Path $resolved "lib\cmake\openblas"),
        (Join-Path $resolved "cmake\openblas")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path (Join-Path $candidate "OpenBLASConfig.cmake")) {
            return (Resolve-Path $candidate).Path
        }
    }

    return $null
}

function Get-CachedOpenBlasDir {
    param(
        [string]$RepoRoot,
        [string]$PresetName
    )

    $cachePath = Join-Path $RepoRoot (Join-Path (Join-Path "build" $PresetName) "CMakeCache.txt")
    if (-not (Test-Path $cachePath)) {
        return $null
    }

    foreach ($line in Get-Content $cachePath) {
        if ($line -match '^OpenBLAS_DIR(?::[^=]+)?=(.+)$') {
            $normalized = Normalize-OpenBlasDir $Matches[1].Trim()
            if ($normalized) {
                return $normalized
            }
        }
    }

    return $null
}

function Find-OpenBlasDir {
    param(
        [string]$RepoRoot,
        [string]$Hint,
        [string]$PresetName
    )

    $normalized = Normalize-OpenBlasDir $Hint
    if ($normalized) {
        return $normalized
    }

    $normalized = Normalize-OpenBlasDir $env:OPENBLAS_DIR
    if ($normalized) {
        return $normalized
    }

    $normalized = Get-CachedOpenBlasDir -RepoRoot $RepoRoot -PresetName $PresetName
    if ($normalized) {
        return $normalized
    }

    $repoParent = Split-Path $RepoRoot -Parent
    $searchRoots = @(
        $RepoRoot,
        (Join-Path $RepoRoot "vendor"),
        (Join-Path $RepoRoot "third_party"),
        (Join-Path $RepoRoot "deps"),
        (Join-Path $RepoRoot ".deps"),
        $repoParent,
        (Join-Path $repoParent "vendor"),
        (Join-Path $repoParent "third_party"),
        (Join-Path $repoParent "deps")
    ) | Select-Object -Unique

    foreach ($root in $searchRoots) {
        if (-not (Test-Path $root)) {
            continue
        }

        $openBlasDirs = Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match 'openblas' }
        foreach ($dir in $openBlasDirs) {
            $normalized = Normalize-OpenBlasDir $dir.FullName
            if ($normalized) {
                return $normalized
            }
        }

        $config = Get-ChildItem -Path $root -Filter "OpenBLASConfig.cmake" -Recurse -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($config) {
            return $config.Directory.FullName
        }
    }

    throw "OpenBLASConfig.cmake not found. Pass -OpenBlasDir, set OPENBLAS_DIR, or place OpenBLAS under vendor/, third_party/, deps/, .deps/ or a sibling directory."
}

function Require-Tool {
    param([string]$Name)

    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "$Name not found on PATH after environment setup."
    }
    return $cmd.Source
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path

Import-VsEnvironment

$cmake = Require-Tool "cmake.exe"
$ctest = Require-Tool "ctest.exe"
$null = Require-Tool "ninja.exe"
$resolvedOpenBlas = Find-OpenBlasDir -RepoRoot $repoRoot -Hint $OpenBlasDir -PresetName $Preset

Write-Host "Repo root    : $repoRoot"
Write-Host "Preset       : $Preset"
Write-Host "OpenBLAS_DIR : $resolvedOpenBlas"
Write-Host "CMake        : $cmake"
Write-Host "CTest        : $ctest"

Push-Location $repoRoot
try {
    if ($Clean) {
        $buildDir = Join-Path $repoRoot (Join-Path "build" $Preset)
        if (Test-Path $buildDir) {
            Remove-Item -Recurse -Force $buildDir
        }
    }

    & $cmake --preset $Preset "-DOpenBLAS_DIR=$resolvedOpenBlas" -DQASR_ENABLE_CPU_BACKEND=ON
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configure failed."
    }

    if (-not $ConfigureOnly) {
        & $cmake --build --preset $Preset
        if ($LASTEXITCODE -ne 0) {
            throw "CMake build failed."
        }

        if (-not $SkipTests) {
            & $ctest --preset $Preset
            if ($LASTEXITCODE -ne 0) {
                throw "CTest failed."
            }
        }
    }
} finally {
    Pop-Location
}