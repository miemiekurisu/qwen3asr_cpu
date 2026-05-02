@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "PRESET=windows-openblas"
set "OPENBLAS_DIR="
set "RUN_TESTS=0"
set "RUN_BENCHMARK=0"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--help" goto usage
if /I "%~1"=="-h" goto usage
if /I "%~1"=="--test" (
    set "RUN_TESTS=1"
    shift
    goto parse_args
)
if /I "%~1"=="--benchmark" (
    set "RUN_BENCHMARK=1"
    shift
    goto parse_args
)
if /I "%~1"=="--preset" (
    if "%~2"=="" goto missing_value
    set "PRESET=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--openblas-dir" (
    if "%~2"=="" goto missing_value
    set "OPENBLAS_DIR=%~2"
    shift
    shift
    goto parse_args
)
echo Unknown argument: %~1
echo.
goto usage

:missing_value
echo Missing value for %~1
echo.
goto usage

:args_done
set "SKIP_TESTS_ARG="
if "%RUN_TESTS%"=="0" set "SKIP_TESTS_ARG=-SkipTests"

if "%OPENBLAS_DIR%"=="" (
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%tools\build_windows_openblas.ps1" -Preset "%PRESET%" -Clean %SKIP_TESTS_ARG%
) else (
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%tools\build_windows_openblas.ps1" -Preset "%PRESET%" -Clean -OpenBlasDir "%OPENBLAS_DIR%" %SKIP_TESTS_ARG%
)
if errorlevel 1 exit /b 1

if "%RUN_BENCHMARK%"=="1" (
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%tools\run_benchmark.ps1"
    if errorlevel 1 exit /b 1
)

exit /b 0

:usage
echo Usage: build.bat [--test] [--benchmark] [--openblas-dir ^<dir^>] [--preset ^<name^>]
echo.
echo Default:
echo   clean configure and compile only, using the windows-openblas preset.
echo.
echo Options:
echo   --test              Run CTest after the clean compile.
echo   --benchmark         Run tools\run_benchmark.ps1 after the clean compile.
echo   --openblas-dir DIR  OpenBLAS install root or OpenBLASConfig.cmake directory.
echo   --preset NAME       CMake preset to use. Default: windows-openblas.
echo   -h, --help          Show this help.
exit /b 0
