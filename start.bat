@echo off
setlocal EnableDelayedExpansion

:: Qwen3-ASR Windows 一键构建脚本
:: 用法: start.bat [--clean] [--test] [--openblas-dir ^<path^>]

set "SCRIPT_DIR=%~dp0"
set "CLEAN_ARG="
set "TEST_ARG="
set "OPENBLAS_ARG="

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--clean" (
    set "CLEAN_ARG=-Clean"
    shift
    goto parse_args
)
if /I "%~1"=="--test" (
    set "TEST_ARG=-Test"
    shift
    goto parse_args
)
if /I "%~1"=="--openblas-dir" (
    set "OPENBLAS_ARG=-OpenBlasDir "%~2""
    shift
    shift
    goto parse_args
)
echo Unknown argument: %~1
echo Usage: start.bat [--clean] [--test] [--openblas-dir ^<path^>]
exit /b 1

:args_done
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%build_all.ps1" %CLEAN_ARG% %TEST_ARG% %OPENBLAS_ARG%
exit /b %ERRORLEVEL%
