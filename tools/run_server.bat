@echo off
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"

set "ARGS="
:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--detach" (
    set "ARGS=!ARGS! -Detach"
) else if /I "%~1"=="--stop" (
    set "ARGS=!ARGS! -Stop"
) else if /I "%~1"=="--status" (
    set "ARGS=!ARGS! -DoHealthCheck"
) else if /I "%~1"=="--https" (
    set "ARGS=!ARGS! -UseHttps"
) else if /I "%~1"=="--https-info" (
    set "ARGS=!ARGS! -HttpsInfo"
) else if /I "%~1"=="--verbose" (
    set "ARGS=!ARGS! -Verb"
) else if /I "%~1"=="--help" (
    set "ARGS=!ARGS! -Help"
)
shift
goto parse_args

:args_done

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
    -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; & '%SCRIPT_DIR%run_server.ps1' %ARGS%"
exit /b %ERRORLEVEL%
