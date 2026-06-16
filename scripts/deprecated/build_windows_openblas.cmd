@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_windows_openblas.ps1" %*
exit /b %ERRORLEVEL%