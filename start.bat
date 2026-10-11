@echo off
rem Starts Qwen Transcribe and opens it in your browser. Close this window to stop it.
title Qwen Transcribe
cd /d "%~dp0"
"%~dp0env\python.exe" -m app %*
if errorlevel 1 pause
