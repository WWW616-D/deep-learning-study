@echo off
REM ==========================================================================
REM  run.cmd -- convenience launcher for this project.
REM
REM  Always runs the project's own venv python, from the project directory, so
REM  you never have to remember either. Forwards every argument verbatim.
REM
REM    run.cmd scripts\status.py
REM    run.cmd -m src.generate --interactive
REM    run.cmd -m src.train_pretrain --preset tiny --max-iters 300
REM    run.cmd tests\test_core.py
REM ==========================================================================

setlocal
cd /d "%~dp0"

set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=%~dp0"

"%~dp0.venv\Scripts\python.exe" %*
exit /b %ERRORLEVEL%
