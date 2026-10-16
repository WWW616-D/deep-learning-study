@echo off
REM ==========================================================================
REM  status.cmd -- DOUBLE-CLICK THIS to see the project status.
REM
REM  Runs scripts/status.py in the project venv and waits for a keypress, so
REM  the window stays open if you launched it from Explorer.
REM
REM  For anything interactive (training, --interactive sampling) use run.cmd
REM  from a real terminal instead.
REM ==========================================================================

chcp 65001 >nul 2>&1
call "%~dp0run.cmd" scripts\status.py
echo.
echo ============================================================
echo  Done. Press any key to close this window.
echo ============================================================
pause >nul
