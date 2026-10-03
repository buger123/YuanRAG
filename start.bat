@echo off
REM RAG Assistant - Windows one-click launcher.
REM Keep this file ASCII-only; non-ASCII bytes break cmd under the default OEM codepage.
REM We avoid `if ... ( pause )` blocks because they can hang cmd.exe when
REM the script is launched from a parent that captures stdout (some shells
REM inject stdin redirection that confuses `pause`).

cd /d "%~dp0"

REM ---- Step 1: virtual environment ----
if not exist ".venv\Scripts\python.exe" goto :create_venv
echo [ok] Using .venv
goto :after_venv
:create_venv
echo [setup] Creating Python virtual environment...
python -m venv .venv
if errorlevel 1 goto :error
echo [ok] Created .venv
:after_venv

REM ---- Step 2: dependencies ----
if exist ".deps_installed" goto :deps_done
echo [setup] Installing Python dependencies (first run may take a few minutes)...
echo [setup] Docling is the slow step on first run.
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :error
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :error
type nul > .deps_installed
echo [ok] Python dependencies installed
goto :deps_after
:deps_done
echo [ok] Python dependencies already installed
:deps_after

REM ---- Step 4: frontend build ----
if exist "src\frontend\dist\index.html" goto :frontend_done
echo [setup] Building frontend (first run)...
call scripts\build_frontend.bat
if errorlevel 1 (
    echo [warning] Frontend build failed. UI will not load; API still works at /docs.
)
goto :frontend_after
:frontend_done
echo [ok] Frontend already built
:frontend_after

REM ---- Step 5: launch ----
echo.
echo ============================================================
echo   RAG Assistant starting on http://127.0.0.1:8765/
echo   Browser will open automatically.
echo   Press Ctrl+C in this window to stop the server.
echo ============================================================
echo.

.venv\Scripts\python.exe -m src.main
goto :eof

:error
echo.
echo [error] Setup failed. Check the messages above.
echo Press any key to close...
pause
exit /b 1
