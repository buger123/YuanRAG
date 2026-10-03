@echo off
REM Build the React frontend into src\frontend\dist
setlocal

cd /d "%~dp0\..\src\frontend"

REM npm on Windows often fails when its global cache directory
REM (under ``C:\Program Files\nodejs\node_cache``) is not writable.
REM We pin a project-local cache to side-step that.
set "NPM_CONFIG_CACHE=%CD%\.npm-cache"

if not exist "node_modules" (
    echo [build] Installing frontend dependencies...
    call npm install --no-audit --no-fund --loglevel=error
    if errorlevel 1 (
        echo [error] npm install failed.
        echo         If this is a permissions issue, try running as Administrator
        echo         or delete the .npm-cache folder and retry.
        exit /b 1
    )
)

echo [build] Building frontend...
call npm run build --silent
if errorlevel 1 (
    echo [error] Frontend build failed.
    exit /b 1
)

echo [build] Frontend built successfully.
exit /b 0
