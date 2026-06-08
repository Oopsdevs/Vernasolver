@echo off
setlocal EnableDelayedExpansion
title VernaSolver Setup
color 0A

echo.
echo  ##############################################
echo  #      VernaSolver - Auto Setup              #
echo  #  Installs Python + all dependencies        #
echo  ##############################################
echo.

:: ── Elevate to admin if needed ───────────────────────────────────────────────
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo  Requesting administrator access...
    powershell -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

set "PROJECT_DIR=%~dp0"
set "VENV=%PROJECT_DIR%venv"
set "PY311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
set "PY_INSTALLER=%TEMP%\python311_setup.exe"

echo  Project folder: %PROJECT_DIR%
echo.

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 1 — Python 3.11
:: ─────────────────────────────────────────────────────────────────────────────
echo  [STEP 1/6]  Checking Python 3.11...

:: Check via launcher first, then direct path
py -3.11 --version >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=*" %%i in ('py -3.11 -c "import sys; print(sys.executable)"') do set "PY=%%i"
    echo             Found: !PY!
    goto :venv
)

if exist "%PY311%" (
    set "PY=%PY311%"
    echo             Found: %PY311%
    goto :venv
)

echo             Not found. Downloading Python 3.11.9 ...
powershell -NoProfile -Command ^
  "$ProgressPreference='SilentlyContinue';" ^
  "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe'" ^
  " -OutFile '%PY_INSTALLER%'"

if not exist "%PY_INSTALLER%" (
    echo.
    echo  [ERROR] Download failed. Check your internet connection and re-run.
    pause & exit /b 1
)

echo             Installing Python 3.11.9 (silent)...
"%PY_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0
del "%PY_INSTALLER%" >nul 2>&1

:: Refresh env so python launcher finds the new install
set "PATH=%LOCALAPPDATA%\Programs\Python\Python311\Scripts;%LOCALAPPDATA%\Programs\Python\Python311;%PATH%"

if exist "%PY311%" (
    set "PY=%PY311%"
    echo             Installed OK.
) else (
    echo  [ERROR] Python installation did not complete. Try running the installer manually:
    echo          https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
    pause & exit /b 1
)

:venv
echo.

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 2 — Wipe old venv + pip cache
:: ─────────────────────────────────────────────────────────────────────────────
echo  [STEP 2/6]  Clearing old environment and pip cache...

if exist "%VENV%" (
    rmdir /s /q "%VENV%"
    echo             Old venv deleted.
)

:: Clear pip cache to avoid stale/broken wheels
"%PY%" -m pip cache purge >nul 2>&1
powershell -NoProfile -Command "Remove-Item -Recurse -Force '$env:LOCALAPPDATA\pip\cache' -ErrorAction SilentlyContinue"
echo             Pip cache cleared.
echo.

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 3 — Create fresh venv
:: ─────────────────────────────────────────────────────────────────────────────
echo  [STEP 3/6]  Creating virtual environment...
"%PY%" -m venv "%VENV%"
if %errorLevel% neq 0 (
    echo  [ERROR] Could not create venv.
    pause & exit /b 1
)
echo             Done.
echo.

set "PIP=%VENV%\Scripts\pip.exe"
set "VPYTHON=%VENV%\Scripts\python.exe"

:: Upgrade pip silently
"%VPYTHON%" -m pip install --upgrade pip --quiet

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 4 — PyTorch (CPU, pinned)
:: ─────────────────────────────────────────────────────────────────────────────
echo  [STEP 4/6]  Installing PyTorch CPU (pinned 2.2.2) ...
echo             This is the largest download (~180 MB). Please wait...
"%PIP%" install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu --quiet
if %errorLevel% neq 0 (
    echo  [ERROR] PyTorch install failed.
    pause & exit /b 1
)
echo             PyTorch installed.
echo.

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 5 — All other dependencies (pinned where needed)
:: ─────────────────────────────────────────────────────────────────────────────
echo  [STEP 5/6]  Installing remaining dependencies...

"%PIP%" install "numpy<2" --quiet
echo             numpy OK

"%PIP%" install transformers==4.44.2 --quiet
if %errorLevel% neq 0 ( echo  [ERROR] transformers failed. & pause & exit /b 1 )
echo             transformers 4.44.2 OK

"%PIP%" install sentence-transformers==3.0.1 --quiet
if %errorLevel% neq 0 ( echo  [ERROR] sentence-transformers failed. & pause & exit /b 1 )
echo             sentence-transformers 3.0.1 OK

"%PIP%" install chromadb --quiet
if %errorLevel% neq 0 ( echo  [ERROR] chromadb failed. & pause & exit /b 1 )
echo             chromadb OK

"%PIP%" install fastapi "uvicorn[standard]" python-multipart python-dotenv --quiet
if %errorLevel% neq 0 ( echo  [ERROR] fastapi/uvicorn failed. & pause & exit /b 1 )
echo             fastapi + uvicorn OK

"%PIP%" install anthropic openai --quiet
echo             anthropic + openai OK

"%PIP%" install pymupdf click --quiet
echo             pymupdf + click OK

echo.

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 6 — Verify
:: ─────────────────────────────────────────────────────────────────────────────
echo  [STEP 6/6]  Verifying installation...
"%VPYTHON%" -c "import torch, sentence_transformers, chromadb, fastapi, fitz; print('  All imports OK  |  Torch', torch.__version__)"
if %errorLevel% neq 0 (
    echo  [ERROR] Verification failed. Please re-run SETUP.bat.
    pause & exit /b 1
)
echo.

:: ─────────────────────────────────────────────────────────────────────────────
:: Write start_server.bat
:: ─────────────────────────────────────────────────────────────────────────────
(
echo @echo off
echo title VernaSolver
echo cd /d "%PROJECT_DIR%"
echo echo.
echo echo  VernaSolver is starting...
echo echo  Open your browser at http://localhost:8000
echo echo  Press Ctrl+C to stop.
echo echo.
echo "%VENV%\Scripts\uvicorn.exe" server:app --reload --port 8000
echo pause
) > "%PROJECT_DIR%start_server.bat"

echo  ##############################################
echo  #   Setup complete!                          #
echo  ##############################################
echo.
echo   Run  start_server.bat  to launch VernaSolver.
echo   Then open:  http://localhost:8000
echo.

set /p "GO=  Start the server now? (y/n): "
if /i "%GO%"=="y" (
    start "" "%PROJECT_DIR%start_server.bat"
)

pause
