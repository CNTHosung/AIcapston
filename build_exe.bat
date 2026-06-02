@echo off
setlocal
cd /d "%~dp0"
echo ============================================
echo  Sensor SNN demo - EXE build (clean venv)
echo ============================================
echo.

rem --- find Python launcher ---
set PY=python
where python >nul 2>nul || set PY=py
%PY% --version >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found on PATH. Install Python 3.10+ from python.org.
  pause
  exit /b 1
)

echo [1/5] Creating clean virtual environment (.venv)...
%PY% -m venv .venv
if errorlevel 1 ( echo [ERROR] venv creation failed. & pause & exit /b 1 )
set VPY=.venv\Scripts\python.exe

echo [2/5] Upgrading pip inside venv...
"%VPY%" -m pip install --upgrade pip

echo [3/5] Installing packages into .venv (first time takes a while)...
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 ( echo [ERROR] pip install failed. & pause & exit /b 1 )

echo [4/5] Sanity check: import torch ...
"%VPY%" -c "import torch, snntorch; print('torch', torch.__version__, 'OK')"
if errorlevel 1 (
  echo.
  echo [ERROR] 'import torch' failed even in a clean venv.
  echo   -^> Install "Microsoft Visual C++ Redistributable (x64)" and retry:
  echo      https://aka.ms/vs/17/release/vc_redist.x64.exe
  pause
  exit /b 1
)

echo [5/5] Building EXE (a few minutes)...
"%VPY%" -m PyInstaller --noconfirm --clean app.spec
if errorlevel 1 ( echo [ERROR] PyInstaller build failed. & pause & exit /b 1 )

echo Copying data and model next to the EXE...
if not exist "dist\SensorSNN" ( echo [ERROR] dist\SensorSNN not found. & pause & exit /b 1 )
xcopy /E /I /Y data "dist\SensorSNN\data" >nul
copy /Y snn_mnist.pt "dist\SensorSNN\" >nul
echo.
echo ============================================
echo  Done!  Run:  dist\SensorSNN\SensorSNN.exe
echo ============================================
pause
