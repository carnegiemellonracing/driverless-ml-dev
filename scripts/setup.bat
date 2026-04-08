@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SKIP_VENV=0"
set "SKIP_INSTALL=0"

for %%A in (%*) do (
  if /I "%%~A"=="--skip-venv" set "SKIP_VENV=1"
  if /I "%%~A"=="--skip-install" set "SKIP_INSTALL=1"
)

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
set "VENV_DIR=%REPO_ROOT%\.venv"
set "REQ_FILE=%REPO_ROOT%\yolo-optimization\requirements.txt"

if not exist "%REQ_FILE%" (
  echo ERROR: requirements file not found: "%REQ_FILE%"
  exit /b 1
)

set "BASE_PY="
where py >nul 2>nul
if %errorlevel%==0 set "BASE_PY=py -3"
if not defined BASE_PY (
  where python >nul 2>nul
  if %errorlevel%==0 set "BASE_PY=python"
)
if not defined BASE_PY (
  echo ERROR: Python was not found. Install Python 3.10+ and try again.
  exit /b 1
)

%BASE_PY% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not %errorlevel%==0 (
  echo ERROR: Python 3.10+ is required.
  exit /b 1
)

if not exist "%REPO_ROOT%\ml_data" mkdir "%REPO_ROOT%\ml_data"
if not exist "%REPO_ROOT%\ml_data\fsoco_raw" mkdir "%REPO_ROOT%\ml_data\fsoco_raw"
if not exist "%REPO_ROOT%\ml_data\fsoco_mod" mkdir "%REPO_ROOT%\ml_data\fsoco_mod"
if not exist "%REPO_ROOT%\ml_data\fsoco_edit" mkdir "%REPO_ROOT%\ml_data\fsoco_edit"
if not exist "%REPO_ROOT%\ml_data\fsoco_edit\img" mkdir "%REPO_ROOT%\ml_data\fsoco_edit\img"
if not exist "%REPO_ROOT%\ml_data\fsoco_edit\ann" mkdir "%REPO_ROOT%\ml_data\fsoco_edit\ann"
if not exist "%REPO_ROOT%\ml_data\fsoco_yolo" mkdir "%REPO_ROOT%\ml_data\fsoco_yolo"
if not exist "%REPO_ROOT%\ml_data\models" mkdir "%REPO_ROOT%\ml_data\models"
if not exist "%REPO_ROOT%\ml_data\experiments" mkdir "%REPO_ROOT%\ml_data\experiments"
if not exist "%REPO_ROOT%\ml_data\experiments\mlflow_tracking" mkdir "%REPO_ROOT%\ml_data\experiments\mlflow_tracking"
echo Created/verified project folders in ml_data\.

set "INSTALL_PY=%BASE_PY%"
if "%SKIP_VENV%"=="0" (
  if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo + %BASE_PY% -m venv "%VENV_DIR%"
    %BASE_PY% -m venv "%VENV_DIR%"
    if not %errorlevel%==0 exit /b %errorlevel%
    echo Created virtual environment at "%VENV_DIR%"
  ) else (
    echo Using existing virtual environment at "%VENV_DIR%"
  )
  set "INSTALL_PY=%VENV_DIR%\Scripts\python.exe"
)

if "%SKIP_INSTALL%"=="0" (
  if "%SKIP_VENV%"=="0" (
    echo + "%INSTALL_PY%" -m pip install --upgrade pip setuptools wheel
    "%INSTALL_PY%" -m pip install --upgrade pip setuptools wheel
    if not %errorlevel%==0 exit /b %errorlevel%
    echo + "%INSTALL_PY%" -m pip install -r "%REQ_FILE%"
    "%INSTALL_PY%" -m pip install -r "%REQ_FILE%"
    if not %errorlevel%==0 exit /b %errorlevel%
  ) else (
    echo + %INSTALL_PY% -m pip install --upgrade pip setuptools wheel
    %INSTALL_PY% -m pip install --upgrade pip setuptools wheel
    if not %errorlevel%==0 exit /b %errorlevel%
    echo + %INSTALL_PY% -m pip install -r "%REQ_FILE%"
    %INSTALL_PY% -m pip install -r "%REQ_FILE%"
    if not %errorlevel%==0 exit /b %errorlevel%
  )
  echo Dependencies installed.
) else (
  echo Skipped dependency install.
)

echo.
echo Setup complete.
if "%SKIP_VENV%"=="0" echo Activate with: .\.venv\Scripts\Activate.ps1
echo Place raw dataset files in: ml_data\fsoco_raw\
echo Then run scripts from repository root.

endlocal
