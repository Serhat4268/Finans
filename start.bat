@echo off

title Finansal Karar Destek Platformu

echo ======================================================================
echo   FINANSAL TAHMIN VE KARAR DESTEK PLATFORMU BASLATILIYOR...
echo ======================================================================

set "PY_CMD=python"

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_CMD=%~dp0.venv\Scripts\python.exe"
    echo [BILGI] Sanal ortam bulundu: .venv
    goto :run
)

if exist "%~dp0venv\Scripts\python.exe" (
    set "PY_CMD=%~dp0venv\Scripts\python.exe"
    echo [BILGI] Sanal ortam bulundu: venv
    goto :run
)

:run
"%PY_CMD%" "%~dp0run.py" %*

if errorlevel 1 (
    echo.
    echo [UYARI] Program bir hata ile sonlandi veya durduruldu.
    pause
)
