@echo off
setlocal enabledelayedexpansion
REM ===========================================================
REM  ACG H. pylori RAG - one-click setup + runner
REM  DOUBLE-CLICK this file. It installs Python if missing,
REM  installs the packages, then runs the pipeline.
REM ===========================================================

cd /d "%~dp0"

echo ==========================================================
echo  ACG PDF - setup / inspect / parse / clean / chunk
echo  Folder: %CD%
echo ==========================================================
echo.

call :FIND_PYTHON
if defined PY goto HAVE_PYTHON

REM ===========================================================
REM  AUTO-INSTALL PYTHON VIA WINGET
REM ===========================================================
echo Python not found. Attempting automatic install...
echo.

where winget >nul 2>&1
if errorlevel 1 (
    echo ==========================================================
    echo  CANNOT AUTO-INSTALL - winget is missing
    echo ==========================================================
    echo.
    echo  Please install Python manually:
    echo    https://www.python.org/downloads/
    echo.
    echo  IMPORTANT: on the first installer screen, tick
    echo    [x] Add python.exe to PATH
    echo.
    echo  Do NOT use the Microsoft Store shortcut - that is the
    echo  placeholder that produced your error.
    echo.
    echo  Then double-click this file again.
    pause
    exit /b 1
)

echo Installing Python 3.12 (this takes 1-3 minutes)...
echo A Windows permission prompt may appear - click Yes.
echo.
winget install -e --id Python.Python.3.12 --scope user --accept-source-agreements --accept-package-agreements
echo.
echo Install step finished. Re-detecting Python...
echo.

call :FIND_PYTHON
if not defined PY (
    echo ==========================================================
    echo  PYTHON STILL NOT DETECTED
    echo ==========================================================
    echo.
    echo  The install may have succeeded but needs a fresh session.
    echo  CLOSE this window and double-click RUN_ME.bat again.
    echo.
    echo  If it still fails, install manually from
    echo    https://www.python.org/downloads/
    echo  and tick "Add python.exe to PATH".
    pause
    exit /b 1
)

:HAVE_PYTHON
echo Found Python: %PY%
%PY% --version
echo.

REM ===========================================================
REM  DISABLE THE STORE STUB HINT (informational only)
REM ===========================================================

REM ===========================================================
REM  PACKAGES
REM ===========================================================
echo Installing packages: pymupdf, transformers ...
echo (first run downloads ~300 MB, be patient)
echo.
%PY% -m pip install --upgrade pip
%PY% -m pip install pymupdf transformers
if errorlevel 1 (
    echo.
    echo [ERROR] Package install failed. Scroll up for the reason.
    pause
    exit /b 1
)
echo.
echo Packages OK.
echo.

REM ===========================================================
REM  FOLDERS + PDF CHECK
REM ===========================================================
if not exist "data\raw"       mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"

set "FOUNDPDF="
for /r %%F in (*.pdf) do set "FOUNDPDF=%%F"
if not defined FOUNDPDF (
    echo [ERROR] No PDF found anywhere under this folder.
    echo Copy the ACG guideline PDF into:  %CD%\data\raw\
    echo Then double-click this file again.
    pause
    exit /b 1
)
echo PDF found: !FOUNDPDF!
echo.

REM ===========================================================
REM  RUN
REM ===========================================================
echo ==========================================================
echo  STEP 1 - INSPECT
echo ==========================================================
%PY% 1_inspect.py       > "data\processed\FULL_OUTPUT.txt" 2>&1

echo. >> "data\processed\FULL_OUTPUT.txt"
echo ========================================================== >> "data\processed\FULL_OUTPUT.txt"
echo  STEP 2 - PARSE + CLEAN + CHUNK >> "data\processed\FULL_OUTPUT.txt"
echo ========================================================== >> "data\processed\FULL_OUTPUT.txt"

echo ==========================================================
echo  STEP 2 - PARSE + CLEAN + CHUNK
echo ==========================================================
%PY% 2_parse_chunk.py  >> "data\processed\FULL_OUTPUT.txt" 2>&1

echo. >> "data\processed\FULL_OUTPUT.txt"
echo ========================================================== >> "data\processed\FULL_OUTPUT.txt"
echo  STEP 3 - RETRIEVAL EVAL >> "data\processed\FULL_OUTPUT.txt"
echo ========================================================== >> "data\processed\FULL_OUTPUT.txt"

echo ==========================================================
echo  STEP 3 - RETRIEVAL EVAL
echo ==========================================================
%PY% 3_eval.py         >> "data\processed\FULL_OUTPUT.txt" 2>&1

type "data\processed\FULL_OUTPUT.txt"

echo.
echo ==========================================================
echo  DONE
echo ==========================================================
echo  Full log : %CD%\data\processed\FULL_OUTPUT.txt
echo  Chunks   : %CD%\data\processed\acg_chunks.json
echo  Sections : %CD%\data\processed\acg_sections.json
echo.
echo  Opening the log in Notepad - copy it back into the chat.
start "" notepad "data\processed\FULL_OUTPUT.txt"

pause
exit /b 0


REM ===========================================================
REM  SUBROUTINE: locate a REAL python
REM  Windows ships a fake python.exe stub in WindowsApps that
REM  only prints "Python was not found". `where python` finds
REM  it, so every candidate must actually be RUN and verified.
REM ===========================================================
:FIND_PYTHON
set "PY="

REM -- 1. the py launcher: ignores the Store stub entirely
py -3 --version >nul 2>&1
if not errorlevel 1 (
    set "PY=py -3"
    goto :eof
)

REM -- 2. python on PATH, verified not to be the stub
for /f "delims=" %%v in ('python --version 2^>^&1') do set "VER=%%v"
if defined VER (
    echo !VER! | find /i "was not found" >nul
    if errorlevel 1 (
        echo !VER! | find /i "Python 3" >nul
        if not errorlevel 1 (
            set "PY=python"
            goto :eof
        )
    )
)

REM -- 3. common install locations (covers a fresh winget install
REM        whose PATH change has not reached this session yet)
for %%D in (
    "%LOCALAPPDATA%\Programs\Python"
    "%ProgramFiles%"
    "%ProgramFiles(x86)%"
    "C:"
) do (
    for /d %%P in ("%%~D\Python3*") do (
        if exist "%%~P\python.exe" if not defined PY set "PY=%%~P\python.exe"
    )
)
goto :eof
