@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) and sys.maxsize > 2**32 else 1)" >nul 2>&1
if not errorlevel 1 (
  py -3 compare.py %*
  goto finished
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) and sys.maxsize > 2**32 else 1)" >nul 2>&1
if not errorlevel 1 (
  python compare.py %*
  goto finished
)
echo Нужен Python 3.10+ x64. Установите его с python.org, затем запустите файл ещё раз.
echo Инструкция: README_RU.md
pause
exit /b 1
:finished
set "RESULT=%ERRORLEVEL%"
echo.
echo Код завершения: %RESULT%. Путь к ZIP-отчёту напечатан выше.
pause
exit /b %RESULT%
