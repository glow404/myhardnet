@echo off
setlocal

rem Generate a locale-independent commit date, e.g. 2026-09-02.
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set "COMMIT_DATE=%%i"

echo Pulling remote changes...
git pull
if errorlevel 1 goto :error

echo Staging changes...
git add .
if errorlevel 1 goto :error

echo Creating commit: %COMMIT_DATE%
git commit -m "%COMMIT_DATE%"
if errorlevel 1 (
    echo No commit was created. This may mean there are no changes to commit.
)

echo Pushing changes...
git push
if errorlevel 1 goto :error

echo.
echo Git synchronization completed.
pause
exit /b 0

:error
echo.
echo Git synchronization stopped because a command failed.
pause
exit /b 1
