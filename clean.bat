@echo off
echo =========================
echo Cleaning __pycache__...
echo =========================

for /d /r %%d in (__pycache__) do (
    if exist "%%d" (
        rd /s /q "%%d" 2>nul
        echo Deleted: %%d
    )
)

echo.
echo Done.
pause