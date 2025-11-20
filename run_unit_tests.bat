@echo off
echo:

where python >nul 2>&1

if %ERRORLEVEL% EQU 0 (

  for %%f in (.\src\acas\*.py) do (
    echo:
    echo ==============================================
    echo %%f
    echo:
    python %%f
    echo:
    echo:
  )

) else (
  
  echo ERROR: "python" must be into the path.

)

pause