@echo off
echo:

where python >nul 2>&1

if %ERRORLEVEL% EQU 0 (

  for %%f in (.\tests\*.py) do (
    echo:
    echo ==============================================
    echo %%f
    echo:
    python %%f
    echo:
    echo:
  )

  REM python ./tests/test.py
  REM python ./tests/test_sim.py
  REM python ./tests/sim_run.py 
  REM python ./tests/sim_run.py -t 240 -x 0 100000 -y 0 100000 -v 1000 700 -b 0 -1.0471975511965976
  REM python ./tests/sim_run.py -t 40 -x 0 10000 -y 0 10000 -v 1000 800 -b 0 -2.0

) else (
  
  echo ERROR: "python" must be into the path.

)

pause