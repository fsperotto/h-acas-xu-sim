@echo off
echo:

where python >nul 2>&1

if %ERRORLEVEL% EQU 0 (

  python ./tests/test.py
  python ./tests/test_sim.py
  python ./tests/sim_run.py 
  python ./tests/sim_run.py -t 240 -x 0 100000 -y 0 100000 -v 1000 700 -b 0 -1.0471975511965976
  python ./tests/sim_run.py -t 40 -x 0 10000 -y 0 10000 -v 1000 800 -b 0 -2.0

) else (
  
  echo ERROR: "python" must be into the path.

)

pause