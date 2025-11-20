# ACAS-Xu Simulator

ONERA Copyright

DO NOT DISTRIBUTE

## installing as a package :

  conda create --name <VENV_NAME> pip conda
  
  conda activate <VENV_NAME>
  
  cd <PATH_TO_ACAS_SIM>
  
  pip install -e .

## running :

Random encounter :

  python ./tests/test_sim.py

Defined encounter :

  python ./tests/sim_run.py -h     (to see parameters) 

  python ./tests/sim_run.py -t 40 -x 0 10000 -y 0 10000 -v 1000 800 -b 0 -2.0
