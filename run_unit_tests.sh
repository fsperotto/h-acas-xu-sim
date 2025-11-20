#!/bin/bash

#check if python3 is installed
if command -v "python3" 2>&1 >/dev/null; then
  OK=0
  TOTAL=0
  #execute each .py in src
  for f in ./src/acas/*.py; do
  # Check if "$f" is a file
    if [ -f "$f" ]; then
      ((TOTAL+=1))
      echo "=============================================="
      echo "$f"
      ## Run the test
      python3 "$f"
      ## Get exist status and count if ok
      status=$?
      if [ $status -eq 0 ]; then
	    ((OK+=1))
	  else
	    echo "EXIT CODE INDICATES ERROR."
	  fi
    fi
  done
  echo "=============================================="
  echo "$TOTAL files tested."
  echo "$OK files OK."
else
  echo "ERROR: python3 command could not be found."
fi