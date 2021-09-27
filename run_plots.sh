PYTHONPATH=$PYTHONPATH:'.'
luigi --module running ProcessData \
  --input-path /home/adam/skim3_005036.hipo \
  --local-scheduler