#!/bin/bash
# Series 3' (docs/plan_part4.md): 3 runs at once, nice -n 10 (day session, user 2026-11-13)
cd /home/ace/projects/ksquares-rl/scripts
source ../.venv/bin/activate
nice -n 10 python graph_s3.py run --config configs/graph_s3p.yaml 2>&1 | tee -a ../runs/graph_s3p.log
