#!/bin/bash
#ps -ef | grep "cms_server.py" | awk '{print $2}' | xargs sudo kill
eval "$(conda shell.bash hook)"
conda activate super

read trial_id
for job in 3; do
    python sdl-client/pytorch/trainingjob.py -tid "${$trial_id}"
    sleep 3
 done
echo "Running scripts in parallel"
wait # This will wait til all scripts finish
echo "Training scripts done running"
