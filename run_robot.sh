#!/bin/bash
set -e
set -u

cuda_id=$1
run_name=$2
task_name=$3

LOG_PATH=/apdcephfs_cq10/share_1150325/ztjiaweixu/legged_cq/$task_name/$run_name
mkdir -p $LOG_PATH

log_root=/apdcephfs_cq10/share_1150325/ztjiaweixu/legged_cq/$task_name

export CUDA_VISIBLE_DEVICES=$cuda_id
tag=$(date "+%Y%m%d%H%M%S")

python -m legged_gym.scripts.train --task=$task_name --log_root=$log_root --run_name=$run_name \
    --headless --jizhi --sim_device cuda --pipeline gpu \
    > $LOG_PATH/${task_name}_${tag}.out 2> $LOG_PATH/${task_name}_${tag}.err &

sleep 2

# ps -ef | grep legged | awk '{print $2}' | xargs kill -9
