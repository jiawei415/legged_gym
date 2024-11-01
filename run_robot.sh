#!/bin/bash
set -e
set -u

cuda_id=$1
task_name=$2
run_id=$3

log_root="/apdcephfs_cq10/share_1150325/ztjiaweixu/legged_cq/$task_name"
# log_root="/home/ztjiaweixu/Code/Robot/legged_gym/logs/$task_name"

export CUDA_VISIBLE_DEVICES=$cuda_id
tag=$(date "+%Y%m%d%H%M%S")

python -m legged_gym.scripts.train --task=$task_name --log_root=$log_root --run_name=$run_id --headless \
    > ~/logs/${task_name}_${tag}.out 2> ~/logs/${task_name}_${tag}.err &

# ps -ef | grep legged | awk '{print $2}' | xargs kill -9
