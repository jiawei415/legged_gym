#!/bin/bash

task_names=$1
run_names=$2
port=$3

if [ -z "$port" ]
then
    port=6007
fi

i=0
logdir_spec=""
# Use `echo` and `tr` to replace commas with spaces, then iterate over the resulting words
for task_name in $(echo "$task_names" | tr ',' ' ')
do
    for run_name in $(echo "$run_names" | tr ',' ' ')
    do
        logdir_spec="${logdir_spec}${task_name}_${run_name}:/apdcephfs_cq10/share_1150325/ztjiaweixu/legged_cq/${task_name}/${run_name},"
        # logdir_spec="${logdir_spec}${run_name}_${task_name}:/apdcephfs_cq10/share_1150325/ztjiaweixu/legged_cq/${task_name}/${run_name},"
        i=$((i+1))
    done
done
echo "${logdir_spec%,}" # Remove the trailing comma

tensorboard --logdir_spec=${logdir_spec%,} --port=${port} --host 0.0.0.0
