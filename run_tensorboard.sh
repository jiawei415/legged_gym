#!/bin/bash

run_names=$1
port=$2

if [ -z "$port" ]
then
    port=6007
fi

i=0
logdir_spec=""
# Use `echo` and `tr` to replace commas with spaces, then iterate over the resulting words
for run_name in $(echo "$run_names" | tr ',' ' ')
do
    # logdir_spec="${logdir_spec}name${i}:/apdcephfs_cq10/share_1150325/ztjiaweixu/legged_cq/random_flat/${run_name},"
    logdir_spec="${logdir_spec}${run_name}:/apdcephfs_cq10/share_1150325/ztjiaweixu/legged_cq/random_flat/${run_name},"
    i=$((i+1))
done
echo "${logdir_spec%,}" # Remove the trailing comma

tensorboard --logdir_spec=${logdir_spec%,} --port=${port} --host 0.0.0.0
