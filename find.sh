#!/bin/bash
set -e
set -u

task_name=$1
time_id=$2
run_id=$3

log_root="/apdcephfs_cq10/share_1150325/ztjiaweixu/legged_cq/${task_name}/${time_id}_${run_id}"

last_file=$(ls -1 "$filePath" | grep .pt | tail -n 1)
ft put $filePath$last_file
# echo $last_file
echo "$last_file" | cut -d '.' -f 1 | cut -d ':' -f 2
