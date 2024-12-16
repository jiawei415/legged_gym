#!/bin/bash
set -e
set -u

task_name=$1
run_name=$2

filePath="/apdcephfs_cq10/share_1150325/ztjiaweixu/legged_cq/${task_name}/${run_name}"

last_file=$(ls -1 "$filePath" | grep .pt | tail -n 1)
ft put $filePath/$last_file
# echo $last_file
# echo ${last_file:6}
echo "${last_file:6}" | cut -d '.' -f 1 | cut -d '.' -f 2
