#!/bin/bash
set -e
set -u

play_id=$1
image_name=$2
image_tag=$3

if [ -z "$image_tag" ]
then
	echo "image tag is not provided"
	image=${image_name}
else
	image=${image_name}:${image_tag}
fi


if [ $play_id -eq 0 ]
then
    echo "running docker without display"
    docker run -it --network=host --gpus=all --name=legged_container leggedrobot:2024093001 /bin/bash
else
    export DISPLAY=$DISPLAY
	echo "setting display to $DISPLAY"
	xhost +
	docker run -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --network=host --gpus=all --name=legged_container leggedrobot:2024093001 /bin/bash
	xhost -
fi
