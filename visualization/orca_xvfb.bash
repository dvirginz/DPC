#!/bin/bash
# set -x #echo on

/usr/bin/xvfb-run --auto-servernum --server-args="-screen 0 640x480x24 +extension RANDR +extension GLX" ~/anaconda3/envs/coup_1/bin/orca "$@"