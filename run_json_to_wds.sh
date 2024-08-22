#/usr/bin/env bash
#
data=$1
outputshard=$2

python tools/prepare_ml_caps_json_to_wds.py --data $data --outputshard $outputshard 
