#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
if [ $# != 1 ] && [ $# != 2 ]; then
    echo "Usage: sh scripts/run_eval_ascend.sh EXPERIMENT_DIR [DEVICE_ID]"
    echo "DEVICE_ID is optional. If not set, it defaults to 0. If you want to set more arguments, run eval_image.py directly."
exit 1
fi
experiment_dir=$1
device_id=0
if [ $# == 2 ]; then
    device_id=$2
fi

nohup python3 eval_image.py --exp-dir $experiment_dir --device-id $device_id > eval.log 2>&1 &
echo "Success! Process has started running in the background. The output will be logged in eval.log."
echo "The generated file will be stored in EXPERIMENT_DIR/eval/"