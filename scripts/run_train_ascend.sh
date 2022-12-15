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
device_id=0
if [ $# != 1 ] && [ $# != 2 ]; then
    echo "Usage: sh scripts/run_train_ascend.sh IMAGE_PATH [DEVICE_ID]"
    echo "DEVICE_ID is optional. If not set, it defaults to 0. If you want to set more arguments, run train_image.py directly."
exit 1
fi
image_path=$1
if [ $# == 2 ]; then
    device_id=$2
fi

nohup python3 train_image.py --image-path $image_path --checkname image --device-id $device_id > train.log 2>&1 &
echo "Success! Process has started running in the background. The output will be logged in train.log."
echo "The generated file will be stored in ./run/image/experiment_*/"