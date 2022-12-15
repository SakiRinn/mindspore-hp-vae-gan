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
# ============================================================================
device_id=0
if [ $# != 1 ] && [ $# != 2 ]; then
    echo "Usage: sh scripts/run_infer_image_310.sh EXPERIMENT_DIR [DEVICE_ID]"
    echo "DEVICE_ID is optional. If not set, it defaults to 0."
    echo "If you want to set a specific scale or checkpoint, run export.py and execute inference."
exit 1
fi

experiment_dir=$1
if [ $# == 2 ]; then
    device_id=$2
fi

# preprocess
# if [ -d $experiment_dir/infer ]; then
#     rm -rf $experiment_dir/infer/
# fi
# echo "Inference files will be stored in EXPERIMENT_DIR/infer/"
# python export.py --exp-dir $experiment_dir --device-id $device_id
# if [ $? -ne 0 ]; then
#     echo "Preprocess and export failed."
#     exit 1
# fi

# compile
export ASCEND_HOME=/usr/local/Ascend/
if [ -d ${ASCEND_HOME}/ascend-toolkit ]; then
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/ascend-toolkit/latest/atc/lib64:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export TBE_IMPL_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:${TBE_IMPL_PATH}:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp
else
    export ASCEND_HOME=/usr/local/Ascend/latest/
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/atc/ccec_compiler/bin:$ASCEND_HOME/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/atc/lib64:$ASCEND_HOME/acllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:$ASCEND_HOME/atc/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/opp
fi

cd ./ascend310_infer/ || exit
sh build.sh
if [ $? -ne 0 ]; then
    echo "Compile app code failed."
    exit 1
fi

# infer
cd - || exit
# if [ -d $experiment_dir/infer/result_Files ]; then
#     rm -rf $experiment_dir/result_Files/
# fi
# if [ -d $experiment_dir/infer/time_Result ]; then
#     rm -rf ./time_Result
# fi
# mkdir result_Files
# mkdir time_Result

./ascend310_infer/out/main --mindir_path=$experiment_dir/infer/netG.mindir \
                           --input0_path=$experiment_dir/infer/noise_init \
                           --input1_path=$experiment_dir/infer/noise_amps \
                           --device_id=$device_id
if [ $? -ne 0 ]; then
    echo "Execute inference failed."
    exit 1
fi

# postprocess
# if [ -d postprocess_Result ]; then
#     rm -rf ./postprocess_Result
# fi
# mkdir postprocess_Result
# python ./postprocess.py --output_path='./postprocess_Result/' --input_dir=$input_dir \
#                             --input_name=$input_name --scale_num=$i
# if [ $? -ne 0 ]; then
#     echo "scale $i: execute post_process failed"
#     exit 1
# fi
