# Contents

- [Contents](#contents)
- [Description](#description)
  - [Citing](#citing)
- [Model architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script description](#script-description)
  - [Script and sample code](#script-and-sample-code)
  - [Training process](#training-process)
  - [Eval process](#eval-process)
  - [Inference Process](#inference-process)
    - [Export MindIR](#export-mindir)
    - [Infer on Ascend310](#infer-on-ascend310)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Description](#contents)

Hp-vae-gan uses a single image or video sample to generate different but similar new samples.

[Paper](https://arxiv.org/abs/2006.12226) Gur S , Benaim S , Wolf L . Hierarchical Patch VAE-GAN: Generating Diverse Videos from a Single Sample[J]. 2020.

## [Citing](#contents)

The BibTex citing format for this repository is as follows:

```BibTex
@article{hp-vae-gan,
  title={Hierarchical Patch VAE-GAN: Generating Diverse Videos from a Single Sample},
  journal={Github repository},
  publisher={Github},
  year={2022},
  howpublished={\url{https://github.com/SakiRinn/mindspore-hp-vae-gan}}
}
```

# [Model architecture](#contents)

The overall network architecture of hp-vae-gan is show below:

[Link](https://arxiv.org/abs/2006.12226)

# [Dataset](#contents)

Just a picture or a video. It can be specified by the user.

- Data format: RGB images.
- Note: We provide a sample dataset in `./data` folder.

# [Environment Requirements](#contents)

- Hardware(Ascend/GPU/CPU)
    - Prepare hardware environment with Ascend, GPU or CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```text
.
├── LICENSE
├── README.md
├── ascend310_infer
│   ├── CMakeLists.txt
│   ├── build.sh
│   ├── inc
│   │   └── utils.h
│   └── src
│       ├── main.cc
│       └── utils.cc
├── data                        # Sample dataset
│   ├── imgs
│   │   └── air_balloons.jpg
│   └── vids
│       └── air_balloons.mp4
├── eval_image.py
├── eval_video.py
├── export.py
├── postprocess.py
├── preprocess.py
├── requirements.txt
├── scripts
│   ├── run_eval_ascend.sh      # script for evaluation on Ascend 910
│   ├── run_infer_310.sh        # script for inference on Ascend 310
│   └── run_train_ascend.sh     # script for training on Ascend 910
├── src
│   ├── __init__.py
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── generate_frames.py
│   │   ├── image.py
│   │   └── video.py
│   ├── modules
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   ├── networks_2d.py
│   │   ├── networks_3d.py
│   │   └── optimizers.py
│   ├── sinFID
│   │   ├── __init__.py
│   │   ├── c3d.py
│   │   ├── fid_score.py
│   │   └── inception.py
│   ├── tools
│   │   ├── __init__.py
│   │   ├── pt2ms.py
│   │   ├── spectral_norm.py
│   │   └── trilinear.py
│   └── utils
│       ├── __init__.py
│       ├── extract.py
│       ├── images.py
│       ├── logger.py
│       ├── progress_bar.py
│       └── saver.py
├── train_image.py
├── train_video.py
└── train_video_baselines.py
```

## [Training process](#contents)

You can start training using python or shell scripts. The usage of shell scripts as follows:

```bash
sh scripts/run_train_ascend.sh IMAGE_PATH [DEVICE_ID]
```

- `IMAGE_PATH`: The filename of the training image.
- `DEVICE_ID`: The number of the Ascend device.

## [Eval process](#contents)

You can start evaluation using python or shell scripts. The usage of shell scripts as follows:

```bash
sh scripts/run_eval_ascend.sh EXPERIMENT_DIR [DEVICE_ID]
```

- `EXPERIMENT_DIR`: The directory to the training output folder.
- `DEVICE_ID`: The number of the Ascend device.

## [Inference Process](#contents)

### [Export MindIR](#contents)

Export MindIR on local.

```shell
python export.py --exp-dir [EXP_DIR] --device-id [DEVICE_ID]
```

- `EXP_DIR`: The directory to the training output folder.
- `DEVICE_ID`: The number of the Ascend device.

### [Infer on Ascend310](#contents)

Before performing inference, the mindir file must bu exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
sh scripts/run_infer_image_310.sh EXPERIMENT_DIR [DEVICE_ID]
```

- `EXPERIMENT_DIR`: The directory to the training output folder.
- `DEVICE_ID`: The number of the Ascend device.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
