
# Repository Overview

Official implementation for the paper **A Novel Multi-Branch Self-Distillation Framework for Optimizing Remote Sensing Change Detection**. This code is built upon the [OpenCD toolbox](https://github.com/likyoo/open-cd) and [MTKD-CD](https://github.com/circleLZY/MTKD-CD).

## News
- **31/3/2024** - The code of MBSD-CD has been open-sourced.

## Dataset
The JL1-CD dataset is now publicly available. You can download the checkpoint files from:

- [Google Drive](https://drive.google.com/drive/folders/1ELoqx7J3GrEFMX5_rRynMjW9-Poxz3Uu?usp=sharing)
- [Baidu Disk](https://pan.baidu.com/s/1_vcO4c5DM5LDuOqLwLrWJg?pwd=5byn)
- [Hugging Face](https://huggingface.co/datasets/circleLZY/JL1-CD)

The SYSU-CD and CDD datasets can be downloaded through [SYSU-CD](https://github.com/liumency/SYSU-CD) and [CDD](https://pan.baidu.com/s/1Xu0kIpThW2koLcyfcJEEfA) (RSAI)
## Usage

### Install

To set up the environment, follow the installation instructions provided in the [OpenCD repository](https://github.com/likyoo/open-cd).

### Training

Below, we use the **Changer-MiT-b0** model as an example:

```bash
python tools/train.py configs/distill-changer/distill-changer_ex_mit-b0_512x512_200k_cgwx.py --work-dir /path/to/save/models/Changer-mit-b0/initial
```

### Testing

Run the following command:

```bash
python test.py <config-file> <checkpoint>
```

#### Checkpoints

All checkpoint files will soon be open sourced.
