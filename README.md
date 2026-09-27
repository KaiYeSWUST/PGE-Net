# PGE-Net
A Physics-Informed Gravity Evolution Network Guided by Uncertainty Decoupling for Precise Water Extraction in Complex Water Areas

## Overview
PGE-Net is an end-to-end semantic segmentation network for precise water
extraction in complex water areas. The model takes four-band RGB+NIR imagery
as input and adopts a SegFormer-based encoder-decoder architecture.

PGE-Net consists of four main components:

- a SegFormer backbone;
- a Near-Infrared Spectral-Guided Feature Interaction Module (NFIM);
- a Triple-Probe Jointly Driven Uncertainty Decoupling Module (TPUD);
- a Dynamic Gravity Evolution Module (DGEM).

NFIM introduces near-infrared spectral information into the shallow feature
stages. TPUD estimates spatial uncertainty and decouples ambiguous features
into boundary and main-body branches. DGEM further refines uncertain water
boundaries through a physics-informed gravity evolution process.

This repository provides the model code, configuration files,
training and inference scripts, visualization tools, and experimental
configuration used in this work. The implementation is developed on top of
the MMSegmentation framework.

## Features

- **Four-band RGB+NIR input:** Supports high-resolution four-band imagery
  containing red, green, blue, and near-infrared channels.

- **SegFormer-based encoder-decoder architecture:** Uses a hierarchical
  SegFormer backbone to extract multiscale features and a customized decoder
  head to produce dense water segmentation predictions.

- **Near-Infrared Spectral-Guided Feature Interaction Module (NFIM):**
  Introduces near-infrared spectral information into the shallow feature
  stages and uses it as a physical prior to enhance water-related features.

- **Triple-Probe Jointly Driven Uncertainty Decoupling Module (TPUD):**
  Estimates spatial uncertainty using semantic, morphological, and
  near-infrared-guided physical probes.

- **Boundary and main-body feature decoupling:** Separates ambiguous
  high-uncertainty boundary regions from relatively stable water-body regions,
  enabling differentiated feature enhancement.

- **Dynamic Gravity Evolution Module (DGEM):**
  Builds a physics-informed gravity and resistance field from spectral,
  albedo, spatial, and feature information to iteratively refine uncertain
  water boundaries.

- **Physics-informed boundary refinement:** Uses a differentiable
  convection-diffusion-style evolution process to guide uncertain semantic
  boundaries toward more accurate water shorelines.

- **MMSegmentation-based implementation:** Follows the configuration-driven
  model construction, training, evaluation, and inference framework provided
  by MMSegmentation.

- **Configurable experiments:** Provides configuration files for model
  construction, dataset settings, training, evaluation, and ablation
  experiments.

- **Inference and visualization:** Provides scripts for applying the trained
  model to input images and generating visualized segmentation results.

## Model Architecture

The detailed architecture is described in the associated paper. The
implementation is organized into the following components:

- **Backbone:** A customized SegFormer MIT backbone with NFIM.
- **Decoder head:** A customized SegFormer decoder head containing TPUD and DGEM.

## Runtime Environment
The code was developed and tested with a specific MMSegmentation
environment. Please install the versions listed in `requirements.txt`.
The installation of PyTorch and MMCV may depend on the local CUDA version.
Please select compatible packages according to the official PyTorch and
OpenMMLab installation instructions.

The code was developed and tested with the following environment:
- Python: 3.8.20
- PyTorch: 2.0.0+cu118
- MMCV: 2.0.0
- MMEngine: 0.10.7
- MMSegmentation: 1.2.2
- GPU: NVIDIA RTX 4070 Super
- Operating system: Windows 11


## Repository Structure
```text
PGE-Net/
├── configs/
│   └── segformer/
│       └── gid5_segformer_mit_b0_qiepian_base_nfim_ufda_dgem_pdes.py
├── mmseg/
│   ├── models/
│   │   ├── backbones/
│   │   │   └── mit.py
│   │   │   └── mit_nfim.py
│   │   └── decode_heads/
│   │   │   └── segformer_head.py
│   │       └── ufda_dgem_segformer_head_pdes.py
│   └── ...
├── tools/
│   ├── train.py
│   ├── gid5_visual_add_nfim_ufda_dgem_map_PDES.py
│   ├── 
│   └── visualize.py
├── data/
│   └── train/
│   │   └── images
│   │   └── masks
│   └── val/
│   │   └── images
│   │   └── masks
├── checkpoints/
│   └── README.md
├── requirements.txt
├── LICENSE
├── NOTICE
└── README.md
```

## Dataset
Using the PGE-Net model, we have created the ZWD dataset for water body extraction and segmentation tasks. The dataset, including its description, download instructions, directory
structure, annotation details, and license, is available at https://github.com/SWUSTKAI/ZWD-DATASET.

## Training
Run the following command from the repository root:
```bash
python tools/train.py configs/segformer/gid5_segformer_mit_b0_qiepian_base_nfim_ufda_dgem_pdes.py
```
Note that this instruction needs to be executed within the **complete mmsegmentation** framework. The link for mmseg is https://github.com/open-mmlab/mmsegmentation

## Inference and Visualization
After training, use the trained checkpoint to generate segmentation predictions and visualization results.
```bash
python tools/gid5_visual_add_nfim_ufda_dgem_map_PDES.py
```
The paths of files such as "config", "checkpoint", "input image", "ground truth" and "output path" should be filled in the code.

Arguments:
- CONFIG_FILE: Path to the model configuration file.
- CHECKPOINT_FILE: Path to the trained PGE-Net checkpoint.
- INPUT_IMG_PATH: Path to the input image or image directory.
- GT_IMG_PATH: Path to the ground truth image.
- OUTPUT_PATH: Directory used to save the prediction and visualization results.

## Citation
The citation information will be added after the associated paper is officially published.

## Acknowledgements
We thank the High-resolution Earth Observation System Sichuan Data & Application Center for providing the basic remote-sensing imagery. We also thank the Sichuan Province Aba Ecological Environment Monitoring Center Station for its assistance during the field investigation.

## License
This repository contains components distributed under different licenses.
1. Original PGE-Net code and newly developed PGE-Net components
   Unless otherwise stated, the original PGE-Net code and newly developed
   PGE-Net components are distributed under the PGE-Net Non-Commercial
   Research License. See LICENSE.
2. MMSegmentation and other third-party components
   The original MMSegmentation components and other third-party components
   remain subject to their respective original licenses. In particular,
   MMSegmentation is distributed under the Apache License, Version 2.0.
   See LICENSE-APACHE-2.0 and NOTICE.
3. ZWD dataset
   The ZWD dataset is not covered by the software license in this
   repository. It is distributed under a separate dataset license specified
   in the ZWD dataset repository:https://github.com/SWUSTKAI/ZWD-DATASET

In case of conflict, the license applicable to the relevant file or
component takes precedence.

