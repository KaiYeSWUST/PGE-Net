# PGE-Net
A Physics-Informed Gravity Evolution Network Guided by Uncertainty Decoupling for Complex Water Precise Extraction

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

This repository provides the model implementation, configuration files,
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

- SegFormer backbone (MIT+NFIM);
- Customized decoder head (TPUD+DGEM).

# Repository Structure
PGE-Net/
├── configs/
├── models/
│   │   ├── backbones/  
│   │   └── decode_heads/
├── tools/
├── data/
├── checkpoints/
├── requirements.txt
└── README.md
