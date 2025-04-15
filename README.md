# UnPuzzle: A Unified Framework for Pathology Image Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Placeholder, assuming MIT License -->

## Overview

<div style="text-align: center;">
  <img width="555" alt="UnPuzzle overview" src="./Docs/images/unpuzzle_overview.jpg">
  <br/>
  <em>Figure 1: High-level overview of the UnPuzzle framework.</em>
</div>

UnPuzzle is an open-source research pipeline designed for **pathology AI**. It provides a modular and unified framework encompassing data pre-processing, model deployment, and task configuration. This enables efficient benchmarking and development for both **Whole Slide Images (WSIs)** and **Region of Interest (ROI)** tasks. The framework supports various learning paradigms like self-supervised learning (SSL), multi-task learning (MTL), and multi-modal learning.

For a detailed description of the project's goals, features, and design, please refer to the [Product Requirements Document](./PRD.md).

The licenses for the incorporated code follow their original sources (see Citing Information section).

## Table of Contents

*   [Overview](#overview)
*   [Project Structure](#project-structure)
*   [Version History](#version-history)
*   [Prerequisites](#prerequisites)
*   [Installation](#installation)
*   [Dataset Preparation](#dataset-preparation)
    *   [Tile Cropping](#3a-tile-cropping)
    *   [Tile Embedding](#3b-tile-embedding)
    *   [MTL Dataset Configuration](#3c-mtl-dataset-configuration)
*   [Running Experiments](#running-experiments)
    *   [WSI Models (MTL Example)](#4a-wsi-models-mtl-example)
    *   [ROI Models (CLS Example)](#4b-roi-models-cls-example)
    *   [ROI Models (MTL Example)](#4c-roi-models-mtl-example)
*   [Contributing](#contributing)
*   [License and Citing Information](#license-and-citing-information)

## Project Structure

The repository is organized into the following key directories:

*   `DataPipe/`: Scripts and tools for data pre-processing, including tile extraction, embedding generation, and dataset configuration.
*   `ModelBase/`: Core implementations of various neural network architectures (e.g., ViT, ABMIL, GigaPath).
*   `PreTraining/`: Modules related to self-supervised or other pre-training strategies.
*   `DownStream/`: Scripts for fine-tuning models on specific downstream tasks (e.g., classification, MTL) at both WSI and ROI levels.
*   `Utils/`: Helper scripts and utility functions, such as decoding prediction results.
*   `Docs/`: Documentation files, images, and detailed guides.
*   `environment.yaml` / `requirements.txt`: Dependency specifications.
*   `README.md`: This file.
*   `PRD.md`: Product Requirements Document.

## Version History

*   2025.2.13: Initial Release

## Prerequisites

*   **Environment:** Ubuntu Server (Recommended)
*   **Hardware:** NVIDIA GPU (e.g., H100) with CUDA Toolkit enabled.
*   **Dataset:** Example uses TCGA-BRCA dataset from GDC. Adjust paths accordingly for your data.

> For detailed tutorials, refer to:
> *   [ROI benchmark doc](./Docs/ROI_benchmark_pipeline.md)
> *   [WSI benchmark doc](./Docs/WSI_benchmark_pipeline.md)
> *   [Dataset design details](./Docs/dataset_design.md)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sagizty/UnPuzzle.git # Updated repo name assuming UnPuzzle
    cd UnPuzzle
    ```

2.  **Install dependencies:** (Using Conda is recommended)
    ```bash
    conda env create -f environment.yaml
    conda activate UnPuzzle # Use the environment name defined in environment.yaml
    pip install -r requirements.txt
    ```

## Dataset Preparation

This section outlines the steps to prepare datasets for use with UnPuzzle, using TCGA-BRCA as an example.

<div style="text-align: center;">
  <img width="1057" alt="Datapipe overview" src="./Docs/images/datapipe_overview.jpg" />
  <br/>
  <em>Figure 2: Overview of the DataPipe process.</em>
</div>

### 3.a Tile Cropping

Extracts smaller image tiles from large WSIs.

```bash
python -u DataPipe/Build_tiles_dataset.py \
    --WSI_dataset_path /path/to/your/WSI/TCGA-BRCA \
    --tiled_WSI_dataset_path /path/to/output/tiles-datasets \
    --edge_size 224 \
    --mode TCGA \
    --target_mpp 0.5
```

### 3.b Tile Embedding

Generates feature embeddings for each tile using a pre-trained model (e.g., `gigapath`).

```bash
# Using default weights (downloaded from web)
python -u DataPipe/Build_embedded_dataset.py \
    --WSI_dataset_path /path/to/output/tiles-datasets \
    --embedded_WSI_dataset_path /path/to/output/tiles-embeddings \
    --model_name gigapath \
    --edge_size 224 \
    --PrefetchDataLoader_num_workers 32 \
    --batch_size 2048 \
    --target_mpp 0.5

# Optional: Using local model weights
python -u DataPipe/Build_embedded_dataset.py \
    --WSI_dataset_path /path/to/output/tiles-datasets \
    --embedded_WSI_dataset_path /path/to/output/tiles-embeddings \
    --model_name gigapath \
    --model_weight_path /path/to/your/gigapath_tile_encoder.pt \
    --edge_size 224 \
    --PrefetchDataLoader_num_workers 32 \
    --batch_size 2048 \
    --target_mpp 0.5
```

### 3.c MTL Dataset Configuration

Sets up the data structure and configuration files needed for Multi-Task Learning experiments based on tile embeddings.

```bash
# Navigate to the DataPipe directory first if necessary
# cd DataPipe
python DataPipe/Slide_dataset_tools.py \
    --root_path /path/to/output/tiles-embeddings/ \
    --task_description_csv /path/to/your/WSI_label/TCGA-BRCA.csv \
    --subject_key image_id \
    --sample_key image_id \
    --split_target_key fold_information \
    --task_setting_folder_name task-settings-5folds \
    --mode basic \
    --dataset_name TCGA-BRCA \
    --tasks_to_run "isup_grade%gleason_score" \
    --fix_random_seed \
    --cls_task isup_grade \
    --k 5
# cd .. # Go back to the root directory if you changed into DataPipe
```
*Note: Ensure paths in commands are updated to your specific locations.*

## Running Experiments

Examples for running WSI and ROI level tasks.

### 4.a WSI Models (MTL Example)

Example using the `ABMIL` model for a WSI-level Multi-Task Learning task on the prepared TCGA-BRCA embeddings.

1.  **Train:**
    ```bash
    python -u MTL_Train.py \
        --model_name ABMIL \
        --tag tcga_brca_mtl \
        --data_path /path/to/output/tiles-embeddings \
        --ROI_feature_dim 1536 \
        --runs_path /path/to/your/runs_TCGA-BRCA \
        --enable_tensorboard \
        --task_description_csv /path/to/output/tiles-embeddings/task-settings-5folds/task_description.csv \
        --task_setting_folder_name task-settings-5folds \
        --slide_id_key slide_id \
        --split_target_key fold_information_5fold-1 \
        --max_tiles 2000 \
        --batch_size 1 \
        --num_workers 64 \
        --num_epochs 100 \
        --warmup_epochs 20 \
        --intake_epochs 50 \
        --gpu_idx 0 \
        --lr 1e-04 \
        --tasks_to_run AJCC_PATHOLOGIC_TUMOR_STAGE_reduced%HISTOLOGICAL_DIAGNOSIS \
        --padding
    ```

2.  **Test:**
    ```bash
    python -u MTL_Test.py \
        --model_name ABMIL \
        --tag tcga_brca_mtl \
        --data_path /path/to/output/tiles-embeddings \
        --ROI_feature_dim 1536 \
        --runs_path /path/to/your/runs_TCGA-BRCA \
        --task_description_csv /path/to/output/tiles-embeddings/task-settings-5folds/task_description.csv \
        --task_setting_folder_name task-settings-5folds \
        --slide_id_key slide_id \
        --split_target_key fold_information_5fold-1 \
        --max_tiles 2000 \
        --batch_size 1 \
        --num_workers 64 \
        --gpu_idx 1 \
        --tasks_to_run AJCC_PATHOLOGIC_TUMOR_STAGE_reduced%HISTOLOGICAL_DIAGNOSIS \
        --padding
    ```

3.  **Decode Results:** Converts the raw prediction outputs into a CSV file.
    ```bash
    # Navigate to Utils directory if necessary
    # cd Utils
    python Utils/Decode_MTL_pred.py \
        --model_name ABMIL \
        --tag tcga_brca_mtl \
        --data_path /path/to/output/tiles-embeddings \
        --runs_path /path/to/your/runs_TCGA-BRCA \
        --WSI_tasks True \
        --task_setting_folder_name task-settings-5folds
    # cd .. # Go back to root directory
    ```

### 4.b ROI Models (CLS Example)

Example using `ViT-h` for an ROI-level classification task on the NCT-CRC dataset.

1.  **Train:**
    ```bash
    python DownStream/ROI_finetune/CLS_Train.py \
        --task_name NCT-CRC \
        --model_name ViT-h \
        --data_path /path/to/your/NCT-CRC-HE-100K \
        --gpu_idx 0 \
        --enable_tensorboard \
        --augmentation_name CellMix-Random \
        --batch_size 128 \
        --num_epochs 1 \
        --lr 0.0001 \
        --num_workers 4 \
        --runs_path /path/to/your/runs/
    ```

2.  **Test:**
    ```bash
    # Note: --runs_path here points to the specific run directory from training
    # Note: --model_path_by_hand points to the saved model checkpoint
    python DownStream/ROI_finetune/CLS_Test.py \
        --task_name NCT-CRC \
        --model_name ViT-h \
        --gpu_idx 0 \
        --data_path /path/to/your/NCT-CRC-HE-100K \
        --data_augmentation_mode 2 \
        --enable_tensorboard \
        --edge_size 384 \
        --batch_size 128 \
        --runs_path /path/to/your/runs/CLS_NCT-CRC/ViT-h_lr_0.0001/ \
        --model_path_by_hand /path/to/your/runs/CLS_NCT-CRC/ViT-h_lr_0.0001/CLS_NCT-CRC_ViT-h_lr_0.0001.pth
    ```

### 4.c ROI Models (MTL Example)

Example using `ViT` for an ROI-level Multi-Task Learning task.

1.  **Train:**
    ```bash
    python DownStream/ROI_finetune/MTL_Train.py \
        --model_name vit \
        --tag ROI_image \
        --data_path /path/to/your/SO/tiled_data \
        --runs_path /path/to/your/runs \
        --enable_tensorboard \
        --task_setting_folder_name task-settings-5folds \
        --split_target_key fold_information_5fold-1 \
        --accum_iter_train 1 \
        --num_epochs 10 \
        --warmup_epochs 5 \
        --intake_epochs 5 \
        --tasks_to_run ACKR1%ACTA2%ADAM12%ADM%AEBP1 \
        --batch_size 128
    ```

2.  **Test:**
    ```bash
    python DownStream/ROI_finetune/MTL_Test.py \
        --model_name vit \
        --tag ROI_image \
        --data_path /path/to/your/SO/tiled_data \
        --runs_path /path/to/your/runs \
        --task_setting_folder_name task-settings-5folds \
        --split_target_key fold_information_5fold-1 \
        --tasks_to_run ACKR1%ACTA2%ADAM12%ADM%AEBP1 \
        --batch_size 128
    ```

3.  **Decode Results:**
    ```bash
    # Navigate to Utils directory if necessary
    # cd Utils
    python Utils/Decode_MTL_pred.py \
        --model_name vit \
        --tag ROI_image \
        --data_path /path/to/your/SO/tiled_data \
        --runs_path /path/to/your/runs \
        --WSI_tasks False \ # Set to False for ROI tasks
        --task_setting_folder_name task-settings-5folds
    # cd .. # Go back to root directory
    ```

## Contributing

Contributions are welcome! Please refer to the [Contribution Guidelines](./CONTRIBUTING.md) (TODO: Create this file) for details on how to submit pull requests, report issues, and suggest improvements.

## License and Citing Information

This project is open-source, likely under the MIT License (confirm and update `LICENSE.txt`).

This repository references and incorporates code from multiple open-source projects. We adhere to their respective original licenses. If you use UnPuzzle or components derived from these works in your research, please cite both UnPuzzle and the relevant original repositories:

*   **UnPuzzle:** (TODO: Add preferred citation format/paper link when available)
*   PuzzleTuning: <https://github.com/sagizty/PuzzleTuning>
*   MAE: <https://github.com/facebookresearch/mae>
*   GigaPath: <https://github.com/prov-gigapath>
*   CLIP Benchmark: <https://github.com/LAION-AI/CLIP_benchmark>
*   Accelerate: <https://github.com/huggingface/accelerate>
*   TorchScale: <https://github.com/microsoft/torchscale>
*   CLAM: <https://github.com/mahmoodlab/CLAM>
*   UNI: <https://github.com/mahmoodlab/UNI>
*   CONCH: <https://github.com/mahmoodlab/CONCH>
*   MUSK: <https://github.com/lilab-stanford/MUSK>