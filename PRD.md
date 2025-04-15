# Product Requirements Document: UnPuzzle

## 1. Introduction

UnPuzzle is an open-source pathology AI research pipeline designed to provide a unified and modular framework for pathology image analysis. It aims to streamline the process from data pre-processing to model deployment and task configuration, facilitating efficient benchmarking and development across Whole Slide Images (WSIs) and Region of Interest (ROI) tasks. The framework supports various advanced learning paradigms, including self-supervised learning, multi-task learning, and multi-modal learning.

## 2. Goals

*   **Provide a Unified Framework:** Offer a single, comprehensive pipeline for common pathology AI research tasks.
*   **Enable Modularity:** Allow researchers to easily swap components (data processing, models, tasks) for flexible experimentation.
*   **Support Diverse Tasks:** Accommodate both WSI-level and ROI-level analysis.
*   **Facilitate Benchmarking:** Streamline the process of comparing different models and approaches on standard datasets.
*   **Incorporate Advanced Techniques:** Support cutting-edge learning paradigms like SSL, MTL, and multi-modal learning.
*   **Promote Open Source Collaboration:** Build a community around reproducible and extensible pathology AI research.

## 3. Features

### 3.1 Data Pipeline (`DataPipe`)

*   **Tile Cropping:** Efficiently extract tiles from WSIs (e.g., `Build_tiles_dataset.py`). Supports configuration of tile size, magnification (MPP), and dataset formats (e.g., TCGA).
*   **Tile Embedding:** Generate feature embeddings from tiles using various pre-trained models (e.g., `gigapath`, ViT) (`Build_embedded_dataset.py`). Supports configurable batch sizes, workers, and loading local model weights.
*   **Dataset Configuration:** Tools for creating task-specific dataset configurations, including multi-task learning (MTL) setups (`Slide_dataset_tools.py`). Handles data splitting (e.g., k-fold cross-validation), task definition, and label mapping.

### 3.2 Model Training and Evaluation

*   **WSI Model Training (`MTL_Train.py`, potentially others):**
    *   Support for WSI-level aggregation models (e.g., ABMIL).
    *   Multi-task learning capabilities.
    *   Configurable hyperparameters (learning rate, epochs, batch size, workers).
    *   TensorBoard integration for monitoring.
    *   Warmup and staged training schedules.
*   **WSI Model Testing (`MTL_Test.py`, potentially others):**
    *   Evaluation pipeline for trained WSI models.
    *   Consistent configuration with training.
*   **ROI Model Fine-tuning (`DownStream/ROI_finetune`):**
    *   Scripts for fine-tuning pre-trained models on ROI-level classification (`CLS_Train.py`, `CLS_Test.py`) and MTL (`MTL_Train.py`, `MTL_Test.py`).
    *   Support for various models (e.g., ViT-h).
    *   Data augmentation options (e.g., `CellMix-Random`).
    *   Configurable training parameters.
*   **Pre-training (`PreTraining` - *Assumption*):** Modules likely exist for self-supervised pre-training of models, although not explicitly detailed in the quick start.
*   **Model Base (`ModelBase` - *Assumption*):** Likely contains the core implementations of different neural network architectures used in the framework (e.g., ViT, ABMIL, GigaPath encoders).

### 3.3 Utilities (`Utils`)

*   **Result Decoding:** Scripts to convert model prediction outputs into human-readable formats (e.g., CSV) (`Decode_MTL_pred.py`).
*   **Other Helper Functions:** (Further exploration needed to detail other utilities).

## 4. Design Considerations

*   **Modularity:** Components should be loosely coupled to allow easy replacement or extension.
*   **Configuration:** Utilize configuration files or command-line arguments extensively for setting up experiments (as shown in README examples).
*   **Scalability:** Design data loading and training pipelines to handle large datasets and leverage parallel processing (e.g., `PrefetchDataLoader_num_workers`, GPU utilization).
*   **Reproducibility:** Ensure experiments can be reproduced through fixed random seeds and clear configuration tracking.
*   **Extensibility:** Allow users to easily add new models, datasets, and tasks.

## 5. Future Considerations / Open Questions

*   Detail the specific models available in `ModelBase`.
*   Explore the functionalities within the `PreTraining` directory.
*   Document the specific data augmentation techniques available.
*   Clarify the supported dataset formats beyond TCGA.
*   Provide more detailed documentation for each module.
*   Add contribution guidelines.
*   Consider adding more example use cases (e.g., different datasets, multi-modal tasks).

*(This PRD is based on the initial analysis of the README.md and project structure. Further refinement will require deeper code analysis.)* 