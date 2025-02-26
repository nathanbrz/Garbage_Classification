# Garbage Classification - Multi-Modal Project

This repository implements a multi-modal garbage classification system that leverages both image and text data to classify items into four categories: **Black**, **Blue**, **Green**, and **TTR**. The system uses transfer learning with a pretrained MobileNetV2 for image feature extraction and a pretrained DistilBERT for text feature extraction. The extracted features are fused and passed to a classifier to output predictions.

> **Note:** Detailed evaluation and performance analysis are provided in the Jupyter Notebook `model_analysis.ipynb`.

---

## Repository Structure

```
Garbage_Classification/
│
├── logs/
│   ├── error.log               # Logs capturing errors during training/evaluation.
│   └── output4.log             # Logs capturing training and validation metrics.
│
├── scripts/
│   ├── dataset_loader.py       # Custom dataset and DataLoader definitions for images and text.
│   ├── model_architecture.py   # Definition of the MultiModalGarbageClassifier model.
│   ├── model_analysis.ipynb    # Jupyter Notebook for in-depth model evaluation and analysis.
│   ├── ModelArchitecture.png   # Diagram illustrating the model architecture.
│   ├── run_script.sh           # SLURM job script for training on the TALC cluster.
│   └── train_model.py          # Main training script.
│
└── README.md                   # This README file.
```

---

## How to Use the Code

### Setup

1. **Environment:**  
   Create and activate a Conda environment (e.g., `pytorch_gpu`) with Python 3.10. Install the following packages:
   - PyTorch (GPU-enabled)
   - torchvision
   - transformers
   - matplotlib, seaborn, scikit-learn
   - JupyterLab (and ipywidgets for interactive notebooks)

   Example commands:
   ```bash
   conda create -n pytorch_gpu python=3.10
   conda activate pytorch_gpu
   conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
   conda install -c conda-forge jupyterlab ipywidgets
   pip install matplotlib seaborn scikit-learn
   ```

2. **Cluster Setup (TALC):**  
   Ensure Git, Conda, and CUDA modules are available. Use the provided `run_script.sh` to submit training jobs on a GPU node.

### Training the Model

- **Local Training:**  
  Run the training script:
  ```bash
  python train_model.py
  ```

- **On the TALC Cluster:**  
  Submit the SLURM job:
  ```bash
  sbatch run_script.sh
  ```

### Evaluating the Model

- Open the Jupyter Notebook `model_analysis.ipynb` in JupyterLab to view:
  - Class distribution and imbalance analysis.
  - Performance metrics (accuracy, precision, recall, F1-score).
  - Confusion matrix and misclassification visualizations.
  
The notebook provides detailed visualizations and insights into model performance.

---

## Files Description

- **logs/**  
  - `error.log`: Contains error messages from training/evaluation.
  - `output4.log`: Contains training and validation logs.

- **scripts/dataset_loader.py**  
  Defines the custom dataset classes and DataLoaders for processing both images and text, including data normalization and tokenization.

- **scripts/model_architecture.py**  
  Implements the `MultiModalGarbageClassifier` model:
  - **Image Branch:** Uses MobileNetV2 to extract a 256-dimensional feature vector from input images (3,224,224), followed by ReLU and LayerNorm.
  - **Text Branch:** Uses DistilBERT to extract the [CLS] token from text, then reduces it from 768 to 256 dimensions via a linear layer, followed by ReLU and LayerNorm.
  - **Fusion & Classification:** Concatenates the two 256-dimensional vectors (resulting in a 512-dimensional vector), passes it through a fusion layer (512 → 512), and finally through a classification layer (512 → 4).

- **scripts/train_model.py**  
  Contains the training loop which:
  - Loads data using the dataset loaders.
  - Trains the model with an optimizer and loss function.
  - Evaluates the model on validation and test sets.
  - Saves the best model weights to `best_model4.pth`.

- **scripts/run_script.sh**  
  A SLURM job script to run training on the TALC cluster. It sets GPU and memory requirements, activates the Conda environment, and executes the training script.

- **scripts/model_analysis.ipynb**  
  A Jupyter Notebook that analyzes the trained model. It includes:
  - Class distribution and imbalance analysis.
  - Performance metrics, confusion matrix, and misclassification visualizations.
  - Discussion of potential improvements.
  
- **scripts/ModelArchitecture.png**  
  A diagram that visually summarizes the model architecture and data flow.

