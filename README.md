# Parkinson's Disease Detection with QSVM

This project implements a pipeline for detecting Parkinson's Disease from medical images (MRI) using a CNN feature extractor followed by a Quantum Support Vector Machine (QSVM). It also supports a classical baseline on voice data.

## Project Structure

```
ML Project/
├─ data/
│  ├─ ntua-parkinson-dataset-master/   # Place image dataset here
│  └─ parkinsons.data                  # Voice dataset (baseline)
├─ src/
│  ├─ data_loader.py                   # Image loading & manifest
│  ├─ cnn_feature_extractor.py         # ResNet18 feature extraction
│  ├─ embeddings_utils.py              # Subject aggregation
│  ├─ pca_qs_pipeline.py               # PCA + QSVM pipeline
│  ├─ quantum_kernel.py                # PennyLane kernel & Nyström
│  ├─ eval_metrics.py                  # Metrics & Plotting
│  └─ run_experiments.py               # Main script
├─ outputs/                            # Results & Artifacts
│  ├─ plots/                           # Generated plots (EDA, ROC, etc.)
│  ├─ eda.html                         # EDA Report
│  └─ ...
├─ notebooks/
│  ├─ eda.ipynb                        # Exploratory Data Analysis
│  └─ quick_smoke_test.ipynb           # Smoke test on voice data
├─ generate_eda_notebook.py            # Script to generate EDA notebook
└─ run_smoke_test_verify.py            # Verification script
```

## Setup

1. **Environment**: Python 3.11+ recommended.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Full Image Pipeline
Run the complete pipeline to extract features, train models, and evaluate:
```bash
python src/run_experiments.py --data_path data/ntua-parkinson-dataset-master --pca_dims 4,8,16
```

**Options**:
- `--pca_dims`: Comma-separated list of PCA dimensions to test (e.g., "4,8,16").
- `--n_qubits`: Number of qubits (default: 0 = auto-match PCA dim).
- `--nystrom_m`: Number of landmarks for Nyström approximation (default: 200).
- `--force_extract`: Force re-running CNN feature extraction.

### 2. Exploratory Data Analysis (EDA)
After running the pipeline (which generates embeddings), you can generate an EDA report:
```bash
# Generate and execute the EDA notebook
python generate_eda_notebook.py && jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb && jupyter nbconvert --to html notebooks/eda.ipynb --output-dir outputs/
```
This produces `outputs/eda.html` and plots in `outputs/plots/eda/`.

### 3. Verification / Smoke Test
To verify the pipeline on a small subset of data (or synthetic data):
```bash
python run_smoke_test_verify.py
```

## Outputs
Results are saved in `outputs/`:
- `embeddings.npy`: Raw CNN features.
- `embeddings_subjects.npy`: Aggregated subject features.
- `results.json`: Detailed metrics for all folds and models.
- `results_summary.csv`: Mean/Std of metrics.
- `plots/`: ROC curves, confusion matrices, and EDA visualizations.
- `eda.html`: HTML report of the EDA.

## Key Features
- **CNN Feature Extraction**: Uses ResNet18 (pretrained) to extract features from MRI slices.
- **Subject Aggregation**: Aggregates slice-level features to subject-level using mean pooling.
- **Quantum Kernel**: Implements a quantum kernel using PennyLane.
- **Nyström Approximation**: Scales the quantum kernel to larger datasets.
- **EDA**: Comprehensive exploratory data analysis of the embeddings.
