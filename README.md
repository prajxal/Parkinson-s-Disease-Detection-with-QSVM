# Quantum vs Classical ML for Parkinson's Disease Detection

This project compares classical machine learning models with quantum kernel methods for Parkinson's disease classification using voice features from the UCI Parkinsons dataset.

## Features

- **Comprehensive EDA**: Summary statistics, histograms, class balance analysis, subject counts, correlation matrices
- **Classical Baselines**: Logistic Regression, Random Forest, and SVM (RBF kernel)
- **Quantum Kernel SVM**: Implementation using PennyLane with angle-encoding quantum feature maps
- **Subject-wise Splitting**: Prevents data leakage by ensuring all recordings from a subject are in the same fold
- **Bootstrap Confidence Intervals**: Statistical analysis with 95% CIs for all metrics
- **Comprehensive Visualizations**: ROC curves, confusion matrices, metric comparisons
- **Model Persistence**: Saves trained models and kernel matrices to disk

## Dataset

### UCI Parkinsons Dataset (Voice Features)
- **Source**: https://archive.ics.uci.edu/ml/datasets/Parkinsons
- **Instances**: 197 voice recordings from 31 people (23 with PD, 8 healthy)
- **Features**: 22 voice measurements
- **Target**: Binary classification (0=healthy, 1=Parkinson's disease)

### Parkinsons Telemonitoring Dataset
- **Source**: https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring
- **Instances**: 5,875 recordings from 42 people with early-stage PD
- **Features**: 16 voice measures + demographic info
- **Target**: Regression (UPDRS scores)

## Installation

1. Clone this repository or download the notebook
2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn pennylane scipy joblib tqdm
```

## Usage

1. Ensure the data files are in the correct location:
   - `parkinsons/parkinsons.data`
   - `parkinsons/telemonitoring/parkinsons_updrs.data`

2. Open the Jupyter notebook:
```bash
jupyter notebook parkinsons_quantum_ml_analysis.ipynb
```

3. Configure hyperparameters in the first code cell (Section 1):
   - `N_QUBITS`: Number of qubits for quantum feature map (default: 4)
   - `N_LAYERS`: Number of layers in quantum circuit (default: 2)
   - `USE_HARDWARE`: Set to `True` if using IBM Quantum hardware
   - `N_SPLITS`: Number of cross-validation folds (default: 5)
   - `PCA_VARIANCE_THRESHOLD`: Variance to retain in PCA (default: 0.95)

4. Run all cells sequentially

## Outputs

The notebook generates the following outputs in the `outputs/` directory:

- **Plots**:
  - `class_distribution.png`: Class balance visualization
  - `subject_analysis.png`: Recordings per subject analysis
  - `feature_histograms.png`: Feature distributions by class
  - `correlation_matrix.png`: Feature correlation heatmap
  - `pca_variance.png`: PCA explained variance plot
  - `metrics_comparison.png`: Model performance comparison
  - `roc_curves.png`: ROC curves for all models
  - `confusion_matrices.png`: Confusion matrices for all models

- **Data Files**:
  - `summary_statistics.csv`: Summary statistics for all features
  - `results_summary.csv`: Model performance metrics with confidence intervals

- **Models**:
  - `model_logistic_regression.pkl`
  - `model_random_forest.pkl`
  - `model_svm_(rbf).pkl`
  - `model_quantum_kernel_svm.pkl`
  - `scaler.pkl`: StandardScaler object
  - `pca.pkl`: PCA object

- **Kernel Matrices**:
  - `quantum_kernel_train.npy`: Training quantum kernel matrix
  - `quantum_kernel_test.npy`: Test quantum kernel matrix

## Using IBM Quantum Hardware

To run on IBM Quantum hardware:

1. Sign up at https://quantum-computing.ibm.com/
2. Get your API token
3. Install Qiskit:
```bash
pip install qiskit qiskit-ibm-provider
```
4. In the notebook, set:
   - `USE_HARDWARE = True`
   - `IBM_DEVICE = 'your_device_name'` (e.g., 'ibmq_lima')
5. Uncomment and configure the IBMQ authentication code in Section 6.1

## Clinical Limitations

⚠️ **Important**: This is a research/educational tool. For clinical diagnosis:
- Requires validation on larger, diverse populations
- Should be used in conjunction with clinical assessment
- Consider demographic and environmental factors
- Voice features alone may not be sufficient for diagnosis

## Methodology

1. **Data Preprocessing**: Standardization and PCA dimensionality reduction
2. **Subject-wise Splitting**: Ensures no data leakage between train/test sets
3. **Cross-Validation**: Stratified K-fold with subject-wise splitting
4. **Classical Models**: Standard scikit-learn implementations
5. **Quantum Kernel**: Angle-encoding feature map with PennyLane
6. **Evaluation**: ROC-AUC, accuracy, sensitivity, specificity with bootstrap CIs

## References

- Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM. 'Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection', BioMedical Engineering OnLine 2007, 6:23
- Tsanas A, Little MA, McSharry PE, Ramig LO. 'Accurate telemonitoring of Parkinson's disease progression by non-invasive speech tests', IEEE Transactions on Biomedical Engineering (2009)

## License

This project is for educational and research purposes only.

# Parkinson-s-Disease-Detection-with-QSVM
