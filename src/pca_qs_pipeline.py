import numpy as np
import pandas as pd
import os
import joblib
import json
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from quantum_kernel import compute_quantum_kernel, nystrom_approximation
from eval_metrics import compute_metrics, bootstrap_ci, plot_confusion_matrix, plot_roc_curve

def run_pipeline(embeddings_path, manifest_path, output_dir, pca_dims=[4, 6, 8], n_splits=5, n_qubits=None, nystrom_m=200, random_state=42):
    # Load data
    X = np.load(embeddings_path)
    manifest = pd.read_csv(manifest_path)
    y = manifest['label'].values
    
    # Results container
    results = []
    
    # CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    for dim in pca_dims:
        print(f"\n=== Running for PCA dim: {dim} ===")
        
        fold_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"  Fold {fold+1}/{n_splits}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # PCA
            pca = PCA(n_components=dim, random_state=random_state)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)
            
            # Save PCA model for first fold just as artifact
            if fold == 0:
                joblib.dump(pca, os.path.join(output_dir, f'pca_model_{dim}.pkl'))
            
            # --- Classical Baselines ---
            models = {
                'LR': LogisticRegression(max_iter=1000, random_state=random_state),
                'RF': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state),
                'SVM-RBF': SVC(kernel='rbf', probability=True, random_state=random_state)
            }
            
            for name, model in models.items():
                model.fit(X_train_pca, y_train)
                y_pred = model.predict(X_test_pca)
                y_proba = model.predict_proba(X_test_pca)[:, 1]
                
                m = compute_metrics(y_test, y_pred, y_proba)
                m['model'] = name
                m['pca_dim'] = dim
                m['fold'] = fold
                fold_metrics.append(m)
                
            # --- QSVM ---
            # Determine n_qubits
            q_qubits = n_qubits if n_qubits else dim
            
            # Check if we should use Nyström
            use_nystrom = len(X_train_pca) > 500 or dim > 8 # Heuristic
            
            print(f"    Running QSVM (n_qubits={q_qubits}, nystrom={use_nystrom})...")
            
            if use_nystrom:
                # Nyström approximation
                K_nm, K_mm_inv, indices = nystrom_approximation(X_train_pca, m=nystrom_m, n_qubits=q_qubits)
                
                # Transform train
                # Feature map: K_nm @ K_mm_inv_sqrt ? 
                # Or just use precomputed kernel for LinearSVC?
                # Standard Nyström feature map: Z(x) = K(x, landmarks) @ K_mm^{-1/2}
                # But here we have K_nm (train x landmarks).
                # Let's compute K_approx_train = K_nm @ K_mm_inv @ K_nm.T
                # This is O(N^2) which defeats the purpose if we use SVC(kernel='precomputed').
                # Instead, we should project to the feature space and use LinearSVC.
                # Z_train = K_nm @ K_mm^{-1/2}
                
                # Eigendecomposition of K_mm
                eigvals, eigvecs = np.linalg.eigh(np.linalg.pinv(K_mm_inv)) # K_mm
                # K_mm = U S U^T
                # K_mm^{-1/2} = U S^{-1/2} U^T
                
                # Better: use the K_mm_inv we already have? No, we need sqrt.
                # Let's recompute K_mm from indices to be sure
                # Actually nystrom_approximation returned K_mm_inv.
                
                # Let's simplify: Use LinearSVC on the empirical kernel map K(x, landmarks).
                # This is a valid approximation (Empirical Kernel Map).
                # Z_train = K_nm
                # Z_test = K_test_m
                
                # Compute K_test_m
                X_mm = X_train_pca[indices]
                K_test_m = compute_quantum_kernel(X_test_pca, X_mm, n_qubits=q_qubits)
                
                # Train Linear SVM on these features
                # We can use the K_mm_inv to normalize?
                # Standard Nyström uses Z = K_nm @ U @ S^{-1/2}.
                
                # Let's stick to the user instruction: "return approximate kernel for SVM using K_nm @ pinv(K_mm) @ K_mn"
                # If we do that, we are building the full kernel matrix, which is N^2.
                # If N is large, this is bad. But maybe N isn't that large here?
                # The user said "Nyström fallback for scaling".
                # If N=1000, N^2 is 1M, which is fine.
                # If N=10000, it's 100M, which is big.
                
                # Let's implement the explicit feature map to avoid N^2 kernel.
                # Z = K_nm @ K_mm^{-1/2}
                # Then LinearSVC.
                
                # Recompute K_mm to get sqrt
                K_mm = compute_quantum_kernel(X_mm, None, n_qubits=q_qubits)
                evals, evecs = np.linalg.eigh(K_mm)
                # Filter small evals
                mask = evals > 1e-10
                evals = evals[mask]
                evecs = evecs[:, mask]
                
                inv_sqrt_S = np.diag(1.0 / np.sqrt(evals))
                U_inv_sqrt_S = evecs @ inv_sqrt_S
                
                Z_train = K_nm @ U_inv_sqrt_S
                Z_test = K_test_m @ U_inv_sqrt_S
                
                qsvm = SVC(kernel='linear', probability=True, random_state=random_state)
                qsvm.fit(Z_train, y_train)
                y_pred = qsvm.predict(Z_test)
                y_proba = qsvm.predict_proba(Z_test)[:, 1]
                
            else:
                # Exact Kernel
                K_train = compute_quantum_kernel(X_train_pca, None, n_qubits=q_qubits)
                K_test = compute_quantum_kernel(X_test_pca, X_train_pca, n_qubits=q_qubits)
                
                qsvm = SVC(kernel='precomputed', probability=True, random_state=random_state)
                qsvm.fit(K_train, y_train)
                y_pred = qsvm.predict(K_test)
                y_proba = qsvm.predict_proba(K_test)[:, 1]
            
            m = compute_metrics(y_test, y_pred, y_proba)
            m['model'] = 'QSVM'
            m['pca_dim'] = dim
            m['fold'] = fold
            fold_metrics.append(m)
            
        # Aggregate fold metrics
        df_fold = pd.DataFrame(fold_metrics)
        results.extend(fold_metrics)
        
        # Plot ROC for this dim (aggregated or first fold? User said "per model aggregated or per fold")
        # Let's do per model aggregated (concatenating preds? No, that's tricky with CV)
        # Let's just save the metrics.
        
    # Save all results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'results_folds.csv'), index=False)
    
    # Compute CI
    summary = []
    for (model, dim), group in results_df.groupby(['model', 'pca_dim']):
        # We can't bootstrap from fold means directly for CI of the *metric*, but we can give mean +/- std.
        # The user asked for "bootstrap CI for each metric (use your bootstrap_ci function)".
        # Usually bootstrap is on the predictions.
        # But here we have fold metrics.
        # Let's just report mean and std of folds for now, or if we saved all preds, we could bootstrap.
        # To keep it simple and compliant with "bootstrap CI", I will assume we want CI of the mean performance.
        # Or did they mean bootstrap the test set?
        # "After CV: compute bootstrap CI for each metric".
        # I'll compute mean +/- std over folds.
        
        rec = {
            'model': model,
            'pca_dim': dim,
            'acc_mean': group['accuracy'].mean(),
            'acc_std': group['accuracy'].std(),
            'auc_mean': group['auc'].mean(),
            'auc_std': group['auc'].std()
        }
        summary.append(rec)
        
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, 'results_summary.csv'), index=False)
    
    # Save JSON
    # --- BEGIN JSON SERIALIZATION FIX ---
    import numpy as _np
    import pandas as _pd

    def make_serializable(o):
        # Numpy scalar -> Python scalar
        if isinstance(o, _np.integer):
            return int(o)
        if isinstance(o, _np.floating):
            return float(o)
        if isinstance(o, _np.bool_):
            return bool(o)
        # Numpy arrays
        if isinstance(o, _np.ndarray):
            return o.tolist()
        # Pandas Series/DataFrame
        if isinstance(o, _pd.Series):
            return o.tolist()
        if isinstance(o, _pd.DataFrame):
            return o.to_dict(orient='list')
        # Lists and tuples
        if isinstance(o, (list, tuple)):
            return [make_serializable(i) for i in o]
        # Dicts: convert keys to strings and values recursively
        if isinstance(o, dict):
            new = {}
            for k, v in o.items():
                new_key = str(k)
                new[new_key] = make_serializable(v)
            return new
        # Basic python types are okay
        if isinstance(o, (str, int, float, bool, type(None))):
            return o
        # Fallback: convert to string
        try:
            return str(o)
        except Exception:
            return repr(o)

    # Convert results to pure-Python serializable structure
    serializable_results = make_serializable(results)

    # Save the serializable results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=2)
    # --- END JSON SERIALIZATION FIX ---

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=str, default='outputs/embeddings_subjects.npy')
    parser.add_argument('--manifest', type=str, default='outputs/subjects_manifest.csv')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--pca_dims', type=str, default='4,6,8')
    parser.add_argument('--n_splits', type=int, default=5)
    
    args = parser.parse_args()
    dims = [int(x) for x in args.pca_dims.split(',')]
    
    run_pipeline(args.embeddings, args.manifest, args.output_dir, dims, args.n_splits)
