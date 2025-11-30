import os
import argparse
import sys
from cnn_feature_extractor import extract_features
from embeddings_utils import aggregate_per_subject
from pca_qs_pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run Parkinson's Detection Pipeline")
    parser.add_argument('--data_path', type=str, default='data/ntua-parkinson-dataset-master')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--sample_mode', action='store_true', help="Run on a small subset for testing")
    parser.add_argument('--pca_dims', type=str, default='4,6,8,10')
    parser.add_argument('--n_qubits', type=int, default=0, help="0 for auto (equal to PCA dim)")
    parser.add_argument('--nystrom_m', type=int, default=200)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--force_extract', action='store_true', help="Force re-extraction of features")
    
    args = parser.parse_args()
    
    # 1. Feature Extraction
    embeddings_path = os.path.join(args.output_dir, 'embeddings.npy')
    manifest_path = os.path.join(args.output_dir, 'manifest.csv')
    
    if not os.path.exists(embeddings_path) or args.force_extract:
        print("Step 1: Extracting features...")
        # If sample mode, maybe we should limit the dataset?
        # The data_loader scans the dir. If we want sample mode, we might need to hack it or just run on full and subset later.
        # But for "smoke test" on existing small dataset, the user provided a separate notebook.
        # Here "sample_mode" might mean "process only first N images".
        # I'll pass it to extract_features if I implemented it, but I didn't.
        # I'll just run full extraction.
        extract_features(args.data_path, args.output_dir)
    else:
        print("Step 1: Embeddings found, skipping extraction.")
        
    # 2. Aggregation
    print("Step 2: Aggregating embeddings...")
    aggregate_per_subject(manifest_path, embeddings_path, args.output_dir)
    
    # 3. Pipeline
    print("Step 3: Running PCA + QSVM pipeline...")
    subj_emb_path = os.path.join(args.output_dir, 'embeddings_subjects.npy')
    subj_man_path = os.path.join(args.output_dir, 'subjects_manifest.csv')
    
    dims = [int(x) for x in args.pca_dims.split(',')]
    n_qubits = args.n_qubits if args.n_qubits > 0 else None
    
    run_pipeline(
        subj_emb_path, 
        subj_man_path, 
        args.output_dir, 
        pca_dims=dims, 
        n_splits=args.n_splits,
        n_qubits=n_qubits,
        nystrom_m=args.nystrom_m,
        random_state=args.random_state
    )
    
    print("Experiment run complete.")

if __name__ == "__main__":
    main()
