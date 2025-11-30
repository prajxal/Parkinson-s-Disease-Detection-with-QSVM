import pandas as pd
import numpy as np
import os
import sys
sys.path.append('src')

from pca_qs_pipeline import run_pipeline

def run_smoke():
    # Load data
    data_path = 'data/parkinsons.data'
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        # Try alternate path
        data_path = 'parkinsons/parkinsons.data'
        if not os.path.exists(data_path):
             print(f"Data not found at {data_path} either.")
             return

    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape}")

    # Prepare data for pipeline
    # Features: all columns except 'name' and 'status'
    features = df.drop(['name', 'status'], axis=1).values.astype(np.float32)
    labels = df['status'].values

    # Create dummy manifest
    manifest = pd.DataFrame({
        'subject_alias': df['name'],
        'label': labels
    })

    # Save to temporary outputs
    output_dir = 'outputs/smoke_test_verify'
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'embeddings_subjects.npy'), features)
    manifest.to_csv(os.path.join(output_dir, 'subjects_manifest.csv'), index=False)

    print("Data prepared.")

    # Run Pipeline
    run_pipeline(
        embeddings_path=os.path.join(output_dir, 'embeddings_subjects.npy'),
        manifest_path=os.path.join(output_dir, 'subjects_manifest.csv'),
        output_dir=output_dir,
        pca_dims=[4],
        n_splits=3,
        n_qubits=4,
        nystrom_m=50
    )
    
    print("Smoke test finished.")

if __name__ == "__main__":
    run_smoke()
