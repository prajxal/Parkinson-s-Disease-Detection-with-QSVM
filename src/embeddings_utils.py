import numpy as np
import pandas as pd
import os

def aggregate_per_subject(manifest_path, embeddings_path, output_dir, method='mean'):
    """
    Aggregates embeddings per subject.
    """
    manifest = pd.read_csv(manifest_path)
    embeddings = np.load(embeddings_path)
    
    if len(manifest) != len(embeddings):
        raise ValueError(f"Manifest length {len(manifest)} != Embeddings length {len(embeddings)}")
        
    subjects = manifest['subject_alias'].unique()
    
    agg_embeddings = []
    agg_labels = []
    agg_subjects = []
    
    print(f"Aggregating embeddings for {len(subjects)} subjects using {method}...")
    
    for subj in subjects:
        # Get indices for this subject
        indices = manifest.index[manifest['subject_alias'] == subj].tolist()
        subj_embeddings = embeddings[indices]
        
        # Get label (assume all images for a subject have same label)
        label = manifest.iloc[indices[0]]['label']
        
        if method == 'mean':
            agg_emb = np.mean(subj_embeddings, axis=0)
        elif method == 'median':
            agg_emb = np.median(subj_embeddings, axis=0)
        elif method == 'max':
            agg_emb = np.max(subj_embeddings, axis=0)
        else:
            raise ValueError("Unknown aggregation method")
            
        agg_embeddings.append(agg_emb)
        agg_labels.append(label)
        agg_subjects.append(subj)
        
    X_subjects = np.vstack(agg_embeddings)
    y_subjects = np.array(agg_labels)
    
    # Save
    np.save(os.path.join(output_dir, 'embeddings_subjects.npy'), X_subjects)
    
    subject_df = pd.DataFrame({
        'subject_alias': agg_subjects,
        'label': agg_labels
    })
    subject_df.to_csv(os.path.join(output_dir, 'subjects_manifest.csv'), index=False)
    
    print(f"Saved aggregated embeddings to {output_dir}")
    return X_subjects, y_subjects, subject_df

if __name__ == "__main__":
    # Test run
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, default='outputs/manifest.csv')
    parser.add_argument('--embeddings', type=str, default='outputs/embeddings.npy')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--method', type=str, default='mean')
    
    args = parser.parse_args()
    
    if os.path.exists(args.manifest) and os.path.exists(args.embeddings):
        aggregate_per_subject(args.manifest, args.embeddings, args.output_dir, args.method)
    else:
        print("Manifest or embeddings not found.")
