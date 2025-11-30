import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import joblib

def generate_visuals(output_dir='outputs'):
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    # Load data
    results_summary = pd.read_csv(os.path.join(output_dir, 'results_summary.csv'))
    with open(os.path.join(output_dir, 'results.json'), 'r') as f:
        results_json = json.load(f)
    results_df = pd.DataFrame(results_json)
    
    # 1. AUC Bar Chart
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='pca_dim', y='auc', hue='model', errorbar='sd')
    plt.title('AUC by Model and PCA Dimension')
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(output_dir, 'plots', 'auc_bar.png'))
    plt.close()
    
    # 2. ROC Curves (Fallback using TPR/FPR points)
    # We don't have full curves, but we have TPR (sensitivity) and TNR (specificity) -> FPR = 1 - Specificity
    plt.figure(figsize=(10, 8))
    
    # Plot chance line
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    
    # We will plot the mean point for each model/dim
    # Or maybe just for the best dim? Let's plot for all.
    markers = {'LR': 'o', 'RF': 's', 'SVM-RBF': '^', 'QSVM': '*'}
    colors = {'LR': 'blue', 'RF': 'green', 'SVM-RBF': 'orange', 'QSVM': 'red'}
    
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        # Aggregate by dim
        for dim in model_data['pca_dim'].unique():
            subset = model_data[model_data['pca_dim'] == dim]
            tpr_mean = subset['sensitivity'].mean()
            fpr_mean = 1 - subset['specificity'].mean()
            auc_mean = subset['auc'].mean()
            
            plt.plot(fpr_mean, tpr_mean, marker=markers.get(model, 'o'), 
                     color=colors.get(model, 'gray'), 
                     markersize=10, 
                     label=f'{model} (Dim {dim}) AUC={auc_mean:.2f}')
            
            # Add error bars if we want? Maybe too cluttered.
            
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Point Estimates (Mean per Model/Dim)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'roc_curves.png'))
    plt.close()
    
    # 3. Confusion Matrices (Fallback)
    # We can reconstruct an approximate aggregate CM from TP, TN, FP, FN sums
    models = results_df['model'].unique()
    dims = results_df['pca_dim'].unique()
    
    # Let's pick the best dimension for each model to show
    best_dims = results_summary.loc[results_summary.groupby('model')['auc_mean'].idxmax()]
    
    fig, axes = plt.subplots(1, len(models), figsize=(4*len(models), 4))
    if len(models) == 1: axes = [axes]
    
    for i, model in enumerate(models):
        # Get best dim for this model
        best_dim = best_dims[best_dims['model'] == model]['pca_dim'].values[0]
        subset = results_df[(results_df['model'] == model) & (results_df['pca_dim'] == best_dim)]
        
        # Sum counts
        tp = subset['tp'].sum()
        tn = subset['tn'].sum()
        fp = subset['fp'].sum()
        fn = subset['fn'].sum()
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f'{model} (Dim {best_dim})')
        axes[i].set_ylabel('True')
        axes[i].set_xlabel('Predicted')
        axes[i].set_xticklabels(['Neg', 'Pos'])
        axes[i].set_yticklabels(['Neg', 'Pos'])
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'confusion_matrices.png'))
    plt.close()
    
    # 4. PCA Variance
    # Try to load pca_model_4.pkl or others
    pca_files = [f for f in os.listdir(output_dir) if f.startswith('pca_model_') and f.endswith('.pkl')]
    if pca_files:
        # Pick the one with largest dim to show more variance info, e.g. 16
        # Sort by dim
        pca_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        target_pca = pca_files[-1]
        
        try:
            pca = joblib.load(os.path.join(output_dir, target_pca))
            
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                     np.cumsum(pca.explained_variance_ratio_), 'bo-')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title(f'PCA Explained Variance ({target_pca})')
            plt.grid(True)
            plt.ylim(0, 1.05)
            plt.savefig(os.path.join(output_dir, 'plots', 'pca_variance.png'))
            plt.close()
        except Exception as e:
            print(f"Failed to plot PCA variance: {e}")
    else:
        print("No PCA model found.")

if __name__ == "__main__":
    generate_visuals()
