import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, roc_curve
import os

def compute_metrics(y_true, y_pred, y_proba=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }
    
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
            metrics['auc'] = auc
        except:
            metrics['auc'] = 0.5
            
    return metrics

def bootstrap_ci(y_true, y_pred, y_proba=None, n_bootstrap=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    
    stats = {k: [] for k in ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']}
    
    for _ in range(n_bootstrap):
        indices = rng.choice(n, n, replace=True)
        yt = y_true[indices]
        yp = y_pred[indices]
        yprob = y_proba[indices] if y_proba is not None else None
        
        m = compute_metrics(yt, yp, yprob)
        for k in stats:
            if k in m:
                stats[k].append(m[k])
                
    results = {}
    for k, v in stats.items():
        if v:
            results[k] = {
                'mean': np.mean(v),
                'lower': np.percentile(v, 2.5),
                'upper': np.percentile(v, 97.5)
            }
            
    return results

def plot_confusion_matrix(y_true, y_pred, title, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(y_true, y_proba, title, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.savefig(output_path)
    plt.close()
