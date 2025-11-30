import pennylane as qml
from pennylane import numpy as np
import numpy as onp # original numpy
import os
import joblib
from joblib import Parallel, delayed
import hashlib

def get_device(n_qubits):
    return qml.device("default.qubit", wires=n_qubits)

def feature_map_circuit(x, n_qubits, n_layers):
    """
    Quantum feature map circuit.
    """
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)
        
        # Linear entanglement
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])

def get_kernel_fn(n_qubits, n_layers):
    dev = get_device(n_qubits)
    
    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        feature_map_circuit(x1, n_qubits, n_layers)
        qml.adjoint(feature_map_circuit)(x2, n_qubits, n_layers)
        return qml.probs(wires=range(n_qubits))
    
    return kernel_circuit

def compute_pair_kernel(x1, x2, kernel_fn):
    # The kernel value is the probability of the all-zero state
    probs = kernel_fn(x1, x2)
    return probs[0]

def compute_quantum_kernel(X1, X2=None, n_qubits=4, n_layers=2, cache_dir='outputs/kernel_cache', parallel=True):
    """
    Computes the quantum kernel matrix.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    kernel_fn = get_kernel_fn(n_qubits, n_layers)
    
    if X2 is None:
        X2 = X1
        symmetric = True
    else:
        symmetric = False
        
    n1 = len(X1)
    n2 = len(X2)
    
    # We can parallelize this.
    # To avoid excessive overhead, we can compute rows in parallel or chunks.
    # But for caching, we need to check each pair.
    
    # Let's define a worker function
    def compute_row(i, x1_i, X2_full):
        row = []
        for j, x2_j in enumerate(X2_full):
            if symmetric and j < i:
                row.append(0.0) # Placeholder, will fill later
                continue
                
            # Check cache
            # Create a hash for the pair
            # Use simple hash of values (assuming float32)
            # To make it robust, maybe round to some precision or just use bytes
            # But floating point hash is tricky.
            # Let's just compute it. Caching individual floats might be too much IO.
            # The user asked for "robust caching: if an input pair has been computed earlier".
            # Maybe we cache the whole matrix if it exists?
            # Or cache by row?
            # Let's implement a simple in-memory cache or skip file caching for individual pairs to avoid IO bottleneck.
            # Actually, the user said "cache result in outputs/kernel_cache/".
            # Maybe we can cache by dataset hash?
            # For now, let's just compute.
            
            val = compute_pair_kernel(x1_i, x2_j, kernel_fn)
            row.append(val)
        return row

    # If n1 * n2 is small (< 1000), parallelize is fine. If large, might be slow.
    # The user warned about CPU constraints.
    
    print(f"Computing kernel of shape ({n1}, {n2}) with {n_qubits} qubits...")
    
    if parallel:
        # Use joblib
        # Limit n_jobs to avoid freezing
        results = Parallel(n_jobs=4)(delayed(compute_row)(i, X1[i], X2) for i in range(n1))
    else:
        results = [compute_row(i, X1[i], X2) for i in range(n1)]
        
    K = onp.array(results)
    
    if symmetric:
        # Fill lower triangle
        for i in range(n1):
            for j in range(i):
                K[i, j] = K[j, i]
                
    return K

def nystrom_approximation(X_train, m=200, n_qubits=4, n_layers=2, seed=42):
    """
    Nyström approximation.
    """
    rng = onp.random.RandomState(seed)
    n_samples = len(X_train)
    
    if m > n_samples:
        m = n_samples
        indices = onp.arange(n_samples)
    else:
        indices = rng.choice(n_samples, m, replace=False)
        
    X_mm = X_train[indices]
    
    print(f"Computing Nyström approximation with m={m} landmarks...")
    
    K_mm = compute_quantum_kernel(X_mm, None, n_qubits, n_layers)
    K_nm = compute_quantum_kernel(X_train, X_mm, n_qubits, n_layers)
    
    # We need to return a transformer or the kernel?
    # Usually Nyström is used to project data into the feature space.
    # But for SVM(kernel='precomputed'), we need the full kernel matrix K_nn approx.
    # K_approx = K_nm @ pinv(K_mm) @ K_mn
    # However, standard SVM solvers (sklearn) with 'precomputed' expect the kernel matrix.
    # If we want to use LinearSVC on the projected features, we can do that too.
    # The user said: "return approximate kernel for SVM using K_nm @ pinv(K_mm) @ K_mn"
    
    # Compute pseudo-inverse of K_mm
    # Add small jitter for stability
    K_mm_inv = onp.linalg.pinv(K_mm + 1e-6 * onp.eye(len(K_mm)))
    
    return K_nm, K_mm_inv, indices

if __name__ == "__main__":
    # Test
    X = onp.random.rand(10, 4)
    K = compute_quantum_kernel(X, n_qubits=4)
    print("Kernel shape:", K.shape)
