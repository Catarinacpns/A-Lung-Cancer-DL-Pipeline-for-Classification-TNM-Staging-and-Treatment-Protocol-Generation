def set_seed(seed=42):
    """Ensure reproducibility by setting the seed across all libraries and enforcing deterministic behavior."""
    import os
    import random
    import numpy as np
    import torch
    
    # Python random seed
    random.seed(seed)
    
    # Numpy random seed
    np.random.seed(seed)
    
    # PyTorch random seeds
    torch.manual_seed(seed)
    
    # Set environmental variables for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # CUDA determinism
    #torch.use_deterministic_algorithms(True, warn_only=True)  # Warn for non-deterministic ops
    
def seed_worker(worker_id):
        np.random.seed(42)
        random.seed(42)