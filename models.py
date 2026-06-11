import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
import numpy as np
from sklearn.model_selection import KFold
from snntorch import functional as SF
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = [
    'ball', 'battery', 'bracket', 'coin', 'empty_can', 'empty_hand', 'full_can',
    'gel', 'lotion', 'mug', 'pen', 'safety_glasses', 'scissors', 'screw_driver',
    'spray_can', 'stapler', 'tape'
]

# FC_SNN_Leaky
class FC_SNN_Leaky(nn.Module):
    def __init__(self, beta=None, spike_grad=surrogate.atan(), num_outputs=17, num_inputs=2*484):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, 3 * 484)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=1.0, learn_beta=True)
        self.fc2 = nn.Linear(3 * 484, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)

    def forward(self, x, syn1_prev=None, mem1_prev=None, mem2_prev=None):
        # x: (batch_size, 2 * num_pixels) = (batch_size, 968)
        batch_size = x.size(0)
        
        # Initialize states for Leaky layer 1 if None
        if mem1_prev is None:
            mem1_init = self.lif1.init_leaky()
            mem1_prev = mem1_init.to(x.device).repeat(batch_size, 1)
        
        # Initialize states for Leaky layer 2 if None
        if mem2_prev is None:
            mem2_init = self.lif2.init_leaky()
            mem2_prev = mem2_init.to(x.device).repeat(batch_size, 1)
    

        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1_prev)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2_prev)

        return spk2, mem2, mem1

def forward_pass_lky(net, data):
    # data: (num_time_steps, batch_size, 2 * num_pixels) = (50, batch_size, 968)
    spk_rec = []
    mem_rec = []  # Collect membrane from output Leaky layer
    mem1_prev = None
    mem2_prev = None
    for step in range(data.size(0)):  # Iterate over time steps
        scaled_data = data[step]  # Scale input to promote spiking
        spk_out, mem2, mem1 = net(scaled_data, mem1_prev, mem2_prev)
        mem_rec.append(mem2)
        spk_rec.append(spk_out)
        mem1_prev = mem1
        mem2_prev = mem2
    return torch.stack(spk_rec), torch.stack(mem_rec)

# FC_SNN_Syn
class FC_SNN_Syn(nn.Module):
    def __init__(self, num_inputs=484, num_outputs=17, fixed_fan_in=None, alpha=0.7, beta=0.7):
        super().__init__()
        
        # Define Layers
        hidden = 3 * num_inputs
        self.fc1 = nn.Linear(num_inputs, hidden, bias=False)
        self.lif1 = snn.Synaptic(alpha=alpha, beta=beta, reset_mechanism='zero')
        
        self.fc2 = nn.Linear(hidden, num_outputs, bias=False)
        self.lif2 = snn.Synaptic(alpha=alpha, beta=beta, output=True, reset_mechanism='zero')

        # --- HARDWARE CONSTRAINT LOGIC ---
        self.fan_in_limit = fixed_fan_in
        
        # 1. Register masks as buffers (always register them to allow safe loading/saving)
        # Initialize with ONES (pass-through) by default
        self.register_buffer('fc1_mask', torch.ones_like(self.fc1.weight))
        self.register_buffer('fc2_mask', torch.ones_like(self.fc2.weight))
        
        # 2. Initialize the masks ONLY IF a limit is set
        if self.fan_in_limit is not None:
            self._init_mask(self.fc1, 'fc1_mask')
            self._init_mask(self.fc2, 'fc2_mask')
        
    def _init_mask(self, layer, mask_name):
        """
        Creates a binary mask where each row (output neuron) has exactly 
        'fan_in_limit' connections selected at random.
        """
        # Safety check: do nothing if limit is None
        if self.fan_in_limit is None:
            return

        n_out, n_in = layer.weight.shape
        
        # Start with all zeros
        new_mask = torch.zeros(n_out, n_in)
        
        for i in range(n_out):
            # If input dimension is smaller than limit, keep all (no pruning needed)
            if n_in <= self.fan_in_limit:
                new_mask[i, :] = 1.0
            else:
                # Randomly select 'fan_in_limit' indices
                perm = torch.randperm(n_in)[:self.fan_in_limit]
                new_mask[i, perm] = 1.0
        
        # Update buffer and weights
        setattr(self, mask_name, new_mask.to(layer.weight.device))
        with torch.no_grad():
            layer.weight.data *= getattr(self, mask_name)

    def enforce_sparsity(self):
        """
        MUST be called after optimizer.step() to zero out gradients 
        applied to pruned connections.
        """
        # Only enforce if we actually have a constraint
        if self.fan_in_limit is not None:
            with torch.no_grad():
                self.fc1.weight.data *= self.fc1_mask
                self.fc2.weight.data *= self.fc2_mask

    def forward(self, x, syn1_prev=None, mem1_prev=None, syn2_prev=None, mem2_prev=None):
        batch_size = x.size(0)
        
        # Initialize states if None
        if syn1_prev is None:
            syn1_prev, mem1_prev = self.lif1.init_synaptic()
            syn1_prev = syn1_prev.to(x.device).repeat(batch_size, 1)
            mem1_prev = mem1_prev.to(x.device).repeat(batch_size, 1)
        
        if syn2_prev is None:
            syn2_prev, mem2_prev = self.lif2.init_synaptic()
            syn2_prev = syn2_prev.to(x.device).repeat(batch_size, 1)
            mem2_prev = mem2_prev.to(x.device).repeat(batch_size, 1)
        
        # Layer 1
        cur1 = self.fc1(x)
        spk1, syn1, mem1 = self.lif1(cur1, syn1_prev, mem1_prev)

        # Layer 2
        cur2 = self.fc2(spk1)
        spk2, syn2, mem2 = self.lif2(cur2, syn2_prev, mem2_prev)

        return spk2, syn2, mem2, syn1, mem1, spk1

def forward_pass_syn(net, data):
    # data: (num_time_steps, batch_size, 2 * num_pixels) = (50, batch_size, 968)
    spk_rec = []
    mem2_rec = []
    mem1_rec = []
    spk1_rec = []  #hidden spikes
    syn1_prev = mem1_prev = syn2_prev = mem2_prev = None

    for step in range(data.size(0)):  # Iterate over time steps
        x_t = data[step] # Scale input to promote spiking
        spk_out, syn2, mem2, syn1, mem1, spk1 = net(x_t, syn1_prev, mem1_prev, syn2_prev, mem2_prev)

        spk_rec.append(spk_out)
        mem2_rec.append(mem2)
        mem1_rec.append(mem1)
        spk1_rec.append(spk1)

        syn1_prev, mem1_prev = syn1, mem1
        syn2_prev, mem2_prev = syn2, mem2
        
    return (
        torch.stack(spk_rec),
        torch.stack(mem2_rec),
        torch.stack(mem1_rec),
        torch.stack(spk1_rec)
    ) 

class FC_SNN_Syn_33(nn.Module):
    def __init__(self, beta=None, alpha=None, sparsity=None, fixed_fan_in=None, spike_grad=surrogate.atan(),
                 num_outputs=None, num_inputs=None):
        super().__init__()
        
        # Layer 1: input → 3 * num_inputs
        self.fc1 = nn.Linear(num_inputs, 3 * num_inputs)
        self.lif1 = snn.Synaptic(
            alpha=alpha, beta=beta, spike_grad=spike_grad,
            threshold=1.0, learn_threshold=True, learn_alpha=True, learn_beta=True
        )
        
        # NEW LAYER: 3*num_inputs → 2*num_inputs
        self.fc2 = nn.Linear(3 * num_inputs, 3 * num_inputs)
        self.lif2 = snn.Synaptic(
            alpha=alpha, beta=beta, spike_grad=spike_grad,
            threshold=1.0, learn_threshold=True, learn_alpha=True, learn_beta=True
        )
        
        # Output layer: 2*num_inputs → num_outputs
        self.fc3 = nn.Linear(3 * num_inputs, num_outputs)
        self.lif3 = snn.Synaptic(
            alpha=alpha, beta=beta, spike_grad=spike_grad,
            learn_alpha=False, learn_beta=False, output=True
        )
        
    def enforce_sparsity(self):
        if hasattr(self, 'fc1_mask') and self.fc1_mask.mean() < 1.0:
            with torch.no_grad():
                self.fc1.weight.data *= self.fc1_mask  # re-zero pruned weights    

    def forward(self, x,
                syn1_prev=None, mem1_prev=None,
                syn2_prev=None, mem2_prev=None,
                syn3_prev=None, mem3_prev=None):
        """
        x: (batch_size, num_inputs)
        Returns: spk3, syn3, mem3, syn2, mem2, syn1, mem1
        """
        batch_size = x.size(0)

        # === Layer 1 ===
        if syn1_prev is None or mem1_prev is None:
            syn1_init, mem1_init = self.lif1.init_synaptic()
            syn1_prev = syn1_init.to(x.device).repeat(batch_size, 1)
            mem1_prev = mem1_init.to(x.device).repeat(batch_size, 1)

        cur1 = self.fc1(x)
        spk1, syn1, mem1 = self.lif1(cur1, syn1_prev, mem1_prev)

        # === NEW Layer 2 (hidden) ===
        if syn2_prev is None or mem2_prev is None:
            syn2_init, mem2_init = self.lif2.init_synaptic()
            syn2_prev = syn2_init.to(x.device).repeat(batch_size, 1)
            mem2_prev = mem2_init.to(x.device).repeat(batch_size, 1)

        cur2 = self.fc2(spk1)
        spk2, syn2, mem2 = self.lif2(cur2, syn2_prev, mem2_prev)

        # === Layer 3 (output) ===
        if syn3_prev is None or mem3_prev is None:
            syn3_init, mem3_init = self.lif3.init_synaptic()
            syn3_prev = syn3_init.to(x.device).repeat(batch_size, 1)
            mem3_prev = mem3_init.to(x.device).repeat(batch_size, 1)

        cur3 = self.fc3(spk2)
        spk3, syn3, mem3 = self.lif3(cur3, syn3_prev, mem3_prev)

        return spk3, syn3, mem3, syn2, mem2, syn1, mem1, spk1
    
def forward_pass_syn_33(net, data):
    """
    data: (num_time_steps, batch_size, num_inputs)
    Returns: spk_rec (output spikes), mem_rec (output membrane)
    """
    spk_rec = []
    mem_rec = []
    mem1_rec = []
    spk1_rec = []  #hidden spikes
    # Initialize hidden states
    syn1_prev = mem1_prev = None
    syn2_prev = mem2_prev = None
    syn3_prev = mem3_prev = None

    for step in range(data.size(0)):
        x_t = data[step]  # (batch_size, num_inputs)
        
        (spk_out, syn3, mem3,
         syn2, mem2,
         syn1, mem1, spk1) = net(
            x_t,
            syn1_prev, mem1_prev,
            syn2_prev, mem2_prev,
            syn3_prev, mem3_prev
        )
        
        spk_rec.append(spk_out)
        mem_rec.append(mem3)
        mem1_rec.append(mem1)
        spk1_rec.append(spk1)

        # Update previous states
        syn1_prev, mem1_prev = syn1, mem1
        syn2_prev, mem2_prev = syn2, mem2
        syn3_prev, mem3_prev = syn3, mem3

    return (
        torch.stack(spk_rec),
        torch.stack(mem_rec),
        torch.stack(mem1_rec),
        torch.stack(spk1_rec)
    ) 

class FC_SNN_Syn_32(nn.Module):
    def __init__(self, beta=None, alpha=None, sparsity=None, fixed_fan_in=None, spike_grad=surrogate.atan(),
                 num_outputs=None, num_inputs=None):
        super().__init__()
        
        # Layer 1: input → 3 * num_inputs
        self.fc1 = nn.Linear(num_inputs, 3 * num_inputs)
        self.lif1 = snn.Synaptic(
            alpha=alpha, beta=beta, spike_grad=spike_grad,
            threshold=1.0, learn_threshold=True, learn_alpha=True, learn_beta=True
        )
        
        # NEW LAYER: 3*num_inputs → 2*num_inputs
        self.fc2 = nn.Linear(3 * num_inputs, 2 * num_inputs)
        self.lif2 = snn.Synaptic(
            alpha=alpha, beta=beta, spike_grad=spike_grad,
            threshold=1.0, learn_threshold=True, learn_alpha=True, learn_beta=True
        )
        
        # Output layer: 2*num_inputs → num_outputs
        self.fc3 = nn.Linear(2 * num_inputs, num_outputs)
        self.lif3 = snn.Synaptic(
            alpha=alpha, beta=beta, spike_grad=spike_grad,
            learn_alpha=False, learn_beta=False, output=True
        )
        
    def enforce_sparsity(self):
        if hasattr(self, 'fc1_mask') and self.fc1_mask.mean() < 1.0:
            with torch.no_grad():
                self.fc1.weight.data *= self.fc1_mask  # re-zero pruned weights
                
    def forward(self, x,
                syn1_prev=None, mem1_prev=None,
                syn2_prev=None, mem2_prev=None,
                syn3_prev=None, mem3_prev=None):
        """
        x: (batch_size, num_inputs)
        Returns: spk3, syn3, mem3, syn2, mem2, syn1, mem1
        """
        batch_size = x.size(0)

        # === Layer 1 ===
        if syn1_prev is None or mem1_prev is None:
            syn1_init, mem1_init = self.lif1.init_synaptic()
            syn1_prev = syn1_init.to(x.device).repeat(batch_size, 1)
            mem1_prev = mem1_init.to(x.device).repeat(batch_size, 1)

        cur1 = self.fc1(x)
        spk1, syn1, mem1 = self.lif1(cur1, syn1_prev, mem1_prev)

        # === NEW Layer 2 (hidden) ===
        if syn2_prev is None or mem2_prev is None:
            syn2_init, mem2_init = self.lif2.init_synaptic()
            syn2_prev = syn2_init.to(x.device).repeat(batch_size, 1)
            mem2_prev = mem2_init.to(x.device).repeat(batch_size, 1)

        cur2 = self.fc2(spk1)
        spk2, syn2, mem2 = self.lif2(cur2, syn2_prev, mem2_prev)

        # === Layer 3 (output) ===
        if syn3_prev is None or mem3_prev is None:
            syn3_init, mem3_init = self.lif3.init_synaptic()
            syn3_prev = syn3_init.to(x.device).repeat(batch_size, 1)
            mem3_prev = mem3_init.to(x.device).repeat(batch_size, 1)

        cur3 = self.fc3(spk2)
        spk3, syn3, mem3 = self.lif3(cur3, syn3_prev, mem3_prev)

        return spk3, syn3, mem3, syn2, mem2, syn1, mem1, spk1

def forward_pass_syn_32(net, data):
    """
    data: (num_time_steps, batch_size, num_inputs)
    Returns: spk_rec (output spikes), mem_rec (output membrane)
    """
    spk_rec = []
    mem_rec = []
    mem1_rec = []
    spk1_rec = []  #hidden spikes
    # Initialize hidden states
    syn1_prev = mem1_prev = None
    syn2_prev = mem2_prev = None
    syn3_prev = mem3_prev = None

    for step in range(data.size(0)):
        x_t = data[step]  # (batch_size, num_inputs)
        
        (spk_out, syn3, mem3,
         syn2, mem2,
         syn1, mem1, spk1) = net(
            x_t,
            syn1_prev, mem1_prev,
            syn2_prev, mem2_prev,
            syn3_prev, mem3_prev
        )
        
        spk_rec.append(spk_out)
        mem_rec.append(mem3)
        mem1_rec.append(mem1)  
        spk1_rec.append(spk1)

        # Update previous states
        syn1_prev, mem1_prev = syn1, mem1
        syn2_prev, mem2_prev = syn2, mem2
        syn3_prev, mem3_prev = syn3, mem3

    return (
        torch.stack(spk_rec),
        torch.stack(mem_rec),
        torch.stack(mem1_rec),
        torch.stack(spk1_rec)
    )


def create_cv_folds(spike_tensors, y_tensors, num_folds=None, num_sessions=None, num_classes=None, random_state=None, retrain=False):
    """Create class-balanced cross-validation folds with separate CV and validation sets.

    Inputs:
        spike_tensors: list or array of samples (each sample is a time × features array)
        y_tensors: list or array of integer labels (same length as spike_tensors)
        num_folds: number of folds to create
        num_frames_per_sample: Number of frames per sample (default: 25)
        frames_per_class_per_session: Total frames per class per session (default: 1200)
        cv_frames: Number of frames per class per session for CV (default: 1000)
        num_sessions: Number of sessions (default: 2)
        num_classes: Number of classes (default: 17)
        random_state: seed for KFold shuffling

    Returns:
        cv_train_data_folds, cv_train_labels_folds, cv_test_data_folds, cv_test_labels_folds,
        val_data, val_labels, all_fold_train_indices, all_fold_test_indices
    """
    num_frames_per_sample = spike_tensors[0].shape[0]
    # Convert to NumPy arrays
    # Ensure spike_tensors is a NumPy array with float32 dtype
    if not isinstance(spike_tensors, np.ndarray) or spike_tensors.dtype != np.float32:
        raise ValueError("spike_tensors must be a NumPy array with dtype float32")
    if num_frames_per_sample is None:
        num_frames_per_sample = spike_tensors.shape[1]  # Infer from the time dimension
    if not isinstance(y_tensors, np.ndarray):
        y_tensors = np.array(y_tensors)

    num_frames_per_sample = spike_tensors[0].shape[0]

    # Calculate samples per class per session
    if num_sessions > 2:
        samples_per_class_per_session = 1100 // num_frames_per_sample
        cv_samples_per_class_per_session = 1000 // num_frames_per_sample
        val_samples_per_class_per_session = samples_per_class_per_session - cv_samples_per_class_per_session
    else:
        samples_per_class_per_session = 1200 // num_frames_per_sample
        cv_samples_per_class_per_session = 1100 // num_frames_per_sample
        val_samples_per_class_per_session = samples_per_class_per_session - cv_samples_per_class_per_session


    # Split indices into CV and validation sets
    cv_indices = []
    val_indices = []
    start_idx = 0
    for cls in range(num_classes):
        for session in range(num_sessions):
            class_session_start = start_idx
            class_session_end = start_idx + samples_per_class_per_session
            cv_indices.extend(range(class_session_start, class_session_start + cv_samples_per_class_per_session))
            val_indices.extend(range(class_session_start + cv_samples_per_class_per_session, class_session_end))
            start_idx = class_session_end

    cv_spike_tensors = spike_tensors[cv_indices]
    cv_y_tensors = y_tensors[cv_indices]
    val_spike_tensors = spike_tensors[val_indices]
    val_y_tensors = y_tensors[val_indices]

    if retrain:
        # Return combined CV train+test data/labels for retraining
        train_data = cv_spike_tensors
        train_labels = cv_y_tensors
        val_data = val_spike_tensors
        val_labels = val_y_tensors
        return train_data, train_labels, val_data, val_labels

    # Perform K-fold CV on the CV set
    unique_classes = np.unique(cv_y_tensors)
    class_ranges = []
    start_idx = 0
    for cls in unique_classes:
        class_mask = (cv_y_tensors == cls)
        class_size = np.sum(class_mask)
        end_idx = start_idx + class_size
        class_ranges.append((start_idx, end_idx))
        start_idx = end_idx

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    all_fold_train_indices = []
    all_fold_test_indices = []

    for fold in range(num_folds):
        fold_train_indices = []
        fold_test_indices = []
        for start, end in class_ranges:
            class_indices = np.arange(start, end)
            splits = list(kf.split(class_indices))
            train_fold, test_fold = splits[fold]
            fold_train_indices.extend(class_indices[train_fold])
            fold_test_indices.extend(class_indices[test_fold])
        all_fold_train_indices.append(np.sort(fold_train_indices))
        all_fold_test_indices.append(np.sort(fold_test_indices))

    # Extract CV train and test data
    cv_train_data_folds = [cv_spike_tensors[indices] for indices in all_fold_train_indices]
    cv_train_labels_folds = [cv_y_tensors[indices] for indices in all_fold_train_indices]
    cv_test_data_folds = [cv_spike_tensors[indices] for indices in all_fold_test_indices]
    cv_test_labels_folds = [cv_y_tensors[indices] for indices in all_fold_test_indices]

    # Validation set is fixed and not folded
    val_data = val_spike_tensors
    val_labels = val_y_tensors

    # Print distributions
    for fold in range(num_folds):
        print(f"Fold {fold + 1}: CV Train samples = {len(all_fold_train_indices[fold])}, "
              f"CV Test samples = {len(all_fold_test_indices[fold])}")
        print(f"  - CV Train class distribution: {np.bincount(cv_train_labels_folds[fold], minlength=num_classes)}")
        print(f"  - CV Test class distribution: {np.bincount(cv_test_labels_folds[fold], minlength=num_classes)}")
    print(f"Validation set: {len(val_data)} samples")
    print(f"  - Validation class distribution: {np.bincount(val_labels, minlength=num_classes)}")
    print(f"\nTotal folds: {num_folds}")
    if len(cv_train_data_folds) > 0:
        print(f"CV Train data shape per fold: {cv_train_data_folds[0].shape}")
        print(f"CV Test data shape per fold: {cv_test_data_folds[0].shape}")
        print(f"Validation data shape: {val_data.shape}")

    return (cv_train_data_folds, cv_train_labels_folds, cv_test_data_folds, cv_test_labels_folds,
            val_data, val_labels, all_fold_train_indices, all_fold_test_indices)


def create_session_cv_folds(spike_tensors, y_tensors, num_folds=None, num_sessions=None,
                            num_classes=None, random_state=None, retrain=False):
    """Create leave-one-session-out folds from 5-session data.

    This function has the same signature as ``create_cv_folds`` but follows a
    session-based split strategy:
    - Uses data from all 5 sessions.
    - Builds 5 folds total.
    - In each fold, one entire session is the validation set.
    - The remaining 4 sessions are the training set.
    - No test set is created.
    - For each class/session block, only the first 1100 frames are used.

    Args:
        spike_tensors: NumPy array of samples (N, T, F), dtype float32.
        y_tensors: NumPy array/list of labels (N,).
        num_folds: Kept for API compatibility. If provided, must be 5.
        num_sessions: Kept for API compatibility. If provided, must be 5.
        num_classes: Number of classes.
        random_state: Kept for API compatibility (unused).
        retrain: Kept for API compatibility (unused).

    Returns:
        train_data_folds, train_labels_folds, val_data_folds, val_labels_folds,
        all_fold_train_indices, all_fold_val_indices
    """
    del random_state, retrain  # API compatibility; not used in session-wise split.

    if not isinstance(spike_tensors, np.ndarray) or spike_tensors.dtype != np.float32:
        raise ValueError("spike_tensors must be a NumPy array with dtype float32")

    if not isinstance(y_tensors, np.ndarray):
        y_tensors = np.array(y_tensors)

    if num_classes is None:
        raise ValueError("num_classes must be provided")

    if num_sessions is None:
        num_sessions = 5
    if num_sessions != 5:
        raise ValueError(f"Expected num_sessions=5, got {num_sessions}")

    if num_folds is None:
        num_folds = 5
    if num_folds != 5:
        raise ValueError(f"Expected num_folds=5, got {num_folds}")

    if len(spike_tensors) != len(y_tensors):
        raise ValueError("spike_tensors and y_tensors must have the same length")

    if len(spike_tensors) % (num_classes * num_sessions) != 0:
        raise ValueError(
            "Dataset size is not divisible by num_classes * num_sessions; "
            "cannot infer class/session block size"
        )

    num_frames_per_sample = spike_tensors[0].shape[0]
    samples_to_use_per_class_session = 1100 // num_frames_per_sample
    samples_per_class_session = len(spike_tensors) // (num_classes * num_sessions)

    if samples_per_class_session < samples_to_use_per_class_session:
        raise ValueError(
            "Not enough samples per class/session to cover first 1100 frames: "
            f"need {samples_to_use_per_class_session}, have {samples_per_class_session}"
        )

    # Collect per-session indices using only the first 1100-frame equivalent samples.
    session_indices = [[] for _ in range(num_sessions)]
    for cls in range(num_classes):
        for session in range(num_sessions):
            block_start = (cls * num_sessions + session) * samples_per_class_session
            block_end = block_start + samples_to_use_per_class_session
            session_indices[session].extend(range(block_start, block_end))

    all_fold_train_indices = []
    all_fold_val_indices = []
    train_data_folds = []
    train_labels_folds = []
    val_data_folds = []
    val_labels_folds = []

    for val_session in range(num_sessions):
        fold_val_indices = np.array(sorted(session_indices[val_session]), dtype=np.int64)
        fold_train_indices = np.array(
            sorted([idx for s in range(num_sessions) if s != val_session for idx in session_indices[s]]),
            dtype=np.int64
        )

        all_fold_train_indices.append(fold_train_indices)
        all_fold_val_indices.append(fold_val_indices)

        train_data_folds.append(spike_tensors[fold_train_indices])
        train_labels_folds.append(y_tensors[fold_train_indices])
        val_data_folds.append(spike_tensors[fold_val_indices])
        val_labels_folds.append(y_tensors[fold_val_indices])

    for fold in range(num_folds):
        print(
            f"Fold {fold + 1}: Train samples = {len(all_fold_train_indices[fold])}, "
            f"Val samples = {len(all_fold_val_indices[fold])}"
        )
        print(
            f"  - Train class distribution: "
            f"{np.bincount(train_labels_folds[fold], minlength=num_classes)}"
        )
        print(
            f"  - Val class distribution: "
            f"{np.bincount(val_labels_folds[fold], minlength=num_classes)}"
        )

    if len(train_data_folds) > 0:
        print(f"\nTotal folds: {num_folds}")
        print(f"Train data shape per fold: {train_data_folds[0].shape}")
        print(f"Validation data shape per fold: {val_data_folds[0].shape}")

    return (
        train_data_folds,
        train_labels_folds,
        val_data_folds,
        val_labels_folds,
        all_fold_train_indices,
        all_fold_val_indices,
    )


def cv_train(cv_train_data_folds=None, cv_train_labels_folds=None, cv_test_data_folds=None, cv_test_labels_folds=None,
             model_class=None, num_folds=8, num_epochs=20, batch_size=16, random_state=42, device=device, verbose=True, 
             num_inputs=2*484, num_outputs=17, loss_fn='ce_rate_mse_membrane', return_weights=False, return_membrane=False, 
             return_spikes=False, alpha=0.7, beta=0.7, fixed_fan_in=None, lr=0.001, exclude_classes=None):
    """Train a given SNN model class using class-balanced cross-validation folds.

    Parameters:
        - model_class: class to instantiate for training (e.g., FC_SNN_Leaky, FC_SNN_Leaky_OneLayer)
        - num_folds, num_epochs, batch_size, random_state, device, verbose
        - num_inputs, num_outputs: passed to model_class if its constructor requires them
        - return_weights (bool): If True, return the first-layer weights after training.

    Returns:
        dict: Dictionary with per-fold and averaged histories, fold indices, and optionally first-layer weights.
    """

    # Histories
    all_loss_hist = []
    all_acc_hist = []
    all_test_loss_hist = []
    all_test_acc_hist = []
    all_conf_matrices = []          # Store confusion matrix for each fold
    all_true_labels = []            # Store true labels for each fold
    all_pred_labels = []            # Store predicted labels for each fold
    all_first_layer_weights = []    # Store first-layer weights for each fold
    all_lif1_params = []            # New list to store alpha, beta, threshold for each fold
    all_mem1_traces = []            # list of (T, B, hidden) per batch
    all_mem1_labels = []
    all_spk1_traces = []
    all_spk2_traces = []
    all_spk1_labels = []
    all_spk2_labels = []
    all_input_spike_counts = []

    # Loss helpers
    loss_ce_count = SF.ce_count_loss()
    loss_ce_max_membrane = SF.ce_max_membrane_loss()
    loss_ce_rate = SF.ce_rate_loss()
    loss_ce_temporal = SF.ce_temporal_loss()
    loss_mse_count = SF.mse_count_loss()
    loss_mse_membrane= SF.mse_membrane_loss()
    loss_mse_temporal = SF.mse_temporal_loss()

    for fold in range(num_folds):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting Fold {fold + 1}/{num_folds} for model {getattr(model_class, '__name__', str(model_class))}")
            print(f"{'='*60}")

        # Get data for this fold
        train_data = cv_train_data_folds[fold]
        train_labels = cv_train_labels_folds[fold]
        test_data = cv_test_data_folds[fold]
        test_labels = cv_test_labels_folds[fold]

        if model_class == FC_SNN_Syn:
            forward_pass = forward_pass_syn
        elif model_class == FC_SNN_Leaky:
            forward_pass = forward_pass_lky
        elif model_class == FC_SNN_Syn_33:
            forward_pass = forward_pass_syn_33
        elif model_class == FC_SNN_Syn_32:
            forward_pass = forward_pass_syn_32

        net = model_class(num_outputs=num_outputs, num_inputs=num_inputs, alpha=alpha, beta=beta, fixed_fan_in=fixed_fan_in).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        fold_loss_hist = []
        fold_acc_hist = []
        fold_test_loss_hist = []
        fold_test_acc_hist = []
        fold_true_labels = []
        fold_pred_labels = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_samples = 0

            # Shuffle train indices every epoch for randomization
            train_indices = list(range(len(train_data)))
            np.random.seed(epoch + fold * num_epochs + random_state)
            np.random.shuffle(train_indices)

            net.train()
            for i in range(0, len(train_indices), batch_size):
                batch_indices = train_indices[i:i+batch_size]
                if len(batch_indices) < batch_size and i + len(batch_indices) < len(train_indices):
                    continue
                data_list = [torch.from_numpy(train_data[idx]).float().to(device) for idx in batch_indices]
                data = torch.stack(data_list, dim=1)  # Shape: [time_steps, batch_size, features]
                targets = torch.tensor(train_labels[batch_indices]).to(device)  # Shape [batch_size]

                spk2_rec, mem2_rec, mem1_rec, spk1_rec = forward_pass(net, data)
                
                if loss_fn == 'ce_count':
                    loss_val = loss_ce_count(spk2_rec, targets)
                elif loss_fn == 'ce_rate':
                    loss_val = loss_ce_rate(spk2_rec, targets)
                elif loss_fn == 'ce_temporal':
                    loss_val = loss_ce_temporal(spk2_rec, targets)
                elif loss_fn == 'ce_max_membrane':
                    loss_val = loss_ce_max_membrane(mem2_rec, targets)
                elif loss_fn == 'mse_count':
                    loss_val = loss_mse_count(spk2_rec, targets)
                elif loss_fn == 'mse_membrane':
                    loss_val = loss_mse_membrane(mem2_rec, targets)
                elif loss_fn == 'mse_temporal':
                    loss_val = loss_mse_temporal(spk2_rec, targets)
                elif loss_fn == 'ce_rate_mse_membrane':
                    loss_val = loss_ce_rate(spk2_rec, targets) + 0.1 * loss_mse_membrane(mem2_rec, targets)
                elif loss_fn == 'ce_rate_mse_count':
                    loss_val = loss_ce_rate(spk2_rec, targets) + 0.1 * loss_mse_count(spk2_rec, targets)
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                if hasattr(net, 'enforce_sparsity'):
                    net.enforce_sparsity()
                epoch_loss += loss_val.item() * len(batch_indices)
                spk2_rec_sum = torch.sum(spk2_rec, dim=0)
                _, predicted_idx = spk2_rec_sum.max(dim=1)
                acc = (predicted_idx == targets).float().mean().item()
                epoch_acc += acc * len(batch_indices)
                num_samples += len(batch_indices)

            if num_samples == 0:
                avg_loss = float('nan')
                avg_acc = float('nan')
            else:
                avg_loss = epoch_loss / num_samples
                avg_acc = epoch_acc / num_samples

            fold_loss_hist.append(avg_loss)
            fold_acc_hist.append(avg_acc)

            if verbose:
                print(f"Fold {fold + 1}, Epoch {epoch + 1} - Avg Train Loss: {avg_loss:.4f}, Avg Train Accuracy: {avg_acc * 100 if not np.isnan(avg_acc) else np.nan:.2f}%")

            # Evaluate on test set
            net.eval()
            test_epoch_loss = 0.0
            test_epoch_acc = 0.0
            test_num_samples = 0
            fold_input_spike_counts = np.zeros(num_inputs, dtype=np.float64) if (return_spikes and epoch == num_epochs - 1) else None
            with torch.no_grad():
                test_indices = list(range(len(test_data)))
                np.random.shuffle(test_indices)
                for i in range(0, len(test_indices), batch_size):
                    batch_indices = test_indices[i:i+batch_size]
                    if len(batch_indices) < batch_size and i + len(batch_indices) < len(test_indices):
                        continue
                    data_list = [torch.from_numpy(test_data[idx]).float().to(device) for idx in batch_indices]
                    data = torch.stack(data_list, dim=1)
                    targets = torch.tensor(test_labels[batch_indices]).to(device)
                    spk2_rec, mem2_rec, mem1_rec, spk1_rec = forward_pass(net, data)
                    if loss_fn == 'ce_count':
                        loss_val = loss_ce_count(spk2_rec, targets)
                    elif loss_fn == 'ce_rate':
                        loss_val = loss_ce_rate(spk2_rec, targets)
                    elif loss_fn == 'ce_temporal':
                        loss_val = loss_ce_temporal(spk2_rec, targets)
                    elif loss_fn == 'ce_max_membrane':
                        loss_val = loss_ce_max_membrane(mem2_rec, targets)
                    elif loss_fn == 'mse_count':
                        loss_val = loss_mse_count(spk2_rec, targets)
                    elif loss_fn == 'mse_membrane':
                        loss_val = loss_mse_membrane(mem2_rec, targets)
                    elif loss_fn == 'mse_temporal':
                        loss_val = loss_mse_temporal(spk2_rec, targets)
                    elif loss_fn == 'ce_rate_mse_membrane':
                        loss_val = loss_ce_rate(spk2_rec, targets) + 0.1 * loss_mse_membrane(mem2_rec, targets)
                    elif loss_fn == 'ce_rate_mse_count':
                        loss_val = loss_ce_rate(spk2_rec, targets) + 0.1 * loss_mse_count(spk2_rec, targets)
                    test_epoch_loss += loss_val.item() * len(batch_indices)
                    spk2_rec_sum = torch.sum(spk2_rec, dim=0)
                    _, predicted_idx = spk2_rec_sum.max(dim=1)
                    acc = (predicted_idx == targets).float().mean().item()
                    test_epoch_acc += acc * len(batch_indices)
                    test_num_samples += len(batch_indices)
                    if epoch == num_epochs - 1:
                        fold_true_labels.extend(targets.cpu().numpy())
                        fold_pred_labels.extend(predicted_idx.cpu().numpy())
                    
                    # === RECORD MEMBRANE & SPIKES ===
                    if return_membrane and epoch == num_epochs - 1:
                        all_mem1_traces.append(mem1_rec.cpu().numpy())  # (T, B, H)
                        all_mem1_labels.extend(targets.cpu().numpy())
                    
                    if return_spikes and epoch == num_epochs - 1:
                        all_spk1_traces.append(spk1_rec.cpu().numpy())  # (T, B, H)
                        all_spk2_traces.append(spk2_rec.cpu().numpy())
                        print(f"Fold {fold}, last epoch: added {spk1_rec.shape} hidden spikes")
                        all_spk1_labels.extend(targets.cpu().numpy())
                        all_spk2_labels.extend(targets.cpu().numpy())
                        input_spikes_batch = data.cpu().numpy().sum(axis=0)  # (batch_size, num_inputs) → sum over time
                        fold_input_spike_counts += input_spikes_batch.sum(axis=0)  # accumulate over batch

            if test_num_samples == 0:
                avg_test_loss = float('nan')
                avg_test_acc = float('nan')
            else:
                avg_test_loss = test_epoch_loss / test_num_samples
                avg_test_acc = test_epoch_acc / test_num_samples

            fold_test_loss_hist.append(avg_test_loss)
            fold_test_acc_hist.append(avg_test_acc)

            if verbose:
                print(f"Fold {fold + 1}, Epoch {epoch + 1} - Avg Test Loss: {avg_test_loss:.4f}, Avg Test Accuracy: {avg_test_acc * 100 if not np.isnan(avg_test_acc) else np.nan:.2f}%")

        # Append fold results
        cm = confusion_matrix(fold_true_labels, fold_pred_labels, labels=range(num_outputs))
        all_conf_matrices.append(cm)
        all_true_labels.append(fold_true_labels)
        all_pred_labels.append(fold_pred_labels)

        all_loss_hist.append(fold_loss_hist)
        all_acc_hist.append(fold_acc_hist)
        all_test_loss_hist.append(fold_test_loss_hist)
        all_test_acc_hist.append(fold_test_acc_hist)

        lif1_params = {
            'alpha': net.lif1.alpha.item(),
            'beta': net.lif1.beta.item(),
            'threshold': net.lif1.threshold.item()
        }
        all_lif1_params.append(lif1_params)

        # Store first-layer weights if requested
        if return_weights:
            all_first_layer_weights.append(net.fc1.weight.data.cpu().numpy())
            
        if return_spikes and fold_input_spike_counts is not None:
            all_input_spike_counts.append(fold_input_spike_counts)
            print(f"Fold {fold}: input spikes accumulated {fold_input_spike_counts.sum()}")

        if verbose:
            print(f"Fold {fold + 1} completed.")
    
    # Aggregate confusion matrices across folds
    aggregated_cm = np.sum(all_conf_matrices, axis=0)
    aggregated_true_labels = np.concatenate(all_true_labels)
    aggregated_pred_labels = np.concatenate(all_pred_labels)

    # Normalize confusion matrix (by row, i.e., true class)
    normalized_cm = aggregated_cm / aggregated_cm.sum(axis=1, keepdims=True)
    normalized_cm = np.nan_to_num(normalized_cm)  # Replace NaN with 0 for empty classes

    # Compute per-class metrics for aggregated results
    precision, recall, f1, _ = precision_recall_fscore_support(
        aggregated_true_labels, aggregated_pred_labels, labels=range(num_outputs), zero_division=0
    )

    class_performance = None
    sorted_classes = None
    if exclude_classes is None:
        # Create class performance dictionary
        class_performance = {
            name: {
                'accuracy': normalized_cm[i, i],
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i]
            } for i, name in enumerate(class_names)
        }

        # Sort classes by accuracy
        sorted_classes = sorted(class_performance.items(), key=lambda x: x[1]['accuracy'], reverse=True)

    # Average across folds (handle possible NaNs)
    avg_loss_hist = np.nanmean(all_loss_hist, axis=0)
    avg_acc_hist = np.nanmean(all_acc_hist, axis=0)
    avg_test_loss_hist = np.nanmean(all_test_loss_hist, axis=0)
    avg_test_acc_hist = np.nanmean(all_test_acc_hist, axis=0)

    if verbose:
        print("\nCross-Validation Summary:")
        print(f"Average Train Accuracy: {np.nanmean(avg_acc_hist) * 100:.2f}% ± {np.nanstd(avg_acc_hist) * 100:.2f}%")
        print(f"Average Test Accuracy: {np.nanmean(avg_test_acc_hist) * 100:.2f}% ± {np.nanstd(avg_test_acc_hist) * 100:.2f}%")

    results = {
        'all_loss_hist': all_loss_hist,
        'all_acc_hist': all_acc_hist,
        'all_test_loss_hist': all_test_loss_hist,
        'all_test_acc_hist': all_test_acc_hist,
        'avg_loss_hist': avg_loss_hist,
        'avg_acc_hist': avg_acc_hist,
        'avg_test_loss_hist': avg_test_loss_hist,
        'avg_test_acc_hist': avg_test_acc_hist,
        'all_conf_matrices': all_conf_matrices,
        'aggregated_cm': aggregated_cm,
        'normalized_cm': normalized_cm,
        'class_performance': class_performance,
        'sorted_classes': sorted_classes
    }

    # Add first-layer weights to results if requested
    if return_weights:
        results['first_layer_weights'] = all_first_layer_weights
        results['all_lif1_params'] = all_lif1_params
    
    if return_membrane:
        if all_mem1_traces:
            mem1_array = np.concatenate(all_mem1_traces, axis=1)      # (T, total_B, H)
            mem1_array = np.transpose(mem1_array, (1, 0, 2))          # (B, T, H)

            results['hidden_membrane']        = mem1_array
            results['hidden_membrane_labels'] = np.array(all_mem1_labels)

            results['hidden_membrane_shape'] = mem1_array.shape

            max_per_fold   = [np.max(tr) for tr in all_mem1_traces]
            mean_per_fold  = [np.mean(tr) for tr in all_mem1_traces]
            results['max_membrane_per_fold']  = np.array(max_per_fold, dtype=float)
            results['mean_membrane_per_fold'] = np.array(mean_per_fold, dtype=float)

            frac_above_05 = (mem1_array > 0.5).mean()
            results['membrane_fraction_above_0.5'] = float(frac_above_05)

            mem_per_sample = mem1_array.max(axis=(1, 2))   # max over T & H
            results['avg_max_membrane_per_test_sample'] = {
                'mean': float(mem_per_sample.mean()),
                'std' : float(mem_per_sample.std())
            }
        else:
            # empty placeholders
            results['hidden_membrane'] = None
            results['hidden_membrane_labels'] = None
            results['hidden_membrane_shape'] = None
            results['max_membrane_per_fold'] = None
            results['mean_membrane_per_fold'] = None
            results['membrane_fraction_above_0.5'] = None
            results['avg_max_membrane_per_test_sample'] = None
            
    if return_spikes:
        if all_spk1_traces:
            print(f"Concatenating {len(all_spk1_traces)} hidden spike traces across folds")
            spk1_array = np.concatenate(all_spk1_traces, axis=1)      # (T, total_B, H)
            print(f"Hidden spike array shape after concatenation: {spk1_array.shape}")
            spk1_array = np.transpose(spk1_array, (1, 0, 2))          # (B, T, H)
            print(f"Hidden spike array shape after transpose: {spk1_array.shape}")

            results['hidden_spikes']          = spk1_array
            results['hidden_spike_labels']    = np.array(all_spk1_labels)
            results['all_spk1_traces'] = all_spk1_traces
            results['all_spk1_labels'] = all_spk1_labels

            results['hidden_spikes_shape']    = spk1_array.shape

            total_spikes_per_fold = [np.sum(traces) for traces in all_spk1_traces]
            results['total_hidden_spikes_per_fold'] = total_spikes_per_fold
            results['total_hidden_spikes']          = np.sum(total_spikes_per_fold)

            spikes_per_sample = spk1_array.sum(axis=(1, 2))          # (B,)
            results['avg_spikes_per_test_sample'] = {
                'mean': float(spikes_per_sample.mean()),
                'std' : float(spikes_per_sample.std())
            }

            T = spk1_array.shape[1]
            firing_rate = spk1_array.mean(axis=(0, 1))               # (H,)
            results['avg_firing_rate_per_neuron'] = {
                'mean': float(firing_rate.mean()),
                'std' : float(firing_rate.std())
            }
        if all_spk2_traces:
            spk2_array = np.concatenate(all_spk2_traces, axis=1)  # (T, total_B, num_classes)
            spk2_array = np.transpose(spk2_array, (1, 0, 2))  # (B, T, num_classes)
            results['output_spikes'] = spk2_array
            results['output_spike_labels'] = np.array(all_spk2_labels)
            results['all_spk2_traces'] = all_spk2_traces
            results['output_spikes_shape'] = spk2_array.shape

            total_spikes_per_output = np.sum(spk2_array, axis=(0, 1))  # (num_classes,)
            results['total_output_spikes'] = np.sum(total_spikes_per_output)
            results['avg_output_spikes_per_test_sample'] = {
                'mean': float(total_spikes_per_output.mean()),
                'std': float(np.std(total_spikes_per_output))
            }
        if all_input_spike_counts:
            # Concatenate across folds: list of (num_inputs,) → array (num_folds, num_inputs)
            input_spike_array = np.stack(all_input_spike_counts, axis=0)
            results['input_spike_counts_per_fold'] = input_spike_array
            results['total_input_spike_counts'] = np.sum(input_spike_array, axis=0)  # (num_inputs,)
            
            # Optional: average per test sample (approximate, assumes similar num samples per fold)
            total_test_samples = sum(len(fold) for fold in cv_test_data_folds)
            results['avg_input_spikes_per_test_sample'] = {
                'mean': float(np.mean(input_spike_array.sum(axis=1)) / (total_test_samples / num_folds)),
                'std': float(np.std(input_spike_array.sum(axis=1)) / (total_test_samples / num_folds))
            }
        else:
            results['input_spike_counts_per_fold'] = None
            results['total_input_spike_counts'] = None
            results['avg_input_spikes_per_test_sample'] = None
    print(f"Returning cv_train results with {len(all_spk1_traces)} hidden trace folds")
    if all_input_spike_counts:
        print(f"  and {len(all_input_spike_counts)} input count arrays")
    else:
        print("  NO input spike counts were collected!")
    return results

def cv_train_topk(cv_train_data_folds, cv_train_labels_folds,
                  cv_test_data_folds, cv_test_labels_folds,
                  model_class=FC_SNN_Syn,
                  num_folds=5, num_epochs=20, batch_size=16,
                  alpha=0.7, beta=0.7,
                  fixed_fan_in=None,
                  topk_accuracy=3,
                  lr=0.001, device=device, verbose=True):
    """
    Lightweight CV training — same loss/optimizer/enforce_sparsity as cv_train,
    but only computes and returns Top-1 and Top-k accuracies (train + test).
    """
    loss_ce = SF.ce_rate_loss()
    loss_mse = SF.mse_membrane_loss()

    # Store per-fold histories
    train_top1_hist = []
    train_topk_hist = []
    test_top1_hist = []
    test_topk_hist = []

    for fold in range(num_folds):
        if verbose:
            print(f"\n{'='*70}")
            print(f"FOLD {fold+1}/{num_folds}")
            print(f"{'='*70}")

        train_data = cv_train_data_folds[fold]
        train_labels = cv_train_labels_folds[fold]
        test_data = cv_test_data_folds[fold]
        test_labels = cv_test_labels_folds[fold]

        forward_pass = forward_pass_syn if model_class == FC_SNN_Syn else forward_pass_lky

        net = model_class(
            num_outputs=len(np.unique(train_labels)),
            num_inputs=train_data[0].shape[1],
            alpha=alpha, beta=beta,
            fixed_fan_in=fixed_fan_in
        ).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        fold_train_top1 = []
        fold_train_topk = []
        fold_test_top1 = []
        fold_test_topk = []

        for epoch in range(num_epochs):
            # === TRAIN ===
            net.train()
            epoch_loss = epoch_top1 = epoch_topk = num_samples = 0.0

            indices = np.random.permutation(len(train_data))
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i+batch_size]
                if len(batch_idx) < batch_size and i + len(batch_idx) < len(indices):
                    continue

                data = torch.stack([torch.from_numpy(train_data[j]).float().to(device) for j in batch_idx], dim=1)
                targets = torch.tensor(train_labels[batch_idx], device=device)

                spk_rec, mem_rec, mem1_rec, spk1_rec = forward_pass(net, data)
                loss = loss_ce(spk_rec, targets) + 0.1 * loss_mse(mem_rec, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if hasattr(net, 'enforce_sparsity'):
                    net.enforce_sparsity()

                epoch_loss += loss.item() * len(batch_idx)

                rates = spk_rec.sum(0)
                pred = rates.max(1)[1]
                epoch_top1 += (pred == targets).sum().item()

                if topk_accuracy > 1:
                    topk_pred = rates.topk(k=topk_accuracy, dim=1)[1]
                    epoch_topk += topk_pred.eq(targets.view(-1, 1)).any(1).sum().item()

                num_samples += len(batch_idx)

            train_acc_top1 = epoch_top1 / num_samples
            train_acc_topk = epoch_topk / num_samples if topk_accuracy > 1 else None

            fold_train_top1.append(train_acc_top1)
            fold_train_topk.append(train_acc_topk)

            # === TEST ===
            net.eval()
            correct_top1 = correct_topk = total = 0
            with torch.no_grad():
                for i in range(0, len(test_data), batch_size):
                    batch_idx = list(range(i, min(i + batch_size, len(test_data))))
                    data = torch.stack([torch.from_numpy(test_data[j]).float().to(device) for j in batch_idx], dim=1)
                    targets = torch.tensor(test_labels[batch_idx], device=device)

                    spk_rec, mem_rec, mem1_rec, spk1_rec = forward_pass(net, data)
                    rates = spk_rec.sum(0)

                    pred = rates.max(1)[1]
                    correct_top1 += (pred == targets).sum().item()

                    if topk_accuracy > 1:
                        topk_pred = rates.topk(k=topk_accuracy, dim=1)[1]
                        correct_topk += topk_pred.eq(targets.view(-1, 1)).any(1).sum().item()

                    total += targets.size(0)

            test_acc_top1 = correct_top1 / total
            test_acc_topk = correct_topk / total if topk_accuracy > 1 else None

            fold_test_top1.append(test_acc_top1)
            fold_test_topk.append(test_acc_topk)

            if verbose:
                msg = f"Epoch {epoch+1:2d} → "
                msg += f"Train Top-1: {train_acc_top1*100:5.2f}%"
                msg += f" | Test Top-1: {test_acc_top1*100:5.2f}%"
                if test_acc_topk is not None:
                    msg += f" | Test Top-{topk_accuracy}: {test_acc_topk*100:5.2f}%"
                print(msg)

        train_top1_hist.append(fold_train_top1)
        train_topk_hist.append(fold_train_topk)
        test_top1_hist.append(fold_test_top1)
        test_topk_hist.append(fold_test_topk)

    # Final averages
    avg_train_top1 = np.mean([np.mean(h) for h in train_top1_hist])
    avg_test_top1  = np.mean([np.mean(h) for h in test_top1_hist])
    avg_train_topk = np.mean([np.mean(h) for h in train_topk_hist]) if topk_accuracy > 1 else None
    avg_test_topk  = np.mean([np.mean(h) for h in test_topk_hist]) if topk_accuracy > 1 else None

    if verbose:
        print(f"\n{'='*70}")
        print(f"FINAL AVERAGE ACCURACY")
        print(f"Train Top-1 : {avg_train_top1*100:5.2f}%")
        print(f"Test  Top-1 : {avg_test_top1*100:5.2f}%")
        if avg_test_topk is not None:
            print(f"Test  Top-{topk_accuracy} : {avg_test_topk*100:5.2f}%")
        print(f"{'='*70}")

    return {
        'avg_train_top1': avg_train_top1,
        'avg_test_top1': avg_test_top1,
        'avg_train_topk': avg_test_topk,
        'avg_test_topk': avg_test_topk,
        'train_top1_hist': train_top1_hist,
        'test_top1_hist': test_top1_hist,
        'train_topk_hist': train_topk_hist,
        'test_topk_hist': test_topk_hist,
        'topk': topk_accuracy
    }

def train(train_data, train_labels, val_data, val_labels, model_class=None, alpha=None, beta=None, num_epochs=20, batch_size=16,
          random_state=42, device=device, verbose=True, num_inputs=None, num_outputs=None, loss_fn='ce_rate_mse_membrane',
          patience=5, min_delta=0.01, lr=0.002, finetune=False, optimizer=None, fixed_fan_in=None):
    """Train a given SNN model class on a single train/validation split with early stopping based on validation accuracy.

    Parameters:
        - train_data, train_labels: Training data and labels
        - val_data, val_labels: Validation data and labels
        - model_class: class to instantiate for training if model is None
        - model: Pre-initialized model to train
        - num_epochs, batch_size, random_state, device, verbose
        - num_inputs, num_outputs: passed to model_class if its constructor requires them
        - loss_fn (str): Loss function to use ('ce', 'combined')
        - patience (int): Number of epochs with no improvement after which training will stop
        - min_delta (float): Minimum change in validation accuracy to qualify as an improvement
        - optimizer: Optional custom optimizer; if None, defaults to Adam with lr=0.001

    Returns:
        dict: Dictionary with training and validation loss/accuracy histories, and best epoch info.
    """
    # Loss helpers
    loss_ce_factory = SF.ce_rate_loss
    loss_mse_factory = SF.mse_membrane_loss

    forward_pass = forward_pass_syn

    
    if finetune is False:
        net = model_class(num_outputs=num_outputs, num_inputs=num_inputs, alpha=alpha, beta=beta, 
                          fixed_fan_in=fixed_fan_in).to(device)

        optimizer = optimizer if optimizer is not None else torch.optim.Adam([param for param in net.parameters() if param.requires_grad], lr=lr)
    else:
        net = model_class
        optimizer = optimizer
        
    loss_ce = loss_ce_factory()
    loss_mse = loss_mse_factory()

    train_loss_hist = []
    train_acc_hist = []
    val_loss_hist = []
    val_acc_hist = []

    best_val_acc = -float('inf')
    best_model_state = None
    patience_counter = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_samples = 0

        train_indices = list(range(len(train_data)))
        np.random.seed(epoch + random_state)
        np.random.shuffle(train_indices)

        net.train()
        for i in range(0, len(train_indices), batch_size):
            batch_indices = train_indices[i:i+batch_size]
            if len(batch_indices) < batch_size and i + len(batch_indices) < len(train_indices):
                continue
            data_list = [torch.from_numpy(train_data[idx]).float().to(device) for idx in batch_indices]
            data = torch.stack(data_list, dim=1)
            targets = torch.tensor(train_labels[batch_indices], dtype=torch.long, device=device)
            spk_rec, mem_rec, mem1_rec, spk1_rec = forward_pass(net, data)
            if loss_fn == 'ce':
                loss_val = loss_ce(spk_rec, targets)
            elif loss_fn == 'ce_rate_mse_membrane':
                loss_val = loss_ce(spk_rec, targets) + 0.1 * loss_mse(mem_rec, targets)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
            if hasattr(net, 'enforce_sparsity'):
                net.enforce_sparsity()
                
            epoch_loss += loss_val.item() * len(batch_indices)
            spk_rec_sum = torch.sum(spk_rec, dim=0)
            _, predicted_idx = spk_rec_sum.max(dim=1)
            acc = (predicted_idx == targets).float().mean().item()
            epoch_acc += acc * len(batch_indices)
            num_samples += len(batch_indices)

        if num_samples == 0:
            avg_loss = float('nan')
            avg_acc = float('nan')
        else:
            avg_loss = epoch_loss / num_samples
            avg_acc = epoch_acc / num_samples
        train_loss_hist.append(avg_loss)
        train_acc_hist.append(avg_acc)

        net.eval()
        val_epoch_loss = 0.0
        val_epoch_acc = 0.0
        val_num_samples = 0
        with torch.no_grad():
            val_indices = list(range(len(val_data)))
            np.random.shuffle(val_indices)
            for i in range(0, len(val_indices), batch_size):
                batch_indices = val_indices[i:i+batch_size]
                if len(batch_indices) < batch_size and i + len(batch_indices) < len(val_indices):
                    continue
                data_list = [torch.from_numpy(val_data[idx]).float().to(device) for idx in batch_indices]
                data = torch.stack(data_list, dim=1)
                targets = torch.tensor(val_labels[batch_indices], dtype=torch.long, device=device)
                spk_rec, mem_rec, mem1_rec, spk1_rec = forward_pass(net, data)
                if loss_fn == 'ce':
                    loss_val = loss_ce(spk_rec, targets)
                elif loss_fn == 'combined':
                    loss_val = loss_ce(spk_rec, targets) + 0.1 * loss_mse(mem_rec, targets)
                val_epoch_loss += loss_val.item() * len(batch_indices)
                spk_rec_sum = torch.sum(spk_rec, dim=0)
                _, predicted_idx = spk_rec_sum.max(dim=1)
                acc = (predicted_idx == targets).float().mean().item()
                val_epoch_acc += acc * len(batch_indices)
                val_num_samples += len(batch_indices)

        if val_num_samples == 0:
            avg_val_loss = float('nan')
            avg_val_acc = float('nan')
        else:
            avg_val_loss = val_epoch_loss / val_num_samples
            avg_val_acc = val_epoch_acc / val_num_samples
        val_loss_hist.append(avg_val_loss)
        val_acc_hist.append(avg_val_acc)

        if verbose:
            print(f"Epoch {epoch + 1} - Avg Train Loss: {avg_loss:.4f}, Avg Train Accuracy: {avg_acc * 100 if not np.isnan(avg_acc) else np.nan:.2f}%")
            print(f"Epoch {epoch + 1} - Avg Val Loss: {avg_val_loss:.4f}, Avg Val Accuracy: {avg_val_acc * 100 if not np.isnan(avg_val_acc) else np.nan:.2f}%")

        if not np.isnan(avg_val_acc) and avg_val_acc > best_val_acc + min_delta:
            best_val_acc = avg_val_acc
            best_model_state = net.state_dict().copy()
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping triggered after epoch {epoch + 1} with no improvement for {patience} epochs.")
                break

    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    if best_epoch > 0:
        final_train_loss = train_loss_hist[best_epoch - 1]
        final_train_acc = train_acc_hist[best_epoch - 1]
        final_val_loss = val_loss_hist[best_epoch - 1]
        final_val_acc = val_acc_hist[best_epoch - 1]
    else:
        final_train_loss = train_loss_hist[-1]
        final_train_acc = train_acc_hist[-1]
        final_val_loss = val_loss_hist[-1]
        final_val_acc = val_acc_hist[-1]

    if verbose:
        print(f"\nFinal Training and Validation Completed (Best Epoch {best_epoch}):")
        print(f"Final Train Loss: {final_train_loss:.4f}, Final Train Accuracy: {final_train_acc * 100:.2f}%")
        print(f"Final Val Loss: {final_val_loss:.4f}, Final Val Accuracy: {final_val_acc * 100:.2f}%")

    results = {
        'train_loss_hist': train_loss_hist,
        'train_acc_hist': train_acc_hist,
        'val_loss_hist': val_loss_hist,
        'val_acc_hist': val_acc_hist,
        'best_epoch': best_epoch,
        'model_state_dict': net.state_dict()
    }

    return results

def train_topk(train_data, train_labels, val_data, val_labels, 
               model_class=FC_SNN_Syn, alpha=None, beta=None, num_epochs=20, batch_size=16,
               random_state=42, device=device, verbose=True, num_inputs=None, num_outputs=None, 
               loss_fn='ce_rate_mse_membrane', patience=5, min_delta=0.01, lr=0.002, finetune=False, optimizer=None,
               topk_accuracy=3, fixed_fan_in=None):
    """
    Lightweight version of train() — same optimization/loss/early stopping, but only returns 
    Top-1 and Top-k accuracies for train and validation. Reports final metrics at best epoch.
    """
    loss_ce_factory = SF.ce_rate_loss
    loss_mse_factory = SF.mse_membrane_loss

    if model_class == FC_SNN_Syn:
        forward_pass = forward_pass_syn
    elif model_class == FC_SNN_Leaky:
        forward_pass = forward_pass_lky
    elif model_class == FC_SNN_Syn_33:
        forward_pass = forward_pass_syn_33
    elif model_class == FC_SNN_Syn_32:
        forward_pass = forward_pass_syn_32

    net = model_class(num_outputs=num_outputs, num_inputs=num_inputs, alpha=alpha, beta=beta, fixed_fan_in=fixed_fan_in).to(device)

    optimizer = optimizer if optimizer is not None else torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=lr
    )
    loss_ce = loss_ce_factory()
    loss_mse = loss_mse_factory()

    # Histories for Top-1 and Top-k
    train_top1_hist = []
    train_topk_hist = []
    val_top1_hist = []
    val_topk_hist = []

    best_val_top1 = -float('inf')
    best_model_state = None
    best_train_top1 = best_train_topk = best_val_topk = 0.0
    patience_counter = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        # === TRAIN ===
        epoch_loss = 0.0
        epoch_top1 = epoch_topk = num_samples = 0

        train_indices = list(range(len(train_data)))
        np.random.seed(epoch + random_state)
        np.random.shuffle(train_indices)

        net.train()
        for i in range(0, len(train_indices), batch_size):
            batch_indices = train_indices[i:i+batch_size]
            if len(batch_indices) < batch_size and i + len(batch_indices) < len(train_indices):
                continue

            data_list = [torch.from_numpy(train_data[idx]).float().to(device) for idx in batch_indices]
            data = torch.stack(data_list, dim=1)
            targets = torch.tensor(train_labels[batch_indices], dtype=torch.long, device=device)

            spk_rec, mem_rec, mem1_rec, spk1_rec = forward_pass(net, data)
            if loss_fn == 'ce':
                loss_val = loss_ce(spk_rec, targets)
            elif loss_fn == 'ce_rate_mse_membrane':
                loss_val = loss_ce(spk_rec, targets) + 0.1 * loss_mse(mem_rec, targets)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
            if hasattr(net, 'enforce_sparsity'):
                net.enforce_sparsity()

            epoch_loss += loss_val.item() * len(batch_indices)

            rates = spk_rec.sum(0)
            pred = rates.max(1)[1]
            epoch_top1 += (pred == targets).sum().item()

            if topk_accuracy > 1:
                topk_pred = rates.topk(k=topk_accuracy, dim=1)[1]
                epoch_topk += topk_pred.eq(targets.view(-1, 1)).any(1).sum().item()

            num_samples += len(batch_indices)

        avg_loss = epoch_loss / num_samples if num_samples > 0 else float('nan')
        avg_train_top1 = epoch_top1 / num_samples if num_samples > 0 else float('nan')
        avg_train_topk = epoch_topk / num_samples if num_samples > 0 and topk_accuracy > 1 else None

        train_top1_hist.append(avg_train_top1)
        if avg_train_topk is not None:
            train_topk_hist.append(avg_train_topk)

        # === VAL ===
        net.eval()
        val_loss = val_top1 = val_topk = val_samples = 0
        with torch.no_grad():
            val_indices = list(range(len(val_data)))
            np.random.shuffle(val_indices)
            for i in range(0, len(val_indices), batch_size):
                batch_indices = val_indices[i:i+batch_size]
                if len(batch_indices) < batch_size and i + len(batch_indices) < len(val_indices):
                    continue

                data_list = [torch.from_numpy(val_data[idx]).float().to(device) for idx in batch_indices]
                data = torch.stack(data_list, dim=1)
                targets = torch.tensor(val_labels[batch_indices], dtype=torch.long, device=device)

                spk_rec, mem_rec, mem1_rec, spk1_rec = forward_pass(net, data)
                if loss_fn == 'ce':
                    loss_val = loss_ce(spk_rec, targets)
                elif loss_fn == 'combined':
                    loss_val = loss_ce(spk_rec, targets) + 0.1 * loss_mse(mem_rec, targets)

                val_loss += loss_val.item() * len(batch_indices)

                rates = spk_rec.sum(0)
                pred = rates.max(1)[1]
                val_top1 += (pred == targets).sum().item()

                if topk_accuracy > 1:
                    topk_pred = rates.topk(k=topk_accuracy, dim=1)[1]
                    val_topk += topk_pred.eq(targets.view(-1, 1)).any(1).sum().item()

                val_samples += len(batch_indices)

        avg_val_loss = val_loss / val_samples if val_samples > 0 else float('nan')
        avg_val_top1 = val_top1 / val_samples if val_samples > 0 else float('nan')
        avg_val_topk = val_topk / val_samples if val_samples > 0 and topk_accuracy > 1 else None

        val_top1_hist.append(avg_val_top1)
        if avg_val_topk is not None:
            val_topk_hist.append(avg_val_topk)

        # Early stopping on Top-1 val accuracy
        if avg_val_top1 > best_val_top1 + min_delta:
            best_val_top1 = avg_val_top1
            best_train_top1 = avg_train_top1
            best_train_topk = avg_train_topk
            best_val_topk = avg_val_topk
            best_model_state = net.state_dict().copy()
            patience_counter = 0
            best_epoch = epoch + 1
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        if verbose:
            msg = f"Epoch {epoch + 1} → Train Loss: {avg_loss:.4f} | Train Top-1: {avg_train_top1*100:5.2f}%"
            msg += f" | Val Loss: {avg_val_loss:.4f} | Val Top-1: {avg_val_top1*100:5.2f}%"
            if avg_val_topk is not None:
                msg += f" | Val Top-{topk_accuracy}: {avg_val_topk*100:5.2f}%"
            if epoch + 1 == best_epoch:
                msg += " ← BEST"
            print(msg)

    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    # Use metrics from the best epoch
    final_train_top1 = best_train_top1 if best_epoch > 0 else train_top1_hist[-1]
    final_val_top1 = best_val_top1 if best_epoch > 0 else val_top1_hist[-1]
    final_train_topk = best_train_topk if best_epoch > 0 and topk_accuracy > 1 else (train_topk_hist[-1] if topk_accuracy > 1 else None)
    final_val_topk = best_val_topk if best_epoch > 0 and topk_accuracy > 1 else (val_topk_hist[-1] if topk_accuracy > 1 else None)

    if verbose:
        print(f"\n{'='*70}")
        print(f"FINAL RESULT (best epoch {best_epoch})")
        print(f"Train Top-1: {final_train_top1*100:.2f}% | Val Top-1: {final_val_top1*100:.2f}%")
        if final_val_topk is not None:
            print(f"Val Top-{topk_accuracy}: {final_val_topk*100:.2f}%")
        print(f"{'='*70}")

    return {
        'train_top1_hist': train_top1_hist,
        'val_top1_hist': val_top1_hist,
        'train_topk_hist': train_topk_hist if topk_accuracy > 1 else None,
        'val_topk_hist': val_topk_hist if topk_accuracy > 1 else None,
        'final_train_top1': final_train_top1,
        'final_val_top1': final_val_top1,
        'final_train_topk': final_train_topk,
        'final_val_topk': final_val_topk,
        'best_epoch': best_epoch,
        'topk': topk_accuracy,
        'model_state_dict': net.state_dict()
    }
