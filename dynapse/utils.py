import numpy as np
import json
import os
import pickle as pkl
from dynapse1constants import NEURONS_PER_CORE

def get_filename_from_params(params_dict: dict, output_dir: str) -> str:
    """Build filename from preprocessing parameters."""
    param_order = ["method", "n", "num_frames", "threshold", "channels", "down_spike", "encoding", "session_id"]
    param_values = []
    
    for param in param_order:
        if param == "session_id":
            value = "_".join(map(str, params_dict.get(param, [0, 1])))
        else:
            value = params_dict.get(param, None)
        param_values.append(value)
    
    param_str = "_".join(f"{k}_{v}" for k, v in zip(param_order, param_values))
    return os.path.join(output_dir, f"spike_data_{param_str}.pkl")

def load_spike_data(filepath: str):
    """Load spike data from pickle file and metadata from JSON."""
    with open(filepath, 'rb') as f:
        data = pkl.load(f)
    
    spike_tensors = np.array(data["spike_tensors"], dtype=np.float32)
    y = data["y"]
    pixel_mask = data["pixel_mask"]
    
    metadata_file = filepath.replace(".pkl", "_meta.json")
    with open(metadata_file, 'r') as f:
        params = json.load(f)
    
    print(f"Loaded: {filepath} | params: {params}")
    return spike_tensors, y, params, pixel_mask

def load_and_prepare_data(preprocess_params, selected_classes, output_dir="preprocessed_data"):
    """
    Load preprocessed data, filter classes, remap labels, and create 80/20 train/test split.
    
    Returns:
        (X_train, y_train, X_test, y_test, num_inputs, num_classes, kept_class_names)
    """
    # Load raw data
    filename = get_filename_from_params(preprocess_params, output_dir)
    print(f"Loading: {filename}")
    
    spike_tensors, y_tensors, params_meta, pixel_mask = load_spike_data(filename)
    spike_tensors = np.array(spike_tensors, dtype=np.float32)
    y_tensors = np.array(y_tensors, dtype=np.int64)
    
    num_samples, num_time_steps, num_inputs = spike_tensors.shape
    print(f"Raw shape: {num_samples} samples, T={num_time_steps}, inputs={num_inputs}")
    
    assert num_inputs <= NEURONS_PER_CORE, f"num_inputs ({num_inputs}) > NEURONS_PER_CORE ({NEURONS_PER_CORE})"
    
    # Class filtering and remapping
    ALL_CLASS_NAMES = [
        'ball', 'battery', 'bracket', 'coin', 'empty_can', 'empty_hand', 'full_can',
        'gel', 'lotion', 'mug', 'pen', 'safety_glasses', 'scissors', 'screw_driver',
        'spray_can', 'stapler', 'tape'
    ]
    
    if selected_classes is None or len(selected_classes) == 0:
        kept_class_names = list(ALL_CLASS_NAMES)
        kept_old_ids = list(range(len(ALL_CLASS_NAMES)))
    else:
        unknown = [n for n in selected_classes if n not in ALL_CLASS_NAMES]
        if unknown:
            raise ValueError(f"Unknown classes: {unknown}")
        kept_class_names = list(selected_classes)
        kept_old_ids = [ALL_CLASS_NAMES.index(n) for n in kept_class_names]
    
    # Filter samples
    mask = np.isin(y_tensors, kept_old_ids)
    spike_tensors = spike_tensors[mask]
    y_old = y_tensors[mask]
    
    # Remap labels to 0..K-1
    old_to_new = {old: new for new, old in enumerate(kept_old_ids)}
    y_tensors = np.array([old_to_new[int(y)] for y in y_old], dtype=np.int64)
    
    # Sort by class label
    orig_idx = np.arange(len(y_tensors), dtype=np.int64)
    order = np.lexsort((orig_idx, y_tensors))
    spike_tensors = spike_tensors[order]
    y_tensors = y_tensors[order]
    
    num_samples, num_time_steps, num_inputs = spike_tensors.shape
    num_classes = len(kept_class_names)
    
    print(f"Kept {num_classes} classes: {kept_class_names}")
    print(f"Class counts: {np.bincount(y_tensors, minlength=num_classes)}")
    
    # 80/20 per-class split
    train_indices = []
    test_indices = []
    
    for cls in np.unique(y_tensors):
        cls_inds = np.where(y_tensors == cls)[0]
        split_idx = int(0.8 * len(cls_inds))
        train_indices.extend(cls_inds[:split_idx])
        test_indices.extend(cls_inds[split_idx:])
    
    train_indices = np.array(train_indices, dtype=np.int64)
    test_indices = np.array(test_indices, dtype=np.int64)
    
    X_train = spike_tensors[train_indices]
    y_train = y_tensors[train_indices]
    X_test = spike_tensors[test_indices]
    y_test = y_tensors[test_indices]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    return X_train, y_train, X_test, y_test, num_inputs, num_classes, kept_class_names

def make_budgeted_fanin_lists(
    pre_size: int,
    post_size: int,
    fanin_budget: int,
    max_weight: int = 1,
    seed: int = None,
):
    """Create (pre_list, post_list) with weights summing to fanin_budget per post-neuron."""
    if fanin_budget > pre_size * max_weight:
        raise ValueError(f"fanin_budget={fanin_budget} infeasible for pre_size={pre_size}, max_weight={max_weight}")
    
    rng = np.random.default_rng(seed)
    pre_list, post_list = [], []
    k_min = int(np.ceil(fanin_budget / max_weight))
    k_max = min(pre_size, fanin_budget)
    
    for post_idx in range(post_size):
        k = int(rng.integers(low=k_min, high=k_max + 1))
        weights = np.ones(k, dtype=int)
        remaining = fanin_budget - k
        
        while remaining > 0:
            candidates = np.where(weights < max_weight)[0]
            pick = int(rng.choice(candidates))
            weights[pick] += 1
            remaining -= 1
        
        pre_indices = np.sort(rng.choice(pre_size, size=k, replace=False))
        for pre_idx, w in zip(pre_indices, weights):
            pre_list.extend([int(pre_idx)] * int(w))
            post_list.extend([int(post_idx)] * int(w))
    
    return pre_list, post_list

def enforce_exact_fanin_budget(int_w: np.ndarray, fanin_budget: int, max_weight: int = 4, seed: int = None) -> np.ndarray:
    """Adjust weight matrix so each column sums exactly to fanin_budget with entries in [0, max_weight]."""
    rng = np.random.default_rng(seed)
    W = np.clip(int_w.copy().astype(int), 0, max_weight)
    
    for j in range(W.shape[1]):
        col = W[:, j]
        s = int(col.sum())
        
        # Decrement where column sum is too large
        while s > fanin_budget:
            idxs = np.where(col > 0)[0]
            i = int(rng.choice(idxs))
            col[i] -= 1
            s -= 1
        
        # Increment where column sum is too small
        while s < fanin_budget:
            idxs = np.where(col < max_weight)[0]
            i = int(rng.choice(idxs))
            col[i] += 1
            s += 1
        
        W[:, j] = col
    
    return W

def build_fpga_stimulus_for_sample(
    sample_tensor,
    base_spikegen_id: int = 0,
    chip_id: int = 0,
    frame_duration_ms: float = 10.0,
):
    """Convert spike tensor (T, F) to FPGA stimulus (spike_times, indices, target_chips).
    
    Spike times are returned in seconds. Multiple spikes in same frame are spaced within that frame.
    """
    T, F = sample_tensor.shape
    assert F <= NEURONS_PER_CORE, f"num_features ({F}) > NEURONS_PER_CORE ({NEURONS_PER_CORE})"
    
    spike_times = []
    indices = []
    target_chips = []
    
    frame_duration_s = frame_duration_ms / 1000.0  # frame duration in seconds
    
    for t in range(T):
        frame_start_s = t * frame_duration_s
        active = np.where(sample_tensor[t] > 0.0)[0]
        
        for f in active:
            # Place spike in the middle of the frame for timing accuracy
            spike_time_s = frame_start_s + frame_duration_s / 2.0
            spike_times.append(float(spike_time_s))
            indices.append(int(base_spikegen_id + f))
            target_chips.append(int(chip_id))
    
    return spike_times, indices, target_chips
def build_teacher_spikes(
    label: int,
    duration_s: float,
    teacher_offset: int,
    teacher_rate_hz: float = 100.0,
):
    """Generate regular teacher spike train for one spike-generator.
    
    Args:
        duration_s: Duration in seconds
        teacher_rate_hz: Spike rate in Hz
    
    Returns spike times in seconds.
    """
    if teacher_rate_hz <= 0:
        return [], []
    
    n_spikes = int(np.floor(duration_s * teacher_rate_hz))
    if n_spikes <= 0:
        return [], []
    
    times = np.linspace(0.0, duration_s, n_spikes, endpoint=False)
    idx = teacher_offset + int(label)
    
    return times.tolist(), [idx] * len(times)

def build_fpga_stimulus_for_sample_with_teacher(
    sample_tensor,
    label: int,
    chip_id: int,
    base_spikegen_id: int,
    teacher_offset: int,
    frame_duration_ms: float,
    teacher_rate_hz: float = 100.0,
):
    """Combine input spikes with teacher forcing for the correct label.
    
    All spike times returned in seconds.
    """
    T, _ = sample_tensor.shape
    duration_s = (T * frame_duration_ms) / 1000.0  # total duration in seconds
    
    # Input spikes from sample
    spike_times, indices, target_chips = build_fpga_stimulus_for_sample(
        sample_tensor,
        base_spikegen_id=base_spikegen_id,
        chip_id=chip_id,
        frame_duration_ms=frame_duration_ms,
    )
    
    # Teacher spikes for the label (in seconds)
    t_times, t_indices = build_teacher_spikes(
        label=int(label),
        duration_s=float(duration_s),
        teacher_offset=int(teacher_offset),
        teacher_rate_hz=float(teacher_rate_hz),
    )
    
    # Merge and sort by time
    spike_times.extend(t_times)
    indices.extend(t_indices)
    target_chips.extend([int(chip_id)] * len(t_times))
    
    order = np.argsort(spike_times)
    spike_times = [spike_times[i] for i in order]
    indices = [indices[i] for i in order]
    target_chips = [target_chips[i] for i in order]
    
    return spike_times, indices, target_chips, float(duration_s)

