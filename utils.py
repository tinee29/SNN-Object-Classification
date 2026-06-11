import numpy as np
import torch
from typing import List, Tuple
import json
import os
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import matplotlib.colors as mcolors
from scipy.spatial.distance import cdist

SAMPLING_FREQUENCY = 100
REFRACTORY_PERIOD_DURATION = 0.01

def get_hand_mask_548() -> np.ndarray:
    """Return the default 548-taxel valid-hand mask as a flat boolean array.

    Returns:
        np.ndarray: Boolean array of shape (1024,) for a 32x32 tactile grid.
    """
    mask = np.array([
        np.ones(32), np.ones(32), np.ones(32),
        np.concatenate((np.zeros(14), np.ones(18))),
        np.concatenate((np.zeros(14), np.ones(18))),
        np.concatenate((np.zeros(14), np.ones(18))),
        np.ones(32), np.ones(32), np.ones(32),
        np.concatenate((np.zeros(14), np.ones(18))),
        np.ones(32), np.ones(32), np.ones(32),
        np.concatenate((np.zeros(14), np.ones(18))),
        np.concatenate((np.zeros(14), np.ones(18))),
        np.ones(32), np.ones(32), np.ones(32),
        np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
        np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
        np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
        np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
        np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
        np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
        np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
        np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
        np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
        np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
        np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
        np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
        np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
        np.concatenate((np.zeros(25), np.ones(4), np.zeros(3)))
    ]).astype(bool)
    return mask.reshape(1024,)

def get_palm_mask_484() -> np.ndarray:
    """Return the palm-only valid mask derived from the 548-taxel hand mask.

    The function removes finger regions using the same geometric convention
    already used in this project.

    Returns:
        np.ndarray: Boolean array of shape (1024,) containing valid palm taxels.
    """
    mask = get_hand_mask_548().reshape((32,32)) 
    finger_mask = np.zeros((32,32))
    finger_mask[:, 0:4] = 1
    finger_mask[28:32, :] = 1
    finger_mask = np.logical_and(mask, finger_mask)
    mask = mask.reshape((1024,)).astype(bool)
    finger_mask = finger_mask.reshape((1024,)).astype(bool)
    valid_palm_mask = np.logical_and(~finger_mask, mask)
    return valid_palm_mask


def bADM(input_signal,threshold_UP,threshold_DOWN,sampling_frequency,refractory_period_duration,return_signal = True):
    """Convert one analog trace to UP/DOWN spike times using bADM thresholding.

    Args:
        input_signal: One-dimensional input trace.
        threshold_UP: Positive threshold.
        threshold_DOWN: Negative threshold.
        sampling_frequency: Signal sampling rate (Hz).
        refractory_period_duration: Refractory period (s).
        return_signal: Legacy flag kept for backward compatibility.

    Returns:
        tuple[np.ndarray, np.ndarray]: UP and DOWN spike timestamps (seconds).
    """
    dt = 1/sampling_frequency
    end_time = len(input_signal)*dt
    times = np.linspace(0,end_time,len(input_signal)).astype(np.float64)
    DC_Voltage = input_signal[0]
    remainder_of_refractory = 0
    spike_t_up =  times[0:2]
    spike_t_dn = times[0:2]
    interpolate_from = 0.0
    interpolation_activation = 0
    intercept_point=0
    
    for i in range(len(times)):
        t = i * dt
        if i == 0:
            continue
        
        slope = ((input_signal[i]-input_signal[i-1])/dt)
        if remainder_of_refractory >= 2*dt:
            remainder_of_refractory = remainder_of_refractory-dt
            interpolation_activation = 1

        else:
            
            if interpolation_activation == 1:
                interpolate_from = (interpolate_from+remainder_of_refractory)
                remainder_of_refractory = 0
                if interpolate_from >= 2*dt:
                    interpolate_from = interpolate_from-dt
                    continue
                interpolate_from = (interpolate_from+remainder_of_refractory)%dt
                Vbelow = (input_signal[i-1] + interpolate_from*slope)
                DC_Voltage = Vbelow
            
                
            else:
                Vbelow = input_signal[i-1]
                interpolate_from = 0

            if DC_Voltage + threshold_UP <= input_signal[i]:
                intercept_point = t - dt + interpolate_from+((threshold_UP+DC_Voltage-Vbelow)/slope)
                spike_t_up = np.append(spike_t_up,intercept_point)
                interpolate_from = dt+intercept_point-t
                remainder_of_refractory = refractory_period_duration 
                interpolation_activation = 1
                continue

            elif DC_Voltage - threshold_DOWN >= input_signal[i]:
                intercept_point = t - dt + interpolate_from+((-threshold_DOWN+DC_Voltage-Vbelow)/slope)
                spike_t_dn = np.append(spike_t_dn,intercept_point)
                interpolate_from = dt+intercept_point-t
                remainder_of_refractory = refractory_period_duration 
                interpolation_activation = 1
                continue

            interpolation_activation = 0
                        
    index =[0,1]
    spike_t_up = np.delete(spike_t_up, index)
    spike_t_dn = np.delete(spike_t_dn, index)


#    if return_signal:
#        sup = np.zeros_like(times);sdw = np.zeros_like(times)
#        sup[np.searchsorted(times,spike_t_up,side='left')] = 1
#        sdw[np.searchsorted(times,spike_t_dn,side='left')] = 1
#        rsig=(threshold_UP * np.cumsum(sup)) + ((-threshold_DOWN) * np.cumsum(sdw)) + input_signal[0]

    return spike_t_up,spike_t_dn #,rsig


def get_true_indices(mask: np.ndarray) -> np.ndarray:
    """
    Convert a boolean mask into a list of indices where the values are True.
    
    Args:
        mask (np.ndarray): A boolean numpy array (e.g., 32x32).
        
    Returns:
        np.ndarray: Array of indices where the mask is True in the flattened array.
    """
    # Ensure the mask is boolean
    mask = mask.astype(bool)
    # Flatten the mask and find indices where values are True
    true_indices = np.where(mask.flatten())[0]
    return true_indices

def decode_matlab_strings(arr: np.ndarray) -> List[str]:
    """Decode MATLAB cell-array-of-char entries into Python strings.

    Args:
        arr (np.ndarray): MATLAB-style array of chars/strings.

    Returns:
        List[str]: Flattened decoded values.
    """
    return ["".join(o.tolist()) if isinstance(o, np.ndarray) else str(o) 
            for o in arr.flatten()]

def save_spike_data(
    spike_tensors: List[np.ndarray],
    y: np.ndarray,
    pixel_mask: np.ndarray,
    output_dir: str,
    params: dict
):
    """
    Save spike tensors and labels with metadata for an experiment.
    
    Args:
        spike_tensors: List of numpy arrays from SmartHandDataset2.get_data()[0]
        y: Numpy array of labels from SmartHandDataset2.get_data()[1]
        output_dir: Directory to save data
        params: Dict of parameters (e.g., {'topn': 300, 'num_frames': 50, ...})
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a unique filename based on parameters
    param_str = "_".join(f"{k}_{v if v is not None else 'None'}" for k, v in params.items() if k != "session_id")
    param_str += f"_session_id_{'_'.join(map(str, params['session_id']))}"
    filename = os.path.join(output_dir, f"spike_data_{param_str}.pkl")
    metadata_file = os.path.join(output_dir, f"spike_data_{param_str}_meta.json")
    
    # Save data as pickle
    with open(filename, 'wb') as f:
        pkl.dump({"spike_tensors": spike_tensors, "y": y, "pixel_mask": pixel_mask}, f)
    
    # Save metadata
    with open(metadata_file, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"Saved data to {filename} and metadata to {metadata_file}")

def get_filename_from_params(params_dict: dict, output_dir: str) -> str:
    """
    Construct the filename based on parameters, mirroring the logic in save_spike_data with a fixed order.
    The order is: topn, num_frames, threshold, subtract_baseline, session_id.
    """
    # Define the fixed parameter order (session_id needs to always be last in this list)
    param_order = ["method", "n", "num_frames", "threshold", "channels", "down_spike", "encoding", "session_id"]
    
    # Extract values in the fixed order, using None or default if not present
    param_values = []
    for param in param_order:
        if param == "session_id":
            value = "_".join(map(str, params_dict.get(param, [0, 1])))  # Default to [0, 1] if missing
        else:
            value = params_dict.get(param, None)
        param_values.append(value if value is not None else None)
    
    # Construct the parameter string with the fixed order
    param_str = "_".join(f"{k}_{v}" for k, v in zip(param_order, param_values))
    filename = os.path.join(output_dir, f"spike_data_{param_str}.pkl")
    return filename

def load_spike_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load spike tensors, labels, and metadata.
    
    Args:
        filepath: Path to .pkl file
    
    Returns:
        Tuple of (spike_tensors, y, params)
    """
    with open(filepath, 'rb') as f:
        data = pkl.load(f)
    spike_tensors = data["spike_tensors"]
    y = data["y"]
    pixel_mask = data["pixel_mask"]
    
    # Convert list of arrays to a single NumPy array with float32 dtype, assuming same shape
    spike_tensors = np.array(spike_tensors, dtype=np.float32)
    
    metadata_file = filepath.replace(".pkl", "_meta.json")
    with open(metadata_file, 'r') as f:
        params = json.load(f)
    
    print(f"Loaded data from {filepath} with params: {params}")
    return spike_tensors, y, params, pixel_mask

def decode_matlab_strings(arr: np.ndarray) -> List[str]:
    """
    Decode MATLAB cell-array-of-char to Python strings efficiently.

    Args:
        arr: MATLAB cell array containing character arrays or strings.

    Returns:
        List of decoded strings.

    Example:
        Input: array([['b', 'a', 'l', 'l'], ['c', 'u', 'p']])
        Output: ['ball', 'cup']
    """
    # NOTE: This later definition intentionally remains in place to preserve
    # the module's existing behavior and public call sites.
    return ["".join(o.tolist()) if isinstance(o, np.ndarray) else str(o) 
            for o in arr.flatten()]

def sample_to_events(signal, base_time_ms, threshold):
    """
    Generate events for a sample (50 frames × 484 taxels) using millisecond timestamps.
    
    Args:
        signal: np.array, shape (50, 484) - Tactile pressures.
        base_time_ms: float - Base timestamp (ms).
    
    Returns:
        np.array - Structured events [(x, i2), (t, f4), (p, i1)].
    """
    events = []
    num_taxels = signal.shape[1]
    taxel_indices = np.arange(num_taxels, dtype=np.int16)
    
    for taxel_idx in taxel_indices:
        taxel_signal = signal[:, taxel_idx]  # (50,)
        spike_t_up, spike_t_dn = bADM(
            taxel_signal,
            threshold_UP=threshold,
            threshold_DOWN=threshold,
            sampling_frequency=SAMPLING_FREQUENCY,  # 100 Hz
            refractory_period_duration=REFRACTORY_PERIOD_DURATION  # 10 ms
        )
        if not (0 <= taxel_idx < 484):
            continue
        for t_sec in spike_t_up:
            if np.isfinite(t_sec) and 0 <= t_sec <= 0.5:
                t_ms = base_time_ms + (t_sec * 1000)  # Convert seconds to ms
                events.append((taxel_idx, t_ms, 1))  # UP: p=1
        for t_sec in spike_t_dn:
            if np.isfinite(t_sec) and 0 <= t_sec <= 0.5:
                t_ms = base_time_ms + (t_sec * 1000)  # Convert seconds to ms
                events.append((taxel_idx, t_ms, -1))  # DOWN: p=-1
    
    if events:
        return np.array(events, dtype=[('x', 'i2'), ('t', 'f4'), ('p', 'i1')])
    return np.empty((0,), dtype=[('x', 'i2'), ('t', 'f4'), ('p', 'i1')])


def plot_convergence(
    file_path,
    save_path,
    plt_title=None,
    plot_std=True,
    include_keys=None,
    metric="acc",
    legend_labels=None,
):
    """Plot convergence curves for accuracy or loss across experiment keys.

    Args:
        file_path (str): Path to the NPZ file.
        save_path (str): Path to save the generated figure.
        plt_title (str, optional): Plot title. If None, a title is generated.
        plot_std (bool): If True, plot mean with shaded +-1 std from folds.
        include_keys (list, optional): Subset of experiment keys to include.
        metric (str): Metric to plot: "acc"/"accuracy" or "loss".
        legend_labels (list[str], optional): Custom labels to use in legend,
            ordered as the plotted result keys.
    """

    metric_key = str(metric).strip().lower()
    if metric_key in {"acc", "accuracy"}:
        train_folds_key, test_folds_key = "train_acc_folds", "test_acc_folds"
        train_avg_key, test_avg_key = "train_acc", "test_acc"
        y_label = "Accuracy"
        metric_title = "Accuracy"
        use_unit_ylim = True
    elif metric_key == "loss":
        train_folds_key, test_folds_key = "train_loss_folds", "test_loss_folds"
        train_avg_key, test_avg_key = "train_loss", "test_loss"
        y_label = "Loss"
        metric_title = "Loss"
        use_unit_ylim = False
    else:
        raise ValueError("metric must be one of {'acc', 'accuracy', 'loss'}")

    # Load the NPZ file with allow_pickle=True to handle the nested dictionary
    data = np.load(file_path, allow_pickle=True)
    all_results = data['data'].item()
    print(f"Available keys in {file_path}: {list(all_results.keys())}")  # Debug: Print top-level keys

    # Filter results based on include_keys if provided
    results = all_results if include_keys is None else {k: v for k, v in all_results.items() if k in include_keys}

    if legend_labels is not None and len(legend_labels) != len(results):
        raise ValueError(
            f"legend_labels length ({len(legend_labels)}) must match number of plotted keys ({len(results)})"
        )

    # Determine the maximum number of epochs for consistent x-axis
    max_epochs = 0
    for result_data in results.values():
        metric_folds = result_data.get(train_folds_key)
        if metric_folds is not None:
            if isinstance(metric_folds, list):
                max_epochs = max(max_epochs, len(metric_folds[0]) if metric_folds else 0)
            else:
                max_epochs = max(max_epochs, metric_folds.shape[1] if metric_folds.size > 0 else 0)
    if max_epochs == 0:
        print("No data to plot. Check keys and metrics in the NPZ file.")
        return

    # Set up the figure with a nicer built-in style and size
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 6), facecolor='white')
    ax = plt.gca()

    # Define a color cycle for better distinction
    colors = plt.cm.tab10(np.linspace(0, 1, len(results) * 2))
    color_idx = 0

    for result_idx, (key, result_data) in enumerate(results.items()):
        display_name = legend_labels[result_idx] if legend_labels is not None else key
        
        # Convert lists to NumPy arrays if necessary
        train_folds = result_data.get(train_folds_key)
        test_folds = result_data.get(test_folds_key)

        if train_folds is not None:
            if isinstance(train_folds, list):
                train_folds = np.array(train_folds) if train_folds else np.array([])
            if test_folds is not None and isinstance(test_folds, list):
                test_folds = np.array(test_folds) if test_folds else np.array([])

        # Plot train metric
        if train_folds is not None and plot_std and train_folds.size > 0:
            mean_train = np.nanmean(train_folds, axis=0)  # Handle NaNs
            std_train = np.nanstd(train_folds, axis=0)
            epochs = np.arange(min(len(mean_train), max_epochs))
            mean_train = mean_train[:len(epochs)]
            std_train = std_train[:len(epochs)]
            ax.plot(epochs, mean_train, label=f"{display_name} Train", color=colors[color_idx], linewidth=2)
            ax.fill_between(epochs, np.nan_to_num(mean_train - std_train), np.nan_to_num(mean_train + std_train),
                            alpha=0.2, color=colors[color_idx])
            color_idx += 1
        elif result_data.get(train_avg_key) is not None:
            epochs = np.arange(min(len(result_data[train_avg_key]), max_epochs))
            ax.plot(epochs, result_data[train_avg_key][:len(epochs)], label=f"{display_name} Train",
                    color=colors[color_idx], linewidth=2)
            color_idx += 1

        # Plot test metric
        if test_folds is not None and plot_std and test_folds.size > 0:
            mean_test = np.nanmean(test_folds, axis=0)
            std_test = np.nanstd(test_folds, axis=0)
            epochs = np.arange(min(len(mean_test), max_epochs))
            mean_test = mean_test[:len(epochs)]
            std_test = std_test[:len(epochs)]
            ax.plot(epochs, mean_test, linestyle='--', label=f"{display_name} Test",
                    color=colors[color_idx], linewidth=2)
            ax.fill_between(epochs, np.nan_to_num(mean_test - std_test), np.nan_to_num(mean_test + std_test),
                            alpha=0.2, color=colors[color_idx])
            color_idx += 1
        elif result_data.get(test_avg_key) is not None:
            epochs = np.arange(min(len(result_data[test_avg_key]), max_epochs))
            ax.plot(epochs, result_data[test_avg_key][:len(epochs)], linestyle='--', label=f"{display_name} Test",
                    color=colors[color_idx], linewidth=2)
            color_idx += 1

        # Quick textual report for the selected metric.
        if result_data.get(train_avg_key) is not None or result_data.get(test_avg_key) is not None:
            train_hist = np.asarray(result_data.get(train_avg_key, []), dtype=float)
            test_hist = np.asarray(result_data.get(test_avg_key, []), dtype=float)
            train_last = train_hist[-1] if train_hist.size > 0 else np.nan
            test_last = test_hist[-1] if test_hist.size > 0 else np.nan
            print(f"{key} ({metric_title}) -> last train: {train_last:.4f}, last test: {test_last:.4f}")

    # Customize the plot
    final_title = plt_title if plt_title else f"Convergence ({metric_title})"
    ax.set_title(final_title, fontsize=14, pad=10, weight='bold')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    # Set exact number of ticks as data points (no decimals)
    ax.set_xticks(np.arange(0, max_epochs, 1))
    ax.set_xticklabels([str(int(x)) for x in np.arange(1, max_epochs + 1, 1)])

    # Adjust layout for readability
    ax.legend(fontsize=10, loc='best', frameon=True, edgecolor='black', fancybox=True)
    ax.grid(True, linestyle='--', alpha=0.7, which='both')
    if use_unit_ylim:
        ax.set_ylim(0, 1)
    ax.tick_params(axis='both', labelsize=10)

    # Create experiments folder if it doesn't exist
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.show()

def analyze_experiment_results(file_path):
    """Analyze experiment results from an NPZ file and save a table with key metrics.

    Args:
        file_path (str): Path to the NPZ file.
    """
    # Load the NPZ file
    try:
        data_container = np.load(file_path, allow_pickle=True)
        all_results = data_container['data'].item()
        print(f"Available keys in {file_path}: {list(all_results.keys())}") 
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Initialize lists to store results
    models = []
    best_train_acc = []
    best_test_acc = []
    avg_train_acc = []
    avg_test_acc = []

    def _to_percent(arr: np.ndarray) -> np.ndarray:
        """Return values in percentage scale.

        Histories may be stored in [0, 1] while logs are shown in [0, 100].
        """
        arr = np.asarray(arr, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return arr
        return arr * 100.0 if np.nanmax(np.abs(finite)) <= 1.5 else arr

    # Process each model/loss
    for model_name, res_data in all_results.items():
        # Retrieve per-fold histories
        train_acc_raw = res_data.get('train_acc_folds')
        test_acc_raw = res_data.get('test_acc_folds')

        if train_acc_raw is None or test_acc_raw is None:
            print(f"Warning: Missing fold data for {model_name}, skipping.")
            continue

        # Robust fold-wise extraction (works even if folds have different lengths).
        try:
            last_epoch_train_acc = np.array([
                float(fold_hist[-1]) for fold_hist in train_acc_raw if len(fold_hist) > 0
            ], dtype=float)
            last_epoch_test_acc = np.array([
                float(fold_hist[-1]) for fold_hist in test_acc_raw if len(fold_hist) > 0
            ], dtype=float)
        except Exception:
            print(f"Warning: Invalid fold format for {model_name}, skipping.")
            continue

        if last_epoch_train_acc.size == 0 or last_epoch_test_acc.size == 0:
            print(f"Warning: Empty fold histories for {model_name}, skipping.")
            continue

        # Convert scales to percentages for consistency with training logs.
        last_epoch_train_acc = _to_percent(last_epoch_train_acc)
        last_epoch_test_acc = _to_percent(last_epoch_test_acc)

        # Best values across all epochs and folds (also in percentage scale).
        train_flat = _to_percent(np.array([
            float(v) for fold_hist in train_acc_raw for v in fold_hist
        ], dtype=float))
        test_flat = _to_percent(np.array([
            float(v) for fold_hist in test_acc_raw for v in fold_hist
        ], dtype=float))

        best_train_acc_val = np.nanmax(train_flat)
        best_test_acc_val = np.nanmax(test_flat)

        avg_train_acc_mean = np.nanmean(last_epoch_train_acc)
        avg_train_acc_std = np.nanstd(last_epoch_train_acc)
        avg_test_acc_mean = np.nanmean(last_epoch_test_acc)
        avg_test_acc_std = np.nanstd(last_epoch_test_acc)

        models.append(model_name)
        best_train_acc.append(f"{best_train_acc_val:.2f}")
        best_test_acc.append(f"{best_test_acc_val:.2f}")
        avg_train_acc.append(f"{avg_train_acc_mean:.2f} ± {avg_train_acc_std:.2f}")
        avg_test_acc.append(f"{avg_test_acc_mean:.2f} ± {avg_test_acc_std:.2f}")

    # Create DataFrame
    table_data = {
        "Model/Loss": models,
        "Best Train Accuracy [%]": best_train_acc,
        "Best Test Accuracy [%]": best_test_acc,
        "Average Train Accuracy [%] (mean ± std)": avg_train_acc,
        "Average Test Accuracy [%] (mean ± std)": avg_test_acc
    }
    df = pd.DataFrame(table_data)

    # Display results
    if not df.empty:
        print("\nExperiment Results Table:")
        print(df.to_string(index=False))
    else:
        print("\nNo valid data found to display.")

    return df


def analyze_topk_experiment_results(file_path, topk_label=3):
    """Analyze Top-k experiment results and report last-epoch fold averages.

    This mirrors ``analyze_experiment_results`` behavior for Top-k outputs:
    it reads per-fold histories and computes the average across folds at the
    last epoch.

    Args:
        file_path (str): Path to the NPZ file.
        topk_label (int): Label to use in output column names (default: 3).

    Returns:
        pd.DataFrame: Table with average last-epoch Top-1 and Top-k validation
        accuracies for each experiment key.
    """
    # Load the NPZ file
    try:
        data_container = np.load(file_path, allow_pickle=True)
        all_results = data_container['data'].item()
        print(f"Available keys in {file_path}: {list(all_results.keys())}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    models = []
    avg_val_top1 = []
    avg_val_topk = []

    def _to_percent(arr: np.ndarray) -> np.ndarray:
        """Return values in percentage scale.

        Histories are often stored in [0, 1] while training logs print [0, 100].
        This helper normalizes output to percentages for easier comparison.
        """
        arr = np.asarray(arr, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return arr
        return arr * 100.0 if np.nanmax(np.abs(finite)) <= 1.5 else arr

    for model_name, res_data in all_results.items():
        # Note: in CV workflows these "test" histories correspond to fold validation sets.
        val_top1_raw = res_data.get('test_top1_hist')
        val_topk_raw = res_data.get('test_topk_hist')

        if val_top1_raw is None or val_topk_raw is None:
            print(f"Warning: Missing top-k fold data for {model_name}, skipping.")
            continue

        # Robust fold-wise last-epoch extraction (works even if fold lengths differ).
        try:
            last_epoch_val_top1 = np.array([
                float(fold_hist[-1]) for fold_hist in val_top1_raw if len(fold_hist) > 0
            ], dtype=float)
        except Exception:
            print(f"Warning: Invalid Top-1 fold format for {model_name}, skipping.")
            continue

        # Some runs may store None entries for top-k when topk_accuracy <= 1.
        try:
            last_epoch_val_topk = np.array([
                np.nan if fold_hist[-1] is None else float(fold_hist[-1])
                for fold_hist in val_topk_raw
                if len(fold_hist) > 0
            ], dtype=float)
        except Exception:
            print(f"Warning: Invalid Top-{topk_label} fold format for {model_name}, skipping.")
            continue

        if last_epoch_val_top1.size == 0 or last_epoch_val_topk.size == 0:
            print(f"Warning: Empty fold histories for {model_name}, skipping.")
            continue

        models.append(model_name)
        last_epoch_val_top1 = _to_percent(last_epoch_val_top1)
        last_epoch_val_topk = _to_percent(last_epoch_val_topk)

        top1_mean = np.nanmean(last_epoch_val_top1)
        top1_std = np.nanstd(last_epoch_val_top1)
        topk_mean = np.nanmean(last_epoch_val_topk)
        topk_std = np.nanstd(last_epoch_val_topk)

        avg_val_top1.append(f"{top1_mean:.2f} ± {top1_std:.2f}")
        avg_val_topk.append(f"{topk_mean:.2f} ± {topk_std:.2f}")

    table_data = {
        "Model/Loss": models,
        "Average Validation Top-1 Accuracy [%] (mean ± std)": avg_val_top1,
        f"Average Validation Top-{topk_label} Accuracy [%] (mean ± std)": avg_val_topk,
    }
    df = pd.DataFrame(table_data)

    if not df.empty:
        print("\nTop-k Experiment Results Table (validation, last epoch, average across folds):")
        print(df.to_string(index=False))
    else:
        print("\nNo valid top-k data found to display.")

    return df

def plot_membrane_one_per_class(file_path, num_neurons=3, seed=None, figsize=(16, 12)):
    """
    Plot membrane potential per class from an NPZ file generated by run_preprocessed_experiment.
    
    Args:
        file_path (str): Path to the NPZ file.
        num_neurons (int): Number of random neurons to plot.
        seed (int, optional): Seed for reproducibility; if None, use random.
        figsize (tuple): Figure size (height increased for better subplot spacing).
    """

    # Load NPZ
    data = np.load(file_path, allow_pickle=True)
    all_results = data['data'].item()

    class_names = [
        'ball', 'battery', 'bracket', 'coin', 'empty_can', 'empty_hand', 'full_can',
        'gel', 'lotion', 'mug', 'pen', 'safety_glasses', 'scissors', 'screw_driver',
        'spray_can', 'stapler', 'tape'
    ]

    # Random seed for neuron selection (different each call if seed=None)
    if seed is None:
        seed = np.random.randint(0, 10000)
    np.random.seed(seed)

    # Loop over varying values (e.g., 'raw', 'spike')
    for var_value, res in all_results.items():
        mem = res.get('hidden_membrane')
        labels = res.get('hidden_membrane_labels')

        if mem is None or labels is None:
            print(f"No membrane data for {var_value}")
            continue

        num_samples, T, N_hidden = mem.shape
        print(f"\nPlotting {var_value}: {num_samples} samples, {T} time steps, {N_hidden} hidden neurons")

        # Randomly select num_neurons neurons
        selected_neurons = np.random.choice(N_hidden, size=num_neurons, replace=False)
        print(f"Selected neurons: {selected_neurons}")

        # Create figure with subplots for selected neurons (taller for better spacing)
        fig, axes = plt.subplots(num_neurons, 1, figsize=figsize, sharex=True)
        if num_neurons == 1:
            axes = [axes]

        # Color map for classes (distinct, readable)
        colors = cm.tab20(np.linspace(0, 1, len(class_names)))

        for n_idx, n in enumerate(selected_neurons):
            ax = axes[n_idx]
            seen = set()

            for cls in range(len(class_names)):
                idxs = np.where(labels == cls)[0]
                if len(idxs) == 0:
                    continue

                # Pick one random sample per class
                sample_idx = np.random.choice(idxs)
                if sample_idx in seen:
                    continue
                seen.add(sample_idx)

                trace = mem[sample_idx, :, n]
                ax.plot(trace, color=colors[cls], linewidth=2, label=class_names[cls])

            # Threshold line
            ax.axhline(1.0, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='Threshold (1.0)')

            ax.set_ylabel(f'Neuron {n}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(-0.5, max(1.5, max(ax.get_ylim()) + 0.2))  # Better y-limits

        axes[-1].set_xlabel('Time Step', fontsize=12, fontweight='bold')
        plt.suptitle(f'{var_value}: Hidden Neuron Membrane — One Sample Per Class\n(Random neurons: {selected_neurons}; Seed: {seed})', 
                     fontsize=14, fontweight='bold')
        
        # Legend (compact, readable)
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

        plt.tight_layout()
        plt.show()


def analyze_spikes(file_path):
    """Print a compact spike-summary report from an experiment NPZ file.

    Args:
        file_path (str): Path to NPZ results containing per-configuration stats.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    data = np.load(file_path, allow_pickle=True)
    all_results = data['data'].item()

    print(f"\n{'='*60}")
    print(f"Spike Analysis – {os.path.basename(file_path)}")
    print(f"{'='*60}")
    print(f"Varying values: {list(all_results.keys())}")

    for value, res in all_results.items():
        print(f"\n--- {value} ---")
        print(f"Total hidden spikes          : {res.get('total_hidden_spikes')}")
        print(f"Spikes per fold (first 8)    : {res.get('total_hidden_spikes_per_fold', [])[:8]} …")

        shape = res.get('hidden_spikes_shape')
        print(f"Hidden spikes array shape    : {shape}")

        # ---- per-sample stats -------------------------------------------------
        avg_sample = res.get('avg_spikes_per_test_sample')
        if avg_sample:
            print(f"Avg spikes per test sample   : {avg_sample['mean']:.2f} ± {avg_sample['std']:.2f}")

        # ---- per-neuron firing rate -------------------------------------------
        avg_rate = res.get('avg_firing_rate_per_neuron')
        if avg_rate:
            print(f"Avg firing rate per neuron   : {avg_rate['mean']:.5f} ± {avg_rate['std']:.5f}")

        print("-" * 45)

def analyze_membrane(file_path):
    """
    Universal membrane diagnostics — works with encoding, sparsity, alpha, etc. experiments.
    """

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    data = np.load(file_path, allow_pickle=True)
    all_res = data['data'].item()
    data.close()

    print(f"Loaded: {os.path.basename(file_path)}")
    print(f"Found {len(all_res)} configurations\n")

    rows = []
    for config_name, r in all_res.items():
        max_fold = r.get('max_membrane_per_fold')
        mean_fold = r.get('mean_membrane_per_fold')
        frac_above_05 = r.get('membrane_fraction_above_0.5')
        avg_max_per_sample = r.get('avg_max_membrane_per_test_sample', {})
        total_spikes = r.get('total_hidden_spikes', 0)

        row = {
            "Config": str(config_name),
            "Max Membrane (best fold)":  f"{np.max(max_fold) if max_fold is not None else np.nan:.3f}",
            "Mean Membrane (avg folds)": f"{np.mean(mean_fold) if mean_fold is not None else np.nan:.3f}",
            "% samples > 0.5":           f"{frac_above_05*100:5.2f}%" if frac_above_05 is not None else "—",
            "Avg peak mem / sample":     f"{avg_max_per_sample.get('mean', np.nan):.3f}",
            "Std peak mem / sample":     f"{avg_max_per_sample.get('std', np.nan):.3f}",
            "Total hidden spikes":       f"{total_spikes:,}" if total_spikes else "0"
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Auto-sort if keys are numeric (sparsity, alpha, etc.)
    try:
        df["sort_key"] = pd.to_numeric(df["Config"], errors='coerce')
        if df["sort_key"].notna().all():
            df = df.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)
    except:
        pass

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)

    print("\n" + "="*88)
    print("           MEMBRANE & SPIKING DIAGNOSTICS")
    print("="*88)
    print(df.to_string(index=False))
    print("="*88)

    print("\nInterpretation:")
    print("  • Max Membrane < 0.8  → too quiet → poor learning")
    print("  • Max Membrane 1.0–2.5 → ideal spiking regime")
    print("  • Max Membrane > 3.0   → exploding → reduce beta or learning rate")
    print("  • % samples > 0.5 > 40% → good temporal integration")
    print("  • Sparsity 0.4–0.6 usually gives best membrane health + accuracy")
    
    
def analyze_weights(file_path):
    """
    Universal weight analysis for any experiment with return_weights=True.
    Focus: input → hidden layer (fc1) connectivity.
    Shows:
      • Intended sparsity (from mask)
      • Actual sparsity (after training)
      • Leakage (pruned connections that became non-zero)
      • Average absolute weight magnitude
    """

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    data = np.load(file_path, allow_pickle=True)
    all_res = data['data'].item()
    data.close()

    print(f"Loaded: {os.path.basename(file_path)}")
    print(f"Found {len(all_res)} configurations\n")

    rows = []
    total_possible_synapses = None  # Will be computed from first weight matrix

    for config_name, r in all_res.items():
        weights_list = r.get('first_layer_weights')  # list of arrays: one per fold

        if weights_list is None or len(weights_list) == 0:
            print(f"  → No weights saved for config '{config_name}' (did you use return_weights=True?)")
            continue

        # Stack all folds: shape (num_folds, hidden_size, num_inputs)
        W = np.stack(weights_list, axis=0)
        if total_possible_synapses is None:
            total_possible_synapses = W.shape[1] * W.shape[2]

        # Flatten across folds and hidden neurons → (num_folds * hidden, num_inputs)
        W_flat = W.reshape(-1, W.shape[2])

        # Actual zeros after training
        actual_zeros = np.abs(W_flat) < 1e-8  # numerical zero
        actual_zero_fraction = actual_zeros.mean()
        actual_nonzero_fraction = 1.0 - actual_zero_fraction

        # Intended zeros: reconstruct the mask used during training
        # We know it was created as: torch.rand(hidden, input) < input_sparsity
        # But we don't have input_sparsity saved → infer from config name if possible
        try:
            # Try to extract sparsity from config name (e.g. "0.40", "0.6")
            intended_keep_ratio = float(str(config_name))
        except:
            intended_keep_ratio = None

        if intended_keep_ratio is not None:
            intended_zero_fraction = 1.0 - intended_keep_ratio
            leakage = np.mean(actual_nonzero_fraction > 1e-3 and intended_keep_ratio < 1.0)
            # More precise leakage: % of connections that should be zero but aren't
            should_be_zero_but_not = (actual_nonzero_fraction > 1e-6) & (intended_keep_ratio < 1.0)
            leakage_pct = np.mean((W_flat != 0) & (np.random.rand(*W_flat.shape) >= intended_keep_ratio)) * 100
            # Actually simpler: just compare actual vs intended
            expected_zero_fraction = 1.0 - intended_keep_ratio
            leakage_fraction = max(0.0, actual_zero_fraction - expected_zero_fraction)
        else:
            intended_zero_fraction = None
            leakage_fraction = None

        avg_abs_weight = np.mean(np.abs(W_flat))
        std_abs_weight = np.std(np.abs(W_flat))
        nonzero_weights = W_flat[W_flat != 0]
        avg_nonzero_weight = np.mean(np.abs(nonzero_weights)) if len(nonzero_weights) > 0 else 0.0

        row = {
            "Config": str(config_name),
            "Intended Keep": f"{intended_keep_ratio:.2f}" if intended_keep_ratio is not None else "—",
            "Actual Keep": f"{actual_nonzero_fraction:.4f}",
            "Actual Zero": f"{actual_zero_fraction:.4f}",
            "Leakage (%)": f"{leakage_fraction*100:5.2f}%" if leakage_fraction is not None else "—",
            "Avg |W| (all)": f"{avg_abs_weight:.5f}",
            "Avg |W| (nonzero)": f"{avg_nonzero_weight:.5f}",
            "Total Synapses": f"{total_possible_synapses:,}",
        }
        rows.append(row)

    if not rows:
        print("No weight data found in any configuration.")
        return

    df = pd.DataFrame(rows)

    # Sort by numeric config if possible (sparsity, alpha, etc.)
    try:
        df["sort_key"] = pd.to_numeric(df["Config"], errors='coerce')
        if df["sort_key"].notna().all():
            df = df.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)
    except:
        pass

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 140)

    print("\n" + "="*100)
    print("           INPUT → HIDDEN LAYER WEIGHT ANALYSIS")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)

    print("\nInterpretation:")
    print("  • Intended Keep = density used during training (e.g. 0.40 = 40% connections kept)")
    print("  • Actual Zero ≈ Intended Zero → mask was perfectly enforced")
    print("  • Leakage < 0.01% → excellent (your mask is strong!)")
    print("  • Higher sparsity → lower Avg |W| (nonzero) → stronger, more selective weights")
    print("  • Best accuracy usually at sparsity where Leakage ≈ 0 and Avg |W| (nonzero) is highest")
    print("\nIn your case: expect leakage ≈ 0.00% — your mask is permanent and perfect.")
    
def analyze_spike_counts_from_file(file_path, n=242, m=242):
    """
    Analyze spike counts from run_preprocessed_experiment NPZ file.
    
    Computes average spike counts across folds for first N neurons (UP/DOWN) and last M neurons (raw).
    
    Args:
        file_path (str): Path to NPZ file from run_preprocessed_experiment with return_spikes=True.
        n (int): Number of first neurons for spike input (UP/DOWN).
        m (int): Number of last neurons for raw pressure input.
    
    Returns:
        dict: Spike analysis summary.
    """
    
    # Load NPZ file
    data = np.load(file_path, allow_pickle=True)
    experiment_results = data['data'].item()
    
    # Expected encodings
    encodings = ['spike']
    
    analysis_summary = {}
    
    for encoding in encodings:
        if encoding not in experiment_results:
            print(f"Warning: Encoding '{encoding}' not found. Skipping.")
            continue
        
        data = experiment_results[encoding]
        
        # Try full traces first
        all_spk1_traces = data.get('all_spk1_traces', [])
        if all_spk1_traces:
            # Compute per fold
            spike_input_counts = []
            raw_input_counts = []
            
            for spk_traces_fold in all_spk1_traces:
                # spk_traces_fold: (T, B, H)
                total_spikes_fold = np.sum(spk_traces_fold, axis=(0, 1))  # (H,)
                
                # First N neurons (UP/DOWN spikes)
                spike_input_fold = np.sum(total_spikes_fold[:n])
                spike_input_counts.append(spike_input_fold)
                
                # Last M neurons (raw pressure)
                raw_input_fold = np.sum(total_spikes_fold[-m:])
                raw_input_counts.append(raw_input_fold)
            
            # Average across folds
            avg_spike_input = np.mean(spike_input_counts)
            avg_raw_input = np.mean(raw_input_counts)
            
            print(f"For '{encoding}' encoding:")
            if encoding == 'raw':
                print(f"  Average spikes for raw neurons (last {m}): {avg_raw_input:.2f}")
            elif encoding == 'spike':
                print(f"  Average spikes for spike neurons (first {n}): {avg_spike_input:.2f}")
            elif encoding == 'hybrid':
                print(f"  Average spikes for spike neurons (first {n}): {avg_spike_input:.2f}")
                print(f"  Average spikes for raw neurons (last {m}): {avg_raw_input:.2f}")
            
            analysis_summary[encoding] = {
                'spike_input_avg': float(avg_spike_input),
                'raw_input_avg': float(avg_raw_input)
            }
        else:
            print(f"Warning: No all_spk1_traces for '{encoding}'. Cannot compute neuron-specific counts.")
            analysis_summary[encoding] = {'spike_input_avg': 0.0, 'raw_input_avg': 0.0}
    
    return analysis_summary


def compute_inference_firing_rates(file_path, timestep_ms=10.0, num_hidden_override=None, verbose=True):
    """Compute per-sample average firing rates for all layers during inference.

    Loads the ``.npz`` file produced by ``run_fanin_experiment`` (called with
    ``return_spikes=True``) and, for each fan-in configuration, reports:

    - Average total input / hidden / output spikes per sample
    - Per-neuron firing rate (Hz) for each layer
    - Estimated synaptic operations (SOPs) per inference, which is the key
      metric for power estimation on event-driven hardware such as DYNAPSE.

    SOPs are estimated as::

        input_SOPs / sample  = avg_input_spikes × (fan_in / num_inputs) × num_hidden
        hidden_SOPs / sample = avg_hidden_spikes × num_outputs
        total_SOPs / sample  = input_SOPs + hidden_SOPs

    The input_SOPs formula reflects that each input spike propagates, on
    average, to ``fan_in / num_inputs × num_hidden`` hidden neurons (random
    fan-in connectivity).  The hidden_SOPs formula assumes a fully-connected
    hidden→output projection.

    Args:
        file_path (str): Path to the ``.npz`` file from ``run_fanin_experiment``.
        timestep_ms (float): Duration of one timestep in milliseconds (default
            10.0 ms, matching 100 Hz tactile frames).
        num_hidden_override (int or None): Override the number of hidden neurons
            (inferred automatically from trace shapes when None).
        verbose (bool): Print a formatted summary table when True.

    Returns:
        pd.DataFrame: One row per fan-in configuration with columns::

            fan_in, num_test_samples, T_steps, sample_duration_ms,
            num_inputs, num_hidden, num_outputs,
            input_spikes_per_sample, hidden_spikes_per_sample, hidden_spikes_std,
            output_spikes_per_sample, output_spikes_std,
            input_rate_hz, hidden_rate_hz, output_rate_hz,
            input_SOPs_per_sample, hidden_SOPs_per_sample, total_SOPs_per_sample
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = np.load(file_path, allow_pickle=True)
    all_results = data['data'].item()

    rows = []

    for key, res in all_results.items():
        # Parse fan-in value from key name (e.g. "fanin_128" → 128)
        try:
            fan_in = int(key.split('_')[1])
        except (IndexError, ValueError):
            fan_in = None

        # ── Hidden layer ──────────────────────────────────────────────────────
        all_spk1_traces = res.get('all_spk1_traces', [])
        if not all_spk1_traces:
            print(f"[{key}] No hidden spike traces — skipping (run with return_spikes=True).")
            continue

        # all_spk1_traces: list of (T, B_i, H) arrays, one entry per batch across all folds
        spk1 = np.concatenate(all_spk1_traces, axis=1)   # (T, total_B, H)
        T, num_samples, H = spk1.shape
        if num_hidden_override is not None:
            H = num_hidden_override

        sample_duration_s = (T * timestep_ms) / 1000.0

        # per-sample hidden spikes: sum over T and H → (B,)
        hidden_per_sample = spk1.sum(axis=(0, 2))
        avg_hidden_spikes = float(hidden_per_sample.mean())
        std_hidden_spikes = float(hidden_per_sample.std())
        hidden_rate_hz = avg_hidden_spikes / (H * sample_duration_s) if (H > 0 and sample_duration_s > 0) else 0.0

        # ── Output layer ──────────────────────────────────────────────────────
        all_spk2_traces = res.get('all_spk2_traces', [])
        if all_spk2_traces:
            spk2 = np.concatenate(all_spk2_traces, axis=1)   # (T, total_B, num_outputs)
            num_outputs = spk2.shape[2]
            output_per_sample = spk2.sum(axis=(0, 2))
            avg_output_spikes = float(output_per_sample.mean())
            std_output_spikes = float(output_per_sample.std())
            output_rate_hz = avg_output_spikes / (num_outputs * sample_duration_s) if (num_outputs > 0 and sample_duration_s > 0) else 0.0
        else:
            num_outputs = None
            avg_output_spikes = std_output_spikes = output_rate_hz = float('nan')

        # ── Input layer ───────────────────────────────────────────────────────
        # Stored as (num_folds, num_inputs) cumulative counts over the last epoch test set.
        input_counts = res.get('input_spike_counts_per_fold')
        num_inputs_inferred = None
        if input_counts is not None and len(input_counts) > 0:
            input_counts = np.array(input_counts)
            num_inputs_inferred = input_counts.shape[1]
            total_input_spikes = float(input_counts.sum())
            avg_input_spikes = total_input_spikes / num_samples
            input_rate_hz = avg_input_spikes / (num_inputs_inferred * sample_duration_s) if (num_inputs_inferred > 0 and sample_duration_s > 0) else 0.0
        else:
            avg_input_spikes = input_rate_hz = float('nan')

        # ── Synaptic operations per inference ─────────────────────────────────
        # input → hidden: random fan-in means each input spike reaches on average
        #   (fan_in / num_inputs) × num_hidden hidden neurons
        if fan_in is not None and num_inputs_inferred is not None and H > 0:
            input_SOPs = avg_input_spikes * (fan_in / num_inputs_inferred) * H
        else:
            input_SOPs = float('nan')

        # hidden → output: fully connected → each hidden spike touches num_outputs weights
        hidden_SOPs = avg_hidden_spikes * num_outputs if num_outputs is not None else float('nan')

        total_SOPs = (
            input_SOPs + hidden_SOPs
            if not (np.isnan(input_SOPs) or np.isnan(hidden_SOPs))
            else float('nan')
        )

        rows.append({
            'fan_in': fan_in,
            'num_test_samples': num_samples,
            'T_steps': T,
            'sample_duration_ms': T * timestep_ms,
            'num_inputs': num_inputs_inferred,
            'num_hidden': H,
            'num_outputs': num_outputs,
            'input_spikes_per_sample': round(avg_input_spikes, 2),
            'hidden_spikes_per_sample': round(avg_hidden_spikes, 2),
            'hidden_spikes_std': round(std_hidden_spikes, 2),
            'output_spikes_per_sample': round(avg_output_spikes, 2),
            'output_spikes_std': round(std_output_spikes, 2),
            'input_rate_hz': round(input_rate_hz, 3),
            'hidden_rate_hz': round(hidden_rate_hz, 3),
            'output_rate_hz': round(output_rate_hz, 3),
            'input_SOPs_per_sample': round(input_SOPs, 1) if not np.isnan(input_SOPs) else float('nan'),
            'hidden_SOPs_per_sample': round(hidden_SOPs, 1) if not np.isnan(hidden_SOPs) else float('nan'),
            'total_SOPs_per_sample': round(total_SOPs, 1) if not np.isnan(total_SOPs) else float('nan'),
        })

    df = pd.DataFrame(rows)
    if not df.empty and df['fan_in'].notna().any():
        df = df.sort_values('fan_in').reset_index(drop=True)

    if verbose:
        if df.empty:
            print("No results found — check that the NPZ was generated with return_spikes=True.")
        else:
            sample_duration_ms = df['sample_duration_ms'].iloc[0]
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 160)
            display_cols = [
                'fan_in', 'num_test_samples',
                'input_spikes_per_sample', 'hidden_spikes_per_sample', 'output_spikes_per_sample',
                'input_rate_hz', 'hidden_rate_hz', 'output_rate_hz',
                'total_SOPs_per_sample',
            ]
            print(f"\n{'='*110}")
            print(f"  INFERENCE FIRING RATE ANALYSIS — {os.path.basename(file_path)}")
            print(f"  Timestep: {timestep_ms} ms  |  Sample duration: {sample_duration_ms:.0f} ms  "
                  f"|  Configurations: {len(df)}")
            print(f"{'='*110}")
            print(df[display_cols].to_string(index=False))
            print('='*110)
            print("\nColumn guide:")
            print("  *_spikes_per_sample  : mean total spike count over all neurons in that layer per inference")
            print("  *_rate_hz            : mean firing rate per neuron (Hz) averaged over test samples")
            print("  total_SOPs_per_sample: estimated synaptic ops = input_spikes×(fan_in/N_in)×N_hid + hidden_spikes×N_out")

    return df

def plot_topk_accuracy_bar(file_path, save_path=None, title="(a)"):
    """
    Plot Top-1 and Top-3 test accuracy bar chart from run_preprocessed_experiment with topk_accuracy=3.

    Uses last-epoch test accuracy per fold → mean and std across folds.
    Matches your attached figure style.

    Args:
        file_path (str): Path to .npz file from run_preprocessed_experiment(topk_accuracy=3)
        save_path (str, optional): Where to save the figure
        title (str): Figure title, e.g. "(a)"

    Returns:
        dict: means and stds
    """

    data = np.load(file_path, allow_pickle=True)
    all_results = data['data'].item()

    # Collect last-epoch test accuracies from all configurations
    top1_last_epoch = []
    top3_last_epoch = []

    for key, res in all_results.items():
        if 'test_top1_hist' in res and 'test_topk_hist' in res:
            # res['test_top1_history'] is list of lists: [fold][epoch]
            top1_per_fold = [fold_hist[-1] for fold_hist in res['test_top1_hist']]
            top3_per_fold = [fold_hist[-1] for fold_hist in res['test_topk_hist']]

            top1_last_epoch.extend(top1_per_fold)
            top3_last_epoch.extend(top3_per_fold)

    if not top1_last_epoch:
        print("No Top-k data found. Did you run with topk_accuracy=3?")
        return None

    top1_vals = np.array(top1_last_epoch) * 100
    top3_vals = np.array(top3_last_epoch) * 100

    top1_mean = top1_vals.mean()
    top1_std = top1_vals.std()
    top3_mean = top3_vals.mean()
    top3_std = top3_vals.std()

    # === Plot ===
    plt.figure(figsize=(5, 7))
    bars = plt.bar(['Top-1', 'Top-3'],
                   [top1_mean, top3_mean],
                   yerr=[top1_std, top3_std],
                   capsize=12,
                   color=['#d62728', '#7f7f7f'],  # red, gray
                   error_kw={'linewidth': 2.5, 'capthick': 2.5})

    # Annotate values on top
    for bar, mean, std in zip(bars, [top1_mean, top3_mean], [top1_std, top3_std]):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                 f'{mean:.1f}±{std:.2f}',
                 ha='center', va='bottom', fontsize=13, fontweight='bold')

    plt.ylabel("Accuracy [%]", fontsize=14)
    plt.title(title, fontsize=16, pad=20)
    plt.ylim(0, 110)
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=14)

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Bar plot saved to {save_path}")

    plt.tight_layout()
    plt.show()

    return {
        'top1_mean': top1_mean,
        'top1_std': top1_std,
        'top3_mean': top3_mean,
        'top3_std': top3_std
    }

def plot_pixel_mask(file_path, active_color='#d32f2f', inactive_color="#d6d6d6",
                    title=None,
                    save_path=None):
    """Plot and optionally save a 32x32 pixel mask using project color semantics.

    Args:
        file_path (str): Path to a saved spike-data `.pkl` file.
        active_color (str): Color used for selected/active taxels.
        inactive_color (str): Color used for valid-but-inactive taxels.
        title (str | None): Optional figure title.
        save_path (str | None): Optional output filename stem under `figures/`.
    """
    
    _, _, _, pixel_mask = load_spike_data(file_path)
 
    if len(pixel_mask) != 1024:
        raise ValueError("pixel_mask must be of length 1024 (32×32 grid).")

    # ────────────────────────────────────────────────────────────────
    # Thesis/report styling
    # ────────────────────────────────────────────────────────────────
    plt.style.use('seaborn-v0_8-white')
    plt.rcParams.update({
        'figure.dpi': 400,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.spines.bottom': False,
    })

    # Reshape masks
    active_grid = pixel_mask.reshape(32, 32).astype(bool)
    valid_grid = get_hand_mask_548().reshape(32, 32)
    
    # Display grid: 0=inactive (white), 1=valid hand (green), 2=active (red)
    display_grid = np.zeros((32, 32), dtype=int)
    display_grid[valid_grid] = 1          # green for valid hand
    display_grid[active_grid] = 2         # red for selected (overrides)

    # Colormap: white → green → red
    cmap = mcolors.ListedColormap(['white', inactive_color, active_color])

    # Rotate only the pixel map clockwise before plotting.
    display_grid = np.rot90(display_grid, k=-1)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.imshow(display_grid, cmap=cmap, origin='upper', interpolation='none')

    # Subtle grid
    ax.set_xticks(np.arange(-0.5, 32, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 32, 1), minor=True)
    ax.grid(which='minor', color='#e0e0e0', linestyle='-', linewidth=0.4, alpha=0.7)
    ax.tick_params(which='minor', size=0)

    # Clean axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Title
    ax.set_title(title, fontsize=24, fontweight='semibold', pad=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(f'figures/{save_path}.png', dpi=400, bbox_inches='tight', facecolor='white')

    plt.show()
    
def get_fps_indices(n_points):
    """
    Performs Farthest Point Sampling (FPS) on the valid pixels.
    
    Args:
        valid_mask (np.array): Boolean mask of shape (1024,) indicating valid taxels.
        n_points (int): Number of points to select.
        
    Returns:
        chosen_indices (np.array): Array of indices (0-1023) selected via FPS.
    """
    # 1. Get (row, col) coordinates of all VALID pixels
    valid_indices = np.where(get_hand_mask_548())[0]
    rows = valid_indices // 32
    cols = valid_indices % 32
    coords = np.stack([rows, cols], axis=1).astype(np.float32) # Shape (N_valid, 2)
    
    # 2. Initialize Selection
    num_valid = len(valid_indices)
    if n_points >= num_valid:
        return valid_indices # Return all if N is larger than available

    # Randomly pick the first point to make it slightly robust (or fix index 0 for determinism)
    # Using a fixed seed ensures reproducibility like "random_state=42"
    rng = np.random.default_rng(seed=42)
    first_idx = rng.integers(0, num_valid)
    
    selected_indices_local = [first_idx]
    
    # Initialize distances: min_dist[i] is distance from point i to the CLOSEST selected point
    # Start with distance to the first point
    dists = cdist(coords, coords[first_idx:first_idx+1], metric='euclidean').flatten()
    
    # 3. Iteratively select farthest points
    for _ in range(n_points - 1):
        # Pick point with largest distance to the current set
        dists_jitter = dists + rng.uniform(-1e-9, 1e-9, size=len(dists))  # tiny noise
        farthest_idx = np.argmax(dists_jitter)
        selected_indices_local.append(farthest_idx)
        
        # Update distances: New dist is min(old_dist, dist_to_new_point)
        new_dists = cdist(coords, coords[farthest_idx:farthest_idx+1], metric='euclidean').flatten()
        dists = np.minimum(dists, new_dists)
        
    # Convert local indices back to global 1024-based indices
    chosen_indices = valid_indices[np.array(selected_indices_local)]
    
    return chosen_indices

def plot_accuracy_comparison_bars(
    method_1_accuracies,
    method_2_accuracies,
    method_1_stds=None,
    method_2_stds=None,
    params=[100, 200, 300, 400],
    width=35,           # bar width in same units as params
    figsize=(10, 6)
):
    """Compare two accuracy series with side-by-side bars and optional error bars.

    Args:
        method_1_accuracies: Mean accuracies for method 1.
        method_2_accuracies: Mean accuracies for method 2.
        method_1_stds: Optional standard deviations for method 1.
        method_2_stds: Optional standard deviations for method 2.
        params: X-axis parameter values.
        width: Legacy width argument (auto-overridden for spacing consistency).
        figsize: Figure size tuple.
    """
    if len(method_1_accuracies) != len(params) or len(method_2_accuracies) != len(params):
        raise ValueError("Both accuracy lists must have the same length as params.")

    if method_1_stds is not None and len(method_1_stds) != len(params):
        raise ValueError("method_1_stds must have the same length as params.")
    if method_2_stds is not None and len(method_2_stds) != len(params):
        raise ValueError("method_2_stds must have the same length as params.")

    x = np.array(params)
    width = (max(x) - min(x)) * 0.08 if len(x) > 1 else 20  # auto-scale width reasonably

    fig, ax = plt.subplots(figsize=figsize)

    method_1_vals = np.array(method_1_accuracies, dtype=float)
    method_2_vals = np.array(method_2_accuracies, dtype=float)
    method_1_err = np.array(method_1_stds, dtype=float) if method_1_stds is not None else None
    method_2_err = np.array(method_2_stds, dtype=float) if method_2_stds is not None else None

    error_style = {'elinewidth': 1.2, 'capsize': 4, 'capthick': 1.2}

    bars_pca = ax.bar(
        x - width/2,
        method_1_vals,
        width,
        yerr=method_1_err,
        label='Method 1',
        color='lightblue',
        edgecolor='blue',
        linewidth=1.2,
        error_kw=error_style
    )
    bars_rnd = ax.bar(
        x + width/2,
        method_2_vals,
        width,
        yerr=method_2_err,
        label='Method 2',
        color='lightgreen',
        edgecolor='green',
        linewidth=1.2,
        error_kw=error_style
    )
    ax.tick_params(axis='both', labelsize=12)

    # Vertical labels inside each bar, near the top, in mean ± std format.
    x_text_offset = width * 0.05

    for i, bar in enumerate(bars_pca):
        h = bar.get_height()
        std_val = method_1_err[i] if method_1_err is not None else 0.0
        label = f"{h:.3f} ± {std_val:.3f}"
        y_pos = max(h - 0.06, h * 0.65)
        ax.text(
            bar.get_x() + bar.get_width() / 2 + x_text_offset,
            y_pos,
            label,
            ha='center',
            va='top',
            rotation=90,
            fontsize=12,
            color='black'
        )

    for i, bar in enumerate(bars_rnd):
        h = bar.get_height()
        std_val = method_2_err[i] if method_2_err is not None else 0.0
        label = f"{h:.3f} ± {std_val:.3f}"
        y_pos = max(h - 0.06, h * 0.65)
        ax.text(
            bar.get_x() + bar.get_width() / 2 + x_text_offset,
            y_pos,
            label,
            ha='center',
            va='top',
            rotation=90,
            fontsize=12,
            color='black'
        )

    ax.set_xlabel('Selected Taxels', fontsize=14)
    ax.set_ylabel('Validation Accuracy', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=14)

    plt.tight_layout()
    plt.savefig('figures/topn_random_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_topk_baseline_vs_ours_bars(
    baseline_top1,
    baseline_top3,
    our_top1,
    our_top3,
    params,
    baseline_top1_stds=None,
    baseline_top3_stds=None,
    our_top1_stds=None,
    our_top3_stds=None,
    figsize=(12, 6),
    save_path=None,
    y_label='Accuracy',
    x_label='Parameter'
):
    """Plot 4 bars per parameter: baseline Top-1/Top-3 and our Top-1/Top-3.

    Per parameter, the bars are placed as:
    baseline Top-1 | baseline Top-3    (gap)    our Top-1 | our Top-3

    Args:
        baseline_top1, baseline_top3, our_top1, our_top3: Lists/arrays of means.
        params: X-axis labels/tick values (same length as accuracy arrays).
        *_stds: Optional std values for error bars, same length as params.
        figsize: Matplotlib figure size.
        save_path: Optional output image path.
        y_label: Y-axis label.
        x_label: X-axis label.
    """
    n = len(params)
    for arr, name in [
        (baseline_top1, 'baseline_top1'),
        (baseline_top3, 'baseline_top3'),
        (our_top1, 'our_top1'),
        (our_top3, 'our_top3'),
    ]:
        if len(arr) != n:
            raise ValueError(f"{name} must have the same length as params.")

    for arr, name in [
        (baseline_top1_stds, 'baseline_top1_stds'),
        (baseline_top3_stds, 'baseline_top3_stds'),
        (our_top1_stds, 'our_top1_stds'),
        (our_top3_stds, 'our_top3_stds'),
    ]:
        if arr is not None and len(arr) != n:
            raise ValueError(f"{name} must have the same length as params.")

    b1 = np.array(baseline_top1, dtype=float)
    b3 = np.array(baseline_top3, dtype=float)
    o1 = np.array(our_top1, dtype=float)
    o3 = np.array(our_top3, dtype=float)

    b1_err = np.array(baseline_top1_stds, dtype=float) if baseline_top1_stds is not None else None
    b3_err = np.array(baseline_top3_stds, dtype=float) if baseline_top3_stds is not None else None
    o1_err = np.array(our_top1_stds, dtype=float) if our_top1_stds is not None else None
    o3_err = np.array(our_top3_stds, dtype=float) if our_top3_stds is not None else None

    x = np.arange(n, dtype=float)
    bar_w = 0.18
    pair_gap = 0.03
    block_gap = 0.12

    # Offsets: two adjacent baseline bars, a gap, then two adjacent our bars.
    off_b1 = -(1.5 * bar_w + pair_gap / 2 + block_gap / 2)
    off_b3 = off_b1 + bar_w + pair_gap
    off_o1 = off_b3 + bar_w + block_gap
    off_o3 = off_o1 + bar_w + pair_gap

    fig, ax = plt.subplots(figsize=figsize)
    error_style = {'elinewidth': 1.1, 'capsize': 4, 'capthick': 1.1}

    bars_b1 = ax.bar(
        x + off_b1, b1, bar_w, yerr=b1_err,
        label='Baseline Top-1', color='#f28e8e', edgecolor='#8b0000', linewidth=1.0, error_kw=error_style
    )
    bars_b3 = ax.bar(
        x + off_b3, b3, bar_w, yerr=b3_err,
        label='Baseline Top-3', color='#f8b4b4', edgecolor='#8b0000', linewidth=1.0, error_kw=error_style
    )
    bars_o1 = ax.bar(
        x + off_o1, o1, bar_w, yerr=o1_err,
        label='Our Study Top-1', color='#93c5fd', edgecolor='#1d4ed8', linewidth=1.0, error_kw=error_style
    )
    bars_o3 = ax.bar(
        x + off_o3, o3, bar_w, yerr=o3_err,
        label='Our Study Top-3', color='#bfdbfe', edgecolor='#1d4ed8', linewidth=1.0, error_kw=error_style
    )

    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', labelsize=11)

    ymax = np.nanmax(np.concatenate([b1, b3, o1, o3])) if n > 0 else 1.0
    ax.set_ylim(0, max(1.05, ymax * 1.15))

    ax.legend(loc='best', fontsize=11, frameon=True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    plt.show()
    
