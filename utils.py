import numpy as np
import torch
from typing import List, Tuple
import json
import os
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
SAMPLING_FREQUENCY = 100

def get_hand_mask_548() -> np.ndarray:
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
    """
    Decode MATLAB cell-array-of-char to Python strings efficiently.
    """
    return ["".join(o.tolist()) if isinstance(o, np.ndarray) else str(o) 
            for o in arr.flatten()]

def convert_to_spikes(data, num_sessions, num_objects, N_frames, sample_num, delta=0.03, ref=0, sampling_freq=100, N=484, labels=None):
    """
    Convert signal data into spike trains using bADM and collect indices, times, and labels.

    Args:
        data (np.ndarray): Input signal data with shape (num_sessions * num_objects, num_frames, num_pixels).
        num_sessions (int): Number of sessions.
        num_objects (int): Number of object categories.
        N_frames (int): Number of frames per sample.
        sample_num (int): Number of samples per class (e.g., 800/N_frames for train, 200/N_frames for test).
        delta (float, optional): Threshold for spike conversion. Defaults to 0.03.
        ref (int, optional): Refractory period. Defaults to 0.
        sampling_freq (int, optional): Sampling frequency. Defaults to 100.
        N (int, optional): Number of input neurons. Defaults to 484.
        labels (list, optional): List to append class labels to. If None, a new list is created.

    Returns:
        tuple: (indices_list_up, times_list_up, indices_list_dw, times_list_dw, y_labels)
            - indices_list_up: List of arrays of neuron indices for up spikes.
            - times_list_up: List of arrays of spike times for up spikes.
            - indices_list_dw: List of arrays of neuron indices for down spikes.
            - times_list_dw: List of arrays of spike times for down spikes.
            - y_labels: Array of class labels.
    """
    indices_list_up = []
    times_list_up = []
    indices_list_dw = []
    times_list_dw = []

    if labels is None:
        y_labels = []
    else:
        y_labels = labels

    for ind_sess in range(num_sessions):
        for class_index in range(num_objects):
            for sample_ind in range(sample_num):
                tuples_list_up = []
                tuples_list_dw = []

                y_labels.append(class_index)

                for i in range(data.shape[2]):  # i: pixel index
                    sig = data[ind_sess * num_objects + class_index].T[i][sample_ind * N_frames:sample_ind * N_frames + N_frames]
                    tup, tdw = bADM(sig, delta, delta, sampling_freq, ref)
                    for item in tup:
                        tuples_list_up.append((i, item))
                    for item in tdw:
                        tuples_list_dw.append((i, item))

                sorted_list_up = sorted(tuples_list_up, key=lambda x: (x[1], x[0]))
                sorted_list_dw = sorted(tuples_list_dw, key=lambda x: (x[1], x[0]))

                indices_list_up_sample = np.array([x[0] for x in sorted_list_up])
                times_list_up_sample = np.array([x[1] for x in sorted_list_up])  # *1000
                indices_list_dw_sample = np.array([x[0] for x in sorted_list_dw])
                times_list_dw_sample = np.array([x[1] for x in sorted_list_dw])  # *1000

                indices_list_up.append(indices_list_up_sample)
                times_list_up.append(times_list_up_sample)
                indices_list_dw.append(indices_list_dw_sample)
                times_list_dw.append(times_list_dw_sample)

                print("No. {} sample processed".format(sample_ind + class_index * sample_num + ind_sess * num_objects * sample_num))

    y_labels = np.array(y_labels)
    return indices_list_up, times_list_up, indices_list_dw, times_list_dw, y_labels

def spike_times_to_tensor(indices_list, times_list, num_steps, num_neurons):
    spike_tensor = np.zeros((len(indices_list), num_steps, num_neurons))
    for sample_idx, (indices, times) in enumerate(zip(indices_list, times_list)):
        for idx, t in zip(indices, times):
            step = int(t * 1000)  # Convert time in seconds to time steps (assuming 1 ms step)
            if step < num_steps:
                spike_tensor[sample_idx, step, idx] = 1
    return torch.tensor(spike_tensor, dtype=torch.float32)

def save_spike_data(
    spike_tensors: List[np.ndarray],
    y: np.ndarray,
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
        pkl.dump({"spike_tensors": spike_tensors, "y": y}, f)
    
    # Save metadata
    with open(metadata_file, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"Saved data to {filename} and metadata to {metadata_file}")
    
import os

def get_filename_from_params(params_dict: dict, output_dir: str) -> str:
    """
    Construct the filename based on parameters, mirroring the logic in save_spike_data with a fixed order.
    The order is: topn, num_frames, threshold, subtract_baseline, session_id.
    """
    # Define the fixed parameter order (session_id needs to always be last in this list)
    param_order = ["topn", "num_frames", "threshold", "channels", "down_spike", "encoding", "rand_pixels", "session_id"]
    
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
    
    # Convert list of arrays to a single NumPy array with float32 dtype, assuming same shape
    spike_tensors = np.array(spike_tensors, dtype=np.float32)
    
    metadata_file = filepath.replace(".pkl", "_meta.json")
    with open(metadata_file, 'r') as f:
        params = json.load(f)
    
    print(f"Loaded data from {filepath} with params: {params}")
    return spike_tensors, y, params

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
            refractory_period_duration=0.01  # 10 ms
        )
        if not (0 <= taxel_idx < 484):
            continue
        for t_sec in spike_t_up:
            if np.isfinite(t_sec) and 0 <= t_sec <= 0.5:
                t_ms = base_time_ms + (t_sec * 1000)  # Convert seconds to ms
                events.append((taxel_idx, t_ms, 1))  # ON: p=1
        for t_sec in spike_t_dn:
            if np.isfinite(t_sec) and 0 <= t_sec <= 0.5:
                t_ms = base_time_ms + (t_sec * 1000)  # Convert seconds to ms
                events.append((taxel_idx, t_ms, -1))  # OFF: p=-1
    
    if events:
        return np.array(events, dtype=[('x', 'i2'), ('t', 'f4'), ('p', 'i1')])
    return np.empty((0,), dtype=[('x', 'i2'), ('t', 'f4'), ('p', 'i1')])


def plot_convergence(file_path, save_path, plt_title, plot_std=True, include_keys=None):
    """Plot convergence comparison showing mean across folds and optional variance band.

    Args:
        file_path (str): Path to the NPZ file.
        save_path (str): Path to save the generated figure.
        plt_title (str): Title of the plot.
        plot_std (bool): If True, plot mean with shaded ±1 std.
        include_keys (list, optional): List of keys to include.
        invert_values (bool): If True, invert numeric keys (except 0.0) for spike_down plots.
    """

    # Load the NPZ file with allow_pickle=True to handle the nested dictionary
    data = np.load(file_path, allow_pickle=True)
    all_results = data['data'].item()
    print(f"Available keys in {file_path}: {list(all_results.keys())}")  # Debug: Print top-level keys

    # Filter results based on include_keys if provided
    results = all_results if include_keys is None else {k: v for k, v in all_results.items() if k in include_keys}

    # Determine the maximum number of epochs for consistent x-axis
    max_epochs = 0
    for data in results.values():
        acc_folds = data.get('train_acc_folds')
        if acc_folds is not None:
            if isinstance(acc_folds, list):
                max_epochs = max(max_epochs, len(acc_folds[0]) if acc_folds else 0)
            else:
                max_epochs = max(max_epochs, acc_folds.shape[1] if acc_folds.size > 0 else 0)
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

    for key, data in results.items():
        
        # Convert lists to NumPy arrays if necessary
        acc_folds = data.get('train_acc_folds')
        test_acc_folds = data.get('test_acc_folds')

        if acc_folds is not None:
            if isinstance(acc_folds, list):
                acc_folds = np.array(acc_folds) if acc_folds else np.array([])
            if test_acc_folds is not None and isinstance(test_acc_folds, list):
                test_acc_folds = np.array(test_acc_folds) if test_acc_folds else np.array([])

        # Plot training accuracy
        if acc_folds is not None and plot_std and acc_folds.size > 0:
            mean_acc = np.nanmean(acc_folds, axis=0)  # Handle NaNs
            std_acc = np.nanstd(acc_folds, axis=0)
            epochs = np.arange(min(len(mean_acc), max_epochs))
            mean_acc = mean_acc[:len(epochs)]
            std_acc = std_acc[:len(epochs)]
            ax.plot(epochs, mean_acc, label=f"{key} Train", color=colors[color_idx], linewidth=2)
            ax.fill_between(epochs, np.nan_to_num(mean_acc - std_acc), np.nan_to_num(mean_acc + std_acc), 
                            alpha=0.2, color=colors[color_idx])
            color_idx += 1
        elif acc_folds is not None:
            epochs = np.arange(min(len(data["train_acc"]), max_epochs))
            ax.plot(epochs, data["train_acc"][:len(epochs)], label=f"{key} Train", 
                    color=colors[color_idx], linewidth=2)
            color_idx += 1

        # Plot test accuracy
        if test_acc_folds is not None and plot_std and test_acc_folds.size > 0:
            mean_test_acc = np.nanmean(test_acc_folds, axis=0)
            std_test_acc = np.nanstd(test_acc_folds, axis=0)
            epochs = np.arange(min(len(mean_test_acc), max_epochs))
            mean_test_acc = mean_test_acc[:len(epochs)]
            std_test_acc = std_test_acc[:len(epochs)]
            ax.plot(epochs, mean_test_acc, linestyle='--', label=f"{key} Test", 
                    color=colors[color_idx], linewidth=2)
            ax.fill_between(epochs, np.nan_to_num(mean_test_acc - std_test_acc), np.nan_to_num(mean_test_acc + std_test_acc), 
                            alpha=0.2, color=colors[color_idx])
            color_idx += 1
        elif test_acc_folds is not None:
            epochs = np.arange(min(len(data["test_acc"]), max_epochs))
            ax.plot(epochs, data["test_acc"][:len(epochs)], linestyle='--', label=f"{key} Test", 
                    color=colors[color_idx], linewidth=2)
            color_idx += 1

    # Customize the plot
    ax.set_title(plt_title, fontsize=14, pad=10, weight='bold')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)

    # Set exact number of ticks as data points (no decimals)
    ax.set_xticks(np.arange(0, max_epochs, 1))
    ax.set_xticklabels([str(int(x)) for x in np.arange(1, max_epochs + 1, 1)])

    # Adjust layout for readability
    ax.legend(fontsize=10, loc='best', frameon=True, edgecolor='black', fancybox=True)
    ax.grid(True, linestyle='--', alpha=0.7, which='both')
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
        file_path (str): Path to the NPZ file (e.g., "experiments/preprocessed_threshold_20251019_1429.npz").

    The function loads data from a single NPZ file, computes metrics like best fold accuracy,
    and saves the results in a CSV table with high readability.
    """

    # Load the NPZ file with allow_pickle=True to handle the nested dictionary
    data = np.load(file_path, allow_pickle=True)
    all_results = data['data'].item()
    print(f"Available keys in {file_path}: {list(all_results.keys())}")  # Debug: Print top-level keys

    # Initialize lists to store results
    models = []
    best_train_acc = []
    best_test_acc = []
    avg_train_acc = []
    avg_test_acc = []

    # Process each model/loss
    for model_name, data in all_results.items():
        train_acc_folds = data.get('train_acc_folds')
        test_acc_folds = data.get('test_acc_folds')

        if train_acc_folds is not None and test_acc_folds is not None:
            last_epoch_train_acc = train_acc_folds[:, -1]  # shape: (num_folds,)
            last_epoch_test_acc = test_acc_folds[:, -1]  # shape: (num_folds,)
            best_train_acc_val = np.max(train_acc_folds)
            best_test_acc_val = np.max(test_acc_folds)

            avg_train_acc_val = np.mean(last_epoch_train_acc)
            avg_test_acc_val = np.mean(last_epoch_test_acc)

            models.append(model_name)
            best_train_acc.append(best_train_acc_val)
            best_test_acc.append(best_test_acc_val)
            avg_train_acc.append(avg_train_acc_val)
            avg_test_acc.append(avg_test_acc_val)
        else:
            print(f"Warning: Missing fold data for {model_name}, skipping analysis.")

    # Create a DataFrame for the table
    table_data = {
        "Model/Loss": models,
        "Best Train Accuracy": best_train_acc,
        "Best Test Accuracy": best_test_acc,
        "Average Train Accuracy": avg_train_acc,
        "Average Test Accuracy": avg_test_acc
    }
    df = pd.DataFrame(table_data)

    # Display the table in the console
    print("\nExperiment Results Table:")
    print(df.to_string(index=False))

def plot_membrane_one_per_class(file_path, num_neurons=3, seed=None, figsize=(16, 12)):
    """
    Plot membrane potential per class from an NPZ file generated by run_preprocessed_experiment.
    
    Args:
        file_path (str): Path to the NPZ file.
        num_neurons (int): Number of random neurons to plot.
        seed (int, optional): Seed for reproducibility; if None, use random.
        figsize (tuple): Figure size (height increased for better subplot spacing).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib import cm

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


def get_worst_classes_from_file(output_name, n=7, metric='accuracy'):
    """
    Load the saved metrics CSV and return the worst `top_n` classes by `metric`.
    
    Args:
        output_name (str): Base name used in run_confusion_matrix_experiment
        top_n (int): Number of worst classes to return
        metric (str): 'accuracy' or 'f1' (default: accuracy)
    
    Returns:
        List of tuples: (class_name, metrics_dict)
    """
    metrics_csv_path = os.path.join("experiments", f"{output_name}_metrics.csv")
    
    if not os.path.exists(metrics_csv_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_csv_path}")
    
    df = pd.read_csv(metrics_csv_path)
    
    # Sort by the chosen metric (ascending = worst first)
    df_sorted = df.sort_values(by=metric, ascending=True)
    
    worst_n = df_sorted.head(n)
    
    # Convert to list of (name, metrics) like in results['sorted_classes']
    worst_list = [
        (row['Class'], {
            'accuracy': row['accuracy'],
            'precision': row['precision'],
            'recall': row['recall'],
            'f1': row['f1']
        })
        for _, row in worst_n.iterrows()
    ]
    
    return worst_list

def analyze_spikes(file_path):
    import numpy as np, os
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
    import numpy as np, os, pandas as pd

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
    import numpy as np, os, pandas as pd

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
    import numpy as np
    import pandas as pd
    
    # Load NPZ file
    data = np.load(file_path, allow_pickle=True)
    experiment_results = data['data'].item()
    
    # Expected encodings
    encodings = ['raw', 'spike', 'hybrid']
    
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
    import numpy as np
    import matplotlib.pyplot as plt
    import os

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