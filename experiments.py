import os
import numpy as np
import pandas as pd
import torch
from utils import load_spike_data, get_filename_from_params
from models import cv_train, FC_SNN_Syn, FC_SNN_Syn_33, create_cv_folds, create_session_cv_folds, train, FC_SNN_Syn_32, cv_train_topk, train_topk
from matplotlib import pyplot as plt
import seaborn as sns
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_loss_experiment(device, num_folds, num_epochs, batch_size, verbose, input_file, output_name, loss_list=None):
    """Run loss function comparison experiments using train().

    This function delegates training to the train(...) function which handles fold
    creation and per-fold training. Returns and saves a single results dict including
    all loss functions' per-epoch averages and per-fold arrays in the experiments folder.

    Args:
        device: The device to run the experiment on (e.g., torch.device("cuda" or "cpu")).
        num_folds (int): Number of cross-validation folds.
        num_epochs (int): Number of training epochs per fold.
        batch_size (int): Batch size for training.
        verbose (bool): If True, print progress messages.

    Returns:
        dict: Dictionary containing results for all loss functions with nested structure.
    """

    output_dir = "preprocessed_data"
    filename = os.path.join(output_dir, input_file)
    spike_tensors, y_tensors, _, _ = load_spike_data(filename)
    
    num_inputs = spike_tensors[0].shape[1]
    num_outputs = len(np.unique(y_tensors))

    (cv_train_data_folds, cv_train_labels_folds, cv_test_data_folds, cv_test_labels_folds, val_data, val_labels, _, _) = create_cv_folds(
             spike_tensors, y_tensors, num_folds=num_folds, num_sessions=2, num_classes=num_outputs,
             random_state=42)

    if loss_list is None:
        print("No loss list provided!")

    all_results = {}

    for loss_name in loss_list:
        if verbose:
            print(f"\n{'#'*70}\nRunning loss function: {loss_name}\n{'#'*70}")

        res = cv_train(
            cv_train_data_folds=cv_train_data_folds,
            cv_train_labels_folds=cv_train_labels_folds,
            cv_test_data_folds=cv_test_data_folds,
            cv_test_labels_folds=cv_test_labels_folds,
            model_class=FC_SNN_Syn,
            num_folds=num_folds,
            num_epochs=num_epochs,
            batch_size=batch_size,
            random_state=42,
            device=device,
            verbose=verbose,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            loss_fn=loss_name
        )

        train_loss_avg = np.array(res['avg_loss_hist'])
        train_acc_avg = np.array(res['avg_acc_hist'])
        test_loss_avg = np.array(res['avg_test_loss_hist'])
        test_acc_avg = np.array(res['avg_test_acc_hist'])

        train_loss_folds = np.array(res['all_loss_hist']) if 'all_loss_hist' in res else None
        train_acc_folds = np.array(res['all_acc_hist']) if 'all_acc_hist' in res else None
        test_loss_folds = np.array(res['all_test_loss_hist']) if 'all_test_loss_hist' in res else None
        test_acc_folds = np.array(res['all_test_acc_hist']) if 'all_test_acc_hist' in res else None

        loss_results = {
            "train_loss": train_loss_avg,
            "train_acc": train_acc_avg,
            "test_loss": test_loss_avg,
            "test_acc": test_acc_avg,
            "train_loss_folds": train_loss_folds,
            "train_acc_folds": train_acc_folds,
            "test_loss_folds": test_loss_folds,
            "test_acc_folds": test_acc_folds,
        }

        all_results[loss_name] = loss_results

    # Create experiments folder if it doesn't exist
    os.makedirs("experiments", exist_ok=True)
    save_path = os.path.join("experiments", output_name)
    # Save the nested dictionary as a single object array
    
    np.savez(save_path + ".npz", data=all_results, allow_pickle=True)
    

    return all_results

def run_model_experiment(device, num_folds, num_epochs, batch_size, verbose, input_file, output_name):
    """Run convergence experiments comparing single- and two-layer models using train().

    This function delegates training to the train(...) function which handles fold
    creation and per-fold training. Returns and saves a single results dict including
    all models' per-epoch averages and per-fold arrays in the experiments folder.

    Args:
        device: The device to run the experiment on (e.g., torch.device("cuda" or "cpu")).
        num_folds (int): Number of cross-validation folds.
        num_epochs (int): Number of training epochs per fold.
        batch_size (int): Batch size for training.
        verbose (bool): If True, print progress messages.

    Returns:
        dict: Dictionary containing results for all models with nested structure.
    """

    output_dir = "preprocessed_data"
    filename = os.path.join(output_dir, input_file)
    spike_tensors, y_tensors, _, _ = load_spike_data(filename)

    num_inputs = spike_tensors[0].shape[1]
    num_outputs = len(np.unique(y_tensors))

    (cv_train_data_folds, cv_train_labels_folds, cv_test_data_folds, cv_test_labels_folds, val_data, val_labels, _, _) = create_cv_folds(
             spike_tensors, y_tensors, num_folds=num_folds, num_sessions=2, num_classes=num_outputs,
             random_state=42, retrain=False
         )

    model_map = {
        "Synaptic": FC_SNN_Syn,
        "Synaptic_3": FC_SNN_Syn_33,
        "Synaptic_3_reduced": FC_SNN_Syn_32
    }

    all_results = {}

    for model_name, model_class in model_map.items():
        if verbose:
            print(f"\n{'#'*70}\nRunning model: {model_name}\n{'#'*70}")

        res = cv_train(
            cv_train_data_folds=cv_train_data_folds,
            cv_train_labels_folds=cv_train_labels_folds,
            cv_test_data_folds=cv_test_data_folds,
            cv_test_labels_folds=cv_test_labels_folds,
            model_class=model_class,
            num_folds=num_folds,
            num_epochs=num_epochs,
            batch_size=batch_size,
            random_state=42,
            device=device,
            verbose=verbose,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            loss_fn='ce_rate_mse_membrane'
        )

        train_acc_avg = np.array(res['avg_acc_hist'])
        train_loss_avg = np.array(res['avg_loss_hist'])
        test_acc_avg = np.array(res['avg_test_acc_hist'])
        test_loss_avg = np.array(res['avg_test_loss_hist'])

        train_acc_folds = np.array(res['all_acc_hist']) if 'all_acc_hist' in res else None
        train_loss_folds = np.array(res['all_loss_hist']) if 'all_loss_hist' in res else None
        test_acc_folds = np.array(res['all_test_acc_hist']) if 'all_test_acc_hist' in res else None
        test_loss_folds = np.array(res['all_test_loss_hist']) if 'all_test_loss_hist' in res else None

        model_results = {
            "train_loss": train_loss_avg,
            "train_acc": train_acc_avg,
            "test_loss": test_loss_avg,
            "test_acc": test_acc_avg,
            "train_loss_folds": train_loss_folds,
            "train_acc_folds": train_acc_folds,
            "test_loss_folds": test_loss_folds,
            "test_acc_folds": test_acc_folds,
        }

        all_results[model_name] = model_results

    # Create experiments folder if it doesn't exist
    os.makedirs("experiments", exist_ok=True)
    save_path = os.path.join("experiments", output_name)
    # Save the nested dictionary as a single object array
    np.savez(save_path + ".npz", data=all_results, allow_pickle=True)

    return all_results

def run_param_experiment(varying_param, varying_values, fixed_params, num_folds=5, num_epochs=20, 
                               batch_size=16, random_state=42, device=device, verbose=True, include_classes=None, 
                               alpha=0.7, beta=0.7, sparsity=None, return_membrane=False, return_spikes=False,
                               topk_accuracy=None, test="all-data"):
    """Run experiments on preprocessed data with varying and fixed parameters.

    Args:
        varying_param (str): Key of the parameter to vary (e.g., "num_frames", "threshold", "subtract_baseline").
        varying_values (list): List of values to test for the varying parameter.
        fixed_params (dict): Dictionary of fixed parameter values (e.g., {"topn": None, "session_id": [0, 1]}).
        num_folds (int): Number of cross-validation folds.
        num_epochs (int): Number of training epochs per fold.
        batch_size (int): Batch size for training.
        alpha (float): Alpha parameter for training.
        beta (float): Beta parameter for SNN.
        random_state (int): Random seed for reproducibility.
        device: The device to run the experiment on (e.g., torch.device("cuda" or "cpu")).
        verbose (bool): If True, print progress messages.
        num_inputs (int): Number of input features.
        num_outputs (int): Number of output classes.
        loss_fn (str): Loss function to use ('ce', 'mse', 'ce_rate_mse_membrane', 'balanced', 'sum').
        include_classes (list): Optional list of class names to include (e.g., ['ball', 'pen']).
        test (str): Test data to use ('all-data', 'inter-sess').
    Returns:
        dict: Dictionary containing results for all parameter variations.
    """

    # Define the output directory and base path
    output_dir = "preprocessed_data"
    results = {}

    class_names = [
        'ball', 'battery', 'bracket', 'coin', 'empty_can', 'empty_hand', 'full_can',
        'gel', 'lotion', 'mug', 'pen', 'safety_glasses', 'scissors', 'screw_driver',
        'spray_can', 'stapler', 'tape'
    ]

    # Iterate over varying parameter values
    for value in varying_values:
        # Create a new parameter dictionary with the varying value
        params_dict = fixed_params.copy()
        params_dict[varying_param] = value

        # Get the filename based on the parameter dictionary
        filename = get_filename_from_params(params_dict, output_dir)
        if not os.path.exists(filename):
            print(f"Skipping missing file: {filename}")
            return None
        
        if verbose:
            print(f"\n{'#'*70}\nRunning experiment for {varying_param} = {value} from {filename}\n{'#'*70}")

        # Load data and parameters
        spike_tensors, y_tensors, data_params, _ = load_spike_data(filename)
        num_outputs = len(class_names)
        
        # Infer number of sessions from dataset metadata when available.
        if isinstance(data_params, dict) and isinstance(data_params.get("session_id", None), (list, tuple)):
            num_sessions_in_data = len(data_params["session_id"])
        else:
            num_sessions_in_data = 2

        # === FILTER BY include_classes ===
        classes_filtered = False
        if include_classes is not None:
            include_indices = [i for i, name in enumerate(class_names) if name in include_classes]
            if not include_indices:
                raise ValueError(f"None of the requested classes {include_classes} were found!")
            
            classes_filtered = True
            mask = np.isin(y_tensors, include_indices)
            spike_tensors = spike_tensors[mask]
            y_tensors = y_tensors[mask]

            # Remap labels to 0..N-1
            unique_old = np.unique(y_tensors)
            class_mapping = {old: new for new, old in enumerate(sorted(unique_old))}
            y_tensors = np.array([class_mapping[y] for y in y_tensors], dtype=np.int64)

            num_outputs = len(include_indices)
            kept_names = [class_names[i] for i in include_indices]

            if verbose:
                print(f"Keeping only classes: {kept_names}")
                print(f"→ New number of classes: {num_outputs}")
        else:
            num_outputs = len(class_names)
            kept_names = class_names
            if verbose:
                print("Using all 17 classes")

        num_inputs = spike_tensors[0].shape[1]
        num_sessions = len(fixed_params.get("session_id", []))

        # Create folds
        if test == "all-data":
            (cv_train_data_folds, cv_train_labels_folds,
            cv_test_data_folds, cv_test_labels_folds, _, _, _, _) = create_cv_folds(
                spike_tensors, y_tensors, num_folds=num_folds, num_sessions=num_sessions_in_data,
                num_classes=num_outputs, random_state=42)
        elif test == "inter-session":
            # create_session_cv_folds returns train and validation folds only.
            # We map validation folds to the "test" slots expected by cv_train/cv_train_topk.
            (cv_train_data_folds, cv_train_labels_folds,
            cv_test_data_folds, cv_test_labels_folds, _, _) = create_session_cv_folds(
                spike_tensors, y_tensors, num_folds=num_folds, num_sessions=5,
                num_classes=num_outputs, random_state=42)
        else:
            raise ValueError("test must be either 'all-data' or 'inter-session'")
        # Train the model for this parameter value using the train function
        if topk_accuracy is not None:
            # Fast path: only accuracy
            res = cv_train_topk(
                cv_train_data_folds=cv_train_data_folds,
                cv_train_labels_folds=cv_train_labels_folds,
                cv_test_data_folds=cv_test_data_folds,
                cv_test_labels_folds=cv_test_labels_folds,
                model_class=FC_SNN_Syn,
                num_folds=num_folds,
                num_epochs=num_epochs,
                batch_size=batch_size,
                alpha=alpha,
                beta=beta,
                topk_accuracy=topk_accuracy,
                device=device,
                verbose=verbose
            )
            model_results = {
            "avg_train_top1": res['avg_train_top1'],
            "avg_test_top1": res['avg_test_top1'],
            "avg_train_topk": res['avg_train_topk'],
            "avg_test_topk": res['avg_test_topk'],
            "train_top1_hist": res['train_top1_hist'],
            "test_top1_hist": res['test_top1_hist'],
            "train_topk_hist": res['train_topk_hist'],
            "test_topk_hist": res['test_topk_hist']
        }
        else:
            res = cv_train(
                cv_train_data_folds=cv_train_data_folds,
                cv_train_labels_folds=cv_train_labels_folds,
                cv_test_data_folds=cv_test_data_folds,
                cv_test_labels_folds=cv_test_labels_folds,
                model_class=FC_SNN_Syn,
                num_folds=num_folds,
                num_epochs=num_epochs,
                batch_size=batch_size,
                random_state=random_state,
                device=device,
                verbose=verbose,
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                loss_fn='ce_rate_mse_membrane',
                alpha=alpha,
                beta=beta,
                return_membrane=return_membrane,
                return_spikes=return_spikes,
            )

            # Extract results from the train function output
            model_results = {
                "train_loss": res['avg_loss_hist'],
                "train_acc": res['avg_acc_hist'],
                "test_loss": res['avg_test_loss_hist'],
                "test_acc": res['avg_test_acc_hist'],
                "train_loss_folds": res['all_loss_hist'],
                "train_acc_folds": res['all_acc_hist'],
                "test_loss_folds": res['all_test_loss_hist'],
                "test_acc_folds": res['all_test_acc_hist']
            }

            if return_spikes:
                model_results['total_hidden_spikes']           = res.get('total_hidden_spikes')
                model_results['total_hidden_spikes_per_fold']  = res.get('total_hidden_spikes_per_fold')
                model_results['hidden_spikes_shape']           = res.get('hidden_spikes_shape')
                model_results['avg_spikes_per_test_sample']    = res.get('avg_spikes_per_test_sample')
                model_results['avg_firing_rate_per_neuron']    = res.get('avg_firing_rate_per_neuron')
                model_results['all_spk1_traces']               = res.get('all_spk1_traces', [])
                model_results['all_spk1_labels']               = res.get('all_spk1_labels', [])

            if return_membrane:
                # ---- forward every key that cv_train creates ----
                model_results['hidden_membrane']                = res.get('hidden_membrane')
                model_results['hidden_membrane_labels']         = res.get('hidden_membrane_labels')

                model_results['hidden_membrane_shape']          = res.get('hidden_membrane_shape')
                model_results['max_membrane_per_fold']          = res.get('max_membrane_per_fold')
                model_results['mean_membrane_per_fold']         = res.get('mean_membrane_per_fold')
                model_results['membrane_fraction_above_0.5']   = res.get('membrane_fraction_above_0.5')
                model_results['avg_max_membrane_per_test_sample'] = res.get('avg_max_membrane_per_test_sample')

        # Store results under the parameter value as key
        results[str(value)] = model_results


    # Create experiments folder if it doesn't exist
    os.makedirs("experiments", exist_ok=True)
    # Save results to a single NPZ file with a timestamp
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    output_path = f"experiments/param_{varying_param}_{timestamp}.npz"
    np.savez(output_path, data=results, allow_pickle=True)
    if verbose:
        print(f"Results saved to: {output_path}")

    return results

def run_confusion_matrix_experiment(device, num_folds, num_epochs, batch_size, verbose, input_file, alpha=None, beta=None):
    """Run experiment to compute and save confusion matrices for a given model and dataset.

    Args:
        device: The device to run the experiment on (e.g., torch.device("cuda" or "cpu")).
        num_folds (int): Number of cross-validation folds.
        num_epochs (int): Number of training epochs per fold.
        batch_size (int): Batch size for training.
        verbose (bool): If True, print progress messages.
        model_class: Model class to train (e.g., FC_SNN_Syn, FC_SNN_Syn_Leaky).
        input_file (str): Path to the preprocessed data file.

    Returns:
        dict: Dictionary containing confusion matrix results and per-class metrics.
    """
    # Define object names for the 17 classes
    class_names = [
        'ball', 'battery', 'bracket', 'coin', 'empty_can', 'empty_hand', 'full_can',
        'gel', 'lotion', 'mug', 'pen', 'safety_glasses', 'scissors', 'screw_driver',
        'spray_can', 'stapler', 'tape'
    ]

    output_dir = "preprocessed_data"
    filename = os.path.join(output_dir, input_file)
    spike_tensors, y_tensors, _, _ = load_spike_data(filename)
    num_inputs = spike_tensors[0].shape[1]
    num_outputs = len(class_names)

    # Create cross-validation folds
    (cv_train_data_folds, cv_train_labels_folds, cv_test_data_folds, cv_test_labels_folds, val_data, val_labels, _, _) = create_cv_folds(
        spike_tensors, y_tensors, num_folds=num_folds, num_sessions=2, num_classes=num_outputs, random_state=42)

    # Run training and collect confusion matrix results
    res = cv_train(
        cv_train_data_folds=cv_train_data_folds,
        cv_train_labels_folds=cv_train_labels_folds,
        cv_test_data_folds=cv_test_data_folds,
        cv_test_labels_folds=cv_test_labels_folds,
        model_class=FC_SNN_Syn,
        num_folds=num_folds,
        num_epochs=num_epochs,
        batch_size=batch_size,
        random_state=42,
        device=device,
        verbose=verbose,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        loss_fn='ce_rate_mse_membrane',
        alpha=alpha,
        beta=beta
    )

    # Extract confusion matrix results
    all_conf_matrices = res['all_conf_matrices']
    aggregated_cm = res['aggregated_cm']
    normalized_cm = res['normalized_cm']
    class_performance = res['class_performance']
    sorted_classes = res['sorted_classes']

    # Create experiments folder if it doesn't exist
    os.makedirs("experiments", exist_ok=True)

    plt.figure(figsize=(13, 11))   # slightly larger figure

    # Main heatmap
    ax = sns.heatmap(
        normalized_cm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0.0,
        vmax=1.0,
        cbar_kws={'label': 'Normalized value', 'ticks': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}
    )

    # ── Font size adjustments ───────────────────────────────────────
    # Tick labels (the class names)
    ax.tick_params(axis='both', which='major', labelsize=12)     # was ~10 by default

    # Axis labels
    ax.set_xlabel('Predicted Label', fontsize=16, labelpad=12)
    ax.set_ylabel('True Label',    fontsize=16, labelpad=12)

    # Title
    plt.title(
        f'Aggregated Normalized Confusion Matrix',
        fontsize=18,               # ← larger title
        pad=18
    )

    # Colorbar adjustments
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)          # larger colorbar tick labels
    cbar.set_label('Normalized value', fontsize=14, labelpad=10)

    # Rotate x-ticks and make sure everything fits nicely
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    # Save with good quality
    cm_plot_path = os.path.join("figures", f"top_{num_inputs}_confusion_matrix_plot.png")
    plt.savefig(cm_plot_path, dpi=400, bbox_inches='tight')   # ↑ higher dpi
    plt.close()

    # Save per-class metrics to CSV
    metrics_df = pd.DataFrame([
        {'Class': name, **metrics} for name, metrics in class_performance.items()
    ])
    metrics_csv_path = os.path.join("experiments", f"top_{num_inputs}_confusion_matrix_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False, float_format='%.4f')

    # Save actual confusion matrix to CSV
    cm_df = pd.DataFrame(aggregated_cm, index=class_names, columns=class_names)
    cm_csv_path = os.path.join("experiments", f"{num_inputs}_confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path, float_format='%.0f')

    if verbose:
        print(f"\nSaved confusion matrix plot to {cm_plot_path}")
        print(f"Saved class metrics to {metrics_csv_path}")
        print(f"Saved confusion matrix to {cm_csv_path}")
        print("\nAggregated Per-Class Performance:")
        print("Class Name".ljust(20) + "Accuracy".ljust(12) + "Precision".ljust(12) + "Recall".ljust(12) + "F1-Score")
        print("-" * 60)
        for name, metrics in sorted_classes:
            print(f"{name.ljust(20)}{metrics['accuracy']:.4f}{metrics['precision']:.4f}{metrics['recall']:.4f}{metrics['f1']:.4f}")

        # ←←← ONLY THESE 3 LINES ADDED ←←←
        print(f"\nClasses ordered by F1-score (best to worst):")
        ordered_classes = [name for name, _ in sorted_classes]
        print(ordered_classes)

    return {
        'all_conf_matrices': all_conf_matrices,
        'aggregated_cm': aggregated_cm,
        'normalized_cm': normalized_cm,
        'class_performance': class_performance,
        'sorted_classes': sorted_classes,
        'class_names': class_names
    }



def run_validation(fixed_params, model=FC_SNN_Syn, alpha=None, beta=None, 
                                 num_epochs=20, batch_size=16, random_state=42, device=device,
                                 verbose=True, loss_fn='ce_rate_mse_membrane', lr=0.001, 
                                 patience=5, min_delta=0.01, include_classes=None, 
                                 topk_accuracy=None, fixed_fan_in=None):
    """
    Perform final training and validation — now with include_classes instead of exclude.

    Args:
        include_classes (list or None): List of class names to KEEP (e.g. ['ball', 'battery', 'coin']).
                                        If None → use all 17 classes.
    """
    output_dir = "preprocessed_data"
    filename = get_filename_from_params(fixed_params, output_dir)
    if not os.path.exists(filename):
        print(f"Skipping missing file: {filename}")
        return None

    if verbose:
        print(f"\n{'#'*70}\nStarting final training from {filename}\n{'#'*70}")

    spike_tensors, y_tensors, _, _ = load_spike_data(filename)
    y_tensors = np.array(y_tensors, dtype=np.int64)

    # === CLASS NAMES ===
    class_names = [
        'ball', 'battery', 'bracket', 'coin', 'empty_can', 'empty_hand', 'full_can',
        'gel', 'lotion', 'mug', 'pen', 'safety_glasses', 'scissors', 'screw_driver',
        'spray_can', 'stapler', 'tape'
    ]

    # === FILTER BY include_classes ===
    classes_filtered = False
    if include_classes is not None:
        include_indices = [i for i, name in enumerate(class_names) if name in include_classes]
        if not include_indices:
            raise ValueError(f"None of the requested classes {include_classes} were found!")
        
        classes_filtered = True
        mask = np.isin(y_tensors, include_indices)
        spike_tensors = spike_tensors[mask]
        y_tensors = y_tensors[mask]

        # Remap labels to 0..N-1
        unique_old = np.unique(y_tensors)
        class_mapping = {old: new for new, old in enumerate(sorted(unique_old))}
        y_tensors = np.array([class_mapping[y] for y in y_tensors], dtype=np.int64)

        num_outputs = len(include_indices)
        kept_names = [class_names[i] for i in include_indices]

        if verbose:
            print(f"Keeping only classes: {kept_names}")
            print(f"→ New number of classes: {num_outputs}")
    else:
        num_outputs = len(class_names)
        kept_names = class_names
        if verbose:
            print("Using all 17 classes")

    num_inputs = spike_tensors[0].shape[1]
    num_sessions = len(fixed_params.get("session_id", []))

    # Train/val split
    train_data, train_labels, val_data, val_labels = create_cv_folds(
        spike_tensors, y_tensors, num_folds=None, num_sessions=num_sessions,
        num_classes=num_outputs, random_state=random_state, retrain=True
    )
    
    if topk_accuracy is not None:
        # Lightweight Top-k version
        results = train_topk(
            train_data=train_data,
            train_labels=train_labels,
            val_data=val_data,
            val_labels=val_labels,
            model_class=model,
            alpha=alpha,
            beta=beta,
            num_epochs=num_epochs,
            batch_size=batch_size,
            random_state=random_state,
            device=device,
            verbose=verbose,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            loss_fn=loss_fn,
            patience=patience,
            min_delta=min_delta,
            lr=lr,
            topk_accuracy=topk_accuracy,
            fixed_fan_in=fixed_fan_in
        )
    else:

        results = train(
            train_data=train_data,
            train_labels=train_labels,
            val_data=val_data,
            val_labels=val_labels,
            model_class=model,
            alpha=alpha,
            beta=beta,
            num_epochs=num_epochs,
            batch_size=batch_size,
            random_state=random_state,
            device=device,
            verbose=verbose,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            loss_fn=loss_fn,
            patience=patience,
            min_delta=min_delta,
            lr=lr,
            fixed_fan_in=fixed_fan_in
        )

    os.makedirs("experiments", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    
    if classes_filtered:
        save_filename = f"experiments/final_training_{num_outputs}_classes_{timestamp}.npz"
    else:
        save_filename = f"experiments/final_training_{timestamp}.npz"
    
    np.savez(save_filename, data=results, allow_pickle=True)
    
    if 'model_state_dict' in results:
        if classes_filtered:
            model_filename = f"models/final_model_{num_outputs}_classes_{timestamp}.pt"
        else:
            model_filename = f"models/final_model_{timestamp}.pt"
        torch.save(results['model_state_dict'], model_filename)
        if verbose:
            print(f"Model state dict saved to: {model_filename}")
    
    if verbose:
        print(f"Results saved to: {save_filename}")

    return results

def run_alpha_experiment(device, num_folds, num_epochs, batch_size, verbose, num_outputs, num_inputs, 
                        input_file, output_name, alpha_values=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """Run alpha parameter comparison experiments using cv_train().

    This function delegates training to cv_train() which handles fold creation and per-fold training.
    Returns and saves a single results dict including all alpha values' per-epoch averages and 
    per-fold arrays in the experiments folder.

    Args:
        device: The device to run the experiment on (e.g., torch.device("cuda" or "cpu")).
        num_folds (int): Number of cross-validation folds.
        num_epochs (int): Number of training epochs per fold.
        batch_size (int): Batch size for training.
        verbose (bool): If True, print progress messages.
        num_outputs (int): Number of output classes.
        num_inputs (int): Number of input features.
        input_file (str): Path to preprocessed spike data file.
        output_name (str): Base name for output NPZ file.
        alpha_values (list): List of alpha values to test [0.1, 0.3, 0.5, 0.7, 0.9].

    Returns:
        dict: Dictionary containing results for all alpha values with nested structure.
    """

    output_dir = "preprocessed_data"
    filename = os.path.join(output_dir, input_file)
    spike_tensors, y_tensors, _, _ = load_spike_data(filename)

    # Create CV folds (fixed across all alpha values)
    (cv_train_data_folds, cv_train_labels_folds, cv_test_data_folds, cv_test_labels_folds, 
     val_data, val_labels, _, _) = create_cv_folds(
         spike_tensors, y_tensors, num_folds=num_folds, num_sessions=2, num_classes=num_outputs,
         random_state=42
     )

    all_results = {}

    for alpha in alpha_values:
        if verbose:
            print(f"\n{'#'*70}")
            print(f"Running alpha experiment: alpha = {alpha}")
            print(f"{'#'*70}")

        # Train with fixed alpha
        res = cv_train(
            cv_train_data_folds=cv_train_data_folds,
            cv_train_labels_folds=cv_train_labels_folds,
            cv_test_data_folds=cv_test_data_folds,
            cv_test_labels_folds=cv_test_labels_folds,
            model_class=FC_SNN_Syn,
            num_folds=num_folds,
            num_epochs=num_epochs,
            batch_size=batch_size,
            random_state=42,
            device=device,
            verbose=verbose,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            loss_fn='ce_rate_mse_membrane',
            alpha=alpha,
            beta=0.3
        )

        # Extract results (EXACT SAME FORMAT as other experiments)
        train_loss_avg = np.array(res['avg_loss_hist'])
        train_acc_avg = np.array(res['avg_acc_hist'])
        test_loss_avg = np.array(res['avg_test_loss_hist'])
        test_acc_avg = np.array(res['avg_test_acc_hist'])

        train_loss_folds = np.array(res['all_loss_hist']) if 'all_loss_hist' in res else None
        train_acc_folds = np.array(res['all_acc_hist']) if 'all_acc_hist' in res else None
        test_loss_folds = np.array(res['all_test_loss_hist']) if 'all_test_loss_hist' in res else None
        test_acc_folds = np.array(res['all_test_acc_hist']) if 'all_test_acc_hist' in res else None

        alpha_results = {
            "train_loss": train_loss_avg,
            "train_acc": train_acc_avg,
            "test_loss": test_loss_avg,
            "test_acc": test_acc_avg,
            "train_loss_folds": train_loss_folds,
            "train_acc_folds": train_acc_folds,
            "test_loss_folds": test_loss_folds,
            "test_acc_folds": test_acc_folds,
        }

        all_results[str(alpha)] = alpha_results

    # Create experiments folder if it doesn't exist
    os.makedirs("experiments", exist_ok=True)
    save_path = os.path.join("experiments", output_name)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    # Save the nested dictionary as a single object array (SAME FORMAT)
    np.savez(save_path + f"_{timestamp}.npz", data=all_results, allow_pickle=True)

    if verbose:
        print(f"\n{'#'*70}")
        print(f"Alpha experiment completed! Results saved to: {save_path}_{timestamp}.npz")
        print(f"{'#'*70}")

    return all_results


def run_beta_experiment(device, num_folds, num_epochs, batch_size, verbose, num_outputs, num_inputs, 
                       input_file, output_name, beta_values=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """Run beta parameter comparison experiments using cv_train().

    This function delegates training to cv_train() which handles fold creation and per-fold training.
    Returns and saves a single results dict including all beta values' per-epoch averages and 
    per-fold arrays in the experiments folder.

    Args:
        device: The device to run the experiment on (e.g., torch.device("cuda" or "cpu")).
        num_folds (int): Number of cross-validation folds.
        num_epochs (int): Number of training epochs per fold.
        batch_size (int): Batch size for training.
        verbose (bool): If True, print progress messages.
        num_outputs (int): Number of output classes.
        num_inputs (int): Number of input features.
        input_file (str): Path to preprocessed spike data file.
        output_name (str): Base name for output NPZ file.
        beta_values (list): List of beta values to test [0.1, 0.3, 0.5, 0.7, 0.9].

    Returns:
        dict: Dictionary containing results for all beta values with nested structure.
    """

    output_dir = "preprocessed_data"
    filename = os.path.join(output_dir, input_file)
    spike_tensors, y_tensors, _, _ = load_spike_data(filename)

    # Create CV folds (fixed across all beta values)
    (cv_train_data_folds, cv_train_labels_folds, cv_test_data_folds, cv_test_labels_folds, 
     val_data, val_labels, _, _) = create_cv_folds(
         spike_tensors, y_tensors, num_folds=num_folds, num_sessions=2, num_classes=num_outputs,
         random_state=42
     )

    all_results = {}

    for beta in beta_values:
        if verbose:
            print(f"\n{'#'*70}")
            print(f"Running beta experiment: beta = {beta}")
            print(f"{'#'*70}")

        # Train with fixed beta
        res = cv_train(
            cv_train_data_folds=cv_train_data_folds,
            cv_train_labels_folds=cv_train_labels_folds,
            cv_test_data_folds=cv_test_data_folds,
            cv_test_labels_folds=cv_test_labels_folds,
            model_class=FC_SNN_Syn,
            num_folds=num_folds,
            num_epochs=num_epochs,
            batch_size=batch_size,
            random_state=42,
            device=device,
            verbose=verbose,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            loss_fn='ce_rate_mse_membrane',
            alpha=0.5,
            beta=beta
        )

        # Extract results (EXACT SAME FORMAT as other experiments)
        train_loss_avg = np.array(res['avg_loss_hist'])
        train_acc_avg = np.array(res['avg_acc_hist'])
        test_loss_avg = np.array(res['avg_test_loss_hist'])
        test_acc_avg = np.array(res['avg_test_acc_hist'])

        train_loss_folds = np.array(res['all_loss_hist']) if 'all_loss_hist' in res else None
        train_acc_folds = np.array(res['all_acc_hist']) if 'all_acc_hist' in res else None
        test_loss_folds = np.array(res['all_test_loss_hist']) if 'all_test_loss_hist' in res else None
        test_acc_folds = np.array(res['all_test_acc_hist']) if 'all_test_acc_hist' in res else None

        beta_results = {
            "train_loss": train_loss_avg,
            "train_acc": train_acc_avg,
            "test_loss": test_loss_avg,
            "test_acc": test_acc_avg,
            "train_loss_folds": train_loss_folds,
            "train_acc_folds": train_acc_folds,
            "test_loss_folds": test_loss_folds,
            "test_acc_folds": test_acc_folds,
        }

        all_results[str(beta)] = beta_results

    # Create experiments folder if it doesn't exist
    os.makedirs("experiments", exist_ok=True)
    save_path = os.path.join("experiments", output_name)
    # Save the nested dictionary as a single object array (SAME FORMAT)
    np.savez(save_path + ".npz", data=all_results, allow_pickle=True)

    if verbose:
        print(f"\n{'#'*70}")
        print(f"Beta experiment completed! Results saved to: {save_path}.npz")
        print(f"{'#'*70}")

    return all_results
def run_alpha_beta_grid_experiment(device, num_folds, num_epochs, batch_size, verbose, input_file,
                                 output_name, alpha_values=[0.1, 0.3, 0.5, 0.7, 0.9], 
                                 beta_values=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """
    Run alpha-beta GRID PARAMETER SWEEP experiment using cv_train().

    Now reports accuracy using **only the last epoch**, averaged across folds.
    Still saves full per-epoch history for plotting.

    Outputs:
    1. NPZ file with full per-epoch results (unchanged)
    2. SUMMARY TABLE with last-epoch avg/max train/test accuracy
    """

    output_dir = "preprocessed_data"
    filename = os.path.join(output_dir, input_file)
    spike_tensors, y_tensors, _, _ = load_spike_data(filename)
    num_inputs = spike_tensors[0].shape[1]
    num_outputs = len(np.unique(y_tensors))

    # Create CV folds (fixed across ALL combinations)
    (cv_train_data_folds, cv_train_labels_folds, cv_test_data_folds, cv_test_labels_folds, 
     val_data, val_labels, _, _) = create_cv_folds(
         spike_tensors, y_tensors, num_folds=num_folds, num_sessions=2, num_classes=num_outputs,
         random_state=42
     )

    all_results = {}
    summary_data = []

    total_combinations = len(alpha_values) * len(beta_values)
    combination_idx = 0

    print(f"\n{'#'*80}")
    print(f"ALPHA-BETA GRID SWEEP: {len(alpha_values)} × {len(beta_values)} = {total_combinations} combinations")
    print(f"{'#'*80}")

    for alpha in alpha_values:
        for beta in beta_values:
            combination_idx += 1
            key = f"alpha_{alpha}_beta_{beta}"
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"[{combination_idx:2d}/{total_combinations}] Running: alpha={alpha}, beta={beta}")
                print(f"{'='*80}")

            res = cv_train(
                cv_train_data_folds=cv_train_data_folds,
                cv_train_labels_folds=cv_train_labels_folds,
                cv_test_data_folds=cv_test_data_folds,
                cv_test_labels_folds=cv_test_labels_folds,
                model_class=FC_SNN_Syn,
                num_folds=num_folds,
                num_epochs=num_epochs,
                batch_size=batch_size,
                random_state=42,
                device=device,
                verbose=verbose,
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                loss_fn='ce_rate_mse_membrane',
                alpha=alpha,
                beta=beta
            )

            # ────────────────────────────────────────────────
            # Extract results
            # ────────────────────────────────────────────────
            train_loss_avg = np.array(res['avg_loss_hist'])
            train_acc_avg  = np.array(res['avg_acc_hist'])
            test_loss_avg  = np.array(res['avg_test_loss_hist'])
            test_acc_avg   = np.array(res['avg_test_acc_hist'])

            train_loss_folds = np.array(res['all_loss_hist']) if 'all_loss_hist' in res else None
            train_acc_folds  = np.array(res['all_acc_hist'])  if 'all_acc_hist' in res else None
            test_loss_folds  = np.array(res['all_test_loss_hist']) if 'all_test_loss_hist' in res else None
            test_acc_folds   = np.array(res['all_test_acc_hist'])  if 'all_test_acc_hist' in res else None

            # Store FULL per-epoch results for plotting (unchanged)
            grid_results = {
                "train_loss": train_loss_avg,
                "train_acc": train_acc_avg,
                "test_loss": test_loss_avg,
                "test_acc": test_acc_avg,
                "train_loss_folds": train_loss_folds,
                "train_acc_folds": train_acc_folds,
                "test_loss_folds": test_loss_folds,
                "test_acc_folds": test_acc_folds,
            }
            all_results[key] = grid_results

            # ────────────────────────────────────────────────
            # SUMMARY METRICS — now using **LAST EPOCH ONLY**
            # ────────────────────────────────────────────────
            last_epoch_idx = -1  # last position = last epoch

            # Average across folds at the last epoch
            last_train_acc = train_acc_avg[last_epoch_idx] if len(train_acc_avg) > 0 else np.nan
            last_test_acc  = test_acc_avg[last_epoch_idx]  if len(test_acc_avg)  > 0 else np.nan

            # If you have per-fold data and want max across folds at last epoch:
            if test_acc_folds is not None and test_acc_folds.ndim >= 2:
                last_test_acc_per_fold = test_acc_folds[:, last_epoch_idx]
                max_last_test_acc = np.nanmax(last_test_acc_per_fold) * 100
            else:
                max_last_test_acc = last_test_acc * 100  # fallback

            if train_acc_folds is not None and train_acc_folds.ndim >= 2:
                last_train_acc_per_fold = train_acc_folds[:, last_epoch_idx]
                max_last_train_acc = np.nanmax(last_train_acc_per_fold) * 100
            else:
                max_last_train_acc = last_train_acc * 100

            # Convert to percentages
            avg_last_train_acc = last_train_acc * 100
            avg_last_test_acc  = last_test_acc  * 100

            summary_data.append({
                'alpha': alpha,
                'beta': beta,
                'last_epoch_avg_train_acc': round(avg_last_train_acc, 2),
                'last_epoch_max_train_acc': round(max_last_train_acc, 2),
                'last_epoch_avg_test_acc':  round(avg_last_test_acc, 2),
                'last_epoch_max_test_acc':  round(max_last_test_acc, 2)
            })

    # ────────────────────────────────────────────────
    # Create & sort summary table
    # ────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('last_epoch_avg_test_acc', ascending=False).reset_index(drop=True)

    print(f"\n{'#'*80}")
    print(f"ALPHA-BETA GRID SWEEP SUMMARY TABLE  (last epoch only)")
    print(f"{'#'*80}")
    print(summary_df.to_string(index=False))

    # Highlight best combination
    best_idx = summary_df['last_epoch_avg_test_acc'].idxmax()
    best_row = summary_df.iloc[best_idx]
    print(f"\n{'*'*80}")
    print(f"BEST COMBINATION (last epoch): alpha={best_row['alpha']}, beta={best_row['beta']}")
    print(f"Last-epoch Avg Test Acc: {best_row['last_epoch_avg_test_acc']}%")
    print(f"Last-epoch Max Test Acc: {best_row['last_epoch_max_test_acc']}%")
    print(f"{'*'*80}")

    # Save summary
    os.makedirs("experiments", exist_ok=True)
    summary_csv = f"experiments/{output_name}_summary_last_epoch.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary table saved to: {summary_csv}")

    # Save full per-epoch results (unchanged)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    save_path = os.path.join("experiments", output_name)
    np.savez(save_path + f"_grid_{timestamp}.npz", data=all_results, allow_pickle=True)
    
    if verbose:
        print(f"\nFull results (all epochs) saved to: {save_path}_grid_{timestamp}.npz")

    return {
        'full_results': all_results,
        'summary_df': summary_df,
        'best_combination': {
            'alpha': float(best_row['alpha']),
            'beta': float(best_row['beta']),
            'last_epoch_avg_test_acc': float(best_row['last_epoch_avg_test_acc'])
        }
    }
    
    
def run_sparsity_experiment(device, num_folds, num_epochs, batch_size, verbose, num_outputs, num_inputs, 
                        input_file, output_name, alpha, beta, sparsity_values=None,
                        return_spikes=None, return_membrane=None, return_weights=None):
    """Run alpha parameter comparison experiments using cv_train().

    This function delegates training to cv_train() which handles fold creation and per-fold training.
    Returns and saves a single results dict including all alpha values' per-epoch averages and 
    per-fold arrays in the experiments folder.

    Args:
        device: The device to run the experiment on (e.g., torch.device("cuda" or "cpu")).
        num_folds (int): Number of cross-validation folds.
        num_epochs (int): Number of training epochs per fold.
        batch_size (int): Batch size for training.
        verbose (bool): If True, print progress messages.
        num_outputs (int): Number of output classes.
        num_inputs (int): Number of input features.
        input_file (str): Path to preprocessed spike data file.
        output_name (str): Base name for output NPZ file.
        alpha_values (list): List of alpha values to test [0.1, 0.3, 0.5, 0.7, 0.9].

    Returns:
        dict: Dictionary containing results for all alpha values with nested structure.
    """

    output_dir = "preprocessed_data"
    filename = os.path.join(output_dir, input_file)
    spike_tensors, y_tensors, _, _ = load_spike_data(filename)

    # Create CV folds (fixed across all alpha values)
    (cv_train_data_folds, cv_train_labels_folds, cv_test_data_folds, cv_test_labels_folds, 
     val_data, val_labels, _, _) = create_cv_folds(
         spike_tensors, y_tensors, num_folds=num_folds, num_sessions=2, num_classes=num_outputs,
         random_state=42
     )

    all_results = {}

    for sparaity in sparsity_values:
        if verbose:
            print(f"\n{'#'*70}")
            print(f"Running sparsity experiment: sparsity = {sparaity}")
            print(f"{'#'*70}")

        # Train with fixed alpha
        res = cv_train(
            cv_train_data_folds=cv_train_data_folds,
            cv_train_labels_folds=cv_train_labels_folds,
            cv_test_data_folds=cv_test_data_folds,
            cv_test_labels_folds=cv_test_labels_folds,
            model_class=FC_SNN_Syn,
            num_folds=num_folds,
            num_epochs=num_epochs,
            batch_size=batch_size,
            random_state=42,
            device=device,
            verbose=verbose,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            loss_fn='ce_rate_mse_membrane',
            alpha=alpha,
            beta=beta,
            input_sparsity=sparaity,
            return_spikes=return_spikes,
            return_membrane=return_membrane,
            return_weights=return_weights
        )

        # Extract results (EXACT SAME FORMAT as other experiments)
        train_loss_avg = np.array(res['avg_loss_hist'])
        train_acc_avg = np.array(res['avg_acc_hist'])
        test_loss_avg = np.array(res['avg_test_loss_hist'])
        test_acc_avg = np.array(res['avg_test_acc_hist'])

        train_loss_folds = np.array(res['all_loss_hist']) if 'all_loss_hist' in res else None
        train_acc_folds = np.array(res['all_acc_hist']) if 'all_acc_hist' in res else None
        test_loss_folds = np.array(res['all_test_loss_hist']) if 'all_test_loss_hist' in res else None
        test_acc_folds = np.array(res['all_test_acc_hist']) if 'all_test_acc_hist' in res else None

        sparsity_results = {
            "train_loss": train_loss_avg,
            "train_acc": train_acc_avg,
            "test_loss": test_loss_avg,
            "test_acc": test_acc_avg,
            "train_loss_folds": train_loss_folds,
            "train_acc_folds": train_acc_folds,
            "test_loss_folds": test_loss_folds,
            "test_acc_folds": test_acc_folds,
        }
        
        if return_spikes:
            sparsity_results['total_hidden_spikes']           = res.get('total_hidden_spikes')
            sparsity_results['total_hidden_spikes_per_fold']  = res.get('total_hidden_spikes_per_fold')
            sparsity_results['hidden_spikes_shape']           = res.get('hidden_spikes_shape')
            sparsity_results['avg_spikes_per_test_sample']    = res.get('avg_spikes_per_test_sample')
            sparsity_results['avg_firing_rate_per_neuron']    = res.get('avg_firing_rate_per_neuron')
            sparsity_results['all_spk1_traces']               = res.get('all_spk1_traces', [])
            sparsity_results['all_spk1_labels']               = res.get('all_spk1_labels', [])

        if return_membrane:
            sparsity_results['hidden_membrane']                = res.get('hidden_membrane')
            sparsity_results['hidden_membrane_labels']         = res.get('hidden_membrane_labels')

            sparsity_results['hidden_membrane_shape']          = res.get('hidden_membrane_shape')
            sparsity_results['max_membrane_per_fold']          = res.get('max_membrane_per_fold')
            sparsity_results['mean_membrane_per_fold']         = res.get('mean_membrane_per_fold')
            sparsity_results['membrane_fraction_above_0.5']   = res.get('membrane_fraction_above_0.5')
            sparsity_results['avg_max_membrane_per_test_sample'] = res.get('avg_max_membrane_per_test_sample')
            
        if return_weights:
            sparsity_results['first_layer_weights']            = res.get('first_layer_weights')
            sparsity_results['all_lif1_params']                = res.get('all_lif1_params')
        
        all_results[str(sparaity)] = sparsity_results


    # Create experiments folder if it doesn't exist
    os.makedirs("experiments", exist_ok=True)
    save_path = os.path.join("experiments", output_name)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    # Save the nested dictionary as a single object array (SAME FORMAT)
    np.savez(save_path + f"_{timestamp}.npz", data=all_results, allow_pickle=True)

    if verbose:
        print(f"\n{'#'*70}")
        print(f"Sparsity experiment completed! Results saved to: {save_path}_{timestamp}.npz")
        print(f"{'#'*70}")

    return all_results


def run_fanin_experiment(
    device, num_folds=5, num_epochs=15, batch_size=16, verbose=True,
    input_file=None,
    alpha=0.7, beta=0.7,
    fanin_values=None,  # e.g. [32, 64, 96, 128, 192, 484]
    return_spikes=None, return_membrane=None, return_weights=None,
    include_classes=None,  # ← NEW: list of class names to KEEP
    test="all-data",
    topk_accuracy=None
):
    # Load data
    spike_tensors, y_tensors, data_params, _ = load_spike_data(os.path.join("preprocessed_data", input_file))
    y_tensors = np.array(y_tensors, dtype=np.int64)

    # Infer number of sessions from dataset metadata when available.
    if isinstance(data_params, dict) and isinstance(data_params.get("session_id", None), (list, tuple)):
        num_sessions_in_data = len(data_params["session_id"])
    else:
        num_sessions_in_data = 2

    # === CLASS NAMES ===
    class_names = [
        'ball', 'battery', 'bracket', 'coin', 'empty_can', 'empty_hand', 'full_can',
        'gel', 'lotion', 'mug', 'pen', 'safety_glasses', 'scissors', 'screw_driver',
        'spray_can', 'stapler', 'tape'
    ]

    # === FILTER: keep only include_classes ===
    classes_filtered = False
    if include_classes is not None:
        include_indices = [i for i, name in enumerate(class_names) if name in include_classes]
        if not include_indices:
            raise ValueError(f"None of the requested classes {include_classes} found!")
        
        classes_filtered = True
        mask = np.isin(y_tensors, include_indices)
        spike_tensors = spike_tensors[mask]
        y_tensors = y_tensors[mask]

        # Remap labels to 0..N-1
        unique_old = np.unique(y_tensors)
        class_mapping = {old: new for new, old in enumerate(sorted(unique_old))}
        y_tensors = np.array([class_mapping[y] for y in y_tensors], dtype=np.int64)

        num_outputs = len(include_indices)
        kept_names = [class_names[i] for i in include_indices]

        if verbose:
            print(f"Keeping classes: {kept_names}")
            print(f"→ New number of classes: {num_outputs}")
    else:
        num_outputs = len(class_names)
        kept_names = class_names
        if verbose:
            print("Using all 17 classes")

    num_inputs = spike_tensors[0].shape[1]
    
    # Create folds
    if test == "all-data":
        (cv_train_data_folds, cv_train_labels_folds,
        cv_test_data_folds, cv_test_labels_folds, _, _, _, _) = create_cv_folds(
            spike_tensors, y_tensors, num_folds=num_folds, num_sessions=num_sessions_in_data,
            num_classes=num_outputs, random_state=42)
    elif test == "inter-session":
        # create_session_cv_folds returns train and validation folds only.
        # We map validation folds to the "test" slots expected by cv_train/cv_train_topk.
        (cv_train_data_folds, cv_train_labels_folds,
        cv_test_data_folds, cv_test_labels_folds, _, _) = create_session_cv_folds(
            spike_tensors, y_tensors, num_folds=num_folds, num_sessions=5,
            num_classes=num_outputs, random_state=42)
    else:
        raise ValueError("test must be either 'all-data' or 'inter-session'")
     
    exclude_classes = False
    all_results = {}
    if include_classes is not None:
        exclude_classes = True
    for fan_in in fanin_values:
        name = f"fanin_{fan_in}"
        print(f"\n{'='*80}")
        print(f"RUNNING: {name.upper()}")
        density = fan_in / num_inputs
        print(f"→ Each hidden neuron receives exactly {fan_in} random inputs → {density:.2%} density")
        print(f"{'='*80}")

        if topk_accuracy is not None:
            res = cv_train_topk(
                cv_train_data_folds=cv_train_data_folds,
                cv_train_labels_folds=cv_train_labels_folds,
                cv_test_data_folds=cv_test_data_folds,
                cv_test_labels_folds=cv_test_labels_folds,
                model_class=FC_SNN_Syn,
                num_folds=num_folds,
                num_epochs=num_epochs,
                batch_size=batch_size,
                alpha=alpha,
                beta=beta,
                fixed_fan_in=fan_in,
                topk_accuracy=topk_accuracy,
                device=device,
                verbose=verbose
            )

            fanin_results = {
                "avg_train_top1": res['avg_train_top1'],
                "avg_test_top1": res['avg_test_top1'],
                "avg_train_topk": res['avg_train_topk'],
                "avg_test_topk": res['avg_test_topk'],
                "train_top1_hist": res['train_top1_hist'],
                "test_top1_hist": res['test_top1_hist'],
                "train_topk_hist": res['train_topk_hist'],
                "test_topk_hist": res['test_topk_hist'],
                "topk": res.get('topk', topk_accuracy)
            }
        else:
            res = cv_train(
                cv_train_data_folds=cv_train_data_folds,
                cv_train_labels_folds=cv_train_labels_folds,
                cv_test_data_folds=cv_test_data_folds,
                cv_test_labels_folds=cv_test_labels_folds,
                model_class=FC_SNN_Syn,
                num_folds=num_folds,
                num_epochs=num_epochs,
                batch_size=batch_size,
                random_state=42,
                device=device,
                verbose=verbose,
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                loss_fn='ce_rate_mse_membrane',
                alpha=alpha,
                beta=beta,
                fixed_fan_in=fan_in,
                return_spikes=return_spikes,
                return_membrane=return_membrane,
                return_weights=return_weights,
                exclude_classes=exclude_classes
            )
            
            # (rest of result extraction unchanged)
            train_loss_avg = np.array(res['avg_loss_hist'])
            train_acc_avg = np.array(res['avg_acc_hist'])
            test_loss_avg = np.array(res['avg_test_loss_hist'])
            test_acc_avg = np.array(res['avg_test_acc_hist'])

            train_loss_folds = np.array(res['all_loss_hist']) if 'all_loss_hist' in res else None
            train_acc_folds = np.array(res['all_acc_hist']) if 'all_acc_hist' in res else None
            test_loss_folds = np.array(res['all_test_loss_hist']) if 'all_test_loss_hist' in res else None
            test_acc_folds = np.array(res['all_test_acc_hist']) if 'all_test_acc_hist' in res else None

            fanin_results = {
                "train_loss": train_loss_avg,
                "train_acc": train_acc_avg,
                "test_loss": test_loss_avg,
                "test_acc": test_acc_avg,
                "train_loss_folds": train_loss_folds,
                "train_acc_folds": train_acc_folds,
                "test_loss_folds": test_loss_folds,
                "test_acc_folds": test_acc_folds,
            }
            
            if return_spikes:
                fanin_results['total_hidden_spikes']           = res.get('total_hidden_spikes')
                fanin_results['total_hidden_spikes_per_fold']  = res.get('total_hidden_spikes_per_fold')
                fanin_results['hidden_spikes_shape']           = res.get('hidden_spikes_shape')
                fanin_results['avg_spikes_per_test_sample']    = res.get('avg_spikes_per_test_sample')
                fanin_results['avg_firing_rate_per_neuron']    = res.get('avg_firing_rate_per_neuron')
                fanin_results['all_spk1_traces']               = res.get('all_spk1_traces', [])
                fanin_results['all_spk1_labels']               = res.get('all_spk1_labels', [])
                
                # Full hidden array
                fanin_results['hidden_spikes']                 = res.get('hidden_spikes')
                fanin_results['hidden_spike_labels']           = res.get('hidden_spike_labels')
                
                # OUTPUT LAYER — missing before!
                fanin_results['all_spk2_traces']               = res.get('all_spk2_traces', [])
                fanin_results['output_spikes']                 = res.get('output_spikes')
                fanin_results['output_spikes_shape']           = res.get('output_spikes_shape')
                
                # Input (already there)
                fanin_results['input_spike_counts_per_fold']   = res.get('input_spike_counts_per_fold')
                fanin_results['total_input_spike_counts']      = res.get('total_input_spike_counts')
                fanin_results['avg_input_spikes_per_test_sample'] = res.get('avg_input_spikes_per_test_sample')
                
            if return_weights:
                fanin_results['first_layer_weights']            = res.get('first_layer_weights')
                fanin_results['all_lif1_params']                = res.get('all_lif1_params')
        
        all_results[name] = fanin_results

    # Save
    os.makedirs("experiments", exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    if classes_filtered:
        save_filename = f"experiments/fan_in_top_{num_inputs}_{num_outputs}_classes_{timestamp}.npz"
    else:
        save_filename = f"experiments/fan_in_{timestamp}.npz"
    np.savez(save_filename, data=all_results, allow_pickle=True)

    if verbose:
        print(f"\n{'#'*70}")
        print(f"Fanin experiment completed! Results saved to: {save_filename}")
        print(f"{'#'*70}")

    return all_results