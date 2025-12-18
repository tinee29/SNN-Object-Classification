import os
import numpy as np
import pandas as pd
import torch
from utils import load_spike_data, get_filename_from_params
from models import cv_train, FC_SNN_Syn, FC_SNN_Syn_33, create_cv_folds, train, FC_SNN_Syn_32, cv_train_topk, train_topk
from matplotlib import pyplot as plt
import seaborn as sns
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_loss_experiment(device, num_folds, num_epochs, batch_size, verbose, num_outputs, num_inputs, input_file, output_name):
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
    spike_tensors, y_tensors, _ = load_spike_data(filename)

    (cv_train_data_folds, cv_train_labels_folds, cv_test_data_folds, cv_test_labels_folds, val_data, val_labels, _, _) = create_cv_folds(
             spike_tensors, y_tensors, num_folds=num_folds, num_sessions=2, num_classes=num_outputs,
             random_state=42)

    loss_map = {
        "ce_only": 'ce',
        "combined": 'combined'
    }

    all_results = {}

    for loss_name, loss_fn in loss_map.items():
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
            loss_fn=loss_fn
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
    spike_tensors, y_tensors, _ = load_spike_data(filename)

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
            loss_fn='combined'
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
                               batch_size=16, random_state=42, device=device, verbose=True, exclude_classes=None, 
                               alpha=0.7, beta=0.7, sparsity=None, return_membrane=False, return_spikes=False,
                               topk_accuracy=None):
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
        loss_fn (str): Loss function to use ('ce', 'mse', 'combined', 'balanced', 'sum').
        exclude_classes (list): Optional list of class names to exclude (e.g., ['ball', 'pen']).

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
            continue

        if verbose:
            print(f"\n{'#'*70}\nRunning experiment for {varying_param} = {value} from {filename}\n{'#'*70}")

        # Load data and parameters
        spike_tensors, y_tensors, _ = load_spike_data(filename)
        num_outputs = len(class_names)

        # Handle class exclusion if specified
        if exclude_classes is not None:
            # Map excluded class names to their indices
            exclude_indices = [i for i, name in enumerate(class_names) if name in exclude_classes]
            if exclude_indices:
                # Filter out excluded classes from y_tensors and corresponding spike_tensors
                mask = ~np.isin(y_tensors, exclude_indices)
                spike_tensors = spike_tensors[mask]
                y_tensors = y_tensors[mask]
                # Update num_outputs by subtracting the number of excluded classes
                num_outputs = len(class_names) - len(exclude_indices)
                # Remap y_tensors to new range (0 to num_outputs-1)
                unique_classes = np.unique(y_tensors)
                class_mapping = {old: new for new, old in enumerate(unique_classes)}
                y_tensors = np.array([class_mapping[y] for y in y_tensors])

        num_inputs = spike_tensors[0].shape[1]
        num_sessions = len(params_dict.get("session_id", []))

        (cv_train_data_folds, cv_train_labels_folds, cv_test_data_folds, cv_test_labels_folds,
         val_data, val_labels, _, _) = create_cv_folds(
             spike_tensors, y_tensors, num_folds=num_folds, num_sessions=num_sessions, num_classes=num_outputs,
             random_state=random_state
         )
        #print(num_inputs, num_outputs)
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
                input_sparsity=sparsity,
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
                loss_fn='combined',
                alpha=alpha,
                beta=beta,
                input_sparsity=sparsity,
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
    spike_tensors, y_tensors, _ = load_spike_data(filename)
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
        loss_fn='combined',
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

    # Plot and save aggregated normalized confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(normalized_cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Aggregated Normalized Confusion Matrix for {FC_SNN_Syn.__name__}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_plot_path = os.path.join("figures", f"top_{num_inputs}_confusion_matrix_plot.png")
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save per-class metrics to CSV
    metrics_df = pd.DataFrame([
        {'Class': name, **metrics} for name, metrics in class_performance.items()
    ])
    metrics_csv_path = os.path.join("experiments", f"top_{num_inputs}_confusion_matrix_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False, float_format='%.4f')

    # Save actual confusion matrix to CSV
    cm_df = pd.DataFrame(aggregated_cm, index=class_names, columns=class_names)
    cm_csv_path = os.path.join("experiments", f"top_{num_inputs}_confusion_matrix.csv")
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



def final_training_and_validation(fixed_params, model=FC_SNN_Syn, alpha=None, beta=None, 
                                 num_epochs=20, batch_size=16, random_state=42, device=device,
                                 verbose=True, loss_fn='combined', lr=0.001, 
                                 patience=5, min_delta=0.01, include_classes=None, topk_accuracy=None):
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

    spike_tensors, y_tensors, _ = load_spike_data(filename)
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
            topk_accuracy=topk_accuracy
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
            lr=lr
        )

    os.makedirs("experiments", exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    
    if classes_filtered:
        save_filename = f"experiments/final_training_{num_outputs}_classes_{timestamp}.npz"
    else:
        save_filename = f"experiments/final_training_{timestamp}.npz"
    
    np.savez(save_filename, data=results, allow_pickle=True)
    
    if verbose:
        print(f"Results saved to: {save_filename}")

    return results

def run_finetuning(fixed_params, num_folds=5, num_epochs=20, batch_size=16,
                   random_state=42, device=device, verbose=True, alpha=None, beta=None, 
                   fine_tune_epochs=10, noise_std=0.01, exclude_classes=None):
    """Run cross-validation training with fixed parameters, extract first-layer weights, and perform fine-tuning.

    Args:
        fixed_params (dict): Dictionary of fixed parameter values (e.g., {"topn": 300, "num_frames": 25, ...}).
        num_folds (int): Number of cross-validation folds.
        num_epochs (int): Number of training epochs per fold for initial training.
        batch_size (int): Batch size for training.
        random_state (int): Random seed for reproducibility.
        device: The device to run the experiment on.
        verbose (bool): If True, print progress messages.
        num_inputs (int): Number of input features.
        num_outputs (int): Number of output classes.
        fine_tune_epochs (int): Number of epochs for fine-tuning.
        noise_std (float): Standard deviation of Gaussian noise added to sampled weights.
        exclude_classes (list): Optional list of class names to exclude (e.g., ['ball', 'pen']).

    Returns:
        dict: Dictionary containing initial training results, fine-tuning results, and fine-tuned model state.
    """

    # Define the output directory
    output_dir = "preprocessed_data"

    # Get the filename based on the parameter dictionary
    filename = get_filename_from_params(fixed_params, output_dir)
    if not os.path.exists(filename):
        print(f"Skipping missing file: {filename}")
        return None

    if verbose:
        print(f"\n{'#'*70}\nRunning fine-tuning experiment with fixed parameters from {filename}\n{'#'*70}")

    # Load data and parameters
    spike_tensors, y_tensors, _ = load_spike_data(filename)

    # Define class names for mapping
    class_names = [
        'ball', 'battery', 'bracket', 'coin', 'empty_can', 'empty_hand', 'full_can',
        'gel', 'lotion', 'mug', 'pen', 'safety_glasses', 'scissors', 'screw_driver',
        'spray_can', 'stapler', 'tape'
    ]
    num_outputs = len(class_names)
    # Handle class exclusion if specified
    classes_excluded = False
    if exclude_classes is not None:
        exclude_indices = [i for i, name in enumerate(class_names) if name in exclude_classes]
        if exclude_indices:
            classes_excluded = True
            if verbose:
                print(f"Excluding classes: {exclude_classes} → indices {exclude_indices}")
            mask = ~np.isin(y_tensors, exclude_indices)
            spike_tensors = spike_tensors[mask]
            y_tensors = y_tensors[mask]
            num_outputs = len(class_names) - len(exclude_indices)
            # Remap labels
            unique_classes = np.unique(y_tensors)
            class_mapping = {old: new for new, old in enumerate(unique_classes)}
            y_tensors = np.array([class_mapping[y] for y in y_tensors])
            if verbose:
                print(f"New number of classes: {num_outputs}")
        else:
            exclude_classes = None  # No valid exclusions

    num_inputs = spike_tensors[0].shape[1]

    (cv_train_data_folds, cv_train_labels_folds, cv_test_data_folds, cv_test_labels_folds,
     val_data, val_labels, _, _) = create_cv_folds(
         spike_tensors, y_tensors, num_folds=num_folds, num_sessions=2, num_classes=num_outputs,
         random_state=random_state
     )

    # Initial cross-validation training to get weights
    initial_results = cv_train(
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
        loss_fn='combined',
        return_weights=True,
        alpha=alpha,
        beta=beta
    )

    # Extract first-layer weights and lif1 parameters
    first_layer_weights = initial_results.get('first_layer_weights', [])
    all_lif1_params = initial_results.get('all_lif1_params', [])
    if not first_layer_weights or not all_lif1_params:
        print("No first-layer weights or lif1 parameters available for fine-tuning.")
        return initial_results

    # Use weights and parameters from the best fold based on test accuracy
    best_fold_idx = np.argmax([np.nanmean(acc) for acc in initial_results['all_test_acc_hist']])
    best_weights = first_layer_weights[best_fold_idx]
    best_lif1_params = all_lif1_params[best_fold_idx]
    if verbose:
        print(f"Using weights and parameters from fold {best_fold_idx + 1} with best test accuracy: {np.nanmean(initial_results['all_test_acc_hist'][best_fold_idx]) * 100:.2f}%")
        print(f"lif1 params: alpha={best_lif1_params['alpha']:.4f}, beta={best_lif1_params['beta']:.4f}, threshold={best_lif1_params['threshold']:.4f}")

    # Create new model for fine-tuning
    fine_tune_model = FC_SNN_Syn(num_outputs=num_outputs, num_inputs=num_inputs, alpha=alpha, beta=beta).to(device)
    fine_tune_model.fc1.weight.data = torch.tensor(best_weights, dtype=torch.float32).to(device)

    # Set lif1 parameters from the best fold
    fine_tune_model.lif1.alpha.data = torch.tensor(best_lif1_params['alpha'], dtype=torch.float32).to(device)
    fine_tune_model.lif1.beta.data = torch.tensor(best_lif1_params['beta'], dtype=torch.float32).to(device)
    fine_tune_model.lif1.threshold.data = torch.tensor(best_lif1_params['threshold'], dtype=torch.float32).to(device)

    # Freeze the first layer and its Synaptic parameters
    for param in fine_tune_model.fc1.parameters():
        param.requires_grad = False
    for param in fine_tune_model.lif1.parameters():
        param.requires_grad = False

    # Set up optimizer for the second layer only
    optimizer = torch.optim.Adam(fine_tune_model.fc2.parameters(), lr=0.001)
    # Load full dataset for fine-tuning with retrain=True
    train_data, train_labels, val_data, val_labels = create_cv_folds(
        spike_tensors, y_tensors, num_folds=None, num_sessions=2, num_classes=num_outputs,
        random_state=random_state, retrain=True
    )

    # Fine-tune using the train function
    fine_tune_results = train(
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        model_class=fine_tune_model,
        num_epochs=fine_tune_epochs,
        batch_size=2,
        random_state=random_state,
        device=device,
        verbose=verbose,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        loss_fn='combined',
        patience=5,
        min_delta=0.01,
        finetune=True,
        optimizer=optimizer
    )

    # Combine results
    combined_results = {
        'initial_results': initial_results,
        'fine_tune_results': fine_tune_results,
        'fine_tuned_model_state': fine_tune_model.state_dict()
    }

    # Create experiments folder if it doesn't exist
    os.makedirs("experiments", exist_ok=True)
    # Save results to a single NPZ file with a timestamp
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    if classes_excluded:
        save_filename = f"experiments/finetuning_{num_outputs}classes_{timestamp}.npz"
    else:
        save_filename = f"experiments/finetuning_{timestamp}.npz"

    np.savez(save_filename, data=combined_results, allow_pickle=True)
    
    if verbose:
        print(f"Results saved to: {save_filename}")

    return combined_results


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
    spike_tensors, y_tensors, _ = load_spike_data(filename)

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
            loss_fn='combined',
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
    spike_tensors, y_tensors, _ = load_spike_data(filename)

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
            loss_fn='combined',
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
def run_alpha_beta_grid_experiment(device, num_folds, num_epochs, batch_size, verbose, num_outputs, num_inputs, 
                                  input_file, output_name, alpha_values=[0.1, 0.3, 0.5, 0.7, 0.9], 
                                  beta_values=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """Run alpha-beta GRID PARAMETER SWEEP experiment using cv_train().

    Tests ALL combinations of alpha × beta values. Outputs:
    1. NPZ file with full per-epoch results (for plotting)
    2. SUMMARY TABLE with avg/max train/test accuracy for ALL combinations

    Args:
        device: The device to run the experiment on.
        num_folds (int): Number of cross-validation folds.
        num_epochs (int): Number of training epochs per fold.
        batch_size (int): Batch size for training.
        verbose (bool): If True, print progress messages.
        num_outputs (int): Number of output classes.
        num_inputs (int): Number of input features.
        input_file (str): Path to preprocessed spike data file.
        output_name (str): Base name for output NPZ file.
        alpha_values (list): List of alpha values to test.
        beta_values (list): List of beta values to test.

    Returns:
        dict: Dictionary containing results for all alpha-beta combinations.
    """

    output_dir = "preprocessed_data"
    filename = os.path.join(output_dir, input_file)
    spike_tensors, y_tensors, _ = load_spike_data(filename)
    num_inputs = spike_tensors[0].shape[1]

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

    # GRID SWEEP: ALL alpha × beta combinations
    for alpha in alpha_values:
        for beta in beta_values:
            combination_idx += 1
            key = f"alpha_{alpha}_beta_{beta}"
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"[{combination_idx:2d}/{total_combinations}] Running: alpha={alpha}, beta={beta}")
                print(f"{'='*80}")

            # Train with FIXED alpha + beta
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
                loss_fn='combined',
                alpha=alpha,
                beta=beta
            )

            # Extract results (SAME FORMAT as other experiments)
            train_loss_avg = np.array(res['avg_loss_hist'])
            train_acc_avg = np.array(res['avg_acc_hist'])
            test_loss_avg = np.array(res['avg_test_loss_hist'])
            test_acc_avg = np.array(res['avg_test_acc_hist'])

            train_loss_folds = np.array(res['all_loss_hist']) if 'all_loss_hist' in res else None
            train_acc_folds = np.array(res['all_acc_hist']) if 'all_acc_hist' in res else None
            test_loss_folds = np.array(res['all_test_loss_hist']) if 'all_test_loss_hist' in res else None
            test_acc_folds = np.array(res['all_test_acc_hist']) if 'all_test_acc_hist' in res else None

            # Store FULL results for plotting
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

            # SUMMARY METRICS (like analyze_experiment)
            avg_train_acc = np.nanmean(train_acc_avg) * 100
            max_train_acc = np.nanmax(train_acc_avg) * 100
            avg_test_acc = np.nanmean(test_acc_avg) * 100
            max_test_acc = np.nanmax(test_acc_avg) * 100

            # Store for summary table
            summary_data.append({
                'alpha': alpha,
                'beta': beta,
                'avg_train_acc': avg_train_acc,
                'max_train_acc': max_train_acc,
                'avg_test_acc': avg_test_acc,
                'max_test_acc': max_test_acc
            })

    # === CREATE SUMMARY TABLE ===
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by avg_test_acc (descending)
    summary_df = summary_df.sort_values('avg_test_acc', ascending=False).reset_index(drop=True)
    
    # Format percentages
    for col in ['avg_train_acc', 'max_train_acc', 'avg_test_acc', 'max_test_acc']:
        summary_df[col] = summary_df[col].round(2)

    # PRINT BEAUTIFUL TABLE
    print(f"\n{'#'*80}")
    print(f"ALPHA-BETA GRID SWEEP SUMMARY TABLE")
    print(f"{'#'*80}")
    print(summary_df.to_string(index=False, float_format='%.2f'))
    
    # HIGHLIGHT BEST COMBINATION
    best_idx = summary_df['avg_test_acc'].idxmax()
    best_row = summary_df.iloc[best_idx]
    print(f"\n{'*'*80}")
    print(f"BEST COMBINATION: alpha={best_row['alpha']}, beta={best_row['beta']}")
    print(f"Avg Test Acc: {best_row['avg_test_acc']}% | Max Test Acc: {best_row['max_test_acc']}%")
    print(f"{'*'*80}")

    # SAVE SUMMARY CSV
    os.makedirs("experiments", exist_ok=True)
    summary_csv = f"experiments/{output_name}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary table saved to: {summary_csv}")

    # === SAVE FULL RESULTS NPZ ===
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    save_path = os.path.join("experiments", output_name)
    np.savez(save_path + f"_grid_{timestamp}.npz", data=all_results, allow_pickle=True)
    
    if verbose:
        print(f"\n{'#'*80}")
        print(f"Grid experiment completed!")
        print(f"Full results saved to: {save_path}_grid_{timestamp}.npz")
        print(f"Summary table saved to: {summary_csv}")
        print(f"Total combinations tested: {total_combinations}")
        print(f"{'#'*80}")

    # Return BOTH full results AND summary
    return {
        'full_results': all_results,
        'summary_df': summary_df,
        'best_combination': {
            'alpha': float(best_row['alpha']),
            'beta': float(best_row['beta']),
            'avg_test_acc': float(best_row['avg_test_acc'])
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
    spike_tensors, y_tensors, _ = load_spike_data(filename)

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
            loss_fn='combined',
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
    num_outputs=17, num_inputs=484,
    input_file=None,
    alpha=0.7, beta=0.7,
    fanin_values=None,  # e.g. [32, 64, 96, 128, 192, 484]
    return_spikes=True, return_membrane=True, return_weights=True,
    include_classes=None,  # ← NEW: list of class names to KEEP
):
    import os, pandas as pd

    # Load data
    spike_tensors, y_tensors, _ = load_spike_data(os.path.join("preprocessed_data", input_file))
    y_tensors = np.array(y_tensors, dtype=np.int64)

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
    (cv_train_data_folds, cv_train_labels_folds,
     cv_test_data_folds, cv_test_labels_folds, _, _, _, _) = create_cv_folds(
         spike_tensors, y_tensors, num_folds=num_folds, num_sessions=2,
         num_classes=num_outputs, random_state=42)
     
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
            loss_fn='combined',
            alpha=alpha,
            beta=beta,
            fixed_fan_in=fan_in,
            input_sparsity=None,
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