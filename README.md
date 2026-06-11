# Spiking Neural Network for Tactile Object Classification

This repository contains a tactile perception pipeline built around spiking neural networks (SNNs) using `snntorch`.
The main experimentation workflow is notebook-driven, while core logic is implemented in Python modules.

## What this project does

- Loads and preprocesses SmartHand tactile recordings.
- Converts tactile time-series into spike/event representations.
- Trains and evaluates SNN models with cross-validation.
- Runs experiment sweeps (loss, architecture, alpha/beta, fan-in, sparsity, Top-k, etc.).
- Produces saved experiment artifacts (`.npz`, confusion matrices, figures, masks).

## Main files and responsibilities

- [snnTorchLearning.ipynb](snnTorchLearning.ipynb): **Primary entry point** for running experiments.
- [experiments.py](experiments.py): High-level experiment runners and orchestration logic.
- [models.py](models.py): SNN model classes, forward passes, training/CV functions.
- [utils.py](utils.py): Data helpers, preprocessing utilities, plotting/analysis helpers.

Related tools:

- [preprocess_data.ipynb](preprocess_data.ipynb): Prepares input datasets used by experiments.
- [PixelSelectorUI.py](PixelSelectorUI.py): Interactive taxel mask selector.
- [SmartHandAverager.py](SmartHandAverager.py): Super-pixel/averaging mapping utility.

## Repository layout

- `experiments/`: saved experiment outputs (`.npz`, confusion matrices, summaries).
- `figures/`: generated plots/figures.
- `masks/`: saved mask files (`.npy`).
- `smarthand_dataset.mat`: source tactile dataset.
- `res95_ranked_taxels.csv`: taxel ranking metadata.

## Quick start (Notebook workflow)

1. Open [snnTorchLearning.ipynb](snnTorchLearning.ipynb).
2. Make sure the notebook kernel is your project environment.
3. Run setup/import cells first.
4. If needed, prepare data with [preprocess_data.ipynb](preprocess_data.ipynb) so required input files exist.
5. Run experiment cells (they call functions from [experiments.py](experiments.py)).
6. Check outputs in `experiments/` and generated plots.

## How experiments are organized

### 1) Data and preprocessing

Utilities in [utils.py](utils.py) handle:

- mask creation/selection,
- spike/event conversion,
- saving/loading preprocessed spike data,
- result analysis/plotting helpers.

### 2) Models and training

[models.py](models.py) contains:

- model architectures (e.g., `FC_SNN_Syn`, variants),
- forward-pass helpers,
- training loops,
- cross-validation fold creation and metrics.

### 3) Experiment orchestration

[experiments.py](experiments.py) contains high-level runners such as:

- `run_loss_experiment(...)`
- `run_model_experiment(...)`
- `run_param_experiment(...)`
- `run_confusion_matrix_experiment(...)`
- `run_alpha_experiment(...)`
- `run_beta_experiment(...)`
- `run_alpha_beta_grid_experiment(...)`
- `run_sparsity_experiment(...)`
- `run_fanin_experiment(...)`

These functions load prepared inputs, call training/CV functions from [models.py](models.py), and save `.npz` outputs to `experiments/`.

## Typical workflow

1. (Optional) Build or edit masks using [PixelSelectorUI.py](PixelSelectorUI.py).
2. Preprocess raw data and save spike datasets.
3. Run experiments from [snnTorchLearning.ipynb](snnTorchLearning.ipynb).
4. Analyze/plot results using helpers in [utils.py](utils.py) and notebooks such as [plots.ipynb](plots.ipynb).

## Outputs

Common artifacts:

- `experiments/*.npz`: experiment result bundles.
- `experiments/*confusion_matrix.csv`: confusion matrices.
- `figures/*.png`: saved figure outputs.
- `masks/*.npy`: saved custom taxel masks.

## Notes

- The notebook [snnTorchLearning.ipynb](snnTorchLearning.ipynb) is the intended place to run most experiments.
- Core experiment logic is in [experiments.py](experiments.py), model/training code in [models.py](models.py), and shared helpers in [utils.py](utils.py).
