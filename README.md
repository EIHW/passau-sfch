# Multimodal Prediction of Spontaneous Humour: A Novel Dataset and Results

Code used for the paper *Multimodal Prediction of Spontaneous Humour: A Novel Dataset and Results*.

Includes scripts for hyperparameter search and/or leave-one-out training & evaluation.

## Data

The Passau-SFCH dataset is available upon request at this [Zenodo-Link](add link here).
Please download the EULA [here](https://drive.google.com/file/d/1vSFvDWokMQsISYIvnIv1wizeLuhKqzvc/view?usp=sharing) 
and send it to lukas\[dot\]christ\[at\]informatik\[dot\]uni-augsburg\[dot\]de. Please note that the EULA must be 
filled and signed by a Professor and the data can only be shared with academic groups for non-commercial research
purposes.

Copy the directories ``features`` and ``labels`` as top-level-directories into the repo. Alternatively, 
adapt ``global_vars.py`` accordingly. 

## Installation

It is highly recommended to create a new Python venv and installing the provided ``requirements.txt``.

## General Information
This repository contains the code to reproduce the experiments in the abovementioned paper. GRUs are used for the humour 
recognition task, while SVMs are employed for sentiment and direction prediction.

Before running any script, the paths in ``global_vars.py`` must be updated.

For all three tasks, the following steps can be executed:
* hyperparameter search on the whole dataset
* evaluating a configuration in a leave-one-out setting: for all 10 coaches in the dataset, train on 9 of them and 
evaluate on the remaining one
* load and evaluate existing checkpoints

All scripts cache the features in the directory specified by ``FEATURE_CACHE_DIR`` in ``global_vars.py``.
Caching can not be disabled (only by deleting the respective directories). 

## Humour (GRUs)

GRU training for the humour task is mainly done with the ``run_rnn_experiments.py`` script.
For a detailed list of parameters, see the ``parse_args`` method.

### Hyperparameter Search

In order to conduct a hyperparameter search, simply specify several values for at least one hyperparameter such as 
``rnn_num_layers`` (see ``parse_args`` for all hyperparameters). The script will execute a grid search and choose the 
best configuration according to the best  validation score. Subsequently, this configuration is used to train and 
evaluate 10 models in a leave-one-out manner.
Note that, by default, all steps are executed ``--num_seeds`` times using incremental seeds (starting from ``--seed``).

### Only Leave-one-out Training and Validation (no HP search)

To skip the hyperparameter search and directly do the leave-one-out training/evaluation, make sure to only specify one 
value for each hyperparameter.

### Results 
Each run will receive an ID based on the features and the current system time. The detailled result information will be 
placed in ``global_vars.RESULT_DIR/humor/rnn/ID``. Analogously, checkpoints and predictions will be stored (if the 
corresponding flags are set).

### Loading and Evaluating Checkpoints

Evaluating on a set of checkpoints can be conducted via ``--eval_cp_only``. The script can then 
load a set of checkpoints (10 checkpoints, one for each leave-one-out setting) and store the evaluation result 
in a json. The unique GRU configuration used to generate the checkpoints must be specified. 
Only one feature can be specified. For the detailled parameters, please see ``parse_args``.


## Sentiment/Direction (SVMs)

In general, the ``run_svm_experiments`` script for sentiment and direction prediction utilising SVMs works 
in the same manner as the script for GRU-based Humour prediction. 

### Hyperparameter Search

Hyperparameters are specified as arguments (see ``parse_args``). As soon as there is more than one value for a 
hyperparameter, a hyperparameter grid search is executed. 

### Only Leave-one-out Training and Validation (no HP search)

Same as for GRUs, no hyperparameter search is carried out if only one configuration is specified via the 
script parameters.

### Results
Each run will receive an ID based on the features and the current system time. The result information will be 
placed in ``global_vars.RESULT_DIR/task/svc/ID``, where ``task` is either ``sentiment`` or ``direction``.

### Loading and Evaluating Checkpoints

Similar to the GRU script, evaluating on a set of checkpoints can be conducted via ``--eval_cp_only``. The script can then 
load a set of checkpoints (10 checkpoints, one for each leave-one-out setting) and store the evaluation result 
in a json. For the detailled parameters, please see ``parse_args``.

## Late Fusion

The ``late_fusion.py`` script ingests prediction csvs`and result jsons produced by ``run_svm_experiments`` or
 ``run_rnn_experiments``.
Late Fusion is conducted for each coach separately, the weights per feature are computed based on the performance of the other 
9 coaches during the leave-one-out training in the underlying experiments. 

``late_fusion.py`` requires the user to specify a ``--task`` (``humor``, ``sentiment``, ``direction``), the 
feature names (``--features``) and the IDs of the underlying unimodal experiments (``--ids``). 
The IDs are corresponding to the experiment's result directory name (the directory that contains ``all_results.json``).

The results will be placed in a ``late_fusion`` subdirectory of ``global_vars.RESULT_DIR``.
