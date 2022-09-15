import json
import pickle
import sys
from argparse import ArgumentParser
from functools import partial
from time import time, ctime
from typing import List, Dict

import numpy as np
import os

from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from tqdm import tqdm

from data_utils.features import load_whole_ds, idx_splitting, load_loo_ds
from data_utils.labels import load_task_data, linspace_weights
from data_utils.log import summarize_results, NpEncoder, unflatten_dict, number_of_configs
from global_vars import RESULT_DIR, PREDICTIONS_DIR, HUMOR, DIRECTION, SENTIMENT, COACHES, CHECKPOINT_DIR

import pandas as pd

from train_eval.evaluation import evaluate_cont


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--svm_cache_size', type=int, default=800)
    parser.add_argument('--name', default=None, type=str)
    parser.add_argument('--task', required=True, choices=[HUMOR, DIRECTION, SENTIMENT])
    parser.add_argument('--summarization_keys', required=False, default=['test.auc.mean'], nargs='+',
                        help='Create excel files for the corresponding results')
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--save_checkpoints', action='store_true')
    ##### FOR CHECKPOINT EVALUATION ####
    parser.add_argument('--eval_cp_only', action='store_true')
    parser.add_argument('--cp_dir', type=str, required= False,
                        help='Directory relative to CHECKPOINTS_DIR/{task} containing the {coach}_left_out.pickle files')
    parser.add_argument('--cp_eval_json', type=str, required=False, help='JSON file in which to store the checkpoint'
                                                                         'evaluation results.')


     ##### FEATURE PARAMETERS #######
    parser.add_argument('--features', nargs='+', default=['egemaps', 'ds', 'bert-4-sentence-level', 'faus'])
    parser.add_argument('--normalize', nargs='+', default=None, type=int)

    ##### HYPERPARAMETERS #######
    parser.add_argument('--C', nargs='+', default=[10., 5., 2., 1.0,0.1,0.01,0.001, 0.0001], type=float,
                        help='For SVC grid search')
    parser.add_argument('--kernel', nargs='+', default=['rbf', 'linear', 'poly', 'sigmoid'])
    parser.add_argument('--poly_degrees', nargs='+', type=int, default=[2,3,4])
    parser.add_argument('--weights', nargs='+', required=False, default=['balanced'], help='Specify either a number to '
                                                                                         'weight the rarer class with or '
                                                                                         '"balanced". For equal weighting, '
                                                                                         'use 1.')
    parser.add_argument('--gamma', nargs='+', default=['auto', 'scale'], choices=['auto', 'scale'],
                        help='Ignored for linear kernel')

    args = parser.parse_args()

    assert (args.normalize is None or len(args.normalize)==len(args.features))
    if args.normalize is None:
        args.normalize = [False for _ in args.features]
    else:
        args.normalize = [n > 0 for n in args.normalize]

    weights = []
    for w in args.weights:
        if not w=='balanced':
            try:
                weights.append(float(w))
            except:
                print("Use floats and/or 'balanced' for --weights")
        else:
            weights.append('balanced')
    args.weights = weights

    if args.name is None:
        args.name = ""
    args.name += ctime(time()).replace(":","-").replace(" ","_")

    return args


def get_num_configs(param_grids:List[Dict]) -> int:
    '''

    Calculates the number of configurations defined by the param_grids list
    :param param_grids:
    :return: number of configurations
    '''
    sum = 0
    for param_grid in param_grids:
        grid_size = 1
        for v in param_grid.values():
            grid_size *= len(v)
        sum += grid_size
    return sum


def get_cont_svc_predictions(svc:SVC, X:np.ndarray) -> np.ndarray:
    '''

    Calculates a score from the continuous svc predictions. It is assumed that the first class can be represented by -1
    (e.g. negative sentiment) and the second by 1
    :param svc: fitted model
    :param X: input data
    :return: scores
    '''
    probs = svc.predict_proba(X)
    assert probs.shape[1] == 2
    return -1 * probs[:, 0] + probs[:, 1]


def checkpoint_eval_only(db:pd.DataFrame, cp_file_dir:str, feature:str, normalize:bool, reduction,
                         target:str, target_json):
    '''

    Loads and evaluates checkpoints (Leave-one-out scenarios)
    :param db: database dataframe
    :param cp_file_dir: directory of the {coach}_leave_out.pickle files
    :param feature: feature name
    :param normalize: normalize?
    :param reduction: reduction method (mean here)
    :param target: target ("sentiment"/"direction")
    :param target_json: json to save result dict in
    :return:
    '''
    res_dct = {}
    for coach in tqdm(COACHES):

        _, (test_X, test_lengths, test_y) = load_loo_ds(
            db, leave_out_coach=coach, feature=feature, target=target, normalize=normalize, reduction=reduction)

        svc = pickle.load(open(os.path.join(cp_file_dir, f'{coach}_left_out.pickle'), 'rb'))
        test_predictions = get_cont_svc_predictions(svc, test_X)

        res_dct[coach] = evaluate_cont(test_y, test_predictions)

    os.makedirs(os.path.dirname(target_json), exist_ok=True)
    json.dump(res_dct, open(target_json, 'w+'))


def svm_experiment(db:pd.DataFrame,
                   param_grid:List[Dict],
                   feature:str,
                   target:str,
                   scoring,
                   normalize=True,
                   reduction=None,
                   num_random_seeds=5,
                   class_weights=None,
                   predict=False,
                   prediction_dir=None,
                   save_checkpoints=False,
                   checkpoint_dir=None) -> dict:
    '''

    Runs a grid search and/or evaluation.
    :param db: database dataframe
    :param param_grid: list of dicts, as input for sklearn GridSearch
    :param feature: feature name
    :param target: target column (sentiment/direction)
    :param scoring: sklearn scorer for GridSearch
    :param normalize: normalize features?
    :param reduction: reduction method (here: mean)
    :param num_random_seeds: number of seeds
    :param class_weights: float to set underrepresented class weight or 'balanced'
    :param predict: whether to save predictions
    :param prediction_dir: directory to save predictions in
    :return: dictionary including the best parameters and the results
    '''
    X, _, y, coach_idxs = load_whole_ds(db, feature, target, normalize=normalize, reduction=reduction)
    if not class_weights is None:
        new_cw = [linspace_weights(y, w) for w in class_weights if type(w) == float]
        if 'balanced' in class_weights:
            new_cw.append('balanced')
        class_weights = new_cw
    else:
        class_weights = ['balanced']

    for params in param_grid:
        params.update({'class_weight':class_weights})

    cv = idx_splitting(coach_idxs)

    num_configs = get_num_configs(param_grid)
    if  num_configs > 1:
        print(f"Starting gridsearch for {number_of_configs(param_grid)} configurations")
        gs = GridSearchCV(SVC(), param_grid, scoring=scoring, cv=cv)
        res = gs.fit(X, y)
        best_params = res.best_params_
        print(f'{ctime(time())}: finished gridsearch - starting eval')
    else:
        print('Only one config given, no hyperparameter search.')
        best_params = {k:v[0] for k,v in param_grid[0].items()}

    # restore and evaluate
    coach_dict = {}
    print(f'Training {"best" if num_configs > 1 else "given"} configuration (leave-one-out scenario)')
    for coach in tqdm(COACHES):

        train_metrics = []
        test_metrics = []
        (train_X, train_lengths, train_y, coach_idxs), (test_X, test_lengths, test_y) = load_loo_ds(
            db, leave_out_coach=coach, feature=feature, target=target, normalize=normalize, reduction=reduction)
        for seed in range(num_random_seeds):

            np.random.seed(seed)
            svc = SVC(**best_params)
            svc.fit(train_X, train_y)

            seed_train_metrics = {}
            seed_test_metrics = {}

            train_predictions = get_cont_svc_predictions(svc, train_X)
            test_predictions = get_cont_svc_predictions(svc, test_X)

            seed_train_metrics.update(evaluate_cont(train_y, train_predictions))
            seed_test_metrics.update(evaluate_cont(test_y, test_predictions))

            train_metrics.append(seed_train_metrics)
            test_metrics.append(seed_test_metrics)

        coach_dict[coach] = {}
        for m in train_metrics[0].keys():
            values = np.array([seed_metrics[m] for seed_metrics in train_metrics])
            coach_dict[coach].update({f'train.{m}.mean': np.mean(values), f'train.{m}.std': np.std(values)})

        for m in test_metrics[0].keys():
            values = np.array([seed_metrics[m] for seed_metrics in test_metrics])
            coach_dict[coach].update({f'test.{m}.mean': np.mean(values), f'test.{m}.std': np.std(values)})

        # save predictions for last seed
        if predict:
            pred_csv = os.path.join(prediction_dir, f'{coach}.csv')
            if len(test_predictions.shape) == 1:
                test_predictions = np.expand_dims(test_predictions, -1)
            pred_dict = {f'pred_{i}':test_predictions[:,i] for i in range(test_predictions.shape[1])}
            pd.DataFrame(pred_dict).to_csv(pred_csv, index=False)
        if save_checkpoints:
            os.makedirs(checkpoint_dir, exist_ok=True)
            cp_file = os.path.join(checkpoint_dir, f'{coach}_left_out.pickle')
            pickle.dump(svc, open(cp_file, 'wb+'))

    return {'best_params': best_params, 'results': unflatten_dict(coach_dict)}


if __name__ == '__main__':
    args = parse_args()

    db, label_mapping, target_col = load_task_data(args.task)

    reduction = partial(np.mean, axis=0)

    if args.eval_cp_only:
        assert len(args.features) == 1, "When evaluating a checkpoint, you may only specify one feature"
        cp_dir = os.path.join(CHECKPOINT_DIR, args.task, args.cp_dir)
        checkpoint_eval_only(db, cp_dir, feature=args.features[0], normalize=args.normalize[0], reduction=reduction,
                             target= target_col, target_json=args.cp_eval_json)
        sys.exit(0)

    param_grids = []

    if 'rbf' in args.kernel:
        param_grids.append({'C':args.C, 'probability':[True], 'kernel':['rbf'],
                            'cache_size': [args.svm_cache_size], 'gamma':args.gamma})
    if 'linear' in args.kernel:
        param_grids.append({'C':args.C, 'probability':[True], 'kernel':['linear'],
                            'cache_size': [args.svm_cache_size]})
    if 'poly' in args.kernel:
        param_grids.append({'C':args.C, 'probability':[True], 'kernel':['poly'],
                            'degree':args.poly_degrees, 'cache_size': [args.svm_cache_size], 'gamma':args.gamma})
    if 'sigmoid' in args.kernel:
        param_grids.append({'C': args.C, 'probability': [True], 'kernel': ['sigmoid'],
                            'cache_size': [args.svm_cache_size], 'gamma':args.gamma})

    experiment_str = os.path.join(args.task, 'svc')

    res_dir = os.path.join(RESULT_DIR, experiment_str, args.name)
    os.makedirs(res_dir, exist_ok=True)

    result_json = os.path.join(res_dir, 'all_results.json')
    coach_excels = [os.path.join(res_dir, f'{key}_coaches.xlsx') for key in args.summarization_keys]
    feature_excels = [os.path.join(res_dir, f'{key}_features.xlsx') for key in args.summarization_keys]

    prediction_dir = None
    if args.save_predictions:
        prediction_dir = os.path.join(PREDICTIONS_DIR, experiment_str, args.name)
        os.makedirs(prediction_dir)

    sorted_label_names = [label_mapping[k] for k in sorted(list(label_mapping.keys()))]

    scoring = make_scorer(roc_auc_score, greater_is_better=True)

    result_dct = {'config': {'params': param_grids,
                             'cli_args': vars(args),
                           'target': target_col,
                           'features': [f'{f} (norm={n})' for f, n in zip(args.features, args.normalize)]},
                'results': {}}

    for i,feature in enumerate(args.features):
        print(f'---{feature}')
        cp_dir = os.path.join(CHECKPOINT_DIR, experiment_str, args.name, feature)
        result_dct['results'][feature] = svm_experiment(db=db, param_grid=param_grids,
                                                        feature=feature, target=target_col, scoring=scoring,
                                                        normalize=args.normalize[i],
                                                        num_random_seeds=args.num_seeds, reduction=reduction,
                                                        class_weights=args.weights, predict=args.save_predictions,
                                                        prediction_dir=prediction_dir,
                                                        save_checkpoints=args.save_checkpoints, checkpoint_dir=cp_dir)
        json.dump(result_dct, open(result_json, 'w+'), cls=NpEncoder)

    json.dump(result_dct, open(result_json, 'w+'))
    for i,summarization_key in enumerate(args.summarization_keys):
        summarize_results(result_dct, coach_target_file=coach_excels[i], feature_target_file=feature_excels[i],
                          key=summarization_key)
