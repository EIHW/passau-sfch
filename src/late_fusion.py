from argparse import ArgumentParser
from itertools import combinations

from sklearn.metrics import roc_auc_score

from data_utils.labels import load_task_data
from global_vars import HUMOR, DIRECTION, SENTIMENT, COACHES, PREDICTIONS_DIR, RESULT_DIR

import os
import pandas as pd
import numpy as np

import json

Z_S = 'z'
MM_S = 'minmax'
STANDARDIZATION = [Z_S, MM_S]

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--task', required=True, choices=[HUMOR, DIRECTION, SENTIMENT])
    parser.add_argument('--features', required=True, nargs='+')
    parser.add_argument('--ids', nargs='+', required=True, help='Must correspond to a directory under predictions/task')
    parser.add_argument('--aliases', nargs='+', required=False)
    parser.add_argument('--unweighted', action='store_true')
    parser.add_argument('--standardization', required=False, choices=STANDARDIZATION)
    parser.add_argument('--name', required=True, type=str)

    args = parser.parse_args()

    assert len(args.ids) > 1
    assert len(args.features) == len(args.ids)

    if args.aliases is None:
        args.aliases = [i.replace("/","-").replace(":","-") for i in args.ids]
    else:
        assert len(args.aliases) == len(args.ids)
    assert (len(set(args.aliases)) == len(args.aliases))

    return args

def preprocess(arr, standardization=None):
    if standardization == Z_S:
        arr = (arr - np.mean(arr)) / np.std(arr)
    elif standardization == MM_S:
        arr =  (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return arr

def load_train_auc(i, feature, coach):
    result_json = os.path.join(RESULT_DIR, args.task, i, 'all_results.json')
    result_dict = json.load(open(result_json, 'r'))['results'][feature]['results'][coach]
    if 'train' in result_dict.keys():
        return result_dict['train']['auc']['mean']
    elif 'best_val' in result_dict.keys():
        return result_dict['best_val']


def load_weights(ids, features):
    coach_weights = {c:[] for c in COACHES}
    for c in COACHES:
        for i,f in zip(ids,features):
            coach_weights[c].append(max(0.5, load_train_auc(i,f,c)) - 0.5)
    return coach_weights



def eval_fusion(coach_preds, coach_gss, comb, weights):
    res_dict = {}
    for c in COACHES:
        c_weights = np.array(weights[c])[list(comb)]
        c_weights = c_weights / np.sum(c_weights)
        arr = coach_preds[c][:,comb]
        arr = arr * c_weights
        arr = np.sum(arr, axis=1)
        res_dict[c] = {'auc': np.round(roc_auc_score(coach_gss[c], arr), 4),
                       'weights': [float(x) for x in c_weights.tolist()]}

    return res_dict


if __name__ == '__main__':
    args = parse_args()

    db, label_mapping, target_col = load_task_data(task=args.task)

    coach_gss = {c: db[db.coach==c][target_col].values for c in COACHES}

    # load predictions
    coach_preds = {c:[] for c in COACHES}
    for idx,i in enumerate(args.ids):
        pred_dir = os.path.join(PREDICTIONS_DIR, args.task, i, args.features[idx])
        for c in COACHES:
            pred_df = pd.read_csv(os.path.join(pred_dir, f'{c}.csv'))
            coach_preds[c].append(preprocess(
                np.expand_dims(pred_df.iloc[:,0].values, -1), standardization=args.standardization))

    for c in COACHES:
        coach_preds[c] = np.hstack(coach_preds[c])


    if args.unweighted:
        weights = {c: np.ones(len(args.ids),) for c in COACHES}
    else:
        weights = load_weights(args.ids, args.features)

    # evaluate all combinations of A,T,V
    all_combinations = []
    idx_list = list(range(0, len(args.ids)))
    for l in range(2, len(args.ids)+1):
        all_combinations.extend(combinations(idx_list, l))

    results = {}
    for comb in all_combinations:
        comb_name = '+'.join([args.aliases[i] for i in comb])
        results[comb_name] = eval_fusion(coach_preds, coach_gss, comb, weights=weights)

    res_dir = os.path.join(RESULT_DIR, 'late_fusion', args.task, "+".join(args.aliases)+"_"+args.name)
    os.makedirs(res_dir, exist_ok=True)
    ret_dict = {'cli_args': vars(args), 'results':results}
    res_df = pd.DataFrame({fusion: {coach: dct['auc'] for coach, dct in ret_dict['results'][fusion].items()}
                           for fusion in ret_dict['results'].keys()})
    json.dump(ret_dict, open(os.path.join(res_dir, 'results.json'), 'w+'))
    res_df.to_excel(os.path.join(res_dir, 'results.xlsx'))

    print('Done')