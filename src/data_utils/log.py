import json
import os

import numpy as np
import pandas as pd

# https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
from global_vars import RESULT_DIR


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def unflatten_dict(dct):
    '''

    Restores the hierarchy in a flat dict with "."-joined keys
    :param dct: flat dictionary
    :return: unflattened dict (hierarchy induced by "." in the keys)
    '''
    if not isinstance(dct, dict):
        return dct
    new_dct = {}

    new_keys = set([k.split(".")[0] for k in dct.keys() if len(k.split(".")) > 1])
    keep_old_keys = set([k.split(".")[0] for k in dct.keys() if len(k.split(".")) ==1])

    sub_keys = {nk:[] for nk in new_keys}
    for k in sorted(dct.keys()):
        if k.split(".")[0] in sorted(new_keys):
            sub_keys[k.split(".")[0]].append(".".join(k.split(".")[1:]))

    for nk in sorted(new_keys):
        sub_dict = {sub_key:dct[f'{nk}.{sub_key}'] for sub_key in sorted(sub_keys[nk])}
        new_dct[nk] = unflatten_dict(sub_dict)

    for ok in sorted(keep_old_keys):
        new_dct[ok] = unflatten_dict(dct[ok])

    return new_dct


def flatten_dict(dct):
    '''

    Makes a dict flat, preserving the hierarchy via "." in keys
    :param dct: dictionary
    :return: flattened dictionary
    '''
    if not isinstance(dct, dict):
        return dct

    new_dict = {}
    for k,v in sorted(dct.items(), key = lambda x: x[0]):
        if not isinstance(v, dict):
            new_dict[k] = v
        else:
            sub_dict = flatten_dict(v)
            for sk,sv in sorted(sub_dict.items(), key= lambda x: x[0]):
                new_dict[f'{k}.{sk}'] = sv

    return new_dict


def __get_stats(values, axis, names):
    '''
    Auxiliary method to process result jsons
    '''
    means = np.round_(np.mean(values, axis=axis), 4)
    stds = np.round_(np.std(values, axis=axis), 4)
    mins = np.round_(np.min(values, axis=axis), 4)
    argmins = np.argmin(values, axis=axis)
    argmins = [names[i] for i in argmins]
    maxs = np.round_(np.max(values, axis=axis), 4)
    argmaxs = np.argmax(values, axis=axis)
    argmaxs = [names[i] for i in argmaxs]
    return {'mean':means, 'std':stds, 'min':mins, 'argmin':argmins, 'max':maxs, 'argmax':argmaxs}


def _process_dimension(dct, key='test.auc.mean'):
    sub_dct = dct['results']
    means = {}
    for f,f_results in sub_dct.items():
        coach_results = {c:flatten_dict(f_results['results'][c]) for c in f_results['results'].keys()}
        try:
            means[f] = {c: coach_results[c][key] for c in f_results['results'].keys()}
        except:
            print(f'Key {key} not available, skip.')
            return None, None
    # means = {f:{c: f_results['results'][c][key] for c in f_results['results'].keys()} for f, f_results in sub_dct.items()}
    coach_df = pd.DataFrame(means)
    coach_df.loc[:,:] = np.round_(coach_df.values, 4)
    num_coaches = coach_df.values.shape[0]
    coaches = coach_df.index.values
    num_features = coach_df.values.shape[1]
    features = list(coach_df.columns)
    # stats per coach
    coach_stats = __get_stats(coach_df.values, axis=1, names = features)
    for key,values in coach_stats.items():
        coach_df[key] = values
    coach_df['coach'] = coaches
    coach_df = coach_df[['coach'] + list(coach_df.columns)[:-1]]
    feature_df = coach_df.iloc[:num_coaches, 1:1+num_features]
    feature_stats = __get_stats(feature_df.values, axis=0, names=coaches)
    feature_df = pd.DataFrame(feature_stats)
    feature_df['feature'] = features
    feature_df = feature_df[['feature'] + list(feature_df.columns)[:-1]]
    #df = pd.concat([coach_df, feature_df])
    return coach_df, feature_df


def summarize_results(result_dict, feature_target_file, coach_target_file, key='test.auc.mean'):

    coach_df, feature_df = _process_dimension(result_dict, key=key)
    if not (coach_df is None or feature_df is None):
        coach_df.to_excel(coach_target_file, index=False, sheet_name=key)
        feature_df.to_excel(feature_target_file, index=False, sheet_name=key)


def number_of_configs(param_grids):
    sum = 0
    for grid in param_grids:
        combinations = 1
        for value_list in grid.values():
            combinations *= len(value_list)
        sum += combinations
    return sum