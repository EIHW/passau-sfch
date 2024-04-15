import os
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from global_vars import HUMOR, DIRECTION, SENTIMENT, LABEL_DIR
import pandas as pd


def db_for_binary_humor_target(db):
    return db, {0:'no humor', 1:'humor'}, 'humor'

def db_for_binary_dimension_discrimination(db, dimension):
    copy_df = db[np.abs(db[dimension]) > 0].copy()
    copy_df[dimension] = np.sign(copy_df[dimension].values).astype(np.int8)
    return copy_df, {-1:'self_directed' if dimension=='direction' else 'negative',
                    1:'others_directed' if dimension=='direction' else 'positive'}, dimension


def linspace_weights(ys, max_weight=5.):
    '''
    
    Assigns weights to a label set, according to a linspace from 1 to ratio, sorted by label frequency descending.
    :param ys: all labels
    :param max_weight: maximum weight (for least frequent label)
    :return: dictionary mapping label to weight
    '''
    counts = Counter(ys)
    srtd = counts.most_common()
    weights = np.linspace(1., max_weight, len(srtd))
    dct = {int(srtd[i][0]): weights[i] for i in range(len(srtd))}
    return dct


def load_task_data(task):
    db = pd.read_csv(os.path.join(LABEL_DIR, 'gold_standard.csv'))
    db['coach'] = [f.split("_")[0] for f in db.file.values]
    db['start'] = db['start'].values - 500

    if task == HUMOR:
        db, label_mapping, target_col = db_for_binary_humor_target(db)
    elif task == DIRECTION:
        db, label_mapping, target_col = db_for_binary_dimension_discrimination(db, 'direction')
    elif task == SENTIMENT:
        db, label_mapping, target_col = db_for_binary_dimension_discrimination(db, 'sentiment')
    else:
        raise NotImplementedError('Invalid task')

    return db, label_mapping, target_col


def stratify(X, lengths, y, idxs, seed, train_size=0.8):
    # stratify per coach
    train_X = []
    train_lengths = []
    train_y = []
    test_X = []
    test_lengths = []
    test_y = []
    for c_start, c_end in zip(idxs[:-1], idxs[1:]):
        coach_X = X[c_start:c_end]
        coach_lengths = lengths[c_start:c_end]
        coach_y = y[c_start:c_end]

        splitter = StratifiedShuffleSplit(n_splits=1, random_state=seed, train_size=train_size)
        splitted = splitter.split(coach_X, coach_y)

        coach_train_idxs, coach_test_idxs = next(iter(splitted))
        train_X.append(coach_X[coach_train_idxs])
        train_lengths.append(coach_lengths[coach_train_idxs])
        train_y.append(coach_y[coach_train_idxs])
        test_X.append(coach_X[coach_test_idxs])
        test_lengths.append(coach_lengths[coach_test_idxs])
        test_y.append(coach_y[coach_test_idxs])

    train_X = np.concatenate(train_X)
    train_lengths = np.concatenate(train_lengths)
    train_y = np.concatenate(train_y)
    test_X = np.concatenate(test_X)
    test_lengths = np.concatenate(test_lengths)
    test_y = np.concatenate(test_y)

    return (train_X, train_lengths, train_y), (test_X, test_lengths, test_y)


def stratify_mm(train_features, train_labels, seed, train_size=0.8):
    # stratify per coach
    Xs1 = []
    Xs2 = []
    ys1 = []
    ys2 = []

    for i in range(len(train_labels)):
        # create pseudo classes: humour in segment or not?
        segments = np.array(sorted(list(set(train_labels[i].keys()))))
        pseudo_y = [np.sum(train_labels[i][seg]) > 0 for seg in segments]
        splitter = StratifiedShuffleSplit(n_splits=1, random_state=seed, train_size=train_size)
        splitted = splitter.split(list(range(len(pseudo_y))), pseudo_y)
        coach_train_idxs, coach_test_idxs = next(iter(splitted))
        coach_train_segments = segments[coach_train_idxs]
        coach_test_segments = segments[coach_test_idxs]
        x_dct_train = {}
        x_dct_test = {}
        for m in ['v', 'a', 't']:
            x_dct_train[m] = {str(seg): train_features[m][i][str(seg)] for seg in coach_train_segments}
            x_dct_test[m] = {str(seg): train_features[m][i][str(seg)] for seg in coach_test_segments}
        Xs1.append(x_dct_train)
        Xs2.append(x_dct_test)
        ys1.append({str(seg): train_labels[i][str(seg)] for seg in coach_train_segments})
        ys2.append({str(seg): train_labels[i][str(seg)] for seg in coach_test_segments})
        print(splitted)

    return (Xs1, ys1), (Xs2, ys2)





