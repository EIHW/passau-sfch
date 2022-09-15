import os
import sys
from argparse import ArgumentParser, Namespace
from collections import Counter
from itertools import product
from time import ctime, time
from typing import Tuple

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from data_utils.features import load_whole_ds, loo_from_whole_ds
from data_utils.labels import load_task_data, stratify
from data_utils.log import unflatten_dict, summarize_results
from global_vars import HUMOR, DIRECTION, SENTIMENT, RESULT_DIR, COACHES, device, PREDICTIONS_DIR, CHECKPOINT_DIR
from models.rnn import GRUClassifier
import numpy as np
import json

import pandas as pd

from train_eval.evaluation import evaluate_cont

from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--name', default=None, type=str)
    parser.add_argument('--task', required=True, choices=[HUMOR, DIRECTION, SENTIMENT])
    parser.add_argument('--summarization_keys', required=False, default=['test.auc.mean'], nargs='+',
                        help='Create excel files for the corresponding results')
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--save_checkpoints', action='store_true')

    ##### FEATURE PARAMETERS #######
    parser.add_argument('--features', nargs='+', default=['egemaps', 'ds', 'bert-4-sentence-level', 'faus'])
    parser.add_argument('--normalize', nargs='+', default=None, type=int, choices=[0,1],
                        help='Specify which features should be normalized. 1 for normalization, 0 for no normalization. '
                             'Must have the same number of arguments as --features.')

    ##### HYPERPARAMETERS #######
    # lstm parameters
    parser.add_argument('--rnn_hidden_dims', type=int, nargs='+', default=[32, 64, 128, 256])
    parser.add_argument('--rnn_num_layers', type=int, nargs='+', default=[1,2,4,8])
    parser.add_argument('--directions', type=str, nargs='+', choices=['uni', 'bi'], default=['uni'])
    # training parameters
    parser.add_argument('--lr', type=float, nargs='+', default=[0.01, 0.001, 0.005, 0.0001, 0.0005])
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--seed', type=int, default = 101)
    parser.add_argument('--rnn_dropout', type=float, nargs='+', default=[0.])
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--devtest_batch_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=3)

    ##### FOR CHECKPOINT EVALUATION ####
    parser.add_argument('--eval_cp_only', action='store_true')
    parser.add_argument('--cp_dir', type=str, required=False,
                        help='Directory relative to CHECKPOINTS_DIR/{task} containing the {coach}_left_out.pickle files')
    parser.add_argument('--cp_eval_json', type=str, required=False, help='JSON file in which to store the checkpoint'
                                                                         'evaluation results.')


    args = parser.parse_args()

    assert (args.normalize is None or len(args.normalize) == len(args.features))
    if args.normalize is None:
        args.normalize = [False for _ in args.features]
    else:
        args.normalize = [n > 0 for n in args.normalize]

    if args.name is None:
        args.name = args.task
    args.name += str(ctime(time())).replace(":","-").replace(" ", "_")

    assert len(args.directions) >= 1
    args.directions = list(set(args.directions))

    return args


class HumorDataset(Dataset):

    def __init__(self, X:np.ndarray, lengths:np.ndarray, y:np.ndarray):
        '''

        :param X: shape (num_examples, seq_len, feature_dim)
        :param lengths: shape (num_examples,)
        :param y: shape (num_examples,)
        '''
        self.X = X.astype(np.float32)
        self.lengths = lengths
        self.y = y
        # y is not necessarily 0,1,2,..
        self.sorted_labels = sorted(set(self.y.tolist()))
        label_mapping = {l:i for i,l in enumerate(self.sorted_labels)}
        self.original_label = {i:l for l,i in label_mapping.items()}
        self.y = np.array([label_mapping[value] for value in self.y]).astype(np.float32)
        self.num_classes = len(self.sorted_labels)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.lengths[item], self.y[item]


def init_model(params:Namespace, seed:int, num_classes:int=2):
    torch.manual_seed(seed)
    model = GRUClassifier(params=params, num_classes=num_classes)
    model.to(device)
    return model


def init_weights(y:np.ndarray):
    '''

    Balances the classes
    :param y: shape (num_examples,)
    :return:
    '''
    counts = {cls:cnt for (cls,cnt) in Counter(y).most_common()}
    classes = sorted(list(np.unique(y)))
    return torch.tensor([y.shape[0]/counts[classes[i]] for i in range(len(classes))])



def training_epoch(model, optimizer, train_loader, loss_fn):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        X, X_lengths, y = batch
        logits = model(X.to(device), X_lengths.to(device)).squeeze()
        loss = loss_fn(logits, y.to(device))
        loss.backward()
        optimizer.step()
    return model

def extract_logits_and_gs(model:GRUClassifier, val_loader:DataLoader):
    '''

    Extract both model predictions (logits) and gold standard values, given a DataLoader
    :param model: model
    :param val_loader: loader
    :return: tensor of logits, numpy array of gold standard values
    '''
    model.eval()

    all_logits = []
    all_ys = []

    with torch.no_grad():
        for batch in val_loader:
            X, X_lengths, y = batch
            logits = model(X.to(device), X_lengths.to(device))
            all_logits.append(logits.detach())
            all_ys.append(y.detach().cpu().numpy())

    all_logits = torch.cat(all_logits)
    all_ys = np.concatenate(all_ys)

    return all_logits, all_ys


def val_score(model:GRUClassifier, val_loader:DataLoader) -> float:
    '''

    Calculates ROC-AUC for the predictions of a model. Used to get a validation score
    :param model: model to evaluate
    :param val_loader: evaluation data
    :return: ROC-AUC score
    '''
    model.eval()
    sigmoid = nn.Sigmoid()

    all_logits, all_ys = extract_logits_and_gs(model, val_loader)

    predictions = sigmoid(all_logits)

    predictions = predictions.squeeze().cpu().numpy()

    return roc_auc_score(all_ys.astype(np.int32), predictions)


def test_evaluation(model:GRUClassifier, test_loader:DataLoader) -> dict:
    '''

    Calculates test metrics for the predictions of a model. Currently identical to evaluation on the validation set
    (ROC-AUC), but allows for computing several metrics.
    :param model: model to evaluate
    :param test_loader: test data
    :return: dictionary of metrics and their value
    '''
    model.eval()
    sigmoid = nn.Sigmoid()
    all_logits, all_ys = extract_logits_and_gs(model, test_loader)

    cont_predictions = sigmoid(all_logits).detach().cpu().numpy()

    return evaluate_cont(all_ys, cont_predictions)


def extract_predictions(model:GRUClassifier, test_loader:DataLoader) -> dict:
    '''

    Extracts dictionary of actual predictions (sigmoid applied to logits)
    :param model: model to extract predictions of
    :param test_loader: DataLoader
    :return: prediction dict (can be used to immediately create a DataFrame)
    '''
    model.eval()
    sigmoid = nn.Sigmoid()
    all_logits, all_ys = extract_logits_and_gs(model, test_loader)

    predictions = sigmoid(all_logits).detach().cpu().numpy()

    if len(predictions.shape) == 1:
        predictions = np.expand_dims(predictions, -1)
    return {f'pred_{i}':predictions[:,i] for i in range(predictions.shape[1])}


def training(model:GRUClassifier, optimizer:torch.optim.Optimizer, loss_fn, epochs, patience, train_loader, val_loader)\
        -> Tuple[float, GRUClassifier]:
    '''

    Training routine
    :return: (best validation score, updated model)
    '''
    best_quality = -1234.
    patience_counter = 0
    best_state_dict = None

    for epoch in range(1, epochs+1):
        model = training_epoch(model, optimizer, train_loader=train_loader, loss_fn=loss_fn)
        epoch_result = val_score(model, val_loader)

        if epoch_result > best_quality:
            best_quality = epoch_result
            patience_counter = 0
            best_state_dict = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter > patience:
                break

    model.load_state_dict(best_state_dict)
    return best_quality, model


def checkpoint_eval_only(db, config, cp_file_dir:str, feature:str, normalize:bool, target:str, target_json:str):
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
    X, lengths, y, coach_idxs = load_whole_ds(db, feature, target, normalize=normalize, reduction=None)
    config.feature_dim = X.shape[-1]
    config.dropout = args.dropout

    res_dct = {}
    for i, coach in tqdm(list(enumerate(COACHES))):
        _, _, (test_X, test_lengths, test_y) = loo_from_whole_ds(X, lengths, y, coach_idxs, leave_out=i)
        test_loader = DataLoader(HumorDataset(test_X, test_lengths, test_y), batch_size=args.devtest_batch_size,
                                 shuffle=False)

        model = init_model(config, seed=1234, num_classes=2)
        state_dict = torch.load(os.path.join(cp_file_dir, f'{coach}_left_out.pt'), map_location=device)
        model.load_state_dict(state_dict)

        res_dct[coach] = test_evaluation(model, test_loader)

    os.makedirs(os.path.dirname(target_json), exist_ok=True)
    json.dump(res_dct, open(target_json, 'w+'))


if __name__ == '__main__':
    args = parse_args()
    db, label_mapping, target_col = load_task_data(task=args.task)

    configurations = [
        Namespace(**{'rnn_hidden_dim': c[0], 'rnn_num_layers': c[1], 'bidirectional': c[2] == 'bi', 'lr': c[3],
                     'rnn_dropout': c[4]}) for c in
        product(args.rnn_hidden_dims, args.rnn_num_layers, args.directions, args.lr,
                args.rnn_dropout)]

    if args.eval_cp_only:
        assert len(args.features) == 1, "When evaluating a checkpoint, you may only specify one feature"
        assert len(configurations) == 1, "When evaluating a checkpoint, please only specify one configuration"

        cp_dir = os.path.join(CHECKPOINT_DIR, args.task, args.cp_dir)
        checkpoint_eval_only(db, config=configurations[0], cp_file_dir=cp_dir, feature=args.features[0],
                             normalize=args.normalize[0], target= target_col, target_json=args.cp_eval_json)
        sys.exit(0)

    experiment_str = os.path.join(args.task, 'rnn')

    res_dir = os.path.join(RESULT_DIR, experiment_str, args.name)
    os.makedirs(res_dir, exist_ok=True)

    result_json = os.path.join(res_dir, 'all_results.json')
    coach_excels = [os.path.join(res_dir, f'{key}_coaches.xlsx') for key in args.summarization_keys]
    feature_excels = [os.path.join(res_dir, f'{key}_features.xlsx') for key in args.summarization_keys]

    pred_dir = os.path.join(PREDICTIONS_DIR, experiment_str, args.name)

    sorted_label_names = [label_mapping[k] for k in sorted(list(label_mapping.keys()))]

    scoring = roc_auc_score

    res_dict = {'config': {'params': {k:vars(args)[k] for k in ['rnn_hidden_dims', 'rnn_num_layers', 'directions',
                                                                'lr', 'rnn_dropout']},
                           'target': target_col,
                           'features': [f'{f} (norm={n})' for f, n in zip(args.features, args.normalize)],
                           'cli_args': vars(args)},
                'results': {}}

    # grid search
    for feature in args.features:
        print(f'---{feature}')
        feature_dict = {}

        X, lengths, y, coach_idxs = load_whole_ds(db, feature, target_col, normalize=args.normalize, reduction=None)
        num_classes = 2

        configurations = [Namespace(**{'rnn_hidden_dim': c[0], 'rnn_num_layers': c[1], 'bidirectional': c[2]=='bi', 'lr': c[3],
                                       'rnn_dropout':c[4]}) for c in
                          product(args.rnn_hidden_dims, args.rnn_num_layers, args.directions, args.lr,
                                  args.rnn_dropout)]
        for config in configurations:
            config.feature_dim = X.shape[-1]
            config.dropout = args.dropout

        best_config = configurations[0]
        best_score = 0

        if len(configurations) > 1:
            print(f"Starting gridsearch for {len(configurations)} configurations")
            for n,config in tqdm(list(enumerate(configurations))):
                #print(f'Hyperparameter Search: Training {n+1} of {len(configurations)} configurations')
                seed_scores = []
                for seed in range(args.seed, args.seed+args.num_seeds):
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    # folds
                    coach_scores = []
                    for i in tqdm(range(len(COACHES))):
                        (train_X, train_lengths, train_y),_, (val_X, val_lengths, val_y) = \
                            loo_from_whole_ds(X, lengths, y, coach_idxs, leave_out=i)
                        train_loader = DataLoader(HumorDataset(train_X, train_lengths, train_y), batch_size=args.train_batch_size,
                                                  shuffle=True)
                        val_loader = DataLoader(HumorDataset(val_X, val_lengths, val_y), batch_size=args.devtest_batch_size,
                                                shuffle=False)
                        model = init_model(config, seed, num_classes)

                        optimizer = AdamW(lr = config.lr, params=model.parameters())

                        weights = init_weights(train_y, config.weights).float()

                        # make sure negative class has weight 1
                        weights = weights / weights[0]
                        loss_fn = BCEWithLogitsLoss(pos_weight=weights[1])

                        coach_scores.append(training(model, optimizer, loss_fn, epochs=args.epochs, patience=args.patience,
                                                     train_loader=train_loader, val_loader=val_loader)[0])
                    seed_scores.append(np.mean(coach_scores))

                config_score = np.mean(seed_scores)
                if config_score > best_score:
                    best_score = config_score
                    best_config = config

            feature_dict['best_params'] = vars(best_config)

        else:
            print('Only one config given, no hyperparameter search.')
        # load best config and test
        print()
        if len(configurations) > 1:
            print(f'{ctime(time())}: finished gridsearch - starting eval')
        # else:
        #     print('Train/test routine for every coach.')

        coach_results = {coach:{'test':{}} for coach in COACHES}

        if args.save_predictions:
            feature_pred_dir = os.path.join(pred_dir, feature)
            os.makedirs(feature_pred_dir, exist_ok=True)

        if args.save_checkpoints:
            cp_dir = os.path.join(CHECKPOINT_DIR, experiment_str, args.name)
            os.makedirs(cp_dir, exist_ok=True)

        print(f'Training {"best" if len(configurations) > 1 else "given"} configuration (leave-one-out scenario)')
        for i,coach in tqdm(list(enumerate(COACHES))):
            (train_X, train_lengths, train_y), tr_coach_idxs, (test_X, test_lengths, test_y) = \
                loo_from_whole_ds(X, lengths, y, coach_idxs, leave_out=i)
            test_loader = DataLoader(HumorDataset(test_X, test_lengths, test_y), batch_size=args.devtest_batch_size,
                                        shuffle=False)
            label_counts = Counter(test_y)
            coach_results[coach]['stats'] = {label_mapping[l]:label_counts[l] for l in label_counts.keys()}
            seed_scores = []

            # which of the n_seeds models for prediction?
            best_test_seed_quality = -1234
            best_test_seed = None
            best_state_dict = None
            test_seed_predictions = {s:None for s in range(args.seed, args.seed+args.num_seeds)}

            for seed in range(args.seed, args.seed+args.num_seeds):
                # stratified sampling to obtain a development set
                (s_train_X, s_train_lengths, s_train_y), (s_val_X, s_val_lengths, s_val_y) = stratify(train_X, train_lengths,
                                                                                          train_y, tr_coach_idxs, seed=seed)
                train_loader = DataLoader(HumorDataset(s_train_X, s_train_lengths, s_train_y),
                                          batch_size=args.train_batch_size,
                                          shuffle=True)
                val_loader = DataLoader(HumorDataset(s_val_X, s_val_lengths, s_val_y), batch_size=args.devtest_batch_size,
                                        shuffle=False)
                model = init_model(best_config, seed, num_classes)

                optimizer = AdamW(lr=best_config.lr, params=model.parameters())

                weights = init_weights(s_train_y).float()
                if num_classes > 2:
                    loss_fn = CrossEntropyLoss(weight=weights.to(device))
                else:
                    # make sure negative class gets weight 1
                    weights = weights / weights[0]
                    loss_fn = BCEWithLogitsLoss(pos_weight=weights[1])

                q, model = training(model, optimizer, loss_fn, epochs=args.epochs, patience=args.patience,
                                             train_loader=train_loader, val_loader=val_loader)
                seed_scores.append(test_evaluation(model, test_loader))
                if q > best_test_seed_quality:
                    best_test_seed_quality = q
                    best_test_seed = seed
                    best_state_dict = model.state_dict()

            coach_results[coach]['best_val'] = best_test_seed_quality
            sss_as_lists = {k:[s[k] for s in seed_scores] for k in seed_scores[0].keys()}
            for k in seed_scores[0].keys():
                coach_results[coach]['test'].update({f'{k}.mean': np.mean(sss_as_lists[k]),
                                             f'{k}.std':np.std(sss_as_lists[k])})
            feature_dict['results'] = unflatten_dict(coach_results)
            res_dict['results'][feature] = feature_dict

            # in case predictions should be saved
            if args.save_predictions:
                prediction_csv = os.path.join(feature_pred_dir, f'{coach}.csv')
                model.load_state_dict(best_state_dict)
                test_predictions = extract_predictions(model, test_loader)
                pd.DataFrame(test_predictions).to_csv(prediction_csv, index=False)

            if args.save_checkpoints:
                cp_file = os.path.join(cp_dir, f'{coach}_left_out.pt')
                torch.save(best_state_dict, cp_file)

        json.dump(res_dict, open(result_json, 'w+'))

    for i,summarization_key in enumerate(args.summarization_keys):
        summarize_results(res_dict, coach_target_file=coach_excels[i], feature_target_file=feature_excels[i],
                          key=summarization_key)