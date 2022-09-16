#import torch
import os
from functools import partial
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import recall_score, roc_auc_score, f1_score

COACHES = ['baum', 'breitenreiter', 'hasenhuttl', 'hecking', 'herrlich', 'kovac', 'nagelsmann', 'schwarz','streich',
           'tedesco']

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
DATA_DIR = os.path.join(ROOT_DIR, 'data')

FEATURE_DIR = os.path.join(ROOT_DIR, 'features')

CACHE_DIR = os.path.join(ROOT_DIR, 'cache')
FEATURE_CACHE_DIR = os.path.join(CACHE_DIR, 'features')

PREDICTIONS_DIR = os.path.join(ROOT_DIR, 'predictions')

RESULT_DIR = os.path.join(ROOT_DIR, 'results')

LABEL_DIR = os.path.join(ROOT_DIR, 'labels')

CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')

# TASKS
HUMOR = 'humor'
SENTIMENT = 'sentiment'
DIRECTION = 'direction'