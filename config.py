from transformers import AutoTokenizer, AutoModel
import vocab
import os
import torch
import logging
import numpy as np
import json
import functools
import random
import operator
import multiprocessing
from sklearn.metrics import f1_score
import copy
from tqdm import tqdm
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    TensorDataset
)
from crf import *
from collections import OrderedDict as odict
import pickle
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
logging.basicConfig(level=logging.INFO)

CONFIG = {
    'bert_path': 'roberta-base',
    'epochs' : 20,
    'lr' : 1e-4,
    'ptmlr' : 5e-6,
    'batch_size' : 1,
    'max_len' : 256,
    'max_value_list' : 16,
    'bert_dim' : 1024,
    'pad_value' : 1,
    'shift' : 1024,
    'dropout' : 0.3,
    'p_unk': 0.1,
    'data_splits' : 20,
    'num_classes' : 7,
    'wp' : 1,
    'wp_pretrain' : 5,
    'data_path' : './MELD/data/MELD/',
    'accumulation_steps' : 8,
    'rnn_layers' : 2,
    'tf_rate': 0.8,
    'aux_loss_weight': 0.3,
    'ngpus' : torch.cuda.device_count(),
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
}

tokenizer = AutoTokenizer.from_pretrained(CONFIG['bert_path'])
_special_tokens_ids = tokenizer('')['input_ids']
CLS = _special_tokens_ids[0]
SEP = _special_tokens_ids[1]
CONFIG['CLS'] = CLS
CONFIG['SEP'] = SEP
