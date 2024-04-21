import os
import numpy as np
import logging
import argparse
import time

import random
import time
import argparse
import dill as pickle 
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score, average_precision_score

from GeNNius import Model, EarlyStopper, shuffle_label_data
from utils import plot_auc

from HG_data import Graph
import numpy as np
import torch
from collections import defaultdict
import resource
import pandas as pd
from HG_model import GNN, GNN_from_raw
from HG_utils import sub_sample1
from sklearn.cluster import KMeans
from scipy.sparse import coo_matrix
from warnings import filterwarnings
filterwarnings("ignore")


seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import torch
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv

# from torch_geometric.utils import accuracy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


import random

import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HGTConv, Linear



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = '../Data/GPCR/hetero_data_gpcr.pt'
data = torch.load(path)
data['drug']['node_id'] = torch.tensor([i for i in range(data['drug'].x.shape[0])])
data['protein']['node_id'] = torch.tensor([i for i in range(data['protein'].x.shape[0])])
data = T.ToUndirected()(data)


# dataname = 'bindingdb'
# dataname = 'drugbank'
# dataname = 'biosnap'
dataname = 'gpcr'

print('='*100)
print('data:', data)
print('='*100)

import random
random.seed(42)
torch.manual_seed(42)
# del data['protein', 'rev_interaction', 'drug'].edge_label 
transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.2,
    is_undirected=True,
    disjoint_train_ratio=0.2,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=True,
    edge_types=("drug", "interaction", "protein"),
    rev_edge_types=("protein", "rev_interaction", "drug"), 
    split_labels=False
)

# train_data, val_data, test_data = transform(data)
# print('*'*100)
# print('train data:', train_data)
# print('*'*100)
# print('val data:', val_data)
# print('*'*100)
# print('test data:', test_data)
# print('*'*100)


# 假设N和M分别是drug和protein的节点数量
N = data['drug']['node_id'].shape[0]
M = data['protein']['node_id'].shape[0]

# 创建空的邻接矩阵
adj_matrix = np.zeros((N+M, N+M))

# 根据边缘索引设置邻接矩阵的值
edge_index1 = data[('drug','interaction','protein')].edge_index
edge_index2 = data[('protein', 'rev_interaction', 'drug')].edge_index
for i in range(edge_index1.shape[1]):
    src = edge_index1[0, i]
    dst = N + edge_index1[1, i]  # protein节点索引需要偏移
    adj_matrix[src, dst] = 1
    adj_matrix[dst, src] = 1  # 无向图，所以需要设置对称位置的值

# for i in range(edge_index2.shape[1]):
#     src = edge_index2[0, i]
#     dst = N + edge_index2[1, i]  # protein节点索引需要偏移
#     adj_matrix[src, dst] = 1
#     adj_matrix[dst, src] = 1  # 无向图，所以需要设置对称位置的值

# 将邻接矩阵保存为txt文件
np.savetxt('../res/'+dataname+'_adj_matrix.txt', adj_matrix, fmt='%d')