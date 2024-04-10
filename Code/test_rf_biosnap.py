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
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score

from GeNNius import Model, EarlyStopper, shuffle_label_data
from utils import plot_auc

from HG_data import Graph

import torch
from collections import defaultdict
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
from sklearn.ensemble import RandomForestClassifier



import random

import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HGTConv, Linear, SAGEConv, GATConv, GCNConv


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = 'Data/BIOSNAP/hetero_data_biosnap.pt'
data = torch.load(path)
data = T.ToUndirected()(data)

print('='*100)
print('data:', data)
print('='*100)

import random
random.seed(42)
# torch.manual_seed(42)

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

train_data, val_data, test_data = transform(data)
print('*'*100)
print('train data:', train_data)
print('*'*100)
print('val data:', val_data)
print('*'*100)
print('test data:', test_data)
print('*'*100)


def data_divide(data):
    labels = data[('drug','interaction','protein')].edge_label.to(device)
    edge_label_index_dict = {('drug','interaction','protein'): data[('drug','interaction','protein')].edge_label_index.to(device) }

    # 预先提取边的索引
    edge_indices = edge_label_index_dict[('drug','interaction','protein')].t()

    # 初始化一个空的列表来存储边特征
    edge_x_list = []

    # 遍历所有边的索引
    for edge_idx in edge_indices:
        # 从train_data中提取边的特征并拼接
        edge_idx_x = torch.cat((data['drug'].x[edge_idx[0]], data['protein'].x[edge_idx[1]]), dim=0)
        edge_x_list.append(edge_idx_x)
    # 将边特征列表转换为张量
    edge_x = torch.stack(edge_x_list, dim=0)

    print('edge_x:',edge_x.shape)
    print('labels:',labels.shape)

    return edge_x, labels
    
train_edge_x, train_y = data_divide(train_data)
val_edge_x, val_y = data_divide(val_data)
test_edge_x, test_y = data_divide(test_data)

train_edge_x, train_y = train_edge_x.cpu(), train_y.cpu()
val_edge_x, val_y = val_edge_x.cpu(), val_y.cpu()
test_edge_x, test_y = test_edge_x.cpu(), test_y.cpu()

# 初始化逻辑回归模型
model = RandomForestClassifier()

acc_list,auc_list,pre_list = [],[],[]
run_time = 10

for i in range(run_time):
    # 测试模型
    model.fit(train_edge_x, train_y)

    # 进行预测
    y_pred_proba = model.predict_proba(test_edge_x)[:, 1]
    y_pred = model.predict(test_edge_x)

    # 计算AUC
    auc = roc_auc_score(test_y, y_pred_proba)
    auc_list.append(auc)

    # 计算Precision
    precision = precision_score(test_y, y_pred)
    pre_list.append(precision)

    # 计算Accuracy
    accuracy = accuracy_score(test_y, y_pred)
    acc_list.append(accuracy)


print(f'RF model: Avg Test Accuracy: {sum(acc_list)/len(acc_list):.4f}',f' Avg Test AUC: {sum(auc_list)/len(auc_list):.4f}',f' Avg Test AUC: {sum(pre_list)/len(pre_list):.4f}')




