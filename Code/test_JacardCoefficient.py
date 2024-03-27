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

path = 'Data/DRUGBANK/hetero_data_drugbank.pt'
data = torch.load(path)
data = T.ToUndirected()(data)

print('='*100)
print('data:', data)
print('='*100)

import random
random.seed(42)

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
    edge_indices = edge_label_index_dict[('drug','interaction','protein')]

    
    print('labels:',labels.shape)

    return labels, edge_indices
    
train_y, train_edge_indices = data_divide(train_data)
val_y, val_edge_indices = data_divide(val_data)
test_y, test_edge_indices = data_divide(test_data)

train_y = train_y.cpu()
val_y = val_y.cpu()
test_y = test_y.cpu()

train_edge_idx_list, val_edge_idx_list, test_edge_idx_list = train_edge_indices.cpu().numpy().tolist(), val_edge_indices.cpu().numpy().tolist(), test_edge_indices.cpu().numpy().tolist()

import networkx as nx

def transform_list(b, a):
    # 找到第一个列表中的最大值
    max_value = max(a)
    
    # 如果第二个列表的长度小于第一个列表中最大值的索引+1，则增加列表的长度
    while len(b) < max_value:
        b.append(max(b) + 1)
    
    return b

def hetero_data_to_nx_graph(pyg_data):
    """
    将 PyG 中的 HeteroData 对象转换为 NetworkX 图对象。

    参数:
        pyg_data (HeteroData): PyG 中的 HeteroData 对象.

    返回:
        G (nx.Graph): 转换后的 NetworkX 图对象.
    """
    # 创建一个空的 NetworkX 图
    G = nx.Graph()

    # 添加节点和边
    drug_nodes = pyg_data['drug'].node_id.cpu().tolist()
    protein_nodes = pyg_data['protein'].node_id.cpu().tolist()
    protein_nodes = transform_list(protein_nodes, drug_nodes)
    edges = pyg_data['drug', 'interaction', 'protein'].edge_index.t().cpu().tolist()

    G.add_nodes_from(drug_nodes, node_type='drug')
    G.add_nodes_from(protein_nodes, node_type='protein')
    G.add_edges_from(edges)

    # 将节点和边的属性从 HeteroData 中复制到 NetworkX 图中
    drug_features = {node: feature.numpy() for node, feature in zip(pyg_data['drug'].node_id.cpu(), pyg_data['drug'].x.cpu())}
    protein_features = {node: feature.numpy() for node, feature in zip(pyg_data['protein'].node_id.cpu(), pyg_data['protein'].x.cpu())}

    nx.set_node_attributes(G, drug_features, 'drug_features')
    nx.set_node_attributes(G, protein_features, 'protein_features')

    return G

# 使用示例
# train_g = hetero_data_to_nx_graph(train_data)
# test_g = hetero_data_to_nx_graph(test_data)
# print(nx.info(train_g), nx.info(test_g))
g = hetero_data_to_nx_graph(data)
print('graph info:', len(g.nodes()), len(g.edges()))

# print(train_edge_idx_list)


acc_list,auc_list,pre_list = [],[],[]
run_time = 10

for i in range(run_time):
    test_preds = []
    for j in range(len(train_edge_idx_list)):
        preds = nx.jaccard_coefficient(g,[tuple(train_edge_idx_list[j][0]),tuple(train_edge_idx_list[j][1])])
        for u,v,p in preds:
            test_preds.append(p)
    
    print(test_preds,len(test_preds))
    # 计算AUC
    y_pred_binary = int(test_preds)
    auc = roc_auc_score(test_y, y_pred_binary)
    auc_list.append(auc)

    # 计算Precision
    precision = precision_score(test_y, y_pred_binary)
    pre_list.append(precision)

    # 计算Accuracy
    accuracy = accuracy_score(test_y, y_pred_binary)
    acc_list.append(accuracy)


print(f'Jacard Coefficient Metric: Avg Test Accuracy: {sum(acc_list)/len(acc_list):.4f}',f' Avg Test AUC: {sum(auc_list)/len(auc_list):.4f}',f' Avg Test AUC: {sum(pre_list)/len(pre_list):.4f}')




