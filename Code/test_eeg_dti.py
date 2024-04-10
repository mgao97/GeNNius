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


seed = 42
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

parser = argparse.ArgumentParser(description='Training GNN on gene cell graph')
parser.add_argument('--data_path', type=str)
parser.add_argument('--epoch', type=int, default=100)
# sampling times
parser.add_argument('--n_batch', type=int, default=25,
                    help='Number of batch (sampled graphs) for each epoch')

parser.add_argument('--drug_rate', type=float, default=0.9)
parser.add_argument('--protein_rate', type=float, default=0.3)

# Result
parser.add_argument('--data_name', type=str,
                    help='The name for dataset')
parser.add_argument('--result_dir', type=str,
                    help='The address for storing the models and optimization results.')
parser.add_argument('--reduction', type=str, default='raw',
                    help='the method for feature extraction, pca, raw, AE')
parser.add_argument('--in_dim', type=int, default=256,
                    help='Number of hidden dimension (AE)')
# GAE
parser.add_argument('--n_hid', type=int,default=64,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int,default=4,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=2,
                    help='Number of GNN layers')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout ratio')
parser.add_argument('--lr', type=float,default=0.01,
                    help='learning rate')

parser.add_argument('--batch_size', type=int,default=16,
                    help='Number of output nodes for training')
parser.add_argument('--layer_type', type=str, default='hgt',
                    help='the layer type for GAE')
parser.add_argument('--loss', type=str, default='kl',
                    help='the loss for GAE')
parser.add_argument('--factor', type=float, default='0.5',
                    help='the attenuation factor')
parser.add_argument('--patience', type=int, default=5,
                    help='patience')
parser.add_argument('--rf', type=float, default='0.0',
                    help='the weights of regularization')
parser.add_argument('--cuda', type=int, default=0,
                    help='cuda 0 use GPU0 else cpu ')
parser.add_argument('--rep', type=str, default='T',
                    help='precision truncation')
parser.add_argument('--AEtype', type=int, default=1,
                    help='AEtype:1 embedding node autoencoder 2:HGT node autoencode')
parser.add_argument('--optimizer', type=str, default='adamw',
                    help='optimizer')

args = parser.parse_args()




import random

import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HGTConv, Linear, SAGEConv, GATConv, GCNConv

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in ['drug', 'protein']:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GCNConv(hidden_channels, hidden_channels)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index, edge_label_index_dict):

        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }
        # print('x_dict:', x_dict)

        x = torch.cat((x_dict['drug'], x_dict['protein']),dim=0).to(device)

        for conv in self.convs:
            x = conv(x, edge_index).relu()

        x_drug = self.lin(x[edge_label_index_dict[('drug','interaction','protein')][0]]).squeeze()
        x_protein = self.lin(x[edge_label_index_dict[('drug','interaction','protein')][1]]).squeeze()

        # print(x_drug, x_protein)
        # print('%'*100)
        

        return torch.sigmoid(x_drug * x_protein.t())


    
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score

def train(model, data, optimizer, device):
    model.train()

    # 获取训练数据
    x_dict = {'drug': data['drug'].x.to(device), 'protein': data['protein'].x.to(device)}
    # edge_index_dict = {('drug','interaction','protein'): data[('drug','interaction','protein')].edge_index.to(device)}
    #                 #    ('protein','rev_interaction','drug'): data[('protein','rev_interaction','drug')].edge_index.to(device)}
    labels = data[('drug','interaction','protein')].edge_label.to(device)
    edge_label_index_dict = {('drug','interaction','protein'): data[('drug','interaction','protein')].edge_label_index.to(device) }
                    #    ('protein','rev_interaction','drug'): data[('protein','rev_interaction','drug')].edge_label_index.to(device)}
    
    
    edge_index = data[('drug','interaction','protein')].edge_index.to(device)
    # 清零梯度
    optimizer.zero_grad()
    
    # 前向传播
    output = model(x_dict, edge_index, edge_label_index_dict).squeeze()
    
    # 计算损失
    loss = F.binary_cross_entropy(output, labels.float())

    # 反向传播
    loss.backward()
    optimizer.step()

    return loss.item()

def test(model, data, device):
    model.eval()

    # 获取测试数据
    x_dict = {'drug': data['drug'].x.to(device), 'protein': data['protein'].x.to(device)}
    # edge_index_dict = {('drug','interaction','protein'): data[('drug','interaction','protein')].edge_index.to(device)}
    #                 #    ('protein','rev_interaction','drug'): data[('protein','rev_interaction','drug')].edge_index.to(device)}
    labels = data[('drug','interaction','protein')].edge_label.to(device)
    edge_label_index_dict = {('drug','interaction','protein'): data[('drug','interaction','protein')].edge_label_index.to(device) }
    # X = torch.cat((data['drug'].x,data['prptein'].x),dim=0).to(device)
    edge_index = data[('drug','interaction','protein')].edge_index.to(device)
    # 前向传播
    with torch.no_grad():
        output = model(x_dict, edge_index, edge_label_index_dict).squeeze()

    # 计算预测结果
    pred = (output >= 0.5).long()

    # 计算准确率
    accuracy = accuracy_score(labels.cpu(), pred.cpu())
    auc = roc_auc_score(labels.cpu(), output.cpu())
    precision = average_precision_score(labels.cpu(), output.cpu())

    return accuracy, auc,precision




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



# 定义模型参数
hidden_channels = 64
out_channels = 1
num_heads = 2
num_layers = 3


# 初始化模型
model = GNN(hidden_channels, out_channels, num_layers).to(device)
print('model:',model)
# 定义优化器
optimizer = Adam(model.parameters(), lr=0.05)

# 训练模型
for epoch in range(1,1001):  # 假设训练10个epoch
    loss = train(model, train_data, optimizer, device)
    if epoch % 50 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss:.4f}')



acc_list,auc_list,pre_list = [],[],[]
run_time = 10

for i in range(run_time):
    # 测试模型
    accuracy, auc, pre = test(model, test_data, device)
    acc_list.append(accuracy)
    auc_list.append(auc)
    pre_list.append(pre)


print(f'avg Test Accuracy: {sum(acc_list)/len(acc_list):.4f}',f' avg Test AUC: {sum(auc_list)/len(auc_list):.4f}',f' avg Test AUC: {sum(pre_list)/len(pre_list):.4f}')





