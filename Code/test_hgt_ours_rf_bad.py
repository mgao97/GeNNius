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
# from HG_model_emb import GNN, GNN_from_raw
from HG_utils import sub_sample1
from sklearn.cluster import KMeans
from scipy.sparse import coo_matrix
from warnings import filterwarnings
filterwarnings("ignore")


import torch
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv

# from torch_geometric.utils import accuracy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


    
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score,precision_score,roc_auc_score

import random
import os.path as osp

import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HGTConv, Linear
from sklearn.ensemble import RandomForestClassifier

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in ['drug', 'protein']:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels*2, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }


        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict, torch.sigmoid(self.lin(torch.cat([x_dict['drug'][edge_label_index_dict[('drug','interaction','protein')][0]], x_dict['protein'][edge_label_index_dict[('drug','interaction','protein')][1]]], dim=1)))
        # return torch.sigmoid(self.lin(torch.cat([x_dict['drug'][edge_label_index_dict[('drug','interaction','protein')][0]], x_dict['protein'][edge_label_index_dict[('drug','interaction','protein')][1]]], dim=1)))



def train(model, data, optimizer, device):
    model.train()

    # 获取训练数据
    x_dict = {'drug': data['drug'].x.to(device), 'protein': data['protein'].x.to(device)}
    edge_index_dict = {('drug','interaction','protein'): data[('drug','interaction','protein')].edge_index.to(device)}
                    #    ('protein','rev_interaction','drug'): data[('protein','rev_interaction','drug')].edge_index.to(device)}
    labels = data[('drug','interaction','protein')].edge_label.to(device)
    edge_label_index_dict = {('drug','interaction','protein'): data[('drug','interaction','protein')].edge_label_index.to(device) }
                    #    ('protein','rev_interaction','drug'): data[('protein','rev_interaction','drug')].edge_label_index.to(device)}
    # 清零梯度
    optimizer.zero_grad()
    
    # 前向传播
    x_dict, output = model(x_dict, edge_index_dict,edge_label_index_dict)
    
    # 计算损失
    loss = F.binary_cross_entropy(output.squeeze(), labels.float())

    # 反向传播
    loss.backward()
    optimizer.step()

    return loss.item()

def eval(model, data, device):
    model.eval()
    min_valid_loss = np.inf

    # 获取测试数据
    x_dict = {'drug': data['drug'].x.to(device), 'protein': data['protein'].x.to(device)}
    edge_index_dict = {('drug','interaction','protein'): data[('drug','interaction','protein')].edge_index.to(device)}
                    #    ('protein','rev_interaction','drug'): data[('protein','rev_interaction','drug')].edge_index.to(device)}
    labels = data[('drug','interaction','protein')].edge_label.to(device)
    edge_label_index_dict = {('drug','interaction','protein'): data[('drug','interaction','protein')].edge_label_index.to(device) }

    # 前向传播
    with torch.no_grad():
        x_dict, output = model(x_dict, edge_index_dict, edge_label_index_dict)

        # 计算损失
        val_loss = F.binary_cross_entropy(output.squeeze(), labels.float())
        if val_loss < min_valid_loss:
            torch.save(model.state_dict(),'hgt-rf.model')
            min_valid_loss = val_loss

    return val_loss.item()

def test(model, data, device):
    model.eval()

    # 获取测试数据
    x_dict = {'drug': data['drug'].x.to(device), 'protein': data['protein'].x.to(device)}
    edge_index_dict = {('drug','interaction','protein'): data[('drug','interaction','protein')].edge_index.to(device)}
                    #    ('protein','rev_interaction','drug'): data[('protein','rev_interaction','drug')].edge_index.to(device)}
    labels = data[('drug','interaction','protein')].edge_label.to(device)
    edge_label_index_dict = {('drug','interaction','protein'): data[('drug','interaction','protein')].edge_label_index.to(device) }

    # 前向传播
    with torch.no_grad():
        x_dict, output = model(x_dict, edge_index_dict, edge_label_index_dict)
    
    return x_dict
    


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

    return edge_x, labels, edge_indices

# 初始化
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载
path = 'Data/DRUGBANK/hetero_data_drugbank.pt'
data = torch.load(path)
data = T.ToUndirected()(data)

print('='*100)
print('data:', data)
print('='*100)

transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.2,
    # num_val=0.2,
    # num_test=0.3,
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

train_edge_x, train_y, train_edge_indices = data_divide(train_data)
val_edge_x, val_y, val_edge_indices = data_divide(val_data)
test_edge_x, test_y, test_edge_indices = data_divide(test_data)

train_edge_x, train_y = train_edge_x.cpu(), train_y.cpu()
val_edge_x, val_y = val_edge_x.cpu(), val_y.cpu()
test_edge_x, test_y = test_edge_x.cpu(), test_y.cpu()



# 定义模型参数
hidden_channels = 64
out_channels = 1
num_heads = 2
num_layers = 2


# 初始化模型
model = HGT(hidden_channels, out_channels, num_heads, num_layers).to(device)
print('model:',model)

# 定义优化器
optimizer = Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
min_valid_loss = np.inf

# 初始化逻辑回归模型
rf_model = RandomForestClassifier()
# import xgboost as xgb
# xgboost_model = xgb.XGBClassifier(learning_rate=0.1, max_depth=5, gamma=1, subsample=0.8)

acc_list, auc_list, pre_list = [],[],[]
run_time = 3




# 模型训练与效果测试
for i in range(run_time):
    # 训练模型
    init_time = time.time()
    for epoch in range(1,1001):  # 假设训练10个epoch 1001->101
        loss = train(model, train_data, optimizer, device)
        val_loss = eval(model, val_data, device)

        if epoch % 100 == 0:
            print(f'Time {i}, Epoch: {epoch+1}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')

    model.load_state_dict(torch.load('hgt-rf.model'))

    # 训练数据
    train_x_dict = test(model, train_data, device)

    train_edge_x_list = []
    # 遍历所有边的索引
    for edge_idx in train_edge_indices:
        # 从test_data中提取边的特征并拼接
        edge_idx_x = torch.cat((train_x_dict['drug'][edge_idx[0]], train_x_dict['protein'][edge_idx[1]]), dim=0)
        train_edge_x_list.append(edge_idx_x)
    # 将边特征列表转换为张量
    edge_x = torch.stack(train_edge_x_list, dim=0)

    train_edge_x_final = torch.cat((edge_x,torch.tensor(train_edge_x).to(device)),dim=1).detach().to('cpu')


    rf_model.fit(train_edge_x_final, train_y)
    end_time = time.time()
    print(f"Elapsed time {(end_time-init_time)/60:.4f} min")

    # 测试模型
    test_x_dict = test(model, test_data, device)

    test_edge_x_list = []
    # 遍历所有边的索引
    for edge_idx in test_edge_indices:
        # 从test_data中提取边的特征并拼接
        edge_idx_x = torch.cat((test_x_dict['drug'][edge_idx[0]], test_x_dict['protein'][edge_idx[1]]), dim=0)
        test_edge_x_list.append(edge_idx_x)
    # 将边特征列表转换为张量
    con_edge_x = torch.stack(test_edge_x_list, dim=0)

    test_edge_x_final = torch.cat((con_edge_x,torch.tensor(test_edge_x).to(device)),dim=1).detach().to('cpu')

    # 进行预测
    y_pred_proba = rf_model.predict_proba(test_edge_x_final)[:, 1]
    y_pred = rf_model.predict(test_edge_x_final)


    print('test data and predicted data:\n')
    # 计算AUC
    auc = roc_auc_score(test_y, y_pred_proba)
    # print('-------------------auc:',auc)
    auc_list.append(auc)

    # 计算Precision
    precision = precision_score(test_y, y_pred)
    # print('-------------------pre:',precision)
    pre_list.append(precision)

    # 计算Accuracy
    accuracy = accuracy_score(test_y, y_pred)
    # print('-------------------acc:',accuracy)
    acc_list.append(accuracy)

    


print(f'avg Test Accuracy: {sum(acc_list)/len(acc_list):.4f}',f' avg Test AUC: {sum(auc_list)/len(auc_list):.4f}', f' avg Test PRE: {sum(pre_list)/len(pre_list):.4f}')





