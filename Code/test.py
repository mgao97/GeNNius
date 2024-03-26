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

def build_graph(adj,feats1,feats2):
    # print(adj.shape[0],adj.shape[1])
    d_index,t_index = adj.shape[0],adj.shape[1]
    # 加上偏移量作为cell的节点标号
    t_index += adj.shape[0]
    edges=torch.tensor([d_index, t_index], dtype=torch.float)
    
    # 这里是直接对graph.edge_list进行修改了，不是副本
    graph = Graph()
    
    s_type, r_type, t_type = ('drug', 'interaction', 'protein')
    elist = graph.edge_list[t_type][s_type][r_type]
    rlist = graph.edge_list[s_type][t_type]['rev_' + r_type]
    # year = 1
    # for s_id, t_id in edges.t().tolist():
    #     elist[t_id][s_id] = year
    #     rlist[s_id][t_id] = year

    print('drug matrix: ',feats1.shape)
    print('protein matrix: ',feats2.shape)
    graph.node_feature['drug'] = torch.tensor(feats1, dtype=torch.float)
    graph.node_feature['protein'] = torch.tensor(feats2, dtype=torch.float)

    # graph.years = np.ones(adj.shape[0]+adj.shape[1])
    return graph


def determine_edge_types(edge_types, node_types, edge_index):
    edge_type_list = []
    edge_type_dict = dict(edge_types)
    
    for i in range(edge_index.size(1)):
        
        source_type = node_types[edge_index[0, i]]
        target_type = node_types[edge_index[1, i]]
        
        if (source_type, target_type) in edge_type_dict:
            edge_type = edge_type_dict[(source_type, target_type)]
        else:
            edge_type = edge_type_dict[(target_type, source_type)]
        
        edge_type_list.append(edge_type)
    
    return edge_type_list


def build_data(adj, feats1, feats2):
    node_type = [0]*adj.shape[0]+[1]*adj.shape[1]
    node_type = torch.LongTensor(node_type)

    edge_list = []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] != 0:
                edge_list.append([i, j])  # 添加起始节点和结束节点
                
    
    # 将列表转换为张量，并确保每一列包含两个元素
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    edge_type = torch.LongTensor([0]*edge_index.shape[1])
    # edge_time = torch.LongTensor([0]*edge_index.shape[1])
    
    x = {'protein': torch.tensor(feats1, dtype=torch.float),
         'drug': torch.tensor(feats2, dtype=torch.float)} 
    # print(len(x['protein']))  # 5000
    # print(len(x['drug']))  # 2713
    # print(len(node_type))  # 7713

    return x, node_type, edge_index, edge_type

device = 'cuda:0'
path = 'Data/DRUGBANK/hetero_data_drugbank.pt'
data = torch.load(path)
# print('data:', data)
data = T.ToUndirected()(data)
del data['protein', 'rev_interaction', 'drug'].edge_label 
print('data:', data)

split = T.RandomLinkSplit(
        num_val= 0.1,
        num_test= 0.2, 
        is_undirected= True,
        add_negative_train_samples= True, # False for: Not adding negative links to train
        neg_sampling_ratio= 1.0, # ratio of negative sampling is 0
        disjoint_train_ratio = 0.2, #
        edge_types=[('drug', 'interaction', 'protein')],
        rev_edge_types=[('protein', 'rev_interaction', 'drug')],
        split_labels=False
    )

train_data, val_data, test_data = split(data)

print('train_data:', train_data)

feats1, feats2 = data['drug']['x'], data['protein']['x']
print('-'* 200)
print(feats1.shape, feats2.shape)

# 获取节点类型
node_types = ('drug', 'protein')

# 获取边的类型
edge_types = ('interaction', 'rev_interaction')

# 初始化邻接矩阵字典
adj = {}

# 定义边的类型列表
edge_types = [('drug', 'interaction', 'protein'), ('protein', 'rev_interaction', 'drug')]

# 初始化邻接矩阵字典
adj = {}

# 为每种边类型构建邻接矩阵
for src_type, rel_type, tgt_type in edge_types:
    print(src_type, rel_type, tgt_type)
    # 获取边的索引
    edge_index = data[(src_type, rel_type, tgt_type)]['edge_index']
    row = edge_index[0]  # 源节点索引
    col = edge_index[1]  # 目标节点索引
    
    # 获取源节点和目标节点的数量
    num_src_nodes = data[src_type]['x'].size(0)
    num_tgt_nodes = data[tgt_type]['x'].size(0)
    
    # 创建邻接矩阵
    # row, col = edge_index.t()
    adj[src_type + '_' + tgt_type] = torch.zeros((num_src_nodes, num_tgt_nodes), dtype=torch.float)
    
    # 填充邻接矩阵
    adj[src_type + '_' + tgt_type][row, col] = 1

# 打印邻接矩阵的形状
for key, mat in adj.items():
    print(f'Adjacency matrix for {key}: shape is {mat.size()}')


g = build_graph(adj['drug_protein'], feats1, feats2)

import anndata as ad
import scanpy as sc

# adata1 = ad.AnnData(adj['drug_protein'].detach().cpu().numpy().transpose(), dtype='int32')
# # print('adata1:',adata1)
# # print('='*100)
# adata1.obsm['AE'] = feats2.detach().cpu().numpy()
# sc.pp.neighbors(adata1, use_rep = 'AE')
# sc.tl.louvain(adata1,resolution = 0.5)
# Cell_Res = coo_matrix((np.ones(adata1.obs['louvain'].shape), np.array([np.arange(feats2.shape[0]),adata1.obs['louvain']])), shape=(feats2.shape[0], len(set(adata1.obs['louvain']))), dtype=int).todense()
# print('Cell_Res:',Cell_Res.shape)
# h = np.zeros((feats2.shape[0],args.n_hid))
# h[:,0:feats1.shape[1]] = Cell_Res
# Cell_Res = torch.tensor(Cell_Res, dtype=torch.float32).to(device)

# print('Cell_Res:\n',Cell_Res)

print("Start sampling!")
np.random.seed(seed)
jobs = []
# cell-drug gene-protein
drug_num=int((adj['protein_drug'].shape[1]*args.drug_rate)/args.n_batch)
protein_num=int((adj['protein_drug'].shape[0]*args.protein_rate)/args.n_batch)
print(f'drug_num: {drug_num}, protein_num: {protein_num}')
for _ in range(args.n_batch):
    p = sub_sample1(g,
                    adj['protein_drug'],
                    drug_num,
                     protein_num,
                    
                    adj['protein_drug'].shape[0],
                    adj['protein_drug'].shape[1])
    jobs.append(p)
print("Sampling end!")
print(len(jobs))

# if (args.reduction != 'raw'):
#     gnn = GNN(conv_name=args.layer_type, in_dim=encoded.shape[1],
#               n_hid=args.n_hid, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
#               num_types=2, num_relations=2, use_RTE=False).to(device)
# else:
#     gnn = GNN_from_raw(conv_name=args.layer_type, in_dim=[encoded.shape[1], encoded2.shape[1]],
#                        n_hid=args.n_hid, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
#                        num_types=2, num_relations=2, use_RTE=False,
#                        AEtype=args.AEtype).to(device)
gnn = GNN_from_raw(conv_name=args.layer_type, in_dim=[20,12],
                       n_hid=args.n_hid, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
                       num_types=2, num_relations=2,
                       AEtype=args.AEtype).to(device)
# gnn = GNN_from_raw(conv_name=args.layer_type, in_dim=[int(feats1.shape[1]), int(feats2.shape[1])],
#                        n_hid=args.n_hid, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
#                        num_types=2, num_relations=2,
#                        AEtype=args.AEtype).to(device)


print('gnn:', gnn)

# default: adamw
if args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(gnn.parameters(), lr=args.lr)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(gnn.parameters(), lr=args.lr)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(gnn.parameters(), lr=args.lr)
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(gnn.parameters(), lr=args.lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', factor=args.factor, patience=args.patience, verbose=True)

# gnn.train()
# for epoch in np.arange(args.epoch):
#     L = 0
#     for job in jobs:
#         feature,edge_list,indxs = job
#         # for key in indxs:
#         #     indxs[key] = torch.tensor(indxs[key]).to(device)

        

#         node_dict = {}
#         node_feature = []
#         node_type = []
#         # node_time = []
#         edge_index = []
#         edge_type = []
#         # edge_time = []

#         node_num = 0
#         types = g.get_types()   # ['gene','cell']
#         for t in types:
#             #print("t in types "+str(t)+"\n")
#             node_dict[t] = [node_num, len(node_dict)]
#             node_num += len(feature[t])
#             if args.reduction == 'raw':
#                 node_feature.append([])
#         # node_dict: {'gene':[0,0],'cell':[134,1]}
            
#             # print('node_dict:', node_dict)

#         for t in types:
#             t_i = node_dict[t][1]
#             #print("feature t:\n")
#             #print("t_i="+str(t_i)+" t="+str(t)+"\n")
#             # print(feature[t].shape)
#             if args.reduction != 'raw':
#                 node_feature += list(feature[t])
#             else:
#                 node_feature[t_i] = torch.tensor(
#                     feature[t], dtype=torch.float32).to(device)

#             # node_time += list(time[t])
#             node_type += [node_dict[t][1] for _ in range(len(feature[t]))]
#         edge_dict = {e[2]: i for i, e in enumerate(g.get_meta_graph())}
#         edge_dict['self'] = len(edge_dict)
#         # {'g_c': 0, 'rev_g_c': 1 ,'self': 2}
#         for target_type in edge_list:
#             for source_type in edge_list[target_type]:
#                 for relation_type in edge_list[target_type][source_type]:
#                     for ii, (ti, si) in enumerate(edge_list[target_type][source_type][relation_type]):
#                         tid, sid = ti + \
#                             node_dict[target_type][0], si + \
#                             node_dict[source_type][0]
#                         edge_index += [[sid, tid]]
#                         edge_type += [edge_dict[relation_type]]

#                         # Our time ranges from 1900 - 2020, largest span is 120.
#                         # edge_time += [node_time[tid] - node_time[sid] + 120]
#                         # edge_time += [120]

#         # print('node_feature:',node_feature)

#         if (args.reduction != 'raw'):
#             node_feature = torch.stack(node_feature)
#             node_feature = torch.tensor(node_feature, dtype=torch.float32)
#             node_feature = node_feature.to(device)

#         # print('node_feature:',node_feature[0].shape, node_feature[1].shape)
        
#         # node_feature = torch.trunc(node_feature*10000)/10000
#         node_type = torch.LongTensor(node_type)
#         # edge_time = torch.LongTensor(edge_time)
#         edge_index = torch.LongTensor(edge_index).t()
#         edge_type = torch.LongTensor(edge_type)
#         if (args.reduction == 'raw'):
#             node_rep, node_decoded_embedding = gnn.forward(node_feature, 
#                                                            node_type.to(device),
#                                                         #    edge_time.to(device),
#                                                            edge_index.to(device),
#                                                            edge_type.to(device),)
#                                                         #    cell_res.to(device))
#         else:
#             node_rep = gnn.forward(node_feature, 
#                                    node_type.to(device),
#                                 #    edge_time.to(device),
#                                    edge_index.to(device),
#                                    edge_type.to(device),)
#                                 #    cell_res.to(device))
        
#         if args.rep == 'T':
#             node_rep = torch.trunc(node_rep*10000000000)/10000000000
#             if args.reduction == 'raw':
#                 for t in types:
#                     t_i = node_dict[t][1]
#                     # print("t_i="+str(t_i))
#                     node_decoded_embedding[t_i] = torch.trunc(
#                         node_decoded_embedding[t_i]*10000000000)/10000000000


#         gene_matrix = node_rep[node_type == 0, ]
#         cell_matrix = node_rep[node_type == 1, ]

#         regularization_loss = 0
#         for param in gnn.parameters():
#             regularization_loss += torch.sum(torch.pow(param, 2))
#         if (args.loss == "kl"):
#             decoder = torch.mm(gene_matrix, cell_matrix.t())
#             decoder = F.normalize(decoder, p=2, dim=-1)
#             # print('decoder:',decoder)
#             # print('adj:',adj)

#             # 确保索引在有效范围内
#             sub_adj = adj['drug_protein'].to(device)
#             indxs_drug = torch.clamp(torch.tensor(indxs['drug']), torch.tensor(0).to(device), torch.tensor(sub_adj.shape[0] - 1).to(device))
#             indxs_protein = torch.clamp(torch.tensor(indxs['protein']), torch.tensor(0), torch.tensor(sub_adj.shape[1] - 1))

#             # 使用 indxs_drug 和 indxs_protein 来选择 'drug_protein' 矩阵中的子矩阵
#             sub_adj = sub_adj[indxs_drug, :][:, indxs_protein]

            
#             # adj = adj[tuple(indxs['protein']), ]
#             # adj = adj[:, indxs['drug']]
#             # adj = torch.tensor(adj, dtype=torch.float32).to(device)
#             # adj = adj['drug_protein'].to(device)[indxs['drug'],][:,indxs['protein']]
            
#             # print('decoder shape:',decoder.shape, 'sub_adj.shape:',sub_adj.shape)
            
            
                  
#             if args.reduction == 'raw':
#                 if epoch % 2 == 0:
#                     loss = F.kl_div(decoder.softmax(
#                         dim=-1).log(), sub_adj.softmax(dim=-1), reduction='sum')+args.rf*regularization_loss
#                 else:
#                     loss = nn.MSELoss()(
#                         node_feature[0], node_decoded_embedding[0])+args.rf*regularization_loss
#                     for t_i in range(1, len(types)):
#                         loss += nn.MSELoss()(node_feature[t_i],
#                                              node_decoded_embedding[t_i])
#             else:
#                 loss = F.kl_div(decoder.softmax(dim=-1).log(),
#                                 adj.softmax(dim=-1), reduction='sum')

#         if (args.loss == "cross"):
#             # negative_sampling not defined
#             print("negative_sampling not defined!")
#             exit()
#             pass

#         L += loss.item()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     scheduler.step(L/(int(sub_adj.shape[0])))
#     print('Epoch :', epoch+1, '|', 'train_loss:%.12f' %
#           (L/(int(sub_adj.shape[0]))/args.n_batch))


state = {'model': gnn.state_dict(), 'optimizer': scheduler.state_dict()}

# state = {'model': gnn.state_dict(), 'optimizer': scheduler.state_dict(),
#          'epoch': epoch}
# torch.save(state, 'gnn_model.pth')

# model.eval()
if (adj['drug_protein'].shape[1]>10000):

    if (adj['drug_protein'].shape[0]>10000):
        ba = 500
    else:
        ba = adj['drug_protein'].shape[0]
else:
    if (adj['drug_protein'].shape[0]>10000):
        ba = 5000
    else:
        ba = adj['drug_protein'].shape[0]
    
gnn.load_state_dict(state['model'])
g_embedding = []
protein_name = []
drug_name = []
attention = []

# ba = 64

with torch.no_grad():
    for i in range(0, adj['drug_protein'].shape[0], ba):
        sub_adj = adj['drug_protein']#[i:(i+ba), :]  
        print('sub adj:', sub_adj, sub_adj.shape)
        print('feats1[i:(ba+i)]:',feats1[:,i:(ba+i)],feats1[:,i:(ba+i)].shape)
        print('feats2:', feats2, feats2.shape)
        # cell_res = Cell_Res
        x,node_type, edge_index,edge_type = build_data(sub_adj,feats1[:,i:(ba+i)],feats2)
        if args.reduction != 'raw':
            node_rep = gnn.forward((torch.cat((x['protein'], x['drug']), 0)).to(device), 
            node_type.to(device),
            edge_index.to(device), edge_type.to(device))
        else:
            node_rep, _ = gnn.forward([x['drug'].to(device), x['protein'].to(device)], 
                                       node_type.to(device),
                                       edge_index.to(device), edge_type.to(device))

        protein_name = protein_name + list(np.array(edge_index[0]+i))
        drug_name = drug_name + list(np.array(edge_index[1]-sub_adj.shape[0]))
        attention.append(gnn.att)
        protein_matrix = node_rep[node_type == 0, ]
        drug_matrix = node_rep[node_type == 1, ]
        g_embedding.append(protein_matrix)

if adj['drug_protein'].shape[0] % ba == 0:
    protein_matrix = np.vstack([i.to('cpu') for i in g_embedding[0:int(adj['drug_protein'].cpu().shape[0]/ba)]])
    attention = np.vstack([i.to('cpu') for i in attention[0:int(adj['drug_protein'].cpu().shape[0]/ba)]])
else:
    final_tensor = np.vstack(g_embedding[0:int(adj['drug_protein'].shape[0]/ba)])
    protein_matrix = np.concatenate((final_tensor, protein_matrix), 0)
    final_attention = np.vstack(attention[0:int(adj['drug_protein'].shape[0]/ba)])
    attention = np.concatenate((final_attention, gnn.att), 0)
drug_matrix = drug_matrix.detach().cpu().numpy()


protein_matrix = np.round(protein_matrix,decimals=4)
drug_matrix = np.round(drug_matrix,decimals=4)
# 保存蛋白质矩阵
np.save('res/protein.npy', protein_matrix)

# 保存药物矩阵
np.save('res/drug.npy', drug_matrix)
# np.savetxt('res/protein.txt', protein_matrix, delimiter=' ')
# np.savetxt('res/drug.txt', drug_matrix, delimiter=' ')