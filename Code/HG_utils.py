from HG_data import Graph
import numpy as np
import torch
from collections import defaultdict
import resource
import pandas as pd

def debuginfoStr(info):
    print(info)
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024)
    print('Mem consumption (GB): '+str(mem))


def loadGAS(data_path):
    df=pd.read_csv(data_path, sep=" ")
    return df.to_numpy()



def build_graph(gene_cell,encoded,encoded2):
    g_index,c_index = np.nonzero(gene_cell)
    # 加上偏移量作为cell的节点标号
    c_index += gene_cell.shape[0]
    edges=torch.tensor([g_index, c_index], dtype=torch.float)
    
    # 这里是直接对graph.edge_list进行修改了，不是副本
    graph = Graph()
    s_type, r_type, t_type = ('gene', 'g_c', 'cell')
    elist = graph.edge_list[t_type][s_type][r_type]
    rlist = graph.edge_list[s_type][t_type]['rev_' + r_type]
    year = 1
    for s_id, t_id in edges.t().tolist():
        elist[t_id][s_id] = year
        rlist[s_id][t_id] = year

    print('gene matrix: ',encoded.shape)
    print('cell matrix: ',encoded2.shape)
    graph.node_feature['gene'] = torch.tensor(encoded, dtype=torch.float)
    graph.node_feature['cell'] = torch.tensor(encoded2, dtype=torch.float)

    graph.years = np.ones(gene_cell.shape[0]+gene_cell.shape[1])
    return graph

def build_data(adj, encoded, encoded2):
    node_type = [0]*adj.shape[0]+[1]*adj.shape[1]
    node_type = torch.LongTensor(node_type)

    g_index,c_index = np.nonzero(adj)
    c_index += adj.shape[0]
    edge_index = torch.tensor([g_index, c_index], dtype=torch.long)
    edge_type = torch.LongTensor([0]*edge_index.shape[1])
    edge_time = torch.LongTensor([0]*edge_index.shape[1])
    
    x = {'gene': torch.tensor(encoded, dtype=torch.float),
         'cell': torch.tensor(encoded2, dtype=torch.float)} 
    # print(len(x['gene']))  # 5000
    # print(len(x['cell']))  # 2713
    # print(len(node_type))  # 7713

    return x,node_type, edge_time, edge_index,edge_type

def norm_rowcol(matrix):
    row_norm=np.sum(matrix,axis=1).reshape(-1,1)
    matrix=matrix/row_norm
    col_norm=np.sum(matrix,axis=0)
    return matrix/col_norm

device = 'cuda'

def sub_sample1(graph, GAS, sampling_size,gene_size,gene_shape,cell_shape):
    cell_indexs=gene_shape+np.random.choice(np.arange(cell_shape),sampling_size,replace=False)
    sub_matrix=GAS[:,cell_indexs-gene_shape]
    # res = Cell_Res[cell_indexs-gene_shape,:]
    sub_matrix = sub_matrix.cpu().numpy()
    gene_indexs=np.nonzero(np.sum(sub_matrix,axis=1))[0]

    sub_matrix=GAS[gene_indexs,:][:,cell_indexs-gene_shape]
    sub_matrix = sub_matrix.cpu().numpy()

    sub_matrix=norm_rowcol(sub_matrix)
    
    _indexs=np.argsort(np.sum(sub_matrix,axis=1))[::-1]
    gene_indexs=gene_indexs[_indexs]
    gene_indexs=gene_indexs[:gene_size]
    gene_indexs = torch.tensor(gene_indexs).to(device)
    
    feature={
        'drug':graph.node_feature['protein'][gene_indexs,:],
        'protein':graph.node_feature['drug'][cell_indexs-gene_shape,:],
    }

    # times={
    #     'drug': np.ones(gene_size),
    #     'protein':np.ones(sampling_size)
    # }

    indxs={
        'drug':gene_indexs,
        'protein':cell_indexs-gene_shape
    }

    edge_list = defaultdict(  # target_type
        lambda: defaultdict(  # source_type
            lambda: defaultdict(  # relation_type
                lambda: []  # [target_id, source_id]
            )))

    for i in range(gene_size):
        edge_list['drug']['drug']['self'].append([i,i])

    for i in range(sampling_size):
        edge_list['protein']['protein']['self'].append([i,i])

    for i,cell_id in enumerate(cell_indexs):
        for j,gene_id in enumerate(gene_indexs):
            if gene_id in graph.edge_list['protein']['drug']['g_c'][cell_id]:
                edge_list['protein']['drug']['g_c'].append([i,j])
                edge_list['drug']['protein']['rev_g_c'].append([j,i])

    return feature, edge_list, indxs