import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import HeteroData

from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import math


# heterogeneous graph neural network
class HGN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads, num_layers):
        super(HGN, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 定义线性层以适应不同类型的节点
        self.drug_lin = nn.Linear(in_channels['drug'], hidden_channels)
        self.protein_lin = nn.Linear(in_channels['protein'], hidden_channels)
        
        # 定义HGN卷积层，这里我们使用HeteroGraphConv，它可以处理异构图中的多种类型的边
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HeteroConv({
                ('drug', 'interaction', 'protein'): GCNConv(in_channels['drug'], hidden_channels, add_self_loops=False),
                ('protein', 'rev_interaction', 'drug'): GCNConv(in_channels['protein'], hidden_channels, add_self_loops=False)
            }, aggr='add'))
        
        # 定义最终的链路预测层，这里我们使用一个简单的线性层来预测边的存在
        self.link_prediction_layer = nn.Linear(hidden_channels * 2, 1)
    
    def forward(self, data):
        # 获取异构图中的节点特征
        drug_x = self.drug_lin(data['drug']['x'])
        protein_x = self.protein_lin(data['protein']['x'])
        
        # 通过HGN卷积层
        for conv in self.convs:
            drug_x = conv(data)[('drug', 'interaction', 'protein')]
            protein_x = conv(data)[('protein', 'rev_interaction', 'drug')]
        
        # 准备链路预测的输入，这里我们简单地拼接两个节点的嵌入
        link_input = torch.cat([drug_x, protein_x], dim=1)
        
        # 通过链路预测层得到预测结果
        link_prediction = self.link_prediction_layer(link_input)
        link_prediction = torch.sigmoid(link_prediction)  # 将输出转换为概率值
        
        return link_prediction


class HGT(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, dropout = 0.2, conv_name = 'hgt', prev_norm = False, last_norm = False):
        super(HGT, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim    = in_dim
        self.n_hid     = n_hid
        self.adapt_ws  = nn.ModuleList()
        self.drop      = nn.Dropout(dropout)
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for l in range(n_layers - 1):
            self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = prev_norm))
        self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = last_norm))

    def forward(self, node_feature, node_type, edge_index, edge_type):
        res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
        meta_xs = self.drop(res)
        del res
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type)
        return meta_xs 


class HGTConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = False, **kwargs):
        super(HGTConv, self).__init__(node_dim=0, aggr='add', **kwargs)

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.total_rel     = num_types * num_relations * num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.use_norm      = use_norm
        # self.use_RTE       = use_RTE
        self.att           = None
        self.res_att        =None
        self.res            =None
        
        
        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        
        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        '''
            TODO: make relation_pri smaller, as not all <st, rt, tt> pair exist in meta relation list.
        '''
        self.relation_pri   = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(num_types))
        self.drop           = nn.Dropout(dropout)
        
        # if self.use_RTE:
        #     self.emb            = RelTemporalEncoding(in_dim)
        
        glorot(self.relation_att)
        glorot(self.relation_msg)
        
    def forward(self, node_inp, node_type, edge_index, edge_type):
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type, \
                              edge_type=edge_type)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type):
        '''
            j: source, i: target; <j, i>
        '''
        data_size = edge_index_i.size(0)
        '''
            Create Attention and Message tensor beforehand.
        '''
        self.res_att     = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg     = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        
        for source_type in range(self.num_types):
            sb = (node_type_j == int(source_type))
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type] 
            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]
                for relation_type in range(self.num_relations):
                    '''
                        idx is all the edges with meta relation <source_type, relation_type, target_type>
                    '''
                    idx = (edge_type == int(relation_type)) & tb
                    if idx.sum() == 0:
                        continue
                    '''
                        Get the corresponding input node representations by idx.
                        Add tempotal encoding to source representation (j)
                    '''
                    target_node_vec = node_inp_i[idx]
                    source_node_vec = node_inp_j[idx]
                    # if self.use_RTE:
                    #     source_node_vec = self.emb(source_node_vec, edge_time[idx])
                    '''
                        Step 1: Heterogeneous Mutual Attention
                    '''
                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(k_mat.transpose(1,0), self.relation_att[relation_type]).transpose(1,0)
                    self.res_att[idx] = (q_mat * k_mat).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk
                    '''
                        Step 2: Heterogeneous Message Passing
                    '''
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(v_mat.transpose(1,0), self.relation_msg[relation_type]).transpose(1,0)   
        '''
            Softmax based on target node's id (edge_index_i). Store attention value in self.att for later visualization.
        '''
        # self.att = softmax(res_att, edge_index_i)
        # res = res_msg * self.att.view(-1, self.n_heads, 1)
        res =res_msg * softmax(self.res_att.view(-1, self.n_heads, 1), edge_index_i)
        # del res_att, res_msg
        return res.view(-1, self.out_dim)


    def update(self, aggr_out, node_inp, node_type):
        '''
            Step 3: Target-specific Aggregation
            x = W[node_type] * gelu(Agg(x)) + x
        '''
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx]))
            '''
                Add skip connection with learnable weight self.skip[t_id]
            '''
            alpha = torch.sigmoid(self.skip[target_type])
            if self.use_norm:
                res[idx] = self.norms[target_type](trans_out * alpha + node_inp[idx] * (1 - alpha))
            else:
                res[idx] = trans_out * alpha + node_inp[idx] * (1 - alpha)
        return res

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)
    
    
    
class DenseHGTConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = True, **kwargs):
        super(DenseHGTConv, self).__init__(node_dim=0, aggr='add', **kwargs)

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.total_rel     = num_types * num_relations * num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.use_norm      = use_norm
        # self.use_RTE       = use_RTE
        self.att           = None
        
        
        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()

        
        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        '''
            TODO: make relation_pri smaller, as not all <st, rt, tt> pair exist in meta relation list.
        '''
        self.relation_pri   = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.drop           = nn.Dropout(dropout)
        
        # if self.use_RTE:
        #     self.emb            = RelTemporalEncoding(in_dim)
        
        glorot(self.relation_att)
        glorot(self.relation_msg)
        
        
        self.mid_linear  = nn.Linear(out_dim,  out_dim * 2)
        self.out_linear  = nn.Linear(out_dim * 2,  out_dim)
        self.out_norm    = nn.LayerNorm(out_dim)
        
    def forward(self, node_inp, node_type, edge_index, edge_type):
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type, \
                              edge_type=edge_type)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type):
        '''
            j: source, i: target; <j, i>
        '''
        data_size = edge_index_i.size(0)
        '''
            Create Attention and Message tensor beforehand.
        '''
        res_att     = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg     = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        
        for source_type in range(self.num_types):
            sb = (node_type_j == int(source_type))
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type] 
            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]
                for relation_type in range(self.num_relations):
                    '''
                        idx is all the edges with meta relation <source_type, relation_type, target_type>
                    '''
                    idx = (edge_type == int(relation_type)) & tb
                    if idx.sum() == 0:
                        continue
                    '''
                        Get the corresponding input node representations by idx.
                        Add tempotal encoding to source representation (j)
                    '''
                    target_node_vec = node_inp_i[idx]
                    source_node_vec = node_inp_j[idx]
                    # if self.use_RTE:
                    #     source_node_vec = self.emb(source_node_vec, edge_time[idx])
                    '''
                        Step 1: Heterogeneous Mutual Attention
                    '''
                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(k_mat.transpose(1,0), self.relation_att[relation_type]).transpose(1,0)
                    res_att[idx] = (q_mat * k_mat).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk
                    '''
                        Step 2: Heterogeneous Message Passing
                    '''
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(v_mat.transpose(1,0), self.relation_msg[relation_type]).transpose(1,0)   
        '''
            Softmax based on target node's id (edge_index_i). Store attention value in self.att for later visualization.
        '''
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        return res.view(-1, self.out_dim)


    def update(self, aggr_out, node_inp, node_type):
        '''
            Step 3: Target-specific Aggregation
            x = W[node_type] * Agg(x) + x
        '''
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx])) + node_inp[idx]
            '''
                Add skip connection with learnable weight self.skip[t_id]
            '''
            if self.use_norm:
                trans_out = self.norms[target_type](trans_out)
                
            '''
                Step 4: Shared Dense Layer
                x = Out_L(gelu(Mid_L(x))) + x
            '''
                
            trans_out     = self.drop(self.out_linear(F.gelu(self.mid_linear(trans_out)))) + trans_out
            res[idx]      = self.out_norm(trans_out)
        return res

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)

    
    
class GeneralConv(nn.Module):
    def __init__(self, conv_name, in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm = True):
        super(GeneralConv, self).__init__()
        self.conv_name = conv_name
        self.res_att = None
        self.res = None
        if self.conv_name == 'hgt':
            self.base_conv = HGTConv(in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm)
        elif self.conv_name == 'dense_hgt':
            self.base_conv = DenseHGTConv(in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm)
        elif self.conv_name == 'gcn':
            self.base_conv = GCNConv(in_hid, out_hid)
        elif self.conv_name == 'gat':
            self.base_conv = GATConv(in_hid, out_hid // n_heads, heads=n_heads)
    def forward(self, meta_xs, node_type, edge_index, edge_type):
        if self.conv_name == 'hgt':
            a=self.base_conv(meta_xs, node_type, edge_index, edge_type)
            self.res_att = self.base_conv.res_att
            self.res = self.base_conv.res
            return a
            # return self.base_conv(meta_xs, node_type, edge_index, edge_type)
        elif self.conv_name == 'gcn':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'gat':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'dense_hgt':
            return self.base_conv(meta_xs, node_type, edge_index, edge_type)
        

# class RelTemporalEncoding(nn.Module):
#     '''
#         Implement the Temporal Encoding (Sinusoid) function.
#     '''
#     def __init__(self, n_hid, max_len = 240, dropout = 0.2):
#         super(RelTemporalEncoding, self).__init__()
#         position = torch.arange(0., max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, n_hid, 2) *
#                              -(math.log(10000.0) / n_hid))
#         emb = nn.Embedding(max_len, n_hid)
#         emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
#         emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
#         emb.requires_grad = False
#         self.emb = emb
#         self.lin = nn.Linear(n_hid, n_hid)
#     def forward(self, x, t):
#         return x + self.lin(self.emb(t))
    
