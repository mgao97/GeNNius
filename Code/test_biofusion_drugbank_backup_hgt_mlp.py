import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric.transforms as T
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import Data
import psutil
import time

# 固定随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------ 数据加载和预处理 ------------------
path = 'Data/DRUGBANK/hetero_data_drugbank.pt'
data = torch.load(path)
data = T.ToUndirected()(data)

smile_llm_emb = torch.load('Data/DRUGBANK/exp_smile_llm_emb.pt', map_location=device)
sequence_llm_emb = torch.load('Data/DRUGBANK/exp_sequence_llm_emb.pt', map_location=device)

# 添加额外嵌入特征
data['drug'].x = torch.cat((data['drug'].x, smile_llm_emb[:data['drug'].x.shape[0]]), dim=1)
data['protein'].x = torch.cat((data['protein'].x, sequence_llm_emb[:data['protein'].x.shape[0]]), dim=1)

# 数据集划分
transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.2,
    is_undirected=True,
    disjoint_train_ratio=0.2,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=True,
    edge_types=("drug", "interaction", "protein"),
    rev_edge_types=("protein", "rev_interaction", "drug")
)
train_data, val_data, test_data = transform(data)

train_data.edge_labels = train_data[('drug','interaction','protein')].edge_label.to(device)
train_data.edge_label_index_dict = {('drug','interaction','protein'): train_data[('drug','interaction','protein')].edge_label_index.to(device)}

val_data.edge_labels = val_data[('drug','interaction','protein')].edge_label.to(device)
val_data.edge_label_index_dict = {('drug','interaction','protein'): val_data[('drug','interaction','protein')].edge_label_index.to(device)}

test_data.edge_labels = test_data[('drug','interaction','protein')].edge_label.to(device)
test_data.edge_label_index_dict = {('drug','interaction','protein'): test_data[('drug','interaction','protein')].edge_label_index.to(device)}




# ------------------ 定义 HGT 模型 ------------------
class HGTWithMLP(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, mlp_hidden_dim):
        super().__init__()

        # HGT 模块
        self.lin_dict = nn.ModuleDict()
        self.norm_dict = nn.ModuleDict()  # 增加归一化层
        for node_type in ['drug', 'protein']:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)
            self.norm_dict[node_type] = nn.BatchNorm1d(hidden_channels)

        self.convs = nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels, data.metadata(), num_heads)
            for _ in range(num_layers)
        ])

        # MLP 分类器
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, mlp_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden_dim, out_channels)
        )

    def forward(self, x_dict, edge_index_dict, edge_label_index_dict):
        x_dict = {node_type: self.norm_dict[node_type](self.lin_dict[node_type](x)).relu_()
                  for node_type, x in x_dict.items()}

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        drug_emb = x_dict['drug'][edge_label_index_dict[('drug', 'interaction', 'protein')][0]]
        protein_emb = x_dict['protein'][edge_label_index_dict[('drug', 'interaction', 'protein')][1]]
        edge_emb = torch.cat([drug_emb, protein_emb], dim=1)

        logits = self.mlp(edge_emb)
        return logits, x_dict


# ------------------ 模型训练与评估逻辑 ------------------
class ModelTrainer:
    def __init__(self, hgt_model):
        self.hgt_model = hgt_model.to(device)

    def train(self, train_data, val_data, optimizer, criterion, scheduler, epochs, patience):
        best_val_auc = 0
        patience_counter = 0

        for epoch in range(epochs):
            self.hgt_model.train()
            optimizer.zero_grad()

            x_dict, edge_index_dict = train_data.x_dict, train_data.edge_index_dict
            
            edge_label_index_dict, edge_labels = train_data.edge_label_index_dict, train_data.edge_labels.to(device)

            logits, _ = self.hgt_model(x_dict, edge_index_dict, edge_label_index_dict)
            loss = criterion(logits.squeeze(), edge_labels.float())
            loss.backward()
            optimizer.step()

            # 验证阶段
            val_auc = self.evaluate(val_data)['AUC']
            scheduler.step(val_auc)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Val AUC: {val_auc:.4f}")

            # 检查早停
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    break

    def evaluate(self, data):
        self.hgt_model.eval()
        with torch.no_grad():
            x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
            edge_label_index_dict, edge_labels = data.edge_label_index_dict, data.edge_labels.to(device)

            logits, _ = self.hgt_model(x_dict, edge_index_dict, edge_label_index_dict)
            probabilities = torch.sigmoid(logits.squeeze())
            predictions = (probabilities > 0.5).long()

            auc = roc_auc_score(edge_labels.cpu().numpy(), probabilities.cpu().numpy())
            precision = precision_score(edge_labels.cpu().numpy(), predictions.cpu().numpy())
            accuracy = accuracy_score(edge_labels.cpu().numpy(), predictions.cpu().numpy())

            return {"AUC": auc, "Precision": precision, "Accuracy": accuracy}


# ------------------ 初始化和运行 ------------------
hidden_channels = 64
out_channels = 1  # 二分类任务
num_heads = 2
num_layers = 2
mlp_hidden_dim = 64
epochs = 400
patience = 10

# 记录训练和测试时间
train_times = []
test_times = []

# 重复运行 5 次
for run in range(5):
    print(f"Run {run + 1}/5")

hgt_model = HGTWithMLP(hidden_channels, out_channels, num_heads, num_layers, mlp_hidden_dim)
trainer = ModelTrainer(hgt_model)

# 优化器、调度器和损失函数
optimizer = Adam(hgt_model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
criterion = nn.BCEWithLogitsLoss()

# 记录训练时间
start_train = time.time()
# 训练模型
trainer.train(train_data, val_data, optimizer, criterion, scheduler, epochs, patience)
end_train = time.time()
train_times.append(end_train - start_train)

# 记录测试时间
start_test = time.time()
# 测试模型
test_metrics = trainer.evaluate(test_data)
end_test = time.time()
test_times.append(end_test - start_test)
print(f"Run {run + 1}/5 - Test AUC: {test_metrics['AUC']:.4f}, Precision: {test_metrics['Precision']:.4f}, Accuracy: {test_metrics['Accuracy']:.4f}")

# 计算平均时间
avg_train_time = np.mean(train_times)
avg_test_time = np.mean(test_times)

print(f"Average Training Time: {avg_train_time:.2f} seconds")
print(f"Average Testing Time: {avg_test_time:.2f} seconds")

# 内存使用情况
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")
