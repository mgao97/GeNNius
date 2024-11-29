import numpy as np
import math
import torch
import torch.utils.data
from torchvision import transforms
from PIL import Image
from torch.utils.data import WeightedRandomSampler


import random
import torch_geometric.transforms as T
random.seed(42)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs, smiles, proteins, label):
        self.imgs = imgs
        self.smiles = smiles
        self.proteins = proteins
        self.label = label

        self.transformImg = transforms.Compose([
            transforms.ToTensor()])

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        # print('item:',item)
        # print(len(self.label))
        # while len(self.label) <= item:
        #     self.label.append(0)
        # print(len(self.label))
        # item = item if item < len(self.label) else item-len(self.label)
        item = item if item < 128 else 128
        img_path = self.imgs[item]
        smiles_feature = self.smiles[item]
        pro_feature = self.proteins[item]
        label_feature = self.label[item]
        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"File not found: {img_path}")
            # 返回一个默认的全黑图像（假设大小为 256x256）
            img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        img = self.transformImg(img)

        # print("img:", len(self.imgs))
        # print("smile:", len(self.smiles))
        # print("pro:", len(self.proteins))
        # print("label:", len(self.label))

        return img, smiles_feature, pro_feature, label_feature


def load_tensor(file_name, dtype):
    return [dtype(d) for d in np.load(file_name)]


def get_label(path,device,mode='train'):
    
    data = torch.load(path)
    data = T.ToUndirected()(data)

    print('='*100)
    print('data:', data)
    print('='*100)

    

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

    if mode == 'train':
        return train_data[('drug','interaction','protein')].edge_label.to(device)
    elif mode == 'val':
        return val_data[('drug','interaction','protein')].edge_label.to(device)
    else:
        return test_data[('drug','interaction','protein')].edge_label.to(device)

def data_loader(batch_size, imgs, smile_name, pro_name, path,device,mode):
    smiles = load_tensor(smile_name, torch.LongTensor)
    proteins = load_tensor(pro_name, torch.LongTensor)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    interactions = get_label(path,device,mode='train').long()


    # interactions = load_tensor(inter_name, torch.LongTensor)

    # print('smiles shape:',smiles[0].shape,len(smiles))
    # print('proteins shape:',proteins[0].shape,len(proteins))
    print('interactions shape:',interactions[0].shape,len(interactions))
    print('='*100)

    dataset = Dataset(imgs, smiles, proteins, interactions)
    # dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 获取正负样本的索引
    # interactions_concat = torch.cat(interactions, dim=0)

    # 获取正负样本的索引
    # positive_indices = np.where(interactions_concat == 1)[0]
    # negative_indices = np.where(interactions_concat == 0)[0]
    positive_indices = np.where(interactions == 1)[0]
    negative_indices = np.where(interactions == 0)[0]

    # 为每个样本分配权重
    # weights = np.zeros(len(interactions_concat))  # 初始化权重数组
    weights = np.zeros(len(interactions))  # 初始化权重数组
    weights[positive_indices] = 1.0              # 正样本权重为1.0
    weights[negative_indices] = 1.0              # 负样本权重为0.5

    # 创建加权采样器
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    # 创建数据加载器
    # dataset_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,sampler=sampler)
    return dataset, dataset_loader


def get_img_path(img_path):
    imgs = []
    with open(img_path, "r") as f:
        lines = f.read().strip().split("\n")
        for line in lines:
            imgs.append(line.split("\t")[0])
    return imgs



def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    return list(parameter_group_vars.values())
