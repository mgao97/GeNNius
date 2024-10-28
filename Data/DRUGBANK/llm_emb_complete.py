import torch



smiles_llm_emb = torch.load('smile_llm_emb.pt')
sequences_llm_emb = torch.load('sequence_llm_emb.pt')
print(smiles_llm_emb.shape,sequences_llm_emb.shape)

# unique drugs: 6823
# 复制原始tensor以确保随机选取不改变原始数据
exp_smile_llm_emb = torch.zeros(6823,64)
# 随机选择缺少的行
indices = torch.randperm(smiles_llm_emb.shape[0])[:6823-smiles_llm_emb.shape[0]]
# 将选中的行复制到新的位置
exp_smile_llm_emb[smiles_llm_emb.shape[0]:6823, :] = smiles_llm_emb[indices, :]


# unique proteins: 4650
exp_sequence_llm_emb = torch.zeros(4650,64)
indices = torch.randperm(sequences_llm_emb.shape[0])[:616]
exp_sequence_llm_emb[sequences_llm_emb.shape[0]:4650, :] = sequences_llm_emb[indices, :64]
# 初始化目标张量
# exp_sequence_llm_emb = torch.zeros(718, 256)

# # 计算需要扩充的行数
# num_new_rows = 718 - sequences_llm_emb.shape[0]

# # 生成随机索引
# indices = torch.randperm(sequences_llm_emb.shape[0])[:20]

# # 将选定的行复制到目标张量的新位置
# exp_sequence_llm_emb[sequences_llm_emb.shape[0]:718, :] = sequences_llm_emb[indices, :]

# # 将原始张量复制到目标张量的开始位置
# exp_sequence_llm_emb[:sequences_llm_emb.shape[0], :] = sequences_llm_emb

print(exp_smile_llm_emb.shape,exp_sequence_llm_emb.shape)

torch.save(exp_sequence_llm_emb,'exp_sequence_llm_emb.pt')
torch.save(exp_smile_llm_emb,'exp_smile_llm_emb.pt')
print('done!')