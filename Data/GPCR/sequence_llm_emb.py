from zhipuai import ZhipuAI

import pandas as pd
import json
import torch


# df = pd.read_csv('smile_llm1.txt',on_bad_lines='skip')

# 定义一个空列表，用来存储每行第一个逗号后的内容
elements_list = []

# 按行读取文件
with open('sequence_llm.txt', 'r', encoding='utf-8') as file:
    for line in file:
        # 使用逗号分割每一行，并提取第一个逗号后的部分
        parts = line.split(',', 1)  # 使用 maxsplit=1 只分割一次
        if len(parts) > 1:  # 确保该行有逗号，避免索引错误
            elements_list.append(parts[1].strip())  # 去除多余的空白字符

sequence_text_dict = dict(zip([i for i in range(len(elements_list))],elements_list))

# 将字典保存为JSON格式
with open('sequence_text_dict.json', 'w', encoding='utf-8') as f:
    json.dump(sequence_text_dict, f, ensure_ascii=False, indent=4)

print("字典已成功保存为JSON格式!")


# client = ZhipuAI(api_key="b98bf11fa3d3abc910f61f831bfc21b8.iEU2fs1qFhVS87aZ")  # 请填写您自己的APIKey

# response = client.embeddings.create(
#     model="embedding-3", #填写需要调用的模型编码
#     input=elements_list,
# )

# print(response.data)

from zhipuai import ZhipuAI

# elements_list = elements_list[:10]


client = ZhipuAI(api_key="b98bf11fa3d3abc910f61f831bfc21b8.iEU2fs1qFhVS87aZ") 

response_batch = []
# response_list = []
all_embeddings = []
# 定义每批处理的行数
batch_size = 4


for i in range(0, len(elements_list), batch_size):
    # 获取当前批次的 SMILES 列表
    batch_smiles = elements_list[i:i + batch_size]
    response = client.embeddings.create(
        model="embedding-3", #填写需要调用的模型编码
        input=batch_smiles,
        dimensions=64,)
    
    for j in range(len(response.data)):
        # 获取响应内容
        
        response_content = response.data[j].embedding
        
        response_batch.append(response_content)
        
# 将每个embedding的list append到all_embeddings中
for response in response_batch:
    all_embeddings.append(response)

# 转换为PyTorch的tensor
tensor_data = torch.tensor(all_embeddings)

# 保存为.pt文件
torch.save(tensor_data, 'sequence_llm_emb.pt')

print("数据已成功保存为 sequence_llm_emb.pt 文件！")
        
# data  = torch.load('smile_llm_emb.pt')
# print(data,data.shape)