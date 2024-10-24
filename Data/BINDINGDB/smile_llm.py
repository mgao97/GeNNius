


from zhipuai import ZhipuAI
import pandas as pd
client = ZhipuAI(api_key="b98bf11fa3d3abc910f61f831bfc21b8.iEU2fs1qFhVS87aZ")  # 请填写您自己的APIKey

# df = pd.read_csv('smile_sequence.txt')
# df_SMILES = df['SMILES'].unique().tolist()[:5]

# response = client.chat.completions.create(
#     model="glm-4",  # 请填写您要调用的模型名称
#     messages=[
#         {"role": "user", 
#          "content": f"""
#         # Role:生物信息专家
#         # Skill:
#             - 掌握蛋白质靶标相互作用，能够准确识别药物的SMILES和英文名称。
#             - 输入的SMILES都是通过 https://pubchem.ncbi.nlm.nih.gov/ 网站数据库获得。
#         # Task:输入药物SMILES，一句话介绍正确的药物英文名称和英文特性，表格形式输出,一共两列，药物一列，英文名称和英文特性一列。
#         # Constraints:不需要输出其他内容。
#         # Example:
#         药物SMILES:C1=C(NC=N1)CCN
#         英文名称和特性:Imidazole, a basic organic compound containing a five-membered ring with two nitrogen atoms at positions 1 and 3, used as a starting material for synthesis in the pharmaceutical industry.

#         # 输入:
#         以下药物列表：{df_SMILES}
#         """
#          },
#     ],
#     stream=False,
# )
# print(response.choices[0].message.content)

# res = response.choices[0].message.content

# with open('smile_llm.txt','a','utf-8') as file:

import pandas as pd

df = pd.read_csv('smile_sequence.txt')
# 假设 df 是你的 DataFrame
df_SMILES = df['SMILES'].unique().tolist()

# 定义每批处理的行数
batch_size = 10

# 打开文件，准备追加内容
with open('test.txt', 'a', encoding='utf-8') as file:
    for i in range(0, len(df_SMILES), batch_size):
        # 获取当前批次的 SMILES 列表
        batch_smiles = df_SMILES[i:i + batch_size]

        # 构建请求内容
        content = f"""
        # Role:生物信息专家
        # Skill:
            - 掌握蛋白质靶标相互作用，能够准确识别药物的SMILES和英文名称。
            - 输入的SMILES都是通过 https://pubchem.ncbi.nlm.nih.gov/ 网站数据库获得。
        # Task:输入药物SMILES，一句话介绍正确的药物英文名称和英文特性，表格形式输出,一共两列，药物一列，英文名称和英文特性一列。
        # Constraints:严格输出表格形式，两列。
        # Example:
        药物SMILES:C1=C(NC=N1)CCN
        英文名称和特性:Imidazole, a basic organic compound containing a five-membered ring with two nitrogen atoms at positions 1 and 3, used as a starting material for synthesis in the pharmaceutical industry.

        # 输入:
        以下药物列表：{batch_smiles}
        """

        # 发送请求
        response = client.chat.completions.create(
            model="glm-4",  # 请填写您要调用的模型名称
            messages=[
                {"role": "user", "content": content},
            ],
            stream=False,
        )

        # 获取响应内容
        response_content = response.choices[0].message.content

        print(response_content)

        # 追加到文件
        file.write(response_content + '\n')

print("内容已成功追加到 test.txt")

