from zhipuai import ZhipuAI
import pandas as pd
client = ZhipuAI(api_key="b98bf11fa3d3abc910f61f831bfc21b8.iEU2fs1qFhVS87aZ")  # 请填写您自己的APIKey


column_names = 'SMILES,SEQUENCE,LABLE\n'

# 以读模式打开文件，并读取所有内容
with open('smile_sequence.txt', 'r') as file:
    lines = file.readlines()
    
# 在第一行添加列名
lines.insert(0, column_names)


# 以写模式打开文件，并写入新的内容
with open('smile_sequence.txt', 'w') as file:
    file.writelines(lines)

print('done!')


df = pd.read_csv('smile_sequence.txt')
df_SMILES = df['SEQUENCE'].unique().tolist()


# 定义每批处理的行数
batch_size = 10


# 打开文件，准备追加内容
with open('sequence_llm.txt', 'a', encoding='utf-8') as file:
    for i in range(0, len(df_SMILES), batch_size):
        # 获取当前批次的 SMILES 列表
        batch_smiles = df_SMILES[i:i + batch_size]


        content= f"""
        # Role:生物信息专家
        # Skill:
            - 掌握蛋白质靶标相互作用，能够准确识别靶标sequence的特性和包含的结构。
            - 输入的target sequence都是通过 网站数据库获得。
        # Task:输入靶标sequence，一句话介绍对应的靶标sequence的特性和包含的结构，按行输出,首先是target，然后是target结构特性和包含子结构的精简说明（英文），target和后面的内容用逗号隔开。
        # Constraints:不需要输出其他内容和有关大模型或其他网站的任何说明。
        # Example:
        靶标Sequence,英文版本的特性和包含的结构

        # 输入:
        以下药物列表：{batch_smiles}
        """


        response = client.chat.completions.create(
            model="glm-4",  # 请填写您要调用的模型名称
            messages=[
            {"role": "user", 
         "content": content},
            ],
            stream=False,
        )
        # 获取响应内容
        response_content = response.choices[0].message.content

        print(response_content)

        # 追加到文件
        file.write(response_content + '\n')

print("内容已成功追加到 sequence llm.txt")