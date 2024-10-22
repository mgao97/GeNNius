import subprocess

# 定义要运行的Python文件列表
python_files = ['label.py', 'smile_to_image.py', 'smile_to_features.py','smile_k_gram.py','protein_k_gram','main.py']

# 遍历列表，按顺序运行每个脚本
for file in python_files:
    try:
        # 使用subprocess.run运行Python脚本
        print(f"正在运行 {file}...")
        result = subprocess.run(['python', file], check=True, text=True, capture_output=True)
        # 打印脚本的输出
        print(f"{file} 的输出：")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        # 如果脚本运行失败，打印错误信息
        print(f"运行 {file} 时出错：")
        print(e.stderr)
    except Exception as e:
        # 捕获其他可能的异常
        print(f"运行 {file} 时发生异常：")
        print(str(e))

print("所有脚本已按顺序运行完毕。")