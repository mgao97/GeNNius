#!/bin/bash
# 定义Python脚本的名称
script="Code/test_hgt_ours_rf_biosnap_lr.py"

# 定义embedding size数组
lr_size=(0.001 0.005 0.01 0.05)

# 定义日志文件前缀
log_prefix="test_hgt_ours_rf_biosnap_lr"

# 遍历embsize数组
for i in {1..4}; do
    lr=${lr_size[$((i-1))]}
    log_file="${log_prefix}-lr${i}-1101.log"

    echo "正在运行: nohup python $script --n_lrs $lr > $log_file 2>&1 &"
    # 使用nohup运行Python脚本，并将输出重定向到日志文件
    # 2>&1 表示将stderr也重定向到stdout
    # & 在命令末尾表示在后台运行
    nohup python $script --lr $lr > $log_file 2>&1 &

    # 保存最后一个后台进程的PID
    last_pid=$!
    # 等待最后一个后台进程完成
    wait $last_pid
done

echo "所有脚本已在后台开始执行。"
