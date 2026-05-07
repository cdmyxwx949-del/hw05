# HW05 手写数字识别实验

## 目录结构
- simple_cnn.py    任务一：极简CNN（来源公众号）
- lenet5.py        任务二：LeNet-5实现
- report.md        实验报告
- debug_notes.md   调试记录
- requirements.txt 依赖环境

## 环境安装
pip install -r requirements.txt

## 运行命令
# 运行任务一（极简CNN）
python simple_cnn.py

# 运行任务二（LeNet-5）
python lenet5.py

## 说明
- 数据集自动下载到 ./data 文件夹
- 自动使用GPU，无GPU则使用CPU
- 训练5轮，自动输出测试准确率
