\# 调试笔记



\## 1. ModuleNotFoundError: No module named 'torchvision'

\- 现象：导入库失败

\- 原因：未安装torchvision

\- 解决：pip install torch torchvision



\## 2. Conda安装卡住

\- 现象：Solving environment 无限等待

\- 原因：conda源慢

\- 解决：改用pip安装



\## 3. 维度不匹配错误

\- 现象：Linear层输入报错

\- 原因：LeNet-5必须是16\*5\*5=400

\- 解决：nn.Linear(400, 120)



\## 4. matplotlib中文乱码

\- 现象：中文显示方框

\- 原因：无中文字体

\- 解决：设置SimHei字体

