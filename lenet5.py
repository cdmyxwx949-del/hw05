#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

# ===================== LeNet-5 模型 =====================
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积层组
        self.features = nn.Sequential(
            # C1: 卷积
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            # S2: 池化
            nn.MaxPool2d(kernel_size=2, stride=2),
            # C3: 卷积
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            # S4: 池化
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 全连接层组
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ===================== 数据加载 =====================
def load_data(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# ===================== 训练函数 =====================
def train(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    train_losses = []
    start = time.time()
    for epoch in range(epochs):
        loss_sum, correct, total = 0,0,0
        for img, lab in train_loader:
            img, lab = img.to(device), lab.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, lab)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            _, pred = out.max(1)
            correct += pred.eq(lab).sum().item()
            total += lab.size(0)
        avg_loss = loss_sum / len(train_loader)
        acc = 100*correct/total
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
    print(f"训练耗时：{time.time()-start:.2f}s")
    return train_losses

# ===================== 测试函数 =====================
def test(model, test_loader, criterion, device):
    model.eval()
    correct, total, loss_sum = 0,0,0
    with torch.no_grad():
        for img, lab in test_loader:
            img, lab = img.to(device), lab.to(device)
            out = model(img)
            loss_sum += criterion(out, lab).item()
            _, pred = out.max(1)
            correct += pred.eq(lab).sum().item()
            total += lab.size(0)
    print(f"\n测试集：Loss={loss_sum/len(test_loader):.4f} | Acc={100*correct/total:.2f}%")

# ===================== 主程序 =====================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用设备：", device)

    train_loader, test_loader = load_data(batch_size=64)
    model = LeNet5().to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n开始训练 LeNet-5...")
    train_losses = train(model, train_loader, criterion, optimizer, device, epochs=5)
    test(model, test_loader, criterion, device)


# In[ ]:





# In[ ]:




