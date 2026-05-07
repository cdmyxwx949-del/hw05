#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. 极简CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(1, 16, 3, 1, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = self.fc(x)
        return x

# 2. 加载数据
def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# 3. 训练函数
def train(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, 准确率: {epoch_acc:.2f}%')
    return train_losses

# 4. 测试函数
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    print(f'测试损失: {test_loss:.4f}, 测试准确率: {accuracy:.2f}%')
    return test_loss, accuracy

# 5. 显示数据集
def display_data(data_loader, num_images=25):
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    grid_size = int(np.ceil(np.sqrt(num_images)))
    plt.figure(figsize=(10, 10))
    for i in range(min(num_images, images.shape[0])):
        plt.subplot(grid_size, grid_size, i+1)
        img = images[i][0].cpu().numpy()
        plt.imshow(img, cmap='gray')
        plt.title(f'标签: {labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 6. 可视化预测
def visualize_predictions(model, test_loader, device, num_images=5):
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    images = images.cpu()
    plt.figure(figsize=(12, 4))
    for i in range(min(num_images, images.shape[0])):
        plt.subplot(1, num_images, i+1)
        img = images[i][0].numpy()
        plt.imshow(img, cmap='gray')
        color = 'green' if predicted[i] == labels[i] else 'red'
        plt.title(f'预测: {predicted[i]}\n真实: {labels[i]}', color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 7. 主函数
def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    train_loader, test_loader = load_data(batch_size=64)
    display_data(train_loader)

    model = SimpleCNN().to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = train(model, train_loader, criterion, optimizer, device, epochs=5)
    test_loss, test_acc = test(model, test_loader, criterion, device)

    visualize_predictions(model, test_loader, device)

    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='训练损失')
    plt.title('训练损失曲线')
    plt.xlabel('轮数')
    plt.ylabel('损失')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()


# In[ ]:




