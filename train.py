import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from models.siamese_network import SiameseNetwork
from utils.data_loader import CustomDataset


# 检查CUDA是否可用
if torch.cuda.is_available():
    # 输出GPU设备信息
    print(f"CUDA is available. GPU device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Training on CPU.")


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 输出设备信息
print(f"Using device: {device}")

# 定义一些超参数
learning_rate = 0.001
num_epochs = 10
batch_size = 32

# 初始化数据集和数据加载器
dataset = CustomDataset(root_dir="E:/image")
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型和优化器，并将它们移到设备上
siamese_model = SiameseNetwork().to(device)
criterion = nn.CosineEmbeddingLoss()
optimizer = optim.Adam(siamese_model.parameters(), lr=learning_rate)

# 在训练循环中
for epoch in range(1, num_epochs + 1):
    for batch, (img1, img_path1) in enumerate(data_loader):
        img2, img_path2 = next(iter(data_loader))
        img1, img2 = img1.to(device), img2.to(device)
        
        # 处理最后一批次
        if len(img1) < batch_size:
            break

        # 模型前向传播
        output1, output2 = siamese_model(img1, img2)
        
        
        # 计算损失
        target = torch.ones(output1.size(0)).to(device)  # 使用 output1 的大小作为目标张量大小
        loss = criterion(output1, output2, target)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印损失
        if (batch + 1) % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Batch [{batch + 1}/{len(data_loader)}], Loss: {loss.item()}")

            # 保存模型
            # checkpoint_name = f'siamese_model_epoch{epoch}_batch{batch + 1}.pth'
            # torch.save(siamese_model.state_dict(), checkpoint_name)
            #
            
# 检查并创建保存模型的文件夹
save_folder = 'saved_models'
os.makedirs(save_folder, exist_ok=True)

# 在训练循环结束后保存最终的模型到指定位置
save_path = os.path.join(save_folder, 'final_siamese_model.pth')
torch.save(siamese_model.state_dict(), save_path)
print(f"Final model saved at: {save_path}")
