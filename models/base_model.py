# models/base_model.py

import torch.nn as nn
import torchvision.models as models


class BaseModel(nn.Module):
	def __init__(self):
		super(BaseModel, self).__init__()
		
		# 使用预训练的ResNet18模型
		resnet18 = models.resnet18(pretrained=True)
		
		# 移除最后的全连接层
		self.resnet_features = nn.Sequential(*list(resnet18.children())[:-1])
		
		# 冻结卷积层的参数，只训练全连接层
		for param in self.resnet_features.parameters():
			param.requires_grad = False
		
		# 全连接层
		self.fc = nn.Linear(512, 512)  # 输入维度是ResNet的输出维度，请根据实际情况调整
	
	def forward(self, x):
		# 提取ResNet的特征
		features = self.resnet_features(x)
		
		# 将特征展平
		features = features.view(features.size(0), -1)
		
		# 全连接层
		output = self.fc(features)
		
		return output
