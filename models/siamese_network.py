# models/siamese_network.py

import torch
import torch.nn as nn
from models.base_model import BaseModel

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.base_model = BaseModel()  # 使用基本模型

    def forward(self, input1, input2):
        output1 = self.base_model(input1)
        output2 = self.base_model(input2)
        return output1, output2
