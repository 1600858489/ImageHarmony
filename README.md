
# Siamese Network for Image Similarity

## 模型训练

### 1. 数据集准备

确保你有一个包含训练数据的文件夹。每个文件夹内应包含两个子文件夹，分别为`positive`和`negative`，用于存放相似和不相似的图像对。
```markdown
dataset/
|-- class1/
|   |-- positive/
|       |-- image1.jpg
|       |-- image2.jpg
|   |-- negative/
|       |-- image3.jpg
|       |-- image4.jpg
|-- class2/
...
```

### 2. 模型训练

使用以下代码训练Siamese Network模型：

```bash
python train_siamese_model.py --data_path /path/to/dataset --output_model_path saved_models/final_siamese_model.pth
```

这将在`/path/to/dataset`目录下的数据上训练模型，并将训练好的模型保存为`saved_models/final_siamese_model.pth`。

## 模型调用

### 1. 导入模型

首先，导入Siamese Network模型：

```python
import torch
from models.siamese_network import SiameseNetwork

model_path = "saved_models/final_siamese_model.pth"
model = SiameseNetwork()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
```

### 2. 图像比对

使用以下代码进行图像比对：

```python
from PIL import Image
from torchvision.transforms import transforms

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# 图片路径
image1_path = "path/to/image1.jpg"
image2_path = "path/to/image2.jpg"

# 预处理图像
image1 = preprocess_image(image1_path)
image2 = preprocess_image(image2_path)

# 比对图像
with torch.no_grad():
    output1, output2 = model(image1, image2)

# 处理输出结果，例如计算相似度
similarity_percentage = # 计算相似度的逻辑

print(f"Image similarity: {similarity_percentage}%")
```

请根据实际情况修改文件路径和计算相似度的逻辑。
```

请注意，这是一个基本示例，具体的模型训练和调用接口可能需要根据你的具体情况进行调整。
