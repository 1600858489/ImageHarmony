# utils/data_loader.py

import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = self._get_image_paths()  # 调用私有方法
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)
        # return min(len(self.image_paths), 1000)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, img_path

    # 私有方法，获取所有图像路径
    def _get_image_paths(self):
        image_paths = []
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(dirpath, filename))
        return image_paths
