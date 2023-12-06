import os
import torch
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import torch.nn.functional as F

# 加载预训练的ResNet模型
resnet_model = models.resnet18(pretrained=True)
resnet_model.eval()


# 图像预处理
def preprocess_image(image_path):
	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	image = Image.open(image_path).convert("RGB")
	return transform(image).unsqueeze(0)


# 提取图像特征
def extract_features(model, image):
	with torch.no_grad():
		features = model(image)
	return features


# 计算相似度
def compute_similarity(features1, features2):
	# 使用余弦相似度计算相似度
	similarity = F.cosine_similarity(features1, features2).item()
	return similarity


if __name__ == "__main__":
	model_path = "saved_models/final_siamese_model.pth"
	model = torch.load(model_path, map_location=torch.device('cpu'))
	
	images_dir = "images"
	image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
	
	for i in range(len(image_files)):
		for j in range(len(image_files)):
			image1_path = os.path.join(images_dir, image_files[i])
			image2_path = os.path.join(images_dir, image_files[j])
			
			# 预处理图像
			image1 = preprocess_image(image1_path)
			image2 = preprocess_image(image2_path)
			
			# 提取图像特征
			features1 = extract_features(resnet_model, image1)
			features2 = extract_features(resnet_model, image2)
			
			# 计算相似度
			similarity = compute_similarity(features1, features2)
			
			print(f"Similarity between {image_files[i]} and {image_files[j]}: {similarity}")
		print("-------------------------------------------------------------------------")
