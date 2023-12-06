import sys

import torch
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListView, \
	QSizePolicy, QFileDialog

from models.siamese_network import SiameseNetwork


class CustomWidget(QWidget):
	def __init__(self):
		super().__init__()
		
		# 创建布局
		main_layout = QHBoxLayout()
		
		# 创建左侧部分
		upload_module = QWidget()
		upload_module_layout = QVBoxLayout()
		
		preview_window = QLabel("图片预览窗口内容")
		preview_window.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		upload_module_layout.addWidget(preview_window)
		
		upload_button = QPushButton("上传")
		upload_button.clicked.connect(self.handle_file_upload)
		upload_module_layout.addWidget(upload_button)
		
		upload_module.setLayout(upload_module_layout)
		main_layout.addWidget(upload_module)
		
		# 创建右侧部分
		compare_buttons = QWidget()
		compare_buttons_layout = QVBoxLayout()
		
		start_button = QPushButton("开始")
		start_button.clicked.connect(self.start_comparison)
		compare_buttons_layout.addWidget(start_button)
		
		pause_button = QPushButton("暂停")
		pause_button.clicked.connect(self.pause_comparison)
		compare_buttons_layout.addWidget(pause_button)
		pause_button.setEnabled(False)
		
		stop_button = QPushButton("终止")
		stop_button.clicked.connect(self.stop_comparison)
		stop_button.setEnabled(False)
		compare_buttons_layout.addWidget(stop_button)
		
		compare_buttons.setLayout(compare_buttons_layout)
		main_layout.addWidget(compare_buttons)
		
		# 添加结果区域
		self.result_area = QListView()
		self.result_model = QStandardItemModel()
		self.result_area.setModel(self.result_model)
		main_layout.addWidget(self.result_area)
		
		self.setLayout(main_layout)
	
	def handle_file_upload(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		file_names, _ = QFileDialog.getOpenFileNames(self, "选择图片文件", "",
		                                             "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)",
		                                             options=options)
		
		if file_names:
			for file_name in file_names:
				self.add_preview_item(file_name)
	
	def add_preview_item(self, file_name):
		# 处理文件上传逻辑
		pixmap = QPixmap(file_name).scaled(QSize(100, 100), Qt.KeepAspectRatio)
		item = QStandardItem()
		item.setData(pixmap, Qt.DecorationRole)
		item.setToolTip(file_name)
		self.result_model.appendRow(item)
	
	def start_comparison(self):
		# 获取上传的图片路径
		image_paths = []
		for row in range(self.result_model.rowCount()):
			item = self.result_model.item(row)
			image_paths.append(item.toolTip())
		
		# 使用模型进行比较
		if image_paths:
			most_similar_images = self.get_most_similar_images(image_paths)
			
			# 在结果列表中显示相似的2-3张图
			for image_path, similarity, similar_images in most_similar_images:
				self.add_result_item(image_path, similarity, similar_images)
		
		# 处理开始比较逻辑
		self.disable_comparison_buttons()
	
	def get_most_similar_images(self, image_paths):
		most_similar_images = []
		
		for image_path in image_paths:
			# 使用模型进行比较，这里需要根据实际情况调整
			similarity, similar_images = self.model.predict(image_path)
			most_similar_images.append((image_path, similarity, similar_images))
		
		return most_similar_images
	
	def add_result_item(self, image_path, similarity, similar_images):
		# 处理添加结果项逻辑
		item = QStandardItem(f"Path: {image_path}\nSimilarity: {similarity:.2%}")
		item.setToolTip(image_path)
		for similar_image in similar_images:
			pixmap = QPixmap(similar_image).scaled(QSize(100, 100), Qt.KeepAspectRatio)
			item.appendRow(QStandardItem(pixmap, ""))
		self.result_model.appendRow(item)
	
	def pause_comparison(self):
		# 处理暂停比较逻辑
		print("暂停比较")
		self.enable_comparison_buttons()
	
	def stop_comparison(self):
		# 处理终止比较逻辑
		print("终止比较")
		self.enable_comparison_buttons()
	
	def disable_comparison_buttons(self):
		# 锁定开始按钮，上传按钮
		self.findChild(QPushButton, "开始").setEnabled(False)
		self.findChild(QPushButton, "上传").setEnabled(False)
		self.findChild(QPushButton, "暂停").setEnabled(True)
		self.findChild(QPushButton, "终止").setEnabled(True)
	
	def enable_comparison_buttons(self):
		# 开放开始按钮，上传按钮
		self.findChild(QPushButton, "开始").setEnabled(True)
		self.findChild(QPushButton, "上传").setEnabled(True)
		self.findChild(QPushButton, "暂停").setEnabled(False)
		self.findChild(QPushButton, "终止").setEnabled(False)


if __name__ == '__main__':
	# 模型初始化
	# 请替换以下路径为你训练好的模型路径
	model_path = "saved_models/final_siamese_model.pth"
	model = SiameseNetwork()  # 请替换为你的 Siamese 模型类
	model.load_state_dict(torch.load(model_path))
	model.eval()
	
	app = QApplication(sys.argv)
	window = CustomWidget()
	window.show()
	sys.exit(app.exec_())
