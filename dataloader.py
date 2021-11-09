import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class MyDataset(Dataset):
	def __init__(self, transform=None):
		super(MyDataset, self).__init__()
		self.input_image_files = sorted(os.listdir("/home/akshay/Downloads/training/rgb"))
		self.output_image_files = sorted(os.listdir("/home/akshay/Downloads/training/depth"))
		self.transform = transform

	def __getitem__(self, idx):
		image_input = Image.open(os.path.join("/home/akshay/Downloads/training/rgb/", self.input_image_files[idx]))
		
		image_output = Image.open(os.path.join("/home/akshay/Downloads/training/depth/", self.output_image_files[idx]))
		training_data = {'input_image': image_input, 'output_image': image_output}


		if self.transform:
			training_data = self.transform(training_data)

		return training_data

	def __len__(self):
		return len(self.input_image_files)	

if __name__ == "__main__":
	pass