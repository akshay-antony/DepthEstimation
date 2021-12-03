import cv2 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mobile_net import Mobile_Net_Unet
from dataloader import KittiDataset
from transforms import RandomHorizontalFlip, Resize_KITTI, ToTensor_KITTI
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from loss import *
from mobile_net_main import find_ssim, find_psnr

def norma(image, maxdepth=256.):
	return maxdepth / image

def train():
	if torch.cuda.is_available():
		device = torch.device("cuda")
	checkpoint = torch.load("/home/akshay/DepthEstimation/checkpoint_28_kitti.pth")

	model = Mobile_Net_Unet()
	model = model.to(device)
	model.load_state_dict(checkpoint['state_dict'])
	custom_transform = transforms.Compose([RandomHorizontalFlip(),
					            		   Resize_KITTI(),
					            		   ToTensor_KITTI()])


	dataset = KittiDataset(custom_transform)
	loss_func = nn.L1Loss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
	optimizer.load_state_dict(checkpoint['optimizer'])
	model.train()

	for i in range(250, 550):
		data = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)
		losses = 0 
		data_no = 0

		for b_no, batch_data in enumerate(data):
			image_input, target_image = batch_data['input_image'], batch_data['output_image']
			image_input = image_input.to(device)

			predicted_image = model(image_input)
			target_image_n = norma(target_image)

			target_image_n = target_image_n.to(device)
			L1_loss = loss_func(predicted_image, target_image_n)
			ssim_loss = torch.clamp((1 - ssim(predicted_image, target_image_n, val_range = 256.0 / 1)) * 0.5, 0, 1)
			loss = ssim_loss + 0.1 * L1_loss
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			out = str(b_no) + "/" + str(90)

			losses += loss.data.item()*image_input.shape[0]
			data_no += image_input.shape[0]

		print("epoch:", i, "loss", (losses/data_no))

	checkpoint = {
    	'epoch': i + 1,
    	'state_dict': model.state_dict(),
   	 	'optimizer': optimizer.state_dict()
		}

	torch.save(checkpoint, "/home/akshay/DepthEstimation/checkpoint_28_kitti.pth")

def validation():
	if torch.cuda.is_available():
		device = torch.device("cuda")

	checkpoint = torch.load("/home/akshay/DepthEstimation/checkpoint_28_kitti.pth")
	model = Mobile_Net_Unet()
	model.load_state_dict(checkpoint['state_dict'])
	model = model.to(device)
	custom_transform = transforms.Compose([RandomHorizontalFlip(),
					            		   Resize_KITTI(),
					            		   ToTensor_KITTI()])


	test_dataset = KittiDataset(custom_transform, filename1="/home/akshay/Downloads/KITTI/data/val/X/", filename2="/home/akshay/Downloads/KITTI/data/val/target/")
	loss_func = nn.L1Loss()
	losses = 0 
	data_no = 0
	psnr_tot = 0 
	loss_u_ = 0
	ssim_tot = 0

	val_data = DataLoader(test_dataset, batch_size=64, shuffle=True)
	model.eval()

	for b_no, batch_data in enumerate(val_data):
	  image_input, target_image = batch_data['input_image'], batch_data['output_image']
	  image_input = image_input.to(device)

	  predicted_image = model(image_input)
	  target_image_n = norma(target_image)

	  psnr_tot += find_psnr(predicted_image, target_image_n)
	  ssim_tot += find_ssim(predicted_image, target_image_n)

	  target_image_n = target_image_n.to(device)
	  L1_loss = loss_func(predicted_image, target_image_n)

	  ssim_loss = torch.clamp((1 - ssim(predicted_image, target_image_n, val_range = 256)) * 0.5, 0, 1)
	  loss = ssim_loss + 0.1 * L1_loss
	  losses += loss.data.item()*image_input.shape[0]
	  data_no += image_input.shape[0]

	print("val loss", (losses / data_no), "psnr", psnr_tot / data_no, "ssim", ssim_tot / data_no)

def test():
	checkpoint = torch.load("/home/akshay/DepthEstimation/checkpoint_28_kitti.pth")
	model = Mobile_Net_Unet()
	model.load_state_dict(checkpoint['state_dict'])
	model = model.to(device)
	custom_transform = transforms.Compose([RandomHorizontalFlip(),
					            		   Resize_KITTI(),
					            		   ToTensor_KITTI()])


if __name__ == "__main__":
	validation()