import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from transforms import RandomHorizontalFlip, Resize, ToTensor
from dataloader import MyDataset
import torchvision
from torchvision import transforms
from PIL import Image
from model import Network
from loss import *
from mobile_net_main import find_ssim, find_psnr


def norma(image, maxdepth=1000.):
	return maxdepth / image

if __name__ == '__main__':
	if torch.cuda.is_available():
		device = torch.device("cuda")

	model = Network()
	model = model.to(device)
	custom_transform = transforms.Compose(
						       [RandomHorizontalFlip(),
					            Resize(),
					            ToTensor()])

	checkpoint = torch.load("/home/akshay/DepthEstimation/checkpoint_24.pth")
	model.load_state_dict(checkpoint['state_dict'])
	epoch = checkpoint['epoch']

	full_dataset = MyDataset(custom_transform)
	train_size = int(0.8 * len(full_dataset))
	test_size = len(full_dataset) - train_size
	train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

	loss_func = nn.L1Loss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
	model.train()
	optimizer.load_state_dict(checkpoint['optimizer'])

	for i in range(10):
		data = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
		losses = 0 
		data_no = 0

		for b_no, batch_data in enumerate(data):
			image_input, target_image = batch_data['input_image'], batch_data['output_image']
			image_input = image_input.to(device)

			predicted_image = model(image_input)
			target_image_n = norma(target_image)

			target_image_n = target_image_n.to(device)
			L1_loss = loss_func(predicted_image, target_image_n)
			ssim_loss = torch.clamp((1 - ssim(predicted_image, target_image_n, val_range = 1000.0 / 10)) * 0.5, 0, 1)
			loss = ssim_loss + 0.1 * L1_loss
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			out = str(b_no) + "/" + str(90)
			#print("epoch=", i, "batch no=", out, "loss=", loss.item(), end="\r")

			losses += loss.data.item()*image_input.shape[0]
			data_no += image_input.shape[0]

		print("epoch:", i, "loss", (losses/data_no))

		losses = 0 
		data_no = 0
		psnr_tot = 0 
		loss_u_ = 0
		ssim_tot = 0

		val_data = DataLoader(test_dataset, batch_size=8, shuffle=True)
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

		  ssim_loss = torch.clamp((1 - ssim(predicted_image, target_image_n, val_range = 1000.0 / 10)) * 0.5, 0, 1)
		  loss = ssim_loss + 0.1 * L1_loss
		  losses += loss.data.item()*image_input.shape[0]
		  data_no += image_input.shape[0]

		print("epoch:", i, "val loss", (losses / data_no), "rmse", loss_u_ / data_no)
		print("epoch", i, "psnr", psnr_tot / data_no, "ssim", ssim_tot / data_no)

	checkpoint = {
    	'epoch': i + 1,
    	'state_dict': model.state_dict(),
   	 	'optimizer': optimizer.state_dict()
		}

	torch.save(checkpoint, "/home/akshay/DepthEstimation/checkpoint_24.pth")
