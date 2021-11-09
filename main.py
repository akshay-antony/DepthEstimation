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
	dataset = MyDataset(custom_transform)
	loss_func = nn.L1Loss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
	model.train()

	for i in range(1000):
		data = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)
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

	checkpoint = {
    	'epoch': i + 1,
    	'state_dict': model.state_dict(),
   	 	'optimizer': optimizer.state_dict()
		}

	torch.save(checkpoint, "/home/akshay/DepthEstimation/checkpoint.pth")
	torch.save(model.state_dict(), "/home/akshay/DepthEstimation/model_weights.pth")
	torch.save(optimizer.state_dict(), "/home/akshay/DepthEstimation/opt_weights.pth")