import torch
from dataloader import MyDataset
from transforms import RandomHorizontalFlip, Resize, ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
from model import Network
import numpy as np
import cv2
from PIL import Image
from matplotlib import cm

def norma(image):
	return 1000.0 / image

if __name__ == '__main__':

	custom_transform = transforms.Compose(
						       [RandomHorizontalFlip(),
					            Resize(),
					            ToTensor()])

	dataset = MyDataset(custom_transform)
	data = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
	model = Network()

	model.load_state_dict(torch.load("/home/akshay/DepthEstimation/model_weights.pth"))
	model.eval()

	for b_no, batch_data in enumerate(data):
		image_input, target_image = batch_data['input_image'], batch_data['output_image']
		predicted_image = model(image_input)
		print(torch.max(predicted_image[0]))
		#predicted_image = np.clip(norma(predicted_image.detach()), 10, 1000) / 100

		target_image = target_image.detach().numpy()
		tr = np.transpose(target_image[0], (1,2,0))
		tr /= np.max(tr)


		predicted_image = predicted_image.detach().numpy()
		pr = np.transpose(predicted_image[0], (1,2,0))
		print(np.max(pr))
		pr = 1000/pr
		pr /= np.max(pr)

		target_image_PIL = Image.fromarray(np.uint8(tr*255)).convert('RGB')
		PIL_image = Image.fromarray(np.uint8(pr*255)).convert('RGB')
		PIL_image.show()
		target_image_PIL.show()
		break