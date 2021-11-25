import torch
from dataloader import MyDataset, dataset_sh
from transforms import RandomHorizontalFlip, Resize, ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
from model import Network
import numpy as np
import cv2
from PIL import Image
from matplotlib import cm
import numpy as np
import time
from mobile_net import Mobile_Net_Unet


def norma(image):
	return 1000.0 / image

if __name__ == '__main__':

	# input_image =cv2.imread("/home/akshay/Downloads/training/rgb/89.jpg")
	# dep = cv2.imread("/home/akshay/Downloads/training/depth/89.png")

	# dep = cv2.resize(dep, (224,224))
	# cv2.imshow("input_imag", dep)
	# input_image = cv2.resize(input_image, (224,224))

	# input_image = np.expand_dims(input_image, axis=0)
	# cv2.imshow("input_image", input_image[0])

	# input_image = input_image.reshape(1,3,input_image.shape[1],input_image.shape[2])


	# input_image = torch.from_numpy(input_image)
	# input_image = input_image.type(torch.float32) 
	# input_image /= 255

	model = Mobile_Net_Unet()

	checkpoint = torch.load("/home/akshay/DepthEstimation/checkpoint_24_mbnet.pth")
	model.load_state_dict(checkpoint['state_dict'])
	model.eval()


	custom_transform = transforms.Compose(
						       [RandomHorizontalFlip(),
					            Resize(),
					            ToTensor()])



	dataset = dataset_sh(custom_transform)
	data = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
	
	
	for b_no, batch_data in enumerate(data):
		image_input, target_image = batch_data['input_image'], batch_data['output_image']
		predicted_image = model(image_input)
		#predicted_image = np.clip(norma(predicted_image.detach()), 10, 1000) / 100
		target_image = target_image.detach().numpy()
		predicted_image = predicted_image.detach().numpy()
		for i in range(16):
			
			tr = np.transpose(target_image[i], (1,2,0))

			print(np.max(tr), np.min(tr))
			tr /= np.max(tr)


			
			pr = np.transpose(predicted_image[i], (1,2,0))
			print(np.min(pr), np.max(pr))
			pr = 1000/pr
			pr /= np.max(pr)

			target_image_PIL = Image.fromarray(np.uint8(tr*255)).convert('RGB')
			PIL_image = Image.fromarray(np.uint8(pr*255)).convert('RGB')
			PIL_image.show()
			target_image_PIL.show()
		break

