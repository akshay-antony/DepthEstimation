import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from transforms import RandomHorizontalFlip, Resize, ToTensor
from dataloader import MyDataset
import torchvision
from torchvision import transforms
from PIL import Image


if __name__ == '__main__':
	if torch.cuda.is_available():
		print("hi")
	else:
		print("no")

	custom_transform = transforms.Compose(
						       [RandomHorizontalFlip(),
					            Resize(),
					            ToTensor()])
	dataset = MyDataset(custom_transform)

	batch_data = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

	for data in batch_data:
		image_input, target_image = data['input_image'], data['output_image']
		target_image = target_image.detach().float() / 1000
		input_image = transforms.ToPILImage()(image_input[0]).convert('RGB')
		target_image = transforms.ToPILImage()(target_image[0]).convert('RGB')
		input_image.show()
		target_image.show()
		break
