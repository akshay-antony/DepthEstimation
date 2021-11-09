import torch
import torchvision
from torchvision import transforms
import random
import numpy as np
from PIL import Image

	
def _is_pil_image(img):
    return isinstance(img, Image.Image)

class RandomHorizontalFlip(object):
	def __call__(self, sample_data):
		input_image, output_image = sample_data['input_image'], sample_data['output_image']

		if random.random() < 0.5:
			input_image = input_image.transpose(Image.FLIP_LEFT_RIGHT)
			output_image = output_image.transpose(Image.FLIP_LEFT_RIGHT)	

		return {'input_image': input_image, 'output_image': output_image}

class Resize(object):
	def __call__(self, sample_data):
		input_image, output_image = sample_data['input_image'], sample_data['output_image']
		input_image = input_image.resize((224,224))
		output_image = output_image.resize((224,224))

		return {'input_image': input_image, 'output_image': output_image}

class ToTensor(object):
	def __call__(self, sample_data):
		input_image, output_image = sample_data['input_image'], sample_data['output_image']
		input_image  = self.to_tensor(input_image)
		output_image = self.to_tensor(output_image).float() * 1000
		output_image = torch.clamp(output_image, min=10, max=1000)

		return {'input_image': input_image, 'output_image': output_image}

	def to_tensor(self, image):
		if not(_is_pil_image(image) or _is_numpy_image(image)):
			raise TypeError('image should be PIL Image or ndarray. Got {}'.format(type(image)))

		if isinstance(image, np.ndarray):
			img = torch.from_numpy(image.transpose((2, 0, 1)))

		if image.mode == 'I':
			img = torch.from_numpy(np.array(image, np.int32, copy=False))
		elif image.mode == 'I;16':
			img = torch.from_numpy(np.array(image, np.int16, copy=False))
		else:
			img = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        
		if image.mode == 'YCbCr':
			nchannel = 3
		elif image.mode == 'I;16':
			nchannel = 1
		else:
			nchannel = len(image.mode)
        
		img = img.view(image.size[1], image.size[0], nchannel)

		img = img.transpose(0, 1).transpose(0, 2).contiguous()
		
		if isinstance(img, torch.ByteTensor):
		    return img.float().div(255)
		else:
		    return img


