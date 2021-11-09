import torch
import torchvision
import numpy as np
import torch.nn as nn


class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.encoder = torchvision.models.vgg19(pretrained=True)
		for param in self.encoder.features.parameters():
			param.requires_grad = False

	def forward(self, x):
		output = [x]
		for k, v in self.encoder.features._modules.items():
			output.append(v(output[-1]))

		return output

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()

		self.uplayer0 = nn.Sequential(
							nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
							nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),				
							nn.Upsample(scale_factor=(2,2), mode='bilinear', align_corners=True), #14*14
							nn.ReLU())

		self.uplayer1 = nn.Sequential(
							nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
							nn.ReLU(),
							nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
							nn.ReLU(),
							nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
							nn.ReLU(),
							nn.Upsample(scale_factor=(2,2), mode='bilinear', align_corners=True), #28*18
							nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
							nn.ReLU(),
							nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
							nn.ReLU(),
							nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
							nn.ReLU(),
							nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
							nn.ReLU(),
                            nn.Upsample(scale_factor=(2,2), mode='bilinear', align_corners=True), #56*56
                            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
				 	 	 	nn.ReLU())

		self.uplayer2 = nn.Sequential(
				            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
							nn.ReLU(),
							nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 
							nn.ReLU(),
							nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
							nn.ReLU(),
							nn.Upsample(scale_factor=(2,2), mode='bilinear', align_corners=True), #112*112
							nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
							nn.ReLU())

		self.uplayer3 = nn.Sequential(
			                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
			                nn.ReLU(),
			                nn.Upsample(scale_factor=(2,2), mode='bilinear', align_corners=True), #224*224
			                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
			                nn.ReLU())

		self.uplayer4 = nn.Sequential(
			                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
			                nn.ReLU(),
			                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
			                )

	def forward(self, features):
		x1 = self.uplayer0(features[-1])
		x2 = self.uplayer1(torch.cat((x1, features[36]), axis=1))
		x3 = self.uplayer2(torch.cat((x2, features[17]), axis=1))
		x4 = self.uplayer3(torch.cat((x3, features[9]), axis=1))
		x5 = self.uplayer4(torch.cat((x4, features[4]), axis=1))
		return x5

			

class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		#self.encoder = 
		self.encoder = Encoder()
		self.decoder = Decoder()

	def forward(self, x):
		x1 = self.encoder(x)
		output = self.decoder(x1)
		return output

if __name__ == '__main__':
	a = torch.randn((1,3,224,224))
	model = Network()
	
	l = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(l/1000000)