import torch
import torch.nn as nn 
import torchvision
from typing import List


class UpSample(nn.Module):
  def __init__(self, in_features=128, out_features=64,  hidden_size=64, extra_layer=True):
    super(UpSample, self).__init__()
    layers = []
    layers.append(nn.Conv2d(in_features, hidden_size, kernel_size=(3,3), padding=(1,1)))
    layers.append(nn.BatchNorm2d(hidden_size))
    layers.append(nn.ReLU())
    
    if extra_layer:
      layers.append(nn.Conv2d(hidden_size, out_features, kernel_size=(3,3), padding=(1,1)))
      layers.append(nn.BatchNorm2d(out_features))
      layers.append(nn.ReLU())
    
    layers.append(nn.ConvTranspose2d(out_features, out_features, kernel_size=(2,2), stride=(2,2)))
    layers.append(nn.ReLU())

    self.net = nn.Sequential(*layers) 

  def forward(self, x):
    return self.net(x)

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.layer = torchvision.models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')


	def forward(self, x):
		features = [x]
		temp = x
		# for k, v in self.layer.features.named_children():
		# 	features.append(v(features[-1]))
		# print(len(features))
		# return features
		for layer_no, (k, v) in enumerate(self.layer.features.named_children()):
			temp = v(temp)
			if layer_no in [1, 2, 6, 13, 18]:
				features.append(temp)
		return features

class Decoder(nn.Module):
	def __init__(self, layer_list=[1280, 160, 96, 64, 32, 24, 16, 3]):
		super(Decoder, self).__init__()
		self.decoder_1 = UpSample(1280, 96, 160)
		self.decoder_2 = UpSample(192, 32, 64)
		self.decoder_3 = UpSample(64, 24, 24, extra_layer=False)
		self.decoder_4 = UpSample(48, 16, 16, extra_layer=False)
		self.decoder_5 = UpSample(32, 3, 3, extra_layer=False)

	def forward(self, features: List[torch.Tensor]):
		x1 = self.decoder_1(features[-1]) #14*14
		x2 = self.decoder_2(torch.cat((x1, features[-2]), dim=1)) #28*28
		x3 = self.decoder_3(torch.cat((x2, features[-3]), dim=1)) #56*56
		x4 = self.decoder_4(torch.cat((x3, features[-4]), dim=1)) #112*112
		x5 = self.decoder_5(torch.cat((x4, features[-5]), dim=1)) #224*224
		return x5 

class Mobile_Net_Unet(nn.Module):
	def __init__(self):
		super(Mobile_Net_Unet, self).__init__()
		self.encoder = Encoder()
		self.decoder = Decoder()

	def forward(self, x):
		features = self.encoder(x)
		output = self.decoder(features)
		return output

if __name__ == '__main__':
	x = torch.randn((1,3,192,224))
	model = Mobile_Net_Unet()
	out = model(x)
	print(out.shape)
	pytorch_total_params = sum(p.numel() for p in model.parameters())
	print(pytorch_total_params/1000000)