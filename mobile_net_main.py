import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from transforms import RandomHorizontalFlip, Resize, ToTensor
from dataloader import MyDataset
import torchvision
from torchvision import transforms
from PIL import Image
from mobile_net import Mobile_Net_Unet
from loss import *
import numpy as np


def norma(image, maxdepth=1000.):
	return maxdepth / image

def image_show(pred, target):
  target_image = np.transpose(target, (1,2,0))
  target_image /= np.max(target_image)

  predicted_image = np.transpose(pred, (1,2,0))
  predicted_image = 1000. / predicted_image
  predicted_image /= np.max(predicted_image)
  
  target_image_PIL = Image.fromarray(np.uint8(target_image*255)).convert('RGB')
  predicted_image_PIL = Image.fromarray(np.uint8(predicted_image*255)).convert('RGB')
  target_image_PIL.show()
  predicted_image_PIL.show()

if __name__ == '__main__':
  if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda found")
 
  model = Mobile_Net_Unet()

  # checkpoint = torch.load("/home/akshay/DepthEstimation/checkpoint_22.pth")
  # model.load_state_dict(checkpoint['state_dict'])
  # epoch = checkpoint['epoch']

  model = model.to(device)
  custom_transform = transforms.Compose(
                     [RandomHorizontalFlip(),
                      Resize(),
                      ToTensor()])
  
  dataset = MyDataset(custom_transform)
  loss_func = nn.L1Loss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  # optimizer.load_state_dict(checkpoint['optimizer'])

  model.train()

  for i in range(0, 1000):
    data = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
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

      if i%10 == 0 and b_no == 1:
        predicted_image_sample = predicted_image.detach().cpu().numpy()
        target_image_sample = target_image.detach().cpu().numpy()
        image_show(predicted_image_sample[0], target_image_sample[0])
        # display(pr)
        # display(tr)

      losses += loss.data.item()*image_input.shape[0]
      data_no += image_input.shape[0]

    print("epoch:", i, "loss", (losses / data_no))

  checkpoint = {
      'epoch': i + 1,
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict()
    }
  torch.save(checkpoint, "/home/akshay/DepthEstimation/checkpoint_24_mbnet.pth")
