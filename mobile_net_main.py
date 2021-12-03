import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from transforms import RandomHorizontalFlip, Resize, ToTensor, ToTensor_custom
from dataloader import MyDataset, dataset_sh
import torchvision
from torchvision import transforms
from PIL import Image
from mobile_net import Mobile_Net_Unet
from loss import *
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def find_psnr(pred, target):
  pred = pred.detach().cpu().numpy()
  target = target.detach().cpu().numpy()
  pred = np.transpose(pred, (0,2,3,1))
  target = np.transpose(target, (0,2,3,1))
  res = 0

  for i in range(pred.shape[0]):
    mse = np.mean((target[i] - pred[i]) ** 2) *(255/100)
    res += 20 * np.log10(255 / np.sqrt(mse))
    #res += peak_signal_noise_ratio(target[i], pred[i], data_range=100)

  return res 

def find_ssim(pred, target):
  pred = pred.detach().cpu().numpy()
  target = target.detach().cpu().numpy()
  pred = np.transpose(pred, (0,2,3,1))
  target = np.transpose(target, (0,2,3,1))
  res = 0

  for i in range(pred.shape[0]):
    res += structural_similarity(target[i], pred[i], data_range=100, multichannel=True)

  return res 

def norma(image, maxdepth=1000.):
	return maxdepth / image

def image_show(pred, target):
  target_image = np.transpose(target, (1,2,0))
  target_image /= np.max(target_image)

  predicted_image = np.transpose(pred, (1,2,0))
  predicted_image = 35. / predicted_image
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

  checkpoint = torch.load("/home/akshay/DepthEstimation/checkpoint_24_mbnet.pth")
  model.load_state_dict(checkpoint['state_dict'])
  epoch = checkpoint['epoch']

  model = model.to(device)
  custom_transform = transforms.Compose(
                     [RandomHorizontalFlip(),
                      Resize(),
                      ToTensor()])
  
  full_dataset = MyDataset(custom_transform)
  
  train_size = int(0.8 * len(full_dataset))
  test_size = len(full_dataset) - train_size
  train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

  loss_func = nn.L1Loss()
  loss_u = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  optimizer.load_state_dict(checkpoint['optimizer'])

  model.train()

  for i in range(0, 10):
    data = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    losses = 0 
    data_no = 0
    model.train()

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

      # if i%10 == 0 and b_no == 0:
      #   predicted_image_sample = predicted_image.detach().cpu().numpy()
      #   target_image_sample = target_image.detach().cpu().numpy()
      #   image_show(predicted_image_sample[0], target_image_sample[0])

      losses += loss.data.item()*image_input.shape[0]
      data_no += image_input.shape[0]

    torch.cuda.empty_cache()
    print("epoch:", i, "train loss", (losses / data_no))
    
    losses = 0 
    data_no = 0
    psnr_tot = 0 
    loss_u_ = 0
    ssim_tot = 0

    val_data = DataLoader(test_dataset, batch_size=12, shuffle=True)
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
      loss_u_ += torch.sqrt(loss_u(predicted_image, target_image_n)).data.item() * image_input.shape[0]

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
  #torch.save(checkpoint, "/home/akshay/DepthEstimation/checkpoint_28_final.pth")
