import numpy as np 
from PIL import Image
import os

if __name__ == '__main__':
	image_files = sorted(os.listdir("/home/akshay/Downloads/KITTI/data/val/y/"))

	for i, image in enumerate(image_files):
		print(i)
		rgba_image = Image.open(os.path.join("/home/akshay/Downloads/KITTI/data/val/y/", image))
		rgba_image.load()
		background = Image.new("RGB", rgba_image.size, (255, 255, 255))
		background.paste(rgba_image, mask = rgba_image.split()[3])
		first_name = image.split(".")[0]
		background.save("/home/akshay/Downloads/KITTI/data/val/target/" + first_name + "jpg", "JPEG", quality=100)