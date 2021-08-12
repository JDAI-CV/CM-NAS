import os
import random
import numpy as np
from PIL import Image
import torch.utils.data as data


class TestData(data.Dataset):
	def __init__(self, test_img_list, test_label, transform=None, img_size=(128,256)):
		test_image = []
		for i in range(len(test_img_list)):
			img = Image.open(test_img_list[i])
			img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
			pix_array = np.array(img)
			test_image.append(pix_array)
		test_image = np.array(test_image)
		self.test_image = test_image
		self.test_label = test_label
		self.transform = transform

	def __getitem__(self, index):
		img, target = self.test_image[index],  self.test_label[index]
		img = self.transform(img)
		return img, target

	def __len__(self):
		return len(self.test_image)
