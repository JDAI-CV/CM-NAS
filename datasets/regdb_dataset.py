import os
import random
import numpy as np
from PIL import Image
import torch.utils.data as data


class RegDBData(data.Dataset):
	def __init__(self, data_root, trial, transform=None, visibleIndex=None, thermalIndex=None, img_size=(128,256)):
		# Load training images (path) and labels
		train_visible_list = os.path.join(data_root, 'idx/train_visible_{}.txt'.format(trial))
		train_thermal_list = os.path.join(data_root, 'idx/train_thermal_{}.txt'.format(trial))

		visible_img_file, train_visible_label = load_data(train_visible_list)
		thermal_img_file, train_thermal_label = load_data(train_thermal_list)

		train_visible_image = []
		for i in range(len(visible_img_file)):
			img = Image.open(os.path.join(data_root, visible_img_file[i]))
			img = img.resize(img_size, Image.ANTIALIAS)
			pix_array = np.array(img)
			train_visible_image.append(pix_array)
		train_visible_image = np.array(train_visible_image) 

		train_thermal_image = []
		for i in range(len(thermal_img_file)):
			img = Image.open(os.path.join(data_root, thermal_img_file[i]))
			img = img.resize(img_size, Image.ANTIALIAS)
			pix_array = np.array(img)
			train_thermal_image.append(pix_array)
		train_thermal_image = np.array(train_thermal_image)

		# BGR to RGB
		self.train_visible_image = train_visible_image  
		self.train_visible_label = train_visible_label

		# BGR to RGB
		self.train_thermal_image = train_thermal_image
		self.train_thermal_label = train_thermal_label

		self.transform = transform
		self.vIndex = visibleIndex
		self.tIndex = thermalIndex

	def __getitem__(self, index):
		img_v, target_v = self.train_visible_image[self.vIndex[index]], self.train_visible_label[self.vIndex[index]]
		img_t, target_t = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

		img_v = self.transform(img_v)
		img_t = self.transform(img_t)

		return img_v, img_t, target_v, target_t

	def __len__(self):
		# return len(self.train_visible_label)
		return len(self.vIndex)


def process_test_regdb(data_root, trial=1, modality='visible'):
	if modality=='visible':
		data_path = os.path.join(data_root, 'idx/test_visible_{}.txt'.format(trial))
	elif modality=='thermal':
		data_path = os.path.join(data_root, 'idx/test_thermal_{}.txt'.format(trial))

	file_image, file_label = load_data(data_path)
	file_image = [os.path.join(data_root, f) for f in file_image]

	return file_image, np.array(file_label)


def load_data(data_path):
	with open(data_path, 'r') as f:
		data_file_list = f.read().splitlines()
		# Get full list of image and labels
		file_image = [s.split(' ')[0] for s in data_file_list]
		file_label = [int(s.split(' ')[1]) for s in data_file_list]

	return file_image, file_label