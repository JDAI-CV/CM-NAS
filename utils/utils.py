import os
import sys
import math
import copy
import shutil
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0 

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def count_parameters_in_MB(model):
	return sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


class Logger(object):
	"""
	Write console output to external text file.
	Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
	"""
	def __init__(self, log_path=None):
		self.console = sys.stdout
		self.file = None
		if log_path is not None:
			mkdir_if_missing(os.path.dirname(log_path))
			self.file = open(log_path, 'w')

	def __del__(self):
		self.close()

	def __enter__(self):
		pass

	def __exit__(self, *args):
		self.close()

	def write(self, msg):
		self.console.write(msg)
		if self.file is not None:
			self.file.write(msg)

	def flush(self):
		self.console.flush()
		if self.file is not None:
			self.file.flush()
			os.fsync(self.file.fileno())

	def close(self):
		self.console.close()
		if self.file is not None:
			self.file.close()


class RandomErasing(object):
	""" Randomly selects a rectangle region in an image and erases its pixels.
		'Random Erasing Data Augmentation' by Zhong et al.
		See https://arxiv.org/pdf/1708.04896.pdf
	Args:
		p: The prob that the Random Erasing operation will be performed.
		sl: Minimum proportion of erased area against input image.
		sh: Maximum proportion of erased area against input image.
		r1: Minimum aspect ratio of erased area.
		mean: Erasing value.
	"""
	def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
		self.p = p
		self.mean = mean
		self.sl = sl
		self.sh = sh
		self.r1 = r1

	def __call__(self, img):
		if random.uniform(0, 1) >= self.p:
			return img

		for attempt in range(100):
			area = img.size()[1] * img.size()[2]

			target_area = random.uniform(self.sl, self.sh) * area
			aspect_ratio = random.uniform(self.r1, 1 / self.r1)

			h = int(round(math.sqrt(target_area * aspect_ratio)))
			w = int(round(math.sqrt(target_area / aspect_ratio)))

			if w < img.size()[2] and h < img.size()[1]:
				x1 = random.randint(0, img.size()[1] - h)
				y1 = random.randint(0, img.size()[2] - w)
				if img.size()[0] == 3:
					img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
					img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
					img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
				else:
					img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
				return img

		return img


class IdentitySampler(Sampler):
	"""Sample person identities evenly in each batch.
	Args:
		train_visible_label, train_thermal_label: labels of two modalities
		visible_idxs_dict, thermal_idxs_dict: indices of each identity 
		num_pos: num of pos per identity in each modality
		batch_size: batch size, and the num of identity per batch is batch_size/num_pos
	"""
	def __init__(self, train_visible_label, train_thermal_label, visible_idxs_dict, thermal_idxs_dict, num_pos, batch_size):
		unique_label = np.unique(train_visible_label)

		N = np.maximum(len(train_visible_label), len(train_thermal_label))
		for j in range(int(N/batch_size)+1):
			batch_idx = np.random.choice(unique_label, batch_size//num_pos, replace=False)
			for i in range(batch_size//num_pos):
				sample_visible = np.random.choice(visible_idxs_dict[batch_idx[i]], num_pos)
				sample_thermal = np.random.choice(thermal_idxs_dict[batch_idx[i]], num_pos)

				if j == 0 and i == 0:
					index_visible = sample_visible
					index_thermal = sample_thermal
				else:
					index_visible = np.hstack((index_visible, sample_visible))
					index_thermal = np.hstack((index_thermal, sample_thermal))

		self.index_visible = index_visible
		self.index_thermal = index_thermal

	def __iter__(self):
		return iter(np.arange(len(self.index_visible)))

	def __len__(self):
		return len(self.index_visible)


def accuracy(output, target, topk=(1,)):
	""" Computes the precision@k for the specified values of k """
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


def save_checkpoint(state, save_root, epoch):
	filename = os.path.join(save_root, 'checkpoint{:03}.pth.tar'.format(epoch))
	torch.save(state, filename)


def mkdir_if_missing(directory):
	if not os.path.exists(directory):
		try:
			os.makedirs(directory)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise  


def create_exp_dir(path, scripts_to_save=None):
	mkdir_if_missing(path)
	print('Experiment dir : {}'.format(path))

	if scripts_to_save is not None:
		mkdir_if_missing(os.path.join(path, 'scripts'))
		for script in scripts_to_save:
			dst_file = os.path.join(path, 'scripts', os.path.basename(script))
			shutil.copyfile(script, dst_file)


def set_seed(seed, cuda=True):
	np.random.seed(seed)
	torch.manual_seed(seed)
	if cuda:
		torch.cuda.manual_seed(seed)


def set_requires_grad(nets, requires_grad=False):
	"""Set requies_grad=Fasle for all the networks to avoid unnecessary computations
	Parameters:
		nets (network list)   -- a list of networks
		requires_grad (bool)  -- whether the networks require gradients or not
	"""
	if not isinstance(nets, list):
		nets = [nets]
	for net in nets:
		if net is not None:
			for param in net.parameters():
				param.requires_grad = requires_grad


def gen_idxs_dict(train_visible_label, train_thermal_label):
	unique_label_visible = np.unique(train_visible_label)
	visible_idxs_dict = {label:[] for label in unique_label_visible}
	for label in unique_label_visible:
		tmp_pos = [k for k,v in enumerate(train_visible_label) if v==label]
		visible_idxs_dict[label] += tmp_pos

	unique_label_thermal = np.unique(train_thermal_label)
	thermal_idxs_dict = {label:[] for label in unique_label_thermal}
	for label in unique_label_thermal:
		tmp_pos = [k for k,v in enumerate(train_thermal_label) if v==label]
		thermal_idxs_dict[label] += tmp_pos
	
	return visible_idxs_dict, thermal_idxs_dict


class EMA():
	def __init__(self, model, decay):
		self.model = model
		self.decay = decay
		self.shadow = {}

	def register(self):
		for name, state in self.model.state_dict().items():
			self.shadow[name] = state.clone()

	def update(self):
		for name, state in self.model.state_dict().items():
			assert name in self.shadow
			new_average = (1.0 - self.decay) * state + self.decay * self.shadow[name]
			self.shadow[name] = new_average.clone()
			del new_average

	def state_dict(self):
		return self.shadow

	def load_state_dict(self, state_dict):
		for name, state in state_dict.items():
			self.shadow[name] = state.clone()