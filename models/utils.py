import torch
import torch.nn as nn


def weights_init_kaiming(m):
	classname = m.__class__.__name__
	if 'Linear' in classname:
		nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
		if m.bias is not None:
			nn.init.constant_(m.bias, 0.0)
	elif 'Conv' in classname:
		nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
		if m.bias is not None:
			nn.init.constant_(m.bias)
	elif 'BatchNorm' in classname:
		if m.affine:
			nn.init.normal_(m.weight, 1.0, 0.01)
			nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
	classname = m.__class__.__name__
	if 'Linear' in classname:
		nn.init.normal_(m.weight, std=0.001)
		if m.bias:
			nn.init.constant_(m.bias, 0.0)