import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from .utils import weights_init_kaiming
from .utils import weights_init_classifier
from .resnet import model_urls, conv3x3, remove_fc
from .resnet import BasicBlock, Bottleneck


class SwitchBatchNorm2d(nn.Module):
	def __init__(self, num_features, switch, eps=1e-05, momentum=0.1, affine=True, 
									 track_running_stats=True, with_relu=True):
		super(SwitchBatchNorm2d, self).__init__()
		self.thermal_bn  = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
		self.visible_bn  = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
		self.sharable_bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
		self.relu = nn.ReLU(inplace=True) if with_relu else None
		self.switch = switch # 0 for split_x, 1 for share_x

	def forward(self, x, mode=0):
		if self.switch == 0:
			# split bn
			if mode == 0:
				x_v, x_t = torch.split(x, x.size(0)//2, dim=0)
				x_v = self.visible_bn(x_v)
				x_t = self.thermal_bn(x_t)
				split_x = torch.cat((x_v, x_t), 0)
			elif mode == 1:
				split_x = self.visible_bn(x)
			elif mode == 2:
				split_x = self.thermal_bn(x)
			split_x = self.relu(split_x) if self.relu is not None else split_x
			return split_x
		elif self.switch == 1:
			# share bn
			share_x = self.sharable_bn(x)
			share_x = self.relu(share_x) if self.relu is not None else share_x
			return share_x
		else:
			raise ValueError('Invalid switch value: {}, must be 0 or 1.'.format(self.switch))


class BasicBlockSwitchBN(nn.Module):
	expansion = 1

	def __init__(self, config, inplanes, planes, stride=1, downsample=None):
		super(BasicBlockSwitchBN, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1   = SwitchBatchNorm2d(planes, config.pop(0), with_relu=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2   = SwitchBatchNorm2d(planes, config.pop(0), with_relu=False)
		self.relu  = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x, mode=0):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out, mode)

		out = self.conv2(out)
		out = self.bn2(out, mode)

		if self.downsample is not None:
		    residual = self.downsample(x, mode)

		out += residual
		out = self.relu(out)

		return out


class BottleneckSwitchBN(nn.Module):
	expansion = 4

	def __init__(self, config, inplanes, planes, stride=1, downsample=None):
		super(BottleneckSwitchBN, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = SwitchBatchNorm2d(planes, config.pop(0), with_relu=True)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
		                       padding=1, bias=False)
		self.bn2 = SwitchBatchNorm2d(planes, config.pop(0), with_relu=True)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = SwitchBatchNorm2d(planes * 4, config.pop(0), with_relu=False)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x, mode=0):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out, mode)

		out = self.conv2(out)
		out = self.bn2(out, mode)

		out = self.conv3(out)
		out = self.bn3(out, mode)

		if self.downsample is not None:
		    residual = self.downsample(x, mode)

		out += residual
		out = self.relu(out)

		return out


class DownsampleSwitchBN(nn.Module):
	def __init__(self, config, in_channels, out_channels, kernel_size, stride, bias):
		super(DownsampleSwitchBN, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, 
							  kernel_size=kernel_size, stride=stride, bias=bias)
		self.bn = SwitchBatchNorm2d(out_channels, config.pop(0), with_relu=False)

	def forward(self, x, mode=0):
		x = self.conv(x)
		x = self.bn(x, mode)

		return x


class ResNetUnfoldSwitchBN(nn.Module):
	def __init__(self, block, layers, config=None, last_stride=1):
		assert config is not None
		self.inplanes = 64
		super(ResNetUnfoldSwitchBN, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = SwitchBatchNorm2d(64, config.pop(0), with_relu=True)
		# self.relu = nn.ReLU(inplace=True)   # add missed relu
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(config, block, 64, layers[0])
		self.layer2 = self._make_layer(config, block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(config, block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(config, block, 512, layers[3], stride=last_stride)

	def _make_layer(self, config, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = DownsampleSwitchBN(config, self.inplanes, planes * block.expansion,
						  				   kernel_size=1, stride=stride, bias=False)

		layers = nn.ModuleList()
		layers.append(block(config, self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(config, self.inplanes, planes))

		return layers

	def forward(self, x, mode=0):
		x = self.conv1(x)
		x = self.bn1(x, mode)
		# x = self.relu(x)    # add missed relu
		x = self.maxpool(x)

		for layer in self.layer1:
			x = layer(x, mode)
		for layer in self.layer2:
			x = layer(x, mode)
		for layer in self.layer3:
			x = layer(x, mode)
		for layer in self.layer4:
			x = layer(x, mode)

		return x

	def load_param(self, pretrained_weights):
		pretrained_state_dict = remove_fc(torch.load(pretrained_weights))
		self.load_state_dict(pretrained_state_dict)

	def load_state_dict(self, pretrained_state_dict):
		for key in pretrained_state_dict:
			if 'bn' in key:
				key_items = key.split('.')
				model_key = '.'.join(key_items[:-1]) + '.thermal_bn.' + key_items[-1]
				self.state_dict()[model_key].copy_(pretrained_state_dict[key])
				model_key = '.'.join(key_items[:-1]) + '.visible_bn.' + key_items[-1]
				self.state_dict()[model_key].copy_(pretrained_state_dict[key])
				model_key = '.'.join(key_items[:-1]) + '.sharable_bn.' + key_items[-1]
				self.state_dict()[model_key].copy_(pretrained_state_dict[key])
			elif 'downsample.0' in key:
				key_items = key.split('.')
				model_key = '.'.join(key_items[:-2]) + '.conv.' + key_items[-1]
				self.state_dict()[model_key].copy_(pretrained_state_dict[key])
			elif 'downsample.1' in key:
				key_items = key.split('.')
				model_key = '.'.join(key_items[:-2]) + '.bn.thermal_bn.' + key_items[-1]
				self.state_dict()[model_key].copy_(pretrained_state_dict[key])
				model_key = '.'.join(key_items[:-2]) + '.bn.visible_bn.' + key_items[-1]
				self.state_dict()[model_key].copy_(pretrained_state_dict[key])
				model_key = '.'.join(key_items[:-2]) + '.bn.sharable_bn.' + key_items[-1]
				self.state_dict()[model_key].copy_(pretrained_state_dict[key])
			else:
				self.state_dict()[key].copy_(pretrained_state_dict[key])


def resnet18unfold_switchbn(pretrained=False, **kwargs):
	"""Constructs a ResNet-18 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNetUnfoldSwitchBN(BasicBlockSwitchBN, [2, 2, 2, 2], **kwargs)
	if pretrained:
		model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet18'])))
	return model


def resnet34unfold_switchbn(pretrained=False, **kwargs):
	"""Constructs a ResNet-34 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNetUnfoldSwitchBN(BasicBlockSwitchBN, [3, 4, 6, 3], **kwargs)
	if pretrained:
		model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet34'])))
	return model


def resnet50unfold_switchbn(pretrained=False, **kwargs):
	"""Constructs a ResNet-50 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNetUnfoldSwitchBN(BottleneckSwitchBN, [3, 4, 6, 3], **kwargs)
	if pretrained:
		model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet50'])))
	return model


def resnet101unfold_switchbn(pretrained=False, **kwargs):
	"""Constructs a ResNet-101 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNetUnfoldSwitchBN(BottleneckSwitchBN, [3, 4, 23, 3], **kwargs)
	if pretrained:
		model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet101'])))
	return model


def resnet152unfold_switchbn(pretrained=False, **kwargs):
	"""Constructs a ResNet-152 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNetUnfoldSwitchBN(BottleneckSwitchBN, [3, 8, 36, 3], **kwargs)
	if pretrained:
		model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet152'])))
	return model




class TwoStreamSwitchBNOp(nn.Module):
	pool_dim = 2048
	feat_dim = 2048

	def __init__(self, num_classes, config, pretrained=False, last_stride=1, dropout_rate=0.0):
		super(TwoStreamSwitchBNOp, self).__init__()
		self.num_classes  = num_classes
		self.dropout_rate = dropout_rate

		self.backbone = resnet50unfold_switchbn(config=config, pretrained=pretrained, last_stride=last_stride)
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.bnneck = nn.BatchNorm1d(self.feat_dim)
		self.bnneck.bias.requires_grad_(False)
		self.classifier = nn.Linear(self.feat_dim, self.num_classes, bias=False)

		self.bnneck.apply(weights_init_kaiming)
		self.classifier.apply(weights_init_classifier)

	def forward(self, x_v, x_t, mode=0):
		if mode == 0:
			x = torch.cat((x_v, x_t), 0)
		elif mode == 1:
			x = x_v
		elif mode == 2:
			x = x_t
		x = self.backbone(x, mode)

		global_feat = self.avgpool(x) # (bs, 2048, 1, 1)
		global_feat = global_feat.view(global_feat.shape[0], -1) # flatten to (bs, 2048)

		feat = self.bnneck(global_feat)

		if self.training:
			# global feature for triplet loss
			if self.dropout_rate > 0:
				feat = F.dropout(feat, p=self.dropout_rate)
			# return global_feat, self.classifier(feat)
			return global_feat, feat, self.classifier(feat)
		else:
			# test with feature before/after BN
			return global_feat, feat
