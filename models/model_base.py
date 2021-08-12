import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet50
from .utils import weights_init_kaiming
from .utils import weights_init_classifier


class Baseline(nn.Module):
	pool_dim = 2048
	feat_dim = 2048

	def __init__(self, num_classes, pretrained=False, last_stride=1, dropout_rate=0.0):
		super(Baseline, self).__init__()
		self.num_classes = num_classes
		self.dropout_rate = dropout_rate

		self.backbone = resnet50(pretrained=pretrained, last_stride=last_stride)
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.bnneck = nn.BatchNorm1d(self.feat_dim)
		self.bnneck.bias.requires_grad_(False)
		self.classifier = nn.Linear(self.feat_dim, self.num_classes, bias=False)

		self.bnneck.apply(weights_init_kaiming)
		self.classifier.apply(weights_init_classifier)

	def forward(self, x, x_tmp=None, mode=None):
		global_feat = self.avgpool(self.backbone(x)) # (bs, 2048, 1, 1)
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
