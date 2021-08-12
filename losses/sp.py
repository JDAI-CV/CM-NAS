import torch
import torch.nn as nn
import torch.nn.functional as F


class SP(nn.Module):
	def __init__(self):
		super(SP, self).__init__()

	def forward(self, feat_v, feat_t):
		feat_v   = feat_v.view(feat_v.size(0), -1)
		G_v      = torch.mm(feat_v, feat_v.t())
		norm_G_v = F.normalize(G_v, p=2, dim=1)

		feat_t   = feat_t.view(feat_t.size(0), -1)
		G_t      = torch.mm(feat_t, feat_t.t())
		norm_G_t = F.normalize(G_t, p=2, dim=1)

		loss = F.mse_loss(norm_G_v, norm_G_t)

		return loss
