import torch
import torch.nn as nn
import torch.nn.functional as F


class CMMD(nn.Module):
	def __init__(self, num_pos):
		super(CMMD, self).__init__()
		self.num_pos = num_pos

	def forward(self, feat_v, feat_t):
		feat_v = feat_v.view(feat_v.size(0), -1)
		feat_v = F.normalize(feat_v, dim=-1)
		feat_v_s = torch.split(feat_v, self.num_pos)

		feat_t = feat_t.view(feat_t.size(0), -1)
		feat_t = F.normalize(feat_t, dim=-1)
		feat_t_s = torch.split(feat_t, self.num_pos)

		losses = [self.mmd_loss(f_v, f_t) for f_v, f_t in zip(feat_v_s, feat_t_s)]
		loss = sum(losses) / len(losses)

		return loss

	def mmd_loss(self, f_v, f_t):
		return (self.poly_kernel(f_v, f_v).mean() + self.poly_kernel(f_t, f_t).mean()
				- 2 * self.poly_kernel(f_v, f_t).mean())

	def poly_kernel(self, a, b):
		a = a.unsqueeze(0)
		b = b.unsqueeze(1)
		res = (a * b).sum(-1).pow(2)
		return res
