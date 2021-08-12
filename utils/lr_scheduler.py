from bisect import bisect_right
import torch


# MultiStepLR with linear warmup
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
	def __init__(
		self,
		optimizer,
		milestones,
		gamma,
		warmup_epochs,
		warmup_factor,
		last_epoch = -1,
	):
		if not list(milestones) == sorted(milestones):
			raise ValueError("Milestones should be a list of increasing ints. Got {}", milestones)

		self.milestones = milestones
		self.gamma = gamma
		self.warmup_epochs = warmup_epochs
		self.warmup_factor = warmup_factor
		super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		warmup_factor = 1
		if self.last_epoch < self.warmup_epochs:
			alpha = self.last_epoch / self.warmup_epochs
			warmup_factor = self.warmup_factor * (1 - alpha) + 1 * alpha

		return [
			base_lr
			* warmup_factor
			* self.gamma ** bisect_right(self.milestones, self.last_epoch)
			for base_lr in self.base_lrs
		]