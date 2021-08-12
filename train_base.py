import os
import sys
import time
import glob
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from utils import AverageMeter, EMA, Logger, accuracy, gen_idxs_dict
from utils import save_checkpoint, create_exp_dir, set_seed
from utils import RandomErasing, IdentitySampler, WarmupMultiStepLR
from utils import eval_sysu, eval_regdb
from datasets import process_query_sysu, process_gallery_sysu, process_test_regdb
from datasets import SYSUData, RegDBData, TestData
from losses import TripletLoss, CrossEntropyLabelSmooth, SP, CMMD
from models import Baseline


parser = argparse.ArgumentParser(description='Cross-Modality ReID Baseline')
# various path
parser.add_argument('--data_root', type=str, required=True, help='dataset root path')
parser.add_argument('--dataset', type=str, required=True, help='dataset name: regdb or sysu')
parser.add_argument('--save', type=str, default='./checkpoints/', help='model and log saving path')
parser.add_argument('--resume', type=str, default='', help='resume from checkpoint')
parser.add_argument('--note', type=str, default='try', help='note for this run')

# training hyper-parameters
parser.add_argument('--print_freq', type=float, default=20, help='print iteration frequency')
parser.add_argument('--test_freq', type=float, default=2, help='test and save epoch frequency')
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--epochs', type=int, default=120, help='num of total training epochs')
parser.add_argument('--steps', type=str, default='[40, 70]', help='steps for lr decreasing')
parser.add_argument('--gamma', type=float, default=0.1, help='scale factor for lr decreasing')
parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epochs')
parser.add_argument('--warmup_factor', type=float, default=0.01, help='warmup factor')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--test_batch', type=int, default=128, help='testing batch size')
parser.add_argument('--num_pos', type=int, default=4, help='num of pos per identity in each modality')
parser.add_argument('--lr', type=float, default=0.01, help='init learning rate')
parser.add_argument('--optim', type=str, default='adam', help='optimizer')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay for sgd or adam')
parser.add_argument('--img_w', type=int, default=128, help='img width')
parser.add_argument('--img_h', type=int, default=256, help='img height')
parser.add_argument('--label_smooth', type=float, default=0.0, help='label smoothing')
parser.add_argument('--last_stride', type=int, default=1, help='last stride for resnet')
parser.add_argument('--dropout_rate', type=float, default=0.0, help='dropout rate for classifier')
parser.add_argument('--ema_decay', type=float, default=0.997, help='whether to use EMA')
parser.add_argument('--sp_lambda', type=float, default=5.0, help='lambda for SP loss')
parser.add_argument('--cmmd_lambda', type=float, default=0.05, help='lambda for CMMD loss')
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--cuda', type=int, default=1)

# hyper parameters
parser.add_argument('--margin', type=float, default=0.4, help='triplet margin')
parser.add_argument('--triplet_feat_norm', type=str, default='no', 
					help='whether normalizing features in triplet loss')
parser.add_argument('--test_feat_norm', type=str, default='yes', 
					help='whether normalizing features in testing')
parser.add_argument('--mode', default='all', type=str, help='all or indoor for sysu')
parser.add_argument('--trial', default=1, type=int, help='trial (only for RegDB dataset)')


args, unparsed = parser.parse_known_args()
args.save = os.path.join(args.save, args.note)
create_exp_dir(args.save, scripts_to_save=glob.glob('*.py') + glob.glob('*.sh'))
sys.stdout = Logger(log_path=os.path.join(args.save, 'log.txt'))


def main():
	# set_seed(args.seed, cuda=args.cuda)
	if args.cuda:
		cudnn.enabled = True
		cudnn.benchmark = True
	print("args = {}".format(args))
	print("unparsed_args = {}".format(unparsed))

	# define transforms
	mean=[0.485, 0.456, 0.406]
	std=[0.229, 0.224, 0.225]
	train_transform = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Pad(10),
		transforms.RandomCrop((args.img_h,args.img_w)),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean,std=std),
		RandomErasing(p=0.5, mean=mean)
	])
	test_transform = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((args.img_h,args.img_w)),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean,std=std),
	])

	# define dataset
	end = time.time()
	if args.dataset == 'sysu':
		# training set
		trainset = SYSUData(args.data_root, transform=train_transform)
		# generate the idx of each person identity
		visible_idxs_dict, thermal_idxs_dict = gen_idxs_dict(trainset.train_visible_label, trainset.train_thermal_label)
		# testing set
		gallery_img, gallery_label, gallery_camid = process_gallery_sysu(args.data_root, mode=args.mode, shot=1, trial=0)
		query_img,   query_label,   query_camid   = process_query_sysu(args.data_root, mode=args.mode)
	elif args.dataset == 'regdb':
		# training set
		trainset = RegDBData(args.data_root, args.trial, transform=train_transform, img_size=(args.img_w,args.img_h))
		# generate the idx of each person identity
		visible_idxs_dict, thermal_idxs_dict = gen_idxs_dict(trainset.train_visible_label, trainset.train_thermal_label)
		# testing set
		gallery_img, gallery_label = process_test_regdb(args.data_root, trial=args.trial, modality='thermal')
		query_img,   query_label   = process_test_regdb(args.data_root, trial=args.trial, modality='visible')
		gallery_camid, query_camid = None, None
	else:
		raise Exception('invalid dataset name......')

	galleryset = TestData(gallery_img, gallery_label, transform=test_transform, img_size=(args.img_w,args.img_h))
	queryset   = TestData(query_img, query_label, transform=test_transform, img_size=(args.img_w,args.img_h))

	# testing data loader
	gallery_loader = data.DataLoader(galleryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
	query_loader   = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
	   
	num_classes = len(np.unique(trainset.train_visible_label))
	nquery      = len(query_label)
	ngallery    = len(gallery_label)
	print('Dataset {} statistics:'.format(args.dataset))
	print('  ------------------------------')
	print('  subset   | # ids | # images')
	print('  ------------------------------')
	print('  visible  | {:5d} | {:8d}'.format(num_classes, len(trainset.train_visible_label)))
	print('  thermal  | {:5d} | {:8d}'.format(num_classes, len(trainset.train_thermal_label)))
	print('  ------------------------------')
	print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
	print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gallery_label)), ngallery))
	print('  ------------------------------')   
	print('Data Loading Time: {}s'.format(int(round(time.time()-end))))


	print('==> Building model......')
	model = Baseline(num_classes, pretrained=True, last_stride=args.last_stride, dropout_rate=args.dropout_rate)
	if args.cuda:
		model = torch.nn.DataParallel(model).cuda()
	# exponential moving average
	if args.ema_decay > 0.0:
		ema = EMA(model, args.ema_decay)
		ema.register()
	else:
		ema = None
	# for resume
	print('==> Done......')

	# initialize optimizer
	ignored_params = list(map(id, model.module.bnneck.parameters())) + \
					 list(map(id, model.module.classifier.parameters()))
	base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())
	if args.optim == 'sgd':
		optimizer = torch.optim.SGD([
			{'params': base_params, 'lr': 0.1*args.lr},
			{'params': model.module.bnneck.parameters(), 'lr': args.lr},
			{'params': model.module.classifier.parameters(), 'lr': args.lr}],
			weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
	elif args.optim == 'adam':
		optimizer = torch.optim.Adam([
			{'params': base_params, 'lr': 0.1*args.lr},
			{'params': model.module.bnneck.parameters(), 'lr': args.lr},
			{'params': model.module.classifier.parameters(), 'lr': args.lr}],
			weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))
	scheduler = WarmupMultiStepLR(optimizer, eval(args.steps), args.gamma, args.warmup_epochs, args.warmup_factor)

	# define loss functions
	if args.label_smooth > 0:
		criterionCE = CrossEntropyLabelSmooth(num_classes, args.label_smooth)
	else:
		criterionCE = nn.CrossEntropyLoss()
	criterionTri = TripletLoss(margin=args.margin, feat_norm=args.triplet_feat_norm)
	criterionSP = SP()
	criterionCMMD = CMMD(args.num_pos)
	if args.cuda:
		criterionCE = criterionCE.cuda()
		criterionTri = criterionTri.cuda()
		criterionSP = criterionSP.cuda()
		criterionCMMD = criterionCMMD.cuda()

	print('==> Start Training......')
	criterions = {'criterionCE':criterionCE, 'criterionTri':criterionTri,
				  'criterionSP':criterionSP, 'criterionCMMD':criterionCMMD}
	gallery = {'gallery_loader':gallery_loader, 'gallery_label':gallery_label, 'gallery_camid':gallery_camid}
	query = {'query_loader':query_loader, 'query_label':query_label, 'query_camid':query_camid}
	for epoch in range(args.epochs):
		# prepare training data loader
		sampler = IdentitySampler(trainset.train_visible_label, trainset.train_thermal_label,
								  visible_idxs_dict, thermal_idxs_dict, args.num_pos, args.batch_size)
		trainset.vIndex = sampler.index_visible
		trainset.tIndex = sampler.index_thermal
		train_loader = data.DataLoader(trainset, batch_size=args.batch_size,
									   sampler=sampler, num_workers=args.workers)

		# scheduler.step()
		current_lr = scheduler.get_lr()[-1]
		print('Epoch: {} lr: {:.6f}'.format(epoch+1, current_lr))

		# train one eopch
		epoch_start_time = time.time()
		train(train_loader, model, ema, optimizer, criterions, epoch)
		epoch_duration = time.time() - epoch_start_time
		print('Epoch time: {}s'.format(int(round(epoch_duration))))

		# testing
		if (epoch + 1) % args.test_freq == 0:
			print('Testing the model......')
			test_start_time = time.time()
			test(gallery, query, model, epoch)
			test_duration = time.time() - test_start_time
			print('Test time: {}s'.format(int(round(test_duration))))

			print('Saving model......')
			save_checkpoint({
				'epoch': epoch+1,
				'model': model.state_dict(),
				'ema': ema.state_dict() if ema is not None else None,
				'optimizer': optimizer.state_dict(),
			}, args.save, epoch+1)

			if ema is not None:
				model.load_state_dict(ema.state_dict())

		scheduler.step()


def train(train_loader, model, ema, optimizer, criterions, epoch):
	batch_time  = AverageMeter()
	data_time   = AverageMeter()
	losses_ce   = AverageMeter()
	losses_tri  = AverageMeter()
	losses_sp   = AverageMeter()
	losses_cmmd = AverageMeter()
	acc         = AverageMeter()

	criterionCE   = criterions['criterionCE']
	criterionTri  = criterions['criterionTri']
	criterionSP   = criterions['criterionSP']
	criterionCMMD = criterions['criterionCMMD']

	model.train()

	end = time.time()
	for idx, (img_v, img_t, target_v, target_t) in enumerate(train_loader, start=1):
		data_time.update(time.time() - end)
		img = torch.cat((img_v, img_t), 0)
		target = torch.cat((target_v, target_t))

		if args.cuda:
			img_v = img_v.cuda(non_blocking=True)
			img_t = img_t.cuda(non_blocking=True)
			img   = img.cuda(non_blocking=True)
			target_v = target_v.cuda(non_blocking=True)
			target_t = target_t.cuda(non_blocking=True)
			target   = target.cuda(non_blocking=True)

		global_feat, feat, logit = model(img)
		feat_v, feat_t = torch.split(feat, img.size(0)//2, dim=0)
		global_feat_v, global_feat_t = torch.split(global_feat, img.size(0)//2, dim=0)
		loss_ce   = criterionCE(logit, target)
		loss_tri  = (criterionTri(global_feat_v, global_feat_v, target_v) +
					 criterionTri(global_feat_t, global_feat_t, target_t) +
					 criterionTri(global_feat_v, global_feat_t, target_t) +
					 criterionTri(global_feat_t, global_feat_v, target_v)) / 4.0
		loss_sp   = criterionSP(feat_v, feat_t) * args.sp_lambda
		loss_cmmd = criterionCMMD(feat_v, feat_t) * args.cmmd_lambda
		loss = loss_ce + loss_tri + loss_sp + loss_cmmd

		prec1, = accuracy(logit, target, topk=(1,))
		losses_ce.update(loss_ce.item(), img.size(0))
		losses_tri.update(loss_tri.item(), img.size(0))
		losses_sp.update(loss_sp.item(), img.size(0))
		losses_cmmd.update(loss_cmmd.item(), img.size(0))
		acc.update(prec1.item(), img.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if ema is not None: ema.update()

		batch_time.update(time.time() - end)
		end = time.time()

		if idx % args.print_freq == 0:
			print('Epoch[{0}]:[{1:03}/{2:03}] '
				  'Time:{batch_time.val:.4f} '
				  'Data:{data_time.val:.4f}  '
				  'CE:{losses_ce.val:.4f}({losses_ce.avg:.4f})  '
				  'Tri:{losses_tri.val:.4f}({losses_tri.avg:.4f})  '
				  'SP:{losses_sp.val:.4f}({losses_sp.avg:.4f})  '
				  'CMMD:{losses_cmmd.val:.4f}({losses_cmmd.avg:.4f})  '
				  'Acc:{acc.val:.2f}({acc.avg:.2f})'.format(
				  epoch+1, idx, len(train_loader), batch_time=batch_time, data_time=data_time,
				  losses_ce=losses_ce, losses_tri=losses_tri, losses_sp=losses_sp, losses_cmmd=losses_cmmd, acc=acc))


def test(gallery, query, model, epoch):
	gallery_loader = gallery['gallery_loader']
	gallery_label  = gallery['gallery_label']
	gallery_camid  = gallery['gallery_camid']
	ngallery       = len(gallery_label)

	query_loader = query['query_loader']
	query_label  = query['query_label']
	query_camid  = query['query_camid']
	nquery       = len(query_label)

	model.eval()
	print('Extracting gallery features...')
	start_time = time.time()
	ptr = 0
	gallery_feats = np.zeros((ngallery, model.module.feat_dim))
	gallery_global_feats = np.zeros((ngallery, model.module.feat_dim))
	with torch.no_grad():
		for idx, (img, _) in enumerate(gallery_loader):
			if args.cuda:
				img = img.cuda(non_blocking=True)
			global_feat, feat = model(img)
			if args.test_feat_norm == 'yes':
				global_feat = F.normalize(global_feat, p=2, dim=1)
				feat  = F.normalize(feat, p=2, dim=1)
			batch_num = img.size(0)
			gallery_feats[ptr:ptr+batch_num,:] = feat.cpu().numpy()
			gallery_global_feats[ptr:ptr+batch_num,:] = global_feat.cpu().numpy()
			ptr = ptr + batch_num
	duration = time.time() - start_time
	print('Extracting time: {}s'.format(int(round(duration))))

	print('Extracting query features...')
	start_time = time.time()
	ptr = 0
	query_feats = np.zeros((nquery, model.module.feat_dim))
	query_global_feats = np.zeros((nquery, model.module.feat_dim))
	with torch.no_grad():
		for idx, (img, _) in enumerate(query_loader):
			if args.cuda:
				img = img.cuda(non_blocking=True)
			global_feat, feat = model(img)
			if args.test_feat_norm == 'yes':
				global_feat = F.normalize(global_feat, p=2, dim=1)
				feat  = F.normalize(feat, p=2, dim=1)
			batch_num = img.size(0)
			query_feats[ptr:ptr+batch_num,:] = feat.cpu().numpy()
			query_global_feats[ptr:ptr+batch_num,:] = global_feat.cpu().numpy()
			ptr = ptr + batch_num
	duration = time.time() - start_time
	print('Extracting time: {}s'.format(int(round(duration))))

	# compute the similarity
	distmat = np.matmul(query_feats, np.transpose(gallery_feats))
	distmat_global = np.matmul(query_global_feats, np.transpose(gallery_global_feats))

	# evaluation
	if args.dataset == 'sysu':
		cmc, mAP = eval_sysu(-distmat, query_label, gallery_label, query_camid, gallery_camid)
		cmc_global, mAP_global = eval_sysu(-distmat_global, query_label, gallery_label, query_camid, gallery_camid)
	elif args.dataset == 'regdb':
		cmc, mAP = eval_regdb(-distmat, query_label, gallery_label)
		cmc_global, mAP_global = eval_regdb(-distmat_global, query_label, gallery_label)
	else:
		raise Exception('invalid dataset name......')

	print('Results - Epoch {}:'.format(epoch+1))
	print('mAP: {:.2%}'.format(mAP))
	for r in [1, 5, 10, 20]:
		print("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r-1]))
	print('mAP_global: {:.2%}'.format(mAP_global))
	for r in [1, 5, 10, 20]:
		print("cmc_global curve, Rank-{:<3}:{:.2%}".format(r, cmc_global[r-1]))


if __name__ == '__main__':
	main()