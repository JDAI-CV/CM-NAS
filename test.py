import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from datasets import process_query_sysu, process_gallery_sysu, process_test_regdb
from datasets import TestData
from models import Baseline, TwoStreamSwitchBNOp
from utils import eval_sysu, eval_regdb
from utils import EMA


parser = argparse.ArgumentParser(description='Cross-Modality ReID Testing')
# various path
parser.add_argument('--data_root', type=str, required=True, help='dataset root path')
parser.add_argument('--dataset', type=str, required=True, help='dataset name: regdb or sysu')
parser.add_argument('--model_type', type=str, required=True, help='model type for testing')
parser.add_argument('--config_path', type=str, default='', help='path of searched config for TwoStreamSwitchBN')
parser.add_argument('--weights', type=str, required=True, help='model weights for testing')

# training hyper-parameters
parser.add_argument('--test_batch', type=int, default=128, help='testing batch size')
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--img_w', type=int, default=128, help='img width')
parser.add_argument('--img_h', type=int, default=256, help='img height')
parser.add_argument('--last_stride', type=int, default=1, help='last stride for resnet')
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--ema', action='store_true', default=False, help='whether to use EMA')

# hyper parameters
parser.add_argument('--test_feat_norm', type=str, default='yes', 
					help='whether normalizing features in testing')
parser.add_argument('--mode', default='all', type=str, help='all or indoor for sysu')
parser.add_argument('--shot', default=1, type=int, help='single or multi shot for sysu')
parser.add_argument('--trial', default=1, type=int, help='trial (only for RegDB dataset)')
parser.add_argument('--tvsearch', action='store_true', help='whether thermal to visible search on RegDB')


def extract_gall_feat(gallery_loader):
	model.eval()
	# print('Extracting gallery features...')
	start_time = time.time()
	ptr = 0
	gallery_feats = np.zeros((ngallery, model.module.feat_dim))
	gallery_global_feats = np.zeros((ngallery, model.module.feat_dim))
	with torch.no_grad():
		for idx, (img, _) in enumerate(gallery_loader):
			if args.cuda:
				img = img.cuda(non_blocking=True)
			global_feat, feat = model(img, img, mode=test_mode[0])
			if args.test_feat_norm == 'yes':
				global_feat = F.normalize(global_feat, p=2, dim=1)
				feat  = F.normalize(feat, p=2, dim=1)
			batch_num = img.size(0)
			gallery_feats[ptr:ptr+batch_num,:] = feat.cpu().numpy()
			gallery_global_feats[ptr:ptr+batch_num,:] = global_feat.cpu().numpy()
			ptr = ptr + batch_num
	duration = time.time() - start_time
	# print('Extracting time: {}s'.format(int(round(duration))))
	return gallery_feats, gallery_global_feats


def extract_query_feat(query_loader):
	model.eval()
	# print('Extracting query features...')
	start_time = time.time()
	ptr = 0
	query_feats = np.zeros((nquery, model.module.feat_dim))
	query_global_feats = np.zeros((nquery, model.module.feat_dim))
	with torch.no_grad():
		for idx, (img, _) in enumerate(query_loader):
			if args.cuda:
				img = img.cuda(non_blocking=True)
			global_feat, feat = model(img, img, mode=test_mode[1])
			if args.test_feat_norm == 'yes':
				global_feat = F.normalize(global_feat, p=2, dim=1)
				feat  = F.normalize(feat, p=2, dim=1)
			batch_num = img.size(0)
			query_feats[ptr:ptr+batch_num,:] = feat.cpu().numpy()
			query_global_feats[ptr:ptr+batch_num,:] = global_feat.cpu().numpy()
			ptr = ptr + batch_num
	duration = time.time() - start_time
	# print('Extracting time: {}s'.format(int(round(duration))))
	return query_feats, query_global_feats


args, unparsed = parser.parse_known_args()
if args.dataset == 'sysu':
	num_classes = 395
	test_mode = [1, 2]
elif args.dataset == 'regdb':
	num_classes = 206
	test_mode = [2, 1]
else:
	raise Exception('Invalid dataset name......')

if args.cuda:
	cudnn.enabled = True
	cudnn.benchmark = True

print('==> Building model......')
if args.model_type == 'baseline':
	model = Baseline(num_classes, pretrained=False, last_stride=args.last_stride, dropout_rate=0.0)
elif args.model_type == 'cm-nas':
	config = open(args.config_path).readline()
	config = [int(x) for x in config.strip().split(' ')]
	model = TwoStreamSwitchBNOp(num_classes, config, pretrained=False, last_stride=args.last_stride, dropout_rate=0.0)
else:
	raise Exception('Invalid model type......')
if args.cuda:
	model = torch.nn.DataParallel(model).cuda()

print('==> Loading weights from checkpoint......')
if os.path.isfile(args.weights):
	checkpoint = torch.load(args.weights)
	if args.ema:
		model.load_state_dict(checkpoint['ema'])
	else:
		model.load_state_dict(checkpoint['model'])
else:
	print('==> No checkpoint found at {}'.format(args.weights))

print('==> Testing......')
# define transforms
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
test_transform = transforms.Compose([
	transforms.ToPILImage(),
	transforms.Resize((args.img_h,args.img_w)),
	transforms.ToTensor(),
	transforms.Normalize(mean=mean,std=std),
])

end = time.time()

if args.dataset == 'sysu':
	query_img, query_label, query_camid = process_query_sysu(args.data_root, mode=args.mode)
	queryset = TestData(query_img, query_label, transform=test_transform, img_size=(args.img_w,args.img_h))
	query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
	nquery = len(query_label)

	query_feats, query_global_feats = extract_query_feat(query_loader)
	
	for trial in tqdm(range(10)):
		gallery_img, gallery_label, gallery_camid = process_gallery_sysu(args.data_root, args.mode, args.shot, trial)
		galleryset = TestData(gallery_img, gallery_label, transform=test_transform, img_size=(args.img_w,args.img_h))
		gallery_loader = data.DataLoader(galleryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
		ngallery = len(gallery_label)

		gallery_feats, gallery_global_feats = extract_gall_feat(gallery_loader)

		# compute the similarity
		distmat = np.matmul(query_feats, np.transpose(gallery_feats))
		distmat_global = np.matmul(query_global_feats, np.transpose(gallery_global_feats))

		# evaluation
		cmc, mAP = eval_sysu(-distmat, query_label, gallery_label, query_camid, gallery_camid)
		cmc_global, mAP_global = eval_sysu(-distmat_global, query_label, gallery_label, query_camid, gallery_camid)

		if trial == 0:
			all_cmc = cmc
			all_mAP = mAP
			all_cmc_global = cmc_global
			all_mAP_global = mAP_global
		else:
			all_cmc += cmc
			all_mAP += mAP
			all_cmc_global += cmc_global
			all_mAP_global += mAP_global

		# print('Test Trial: {}, Shot = {}'.format(trial, args.shot))
		# print('mAP: {:.2%} | Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%} | Rank-20: {:.2%}'.format(
		# 	mAP, cmc[0], cmc[4], cmc[9], cmc[19]))
		# print('mAP_global: {:.2%} | Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%} | Rank-20: {:.2%}'.format(
		# 	mAP_global, cmc_global[0], cmc_global[4], cmc_global[9], cmc_global[19]))

	cmc = all_cmc / 10
	mAP = all_mAP / 10
	cmc_global = all_cmc_global / 10
	mAP_global = all_mAP_global / 10

	print('All Average (Shot = {}):'.format(args.shot))
	print('mAP:        {:.2%} | Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%} | Rank-20: {:.2%}'.format(
		mAP, cmc[0], cmc[4], cmc[9], cmc[19]))
	print('mAP_global: {:.2%} | Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%} | Rank-20: {:.2%}'.format(
		mAP_global, cmc_global[0], cmc_global[4], cmc_global[9], cmc_global[19]))

elif args.dataset == 'regdb':
	gallery_img, gallery_label = process_test_regdb(args.data_root, trial=args.trial, modality='thermal')
	query_img,   query_label   = process_test_regdb(args.data_root, trial=args.trial, modality='visible')

	galleryset = TestData(gallery_img, gallery_label, transform=test_transform, img_size=(args.img_w,args.img_h))
	queryset   = TestData(query_img, query_label, transform=test_transform, img_size=(args.img_w,args.img_h))

	gallery_loader = data.DataLoader(galleryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
	query_loader   = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

	ngallery = len(gallery_label)
	nquery   = len(query_label)

	gallery_feats, gallery_global_feats = extract_gall_feat(gallery_loader)
	query_feats, query_global_feats = extract_query_feat(query_loader)

	if args.tvsearch:
		# compute the similarity
		distmat = np.matmul(gallery_feats, np.transpose(query_feats))
		distmat_global = np.matmul(gallery_global_feats, np.transpose(query_global_feats))
		# evaluation
		cmc, mAP = eval_regdb(-distmat, gallery_label, query_label)
		cmc_global, mAP_global = eval_regdb(-distmat_global, gallery_label, query_label)
	else:
		# compute the similarity
		distmat = np.matmul(query_feats, np.transpose(gallery_feats))
		distmat_global = np.matmul(query_global_feats, np.transpose(gallery_global_feats))
		# evaluation
		cmc, mAP = eval_regdb(-distmat, query_label, gallery_label)
		cmc_global, mAP_global = eval_regdb(-distmat_global, query_label, gallery_label)

	if args.tvsearch:
		print('Test Trial: {}, Thermal to Visible'.format(args.trial))
	else:
		print('Test Trial: {}, Visible to Thermal'.format(args.trial))
	print('mAP: {:.2%} | Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%} | Rank-20: {:.2%}'.format(
		mAP, cmc[0], cmc[4], cmc[9], cmc[19]))
	print('mAP_global: {:.2%} | Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%} | Rank-20: {:.2%}'.format(
		mAP_global, cmc_global[0], cmc_global[4], cmc_global[9], cmc_global[19]))
