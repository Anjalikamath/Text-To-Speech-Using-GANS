import torch
from torch import optim
from torch.autograd import Variable
import numpy as np
import pickle
from utils import Hps
from utils import DataLoader
from utils import Logger
from utils import SingleDataset
from solver import Solver
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', default=True, action='store_true')
	parser.add_argument('--test', default=False, action='store_true')
	parser.add_argument('--load_model', default=True)#, action='store_true')
	parser.add_argument('-flag', default='train')
	parser.add_argument('-hps_path', default='./hps/vctk.json')
	parser.add_argument('-load_model_path', default='/home/anjali/Desktop/sem5/BIGDATA/Voice-Conversion/pkl/model.pkl-1000')
	parser.add_argument('-dataset_path', default='./vctk_test.h5')
	parser.add_argument('-index_path', default='./index.json')
	parser.add_argument('-output_model_path', default='./pkl')
	args = parser.parse_args()
	hps = Hps()
	hps.load(args.hps_path)
	hps_tuple = hps.get_tuple()
	dataset = SingleDataset(args.dataset_path,
			args.index_path,
			seg_len=hps_tuple.seg_len)
	'''
	dataset=SingleDataset(args.load_model_path,args.index_path,seg_len=hps_tuple.seg_len)'''

	data_loader = DataLoader(dataset)

	solver = Solver(hps_tuple, data_loader)
	torch.cuda.empty_cache()

		#print(args.load_model_path)
	if args.train:
		if args.load_model:
		
			
			#a,b,c=solver.load_model(args.load_model_path) #or a,b,c=solver.load_model(args.load_model_path)
			mode='train'
			iteration=1000
			solver.train(args.output_model_path, args.flag, mode='pretrain_G')
			solver.train(args.output_model_path, args.flag, mode='pretrain_D')
			solver.train(args.output_model_path, args.flag, mode='train')
			solver.train(args.output_model_path, args.flag, mode='patchGAN')
			
