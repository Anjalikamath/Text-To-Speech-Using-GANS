import os
import sys
import h5py
import torch
import numpy as np
from utils import Hps
from solver import Solver
from scipy.io import wavfile
from torch.autograd import Variable
from preprocess.tacotron.norm_utils import spectrogram2wav

def sp2wav(sp): 
	exp_sp = sp
	wav_data = spectrogram2wav(exp_sp)
	return wav_data

def convert_sp(sp, c, solver, gen = True):
	c_var = Variable(torch.from_numpy(np.array([c]))).cuda()
	sp_tensor = torch.from_numpy(np.expand_dims(sp, axis = 0))
	sp_tensor = sp_tensor.type(torch.FloatTensor)
	converted_sp = solver.test_step(sp_tensor, c_var, gen = gen)
	converted_sp = converted_sp.squeeze(axis = 0).transpose((1, 0))
	return converted_sp

def get_model(hps_path, model_path):
	hps = Hps()
	hps.load(hps_path)
	hps_tuple = hps.get_tuple()
	solver = Solver(hps_tuple, None, None)
	solver.load_model(model_path)
	return solver
utt=[]
def convert_all_sp(h5_path, src_speaker, tar_speaker, solver, dir_path,
					dset = 'test', gen = True, max_n =len(utt),
					speaker_used_path = './hps/en_speaker_used.txt'):
	# read speaker id file
	with open(speaker_used_path) as f:
		speakers = [line.strip() for line in f]
		#print(speakers)
		speaker2id = {speaker:i for i, speaker in enumerate(speakers)}
		#print(speaker2id)

	with h5py.File(h5_path, 'r') as f_h5:
		c = 0
		#for utt_id in f_h5[f'{dset}/{src_speaker}']:
		for utt_id in utt:
			print(utt_id)
			#print(f_h5[f'{dset}/{src_speaker}'])
			sp = f_h5[f'{dset}/{src_speaker}/{utt_id}/lin'][()]
			#print(sp)
			converted_sp = convert_sp(sp, speaker2id[tar_speaker], solver, gen=gen)
			#print(speaker2id[tar_speaker])
			wav_data = sp2wav(converted_sp)
			wav_path = os.path.join(dir_path, f'{src_speaker}_{tar_speaker}_{utt_id}.wav')
			wavfile.write(wav_path, 16000, wav_data)
			c += 1
			if c >= max_n:
				break

if __name__ == '__main__':
	#h5_path = './vctk_test.h5'
	h5_path = '/home/anjali/Downloads/vctk.h5'
	root_dir = './results'
	model_path = './pkl/model.pkl'
	hps_path = './hps/vctk.json'
	dset='test'
	solver = get_model(hps_path = hps_path, model_path = model_path)
	#speakers = ['225', '226', '227', '228', '229']
	#max_n = 5
	'''take speaker1,speaker2,number of sentences to be used from speaker1, the sentence ids'''
	if len(sys.argv) == 4:
		speaker1 =(sys.argv[1])#speakers[:min(5, int(sys.argv[1]))]
		speaker2=(sys.argv[2])
		max_n=int(sys.argv[3])
		print("enter number of speakers n %d" % (max_n))
		#utt=[]
	for i in range(max_n):
		#utt.append(int(sys.argv[(i+3)]))
		utt.append((input("enter the sentence id for speaker 1 ")))
	print(utt)
		#max_n = min(5, int(sys.argv[2]))
	#if speaker1==speaker2:
	#	pass
	#else:
	dir_path=os.path.join(root_dir,f'p{speaker1}_{speaker2}')
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)
	
	convert_all_sp(h5_path, speaker1,speaker2,solver, dir_path, max_n =len(utt))
	'''for speaker_A in speakers:
		for speaker_B in speakers:
			if speaker_A != speaker_B:
				dir_path = os.path.join(root_dir, f'p{speaker_A}_p{speaker_B}')
				if not os.path.exists(dir_path):
					os.makedirs(dir_path)
				convert_all_sp(h5_path, speaker_A, speaker_B,
								solver, dir_path, max_n = max_n)'''
