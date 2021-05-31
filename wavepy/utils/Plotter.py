import logging
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

cuda = torch.cuda.is_available()

class Plotter(object):

	def __init__(self, config, epoch, batch_step, plt_dir='./Plots'):
		self.config = config
		self.epoch = str(epoch).zfill(4)
		self.batch_step = str(batch_step).zfill(4)
		self.logger = logging.getLogger(__name__)

		if self.config['precision'] == 'single':
		    torch.set_default_tensor_type('torch.FloatTensor')
		else:
		    torch.set_default_tensor_type('torch.DoubleTensor')

		if config['bidirectional']:
			network_direc = 'bidirectional'
		else:
			network_direc = 'unidirectional'
		strain = config['strain'].upper()
		rcell = config['rnn_cell'].upper()
		data_token = config['path'].split('_')[-1]
		hiddens = str(config['num_stacked_layer'])+'n'+str(config['hidden_size'])
		tfr = 'TFR'+str(config['teacher_forcing_ratio']).replace('.', '')
		G_model_structure = strain+'_'+network_direc+'_' \
							+rcell+'_'+hiddens+'_'+tfr \
							+'_'+data_token
		if config['start_token'] == 'input':
			G_model_structure += '_ST_input'

		self.plt_path = os.path.join(plt_dir, G_model_structure)

		if not os.path.exists(self.plt_path):
			os.makedirs(self.plt_path)

	def _get_go_stop_prob(self, lengths, targets):
		sup = 5.
		ans = torch.zeros_like(targets)
		for i, L in enumerate(lengths):
			L = int(L)-1
			x = torch.linspace(0, L, L)
			y = 1.0 - 0.5*(x/L)**sup
			ans[i, :L] = y.view(-1, 1)
		return ans.cpu().numpy()

	def plot_waves(self, inputs, targets, pwaves, votes, input_lens, target_lens, 
		cutoff_index, batch_size=16, batch_plot=False, random_index=False):
		vote_answer = self._get_go_stop_prob(target_lens, targets)
		inputs = nn.utils.rnn.pad_packed_sequence(inputs)[0].cpu().numpy()
		targets = targets.cpu().numpy()
		pwaves = pwaves.cpu().numpy()
		votes = votes.cpu().numpy()

		if batch_plot:
			if random_index:
				raise ValueError("Both random index and batch plot can't be True at the same time")

			psize = min(int(np.ceil(np.sqrt(batch_size))), 4)
			iter_size = min(16, batch_size)

			idxs = np.arange(batch_size)
			np.random.shuffle(idxs)
			rand_idxs = idxs[:iter_size]

			plt.figure(0)
			plt.cla()
			plt.figure(1)
			plt.cla()

			for i in xrange(iter_size):
				cutoff_idx = int(cutoff_index[rand_idxs[i]])
				tgt = targets[rand_idxs[i]].ravel()[:int(target_lens[rand_idxs[i]].item())]
				inp_cut_len = len(tgt)
				inp = inputs[:, rand_idxs[i]].ravel()[:int(input_lens[rand_idxs[i]].item())][-inp_cut_len:]
				inp_len = len(inp)
				inp_x = range(inp_len)

				tgt_len = len(tgt)
				tgt_x = range(inp_len-4, inp_len+tgt_len-4)

				prd = pwaves[:,rand_idxs[i]].ravel()[:cutoff_idx]
				prd_len = len(prd)
				prd_x = range(inp_len-4, inp_len+prd_len-4)

				vtans = vote_answer[rand_idxs[i]].ravel()
				vtprd = votes[:, rand_idxs[i]].ravel()
				vtcprd = vtprd[:cutoff_idx]

				plt.figure(0)
				plt.subplot(psize, psize, i+1)
				plt.plot(inp_x, inp, color='g', label='validation: input')
				plt.plot(tgt_x, tgt, color='b', label='validation: target')
				plt.plot(prd_x, prd, color='r', label='validation: generated')
				# plt.legend()
				plt.tight_layout()

				plt.figure(1)
				plt.subplot(psize, psize, i+1)
				plt.plot(vtans, color='b', label='validation: vote answer')
				plt.plot(vtprd, color='y', label='validation: vote prediction')
				plt.plot(vtcprd, color='r', label='validation: vote cutoff prediction')
				# plt.legend()
				plt.tight_layout()

		else:
			if random_index:
				index = np.random.randint(batch_size)
			else:
				index = 0

			cutoff_idx = int(cutoff_index[index])
			tgt = targets[index].ravel()[:int(target_lens[index].item())]
			inp_cut_len = len(tgt)
			inp = inputs[:, index].ravel()[:int(input_lens[i].item())][-inp_cut_len:]
			inp_len = len(inp)
			inp_x = range(inp_len)

			tgt_len = len(tgt)
			tgt_x = range(inp_len-1, inp_len+tgt_len-1)

			prd = pwaves[:,index].ravel()[:cutoff_idx]
			prd_len = len(prd)
			prd_x = range(prd_len-1, inp_len+prd_len-1)

			vtans = vote_answer[index].ravel()
			vtprd = votes[:, index].ravel()[:cutoff_idx]

			plt.figure(0)
			plt.subplot(psize, psize, i+1)
			plt.plot(inp_x, inp, color='g', label='validation: input')
			plt.plot(tgt_x, tgt, color='b', label='validation: target')
			plt.plot(prd_x, prd, color='r', label='validation: generated')
			plt.ylim(-1, 1)
			plt.xlim(0, 300)
			plt.tight_layout()

			plt.figure(1)
			plt.subplot(psize, psize, i+1)
			plt.plot(vtans, color='b', label='validation: vote answer')
			plt.plot(vtprd, color='r', label='validation: vote prediction')
			plt.ylim(0, 1)
			plt.xlim(0, 300)
			plt.tight_layout()

		plt.figure(0)
		plt.savefig(os.path.join(self.plt_path, 'Ep{}_Bs{}_val_wave_prediction'.format(self.epoch, self.batch_step)))
		plt.close()
		plt.figure(1)
		plt.savefig(os.path.join(self.plt_path, 'Ep{}_Bs{}_val_vote_prediction'.format(self.epoch, self.batch_step)))
		plt.close()