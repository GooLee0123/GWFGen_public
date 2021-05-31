import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
	'''
		Fully Convoluntional Network
	'''

	def __init__(self, in_channel=1, out_channel=128):
		super(FCN, self).__init__()

		self.in_channel = in_channel
		self.out_channel = out_channel

		self.main = nn.Sequential(
			# input size is in_channel(1) x inp_size x vote_size
			nn.Conv2d(in_channel, out_channel,
					kernel_size=(2, 1), stride=(2, 1),
					padding=(5, 0), bias=False),
			nn.LeakyReLU(0.2),
			# state size: out_channel x (inp_size/2) x vote_size
			nn.Conv2d(out_channel, out_channel*2,
					kernel_size=(2, 1), stride=(2, 1),
					padding=(5, 0), bias=False),
			nn.BatchNorm2d(out_channel*2),
			nn.LeakyReLU(0.2),
			# state size: (out_channel*2) x (inp_size/4) x vote_size
			nn.Conv2d(out_channel*2, out_channel*4,
					kernel_size=(2, 1), stride=(2, 1),
					padding=(5, 0), bias=False),
			nn.BatchNorm2d(out_channel*4),
			nn.LeakyReLU(0.2),
			# state size: (out_channel*4) x (inp_size/8) x vote_size
			nn.Conv2d(out_channel*4, 1,
					kernel_size=(2, 1), stride=(2, 1),
					padding=(5, 0), bias=False),
			# nn.BatchNorm2d(out_channel*8),
			# nn.LeakyReLU(0.2),
			# state size : (out_channel*8) x (inp_size/16) x vote_size
			# nn.Conv2d(out_channel*8, 1,
			# 		kernel_size=(10, 1), stride=(2, 1),
			# 		padding=(1, 0), bias=False),
			# state size : 1 x (inp_size/32) x vote_size
			nn.AvgPool2d(kernel_size=(5, 1), stride=(2,1), padding=(1,0)))

		self.global_pooling = F.adaptive_avg_pool2d

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				m.weight.data.normal_(0.0, 0.02)
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self, inputs):
		inputs = self._validate_args(inputs)
		output = self.main(inputs)
		# print output.size()
		output = self.global_pooling(output, (1,1))
		# print output.size()

		return output.squeeze()

	def _validate_args(self, inputs):
		# print inputs.size()
		seq_length = inputs.size(0)
		batch_size = inputs.size(1)
		inp_length = inputs.size(2)
		inputs = inputs.view(batch_size, self.in_channel,
							seq_length*inp_length, -1)
		return inputs

# class RNN(nn.Module):
# 	'''
# 		Recurrent Neural Network
# 	'''

# 	def __init__(self, rnn_cell):
# 		super(RNN, self).__init__()

# 		if rnn_cell.lower() == 'lstm':
# 			self.rnn_cell = nn.LSTM
# 		elif rnn_cell.lower() == 'gru':
# 			self.rnn_cell = nn.GRU

# 		self.main = nn.Sequential(
# 			)


# 	def forward(self, inputs):
# 		output = self.main()

# 		return output


