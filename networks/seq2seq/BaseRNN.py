import torch
import torch.nn as nn

class BaseRNN(nn.Module):

	def __init__(self, inp_size, hidden_size, input_dropout_p,
				dropout_p, num_stacked_layer, rnn_cell):
		super(BaseRNN, self).__init__()
		self.inp_size = inp_size
		self.hidden_size = hidden_size
		self.input_dropout = nn.Dropout(p=input_dropout_p)
		self.dropout = nn.Dropout(p=dropout_p)
		self.num_stacked_layer = num_stacked_layer
		if rnn_cell.lower() == 'lstm':
			self.rnn_cell = nn.LSTM
		elif rnn_cell.lower() == 'gru':
			self.rnn_cell = nn.GRU
		elif rnn_cell.lower() == 'rnn':
			# print "yeah RNN"
			self.rnn_cell = nn.RNN
		else:
			raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

	def forward(self, *args, **kwargs):
		raise NotImplementedError()
