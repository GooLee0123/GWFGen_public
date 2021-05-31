import logging
import random

import numpy as np

import torch
import torch.nn as nn

from .BaseRNN import BaseRNN

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device

class VoterRNN(BaseRNN):
    """
    """

    def __init__(self, vote_size, hidden_size, num_stacked_layer=4, input_dropout_p=0,
                dropout_p=0, bidirectional=False, rnn_cell='gru', start_token='zero',
                precision='single', isize=1, transfer=False):
        super(VoterRNN, self).__init__(vote_size, hidden_size, input_dropout_p,
                                        dropout_p, num_stacked_layer, rnn_cell)

        if precision == 'single':
            torch.set_default_tensor_type('torch.FloatTensor')
            self.dtype = np.float32
        else:
            torch.set_default_tensor_type('torch.DoubleTensor')
            self.dtype = np.float64

        self.bidirectional_encoder = bidirectional
        self.vote_size = vote_size
        self.hidden_size = hidden_size
        self.start_token = start_token
        self.isize = isize
        self.logger = logging.getLogger(__name__)

        self.rnn = self.rnn_cell(vote_size, hidden_size, num_stacked_layer,
                                batch_first=True, dropout=dropout_p)
        if transfer:
            self.logger.info("Set the requires_grad of voter rnn cells to false")
            for param in self.rnn.parameters():
                param.requires_grad = False
        for L in self.rnn.named_parameters():
            if 'weight' in L[0]:
                torch.nn.init.xavier_normal_(L[1])
            else:
                torch.nn.init.normal_(L[1])

        self.vote = nn.Linear(self.hidden_size, self.vote_size)
        torch.nn.init.xavier_normal_(self.vote.weight)
        torch.nn.init.normal_(self.vote.bias)

    def forward_step(self, input_var, h, votefunc=torch.sigmoid):
        batch_size = input_var.size(0)

        dropped_input_var = self.input_dropout(input_var)
        output, hidden = self.rnn(dropped_input_var, h)

        vote = self.vote(output.contiguous().view(-1, self.hidden_size)) \
                                .view(-1, batch_size, self.vote_size)

        return vote, hidden

    def forward(self, inputs, encoder_hidden,
                votefunc=torch.sigmoid, teacher_forcing_ratio=0.5):
        inputs, batch_size, max_length, _train = self._validate_args(
                                                    inputs,
                                                    encoder_hidden,
                                                    teacher_forcing_ratio)

        voter_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        voter_input = inputs[:, 0].unsqueeze(1)

        cutoff_index = np.zeros(batch_size, dtype=self.dtype)-1
        for di in xrange(max_length):
            voter_output, voter_hidden = self.forward_step(voter_input,
                                                        voter_hidden,
                                                        votefunc=votefunc)

            if di == 0:
                voter_outputs = voter_output
            else:
                voter_outputs = torch.cat((voter_outputs, voter_output), 0)

            if use_teacher_forcing:
                voter_input = inputs[:, di+1].unsqueeze(1)
            else:
                voter_input = voter_output.view(voter_output.size(1), -1, 1)

            # Get cutoff index
            if _train == False:
                vote_max = np.array(voter_output.cpu() < 0.5).ravel()
                nonvary_index = cutoff_index < 0.
                vary_index = map(bool, vote_max * nonvary_index)

                cutoff_index[vary_index] = di+1

        return voter_outputs, cutoff_index

    def _init_state(self, encoder_hidden):
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, teacher_forcing_ratio):
        if teacher_forcing_ratio == 0:
            _train = False
            if self.rnn_cell is nn.LSTM:
                batch_size = encoder_hidden[0].size(1)
            elif self.rnn_cell is nn.GRU:
                batch_size = encoder_hidden.size(1)

            if self.start_token == 'zero':
                inputs = torch.tensor([[0.]*self.vote_size]*batch_size) \
                                        .view(batch_size, 1, self.vote_size)
            else:
                inputs = torch.tensor([[1.]*self.vote_size]*batch_size) \
                                        .view(batch_size, 1, self.vote_size)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = int(800/self.isize)
        else:
            _train = True
            batch_size = inputs.size(0)
            max_length = inputs.size(1)-1

        return inputs, batch_size, max_length, _train
