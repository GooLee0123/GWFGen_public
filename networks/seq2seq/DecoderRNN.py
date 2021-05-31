import random
import time
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .BaseRNN import BaseRNN

class DecoderRNN(BaseRNN):
    """
    """

    def __init__(self, inp_size, hidden_size, num_stacked_layer=4, input_dropout_p=0,
                dropout_p=0, bidirectional=False, rnn_cell='gru', start_token='zero',
                precision='single', transfer=False):
        super(DecoderRNN, self).__init__(inp_size, hidden_size, input_dropout_p,
                                        dropout_p, num_stacked_layer, rnn_cell)
        if precision == 'single':
            torch.set_default_tensor_type('torch.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.DoubleTensor')

        self.bidirectional_encoder = bidirectional
        self.output_size = inp_size
        self.hidden_size = hidden_size
        self.start_token = start_token
        self.transfer = transfer
        self.logger = logging.getLogger(__name__)

        self.rnn = self.rnn_cell(inp_size, hidden_size, num_stacked_layer,
                                batch_first=True, dropout=dropout_p)
        if transfer:
            self.logger.info("Set the requires_grad of decoder rnn cells to false")
            for param in self.rnn.parameters():
                param.requires_grad = False

        for L in self.rnn.named_parameters():
            if 'weight' in L[0]:
                torch.nn.init.xavier_normal_(L[1])
            else:
                torch.nn.init.normal_(L[1])

        self.out = nn.Linear(self.hidden_size, self.output_size)
        torch.nn.init.xavier_normal_(self.out.weight)
        torch.nn.init.normal_(self.out.bias)

    def forward_step(self, input_var, h, outfunc=F.tanh):
        batch_size = input_var.size(0)

        dropped_input_var = self.input_dropout(input_var)
        output, hidden = self.rnn(dropped_input_var, h)

        predicted_wave = self.out(output.contiguous().view(-1, self.hidden_size)) \
                                .view(-1, batch_size, self.output_size)
                                
        # predicted_wave = torch.where(predicted_wave > 1, torch.tensor([1.]).cuda(), predicted_wave)
        # predicted_wave = torch.where(predicted_wave < -1, torch.tensor([-1.]).cuda(), predicted_wave)

        return predicted_wave, hidden

    def forward(self, inputs, encoder_hidden,
                outfunc=F.tanh, teacher_forcing_ratio=0):
        inputs, batch_size, max_length = self._validate_args(inputs, 
                                                    encoder_hidden,
                                                    teacher_forcing_ratio)

        # estime = time.time()
        decoder_hidden = self._init_state(encoder_hidden)
        # print max_length

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        
        decoder_input = inputs[:, 0].unsqueeze(1)
        #decoder_input = inputs[:, 0]
        for di in xrange(max_length):
            decoder_output, decoder_hidden = self.forward_step(decoder_input,
                                                                decoder_hidden,
                                                                outfunc=outfunc)
            if di == 0:
                decoder_outputs = decoder_output
            else:
                decoder_outputs = torch.cat((decoder_outputs, decoder_output), 0)

            if use_teacher_forcing:
                decoder_input = inputs[:, di+1].unsqueeze(1)
            else:
                decoder_input = decoder_output.view(decoder_output.size(1), -1, 1)

        # eetime = time.time()
        # edur = eetime-estime
        # self.logger.info('except_validate_args took {}'.format(edur))
        return decoder_outputs

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
        if teacher_forcing_ratio != 0:
            batch_size = inputs.size(0)
            max_length = inputs.size(1)-1
        else:
            if self.rnn_cell is nn.LSTM:
                batch_size = encoder_hidden[0].size(1)
            elif self.rnn_cell is nn.GRU:
                batch_size = encoder_hidden.size(1)
            max_length = int(800/self.inp_size)
            if self.start_token == 'zero':
                inputs = torch.tensor([[0.]*self.output_size]*batch_size)\
                                       .view(batch_size, 1, self.inp_size)
            else:
                inputs = inputs[:, 0].unsqueeze(1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
        return inputs, batch_size, max_length
