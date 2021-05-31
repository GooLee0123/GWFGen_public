import math
import copy

import torch
import torch.nn as nn

from .BaseRNN import BaseRNN

class EncoderRNN(BaseRNN):
    r"""
    Applies a multi-layer RNN to an input sequence.

    Args:
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encodr (defulat False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`

    Examples::

         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)
    """

    def __init__(self, inp_size, hidden_size, 
                num_stacked_layer=5, input_dropout_p=0, dropout_p=0,
                bidirectional=False, rnn_cell='gru', variable_lengths=True,
                precision='single'):
        super(EncoderRNN, self).__init__(inp_size, hidden_size, input_dropout_p,
                                        dropout_p, num_stacked_layer, rnn_cell)

        if precision == 'single':
            torch.set_default_tensor_type('torch.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.DoubleTensor')

        self.variable_lengths = variable_lengths

        self.rnn = self.rnn_cell(inp_size, hidden_size, num_stacked_layer,
                                batch_first=True, bidirectional=bidirectional,
                                dropout=dropout_p)

        for L in self.rnn.named_parameters():
            if 'weight' in L[0]:
                torch.nn.init.xavier_normal_(L[1])
            else:
                torch.nn.init.normal_(L[1])

    def forward(self, input_var):
        # print "Encoder"
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len, input_len): tensor containing the features of the input sequence.

        Returns: output, hidden
            - **hidden** (num_layers * num_directions, batch, 
            hidden_size): variable containing the features in the hidden state h
        """
        _, hidden = self.rnn(input_var)

        return hidden
