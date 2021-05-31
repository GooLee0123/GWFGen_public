import time
import torch
import logging
import numpy as np
import torch.nn as nn

class Seq2seq(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

    Inputs: input_variable, target_variable, teacher_forcing_ratio
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.

    """

    def __init__(self, encoder, decoder, voter,
              decode_out_func=torch.tanh, vote_out_func=torch.sigmoid):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.voter = voter

        self.logger = logging.getLogger(__name__)

        self.decode_out_func = decode_out_func
        self.vote_out_func = vote_out_func

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()
        self.voter.rnn.flatten_parameters()

    def forward(self, input_variable, target_variable,
                vote_answer=None, teacher_forcing_ratio=0):
        # estime = time.time()
        encoder_hidden = self.encoder(input_variable)
        # eetime = time.time()
        # edur = eetime-estime
        # self.logger.info('encoding took {}'.format(edur))

        # estime = time.time()
        decoder_output = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              outfunc=self.decode_out_func,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        # eetime = time.time()
        # edur = eetime-estime
        # self.logger.info('decoding took {}'.format(edur))

        # estime = time.time()
        voter_results = self.voter(inputs=vote_answer,
                            encoder_hidden=encoder_hidden,
                            votefunc=self.vote_out_func,
                            teacher_forcing_ratio=teacher_forcing_ratio)
        # eetime = time.time()
        # edur = eetime-estime
        # self.logger.info('voting took {}'.format(edur))

        return decoder_output, voter_results
