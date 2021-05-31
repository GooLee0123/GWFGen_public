import os
import time
import logging
import itertools

import numpy as np
import torch

class Predictor(object):

    GENERATED_DIR_NAME = 'generated'

    def __init__(self, seq2seq, inputs, targets, batch_size, config, inputSI,
                target_lengths=None, unsorted_target=None, sampling_rate=4096.,
                precision='single', isize=1):
        if precision == 'single':
            torch.set_default_tensor_type('torch.FloatTensor')
            self.dtype = torch.float32
        else:
            torch.set_default_tensor_type('torch.DoubleTensor')
            self.dtype = torch.float64

        self.logger = logging.getLogger(__name__)
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.config = config
        self.inputSI = inputSI
        self.us_target = unsorted_target
        self.sampling_rate = sampling_rate
        self.target_lengths = target_lengths
        self.cuda = torch.cuda.is_available()
        self.isize = isize

        if self.cuda:
            self.seq2seq = seq2seq.cuda()
        else:
            self.seq2seq = seq2seq.cpu()

        self.seq2seq.eval()

    def _get_masking_weights(self, sample, lengths, batch_size, vote=False):
        sample_size = sample.size()
        seq_num = sample_size[0]
        bsize = sample_size[1]

        if vote:
            rep = 1
            sizes = (seq_num, bsize, 1)
        else:
            rep = self.isize
            sizes = sample_size

        weight = sample.new_ones(sizes, dtype=self.dtype)
        pad = sample.new_zeros(sizes, dtype=self.dtype)
        lst = range(0, seq_num)

        idxs = torch.tensor(list(itertools.chain.from_iterable(
            itertools.repeat(x, bsize*rep) for x in lst)),
            dtype=self.dtype).view(sizes)

        for b in range(batch_size):
            weight[:, b] = torch.where(idxs[:, b] >= lengths[b].cpu(),
                                       idxs[:, b].cpu(), weight[:, b].cpu())
            weight[:, b] = torch.where(idxs[:, b] < lengths[b].cpu(),
                                       weight[:, b].cpu(), pad[:, b].cpu())

        return weight

    def _write(self, decoder_output, cutoffs):
        data_token = self.config['path'].split('_')[-1]
        if not 'bias' in data_token:
            data_token = 'bias_'+data_token
        alpha_token = 'alpha_'+str(self.config['alpha'])
        rcell = self.config['rnn_cell'].upper()
        hiddens = str(self.config['num_stacked_layer'])+'n'+str(self.config['hidden_size'])
        tfr = 'TFR'+str(self.config['teacher_forcing_ratio']).replace('.', '')
        itoken = 'Isize_'+str(self.isize)
        model_structure = rcell+'_'+hiddens+'_'+tfr+'_'+data_token+'_'+alpha_token+'_'+itoken
        if self.config['start_token'] == 'input':
            model_structure += '_ST_input'

        gdirs = os.path.join(self.GENERATED_DIR_NAME, model_structure)
        gpath = os.path.join(gdirs, 'test_data_prediction')

        if not os.path.exists(gdirs):
            os.makedirs(gdirs)

        batch_size = decoder_output.shape[1]

        with open(gpath, 'w') as gtf:
            for i in xrange(batch_size):
                data = np.array(decoder_output[:, i])
                data = data.ravel()
                cutoff = int(cutoffs[i])*self.isize
                cdata = data[:cutoff]
                np.savetxt(gtf, cdata.reshape(1, -1))

        return gpath

    def _unsort(self, data, idx):
        unsort_data = np.zeros(data.size())
        unsort_idx = np.zeros(len(idx), dtype=np.int)
        unsort_check = np.zeros(len(idx))
        dlen = len(idx)
        for i in xrange(dlen):
            unsort_data[:, self.inputSI[i]] = data[:, i].cpu()
            unsort_idx[self.inputSI[i]] = int(idx[i])

        return unsort_data, unsort_idx

    def _get_min_overlap(self, decoder_output, cutoffs):
        batch_size = decoder_output.shape[1]

        overlap_bowl = []
        for i in xrange(batch_size):
            cutoff = int(cutoffs[i])*self.isize

            # print(self.s_target[i])
            # print(self.us_target[i])
            if batch_size == 1:
                twave = np.array(self.s_target[i].cpu()).ravel()
            else:
                twave = np.array(self.us_target[i]).ravel()
                # twave = np.array(targets[i]).ravel()[self.isize:tlenth*self.isize]

            gwave = np.array(decoder_output[:, i])
            gwave = gwave.ravel()
            gwave = gwave[:cutoff]

            mlen = min(len(twave), len(gwave))
            tw = twave[:mlen]
            gw = gwave[:mlen]
            df = self.sampling_rate / mlen

            t_ff = np.fft.fft(tw)[:mlen//2]
            g_ff = np.fft.fft(gw)[:mlen//2]

            t_norm_term = np.sqrt(np.trapz(t_ff*np.conjugate(t_ff), dx=df))
            g_norm_term = np.sqrt(np.trapz(g_ff*np.conjugate(g_ff), dx=df))
            norm_term = t_norm_term*g_norm_term

            temp_overlap = abs(np.trapz(g_ff*np.conjugate(t_ff), dx=df)/norm_term)

            overlap_bowl.append(temp_overlap)

        return min(overlap_bowl)

    def predict(self, batch_iter, return_overlap=False):
        if return_overlap and self.batch_size == 1:
            raise ValueError("Batch size for all checkpoint search should be bigger than test data size")
        self.seq2seq.train(False)
        self.seq2seq.eval()
        num_data = 1
        for _ in xrange(batch_iter):
            if self.config['search'] == 'latest':
                batch_test_input, batch_test_padded, batch_test_ilength = next(self.inputs)
                batch_test_target, target_lengths, target_rlengths = next(self.targets)

                test_inp_last = torch.tensor([batch_test_padded.squeeze()[li, lj-2] for li, lj in enumerate(batch_test_ilength)]).view(-1)
                batch_test_target[:, 0] = test_inp_last.view(-1, 1)

            elif self.config['search'] == 'every':
                batch_test_input = self.inputs
                batch_test_target = self.targets
                target_lengths = self.target_lengths

            if self.cuda:
                batch_test_input = batch_test_input.cuda()

            stime = time.time()
            with torch.no_grad():
                test_decoder_output, test_voter_results = self.seq2seq(batch_test_input, batch_test_target, teacher_forcing_ratio=0)
            etime = time.time()
            duration = etime - stime

            masking_weight = self._get_masking_weights(test_decoder_output, target_lengths, test_decoder_output.size(1))
            masked_test_decoder_outputs = masking_weight*test_decoder_output
            masked_test_decoder_outputs = masked_test_decoder_outputs.contiguous()

            if self.batch_size != 1:
                masked_test_decoder_outputs, test_voter_results = self._unsort(masked_test_decoder_outputs, test_voter_results[1])
            else:
                self.s_target = batch_test_target
                test_voter_results = test_voter_results[1]
                masked_test_decoder_outputs = masked_test_decoder_outputs.cpu()

            if return_overlap:
                min_overlap = self._get_min_overlap(masked_test_decoder_outputs, test_voter_results)
                self.logger.info("minimum overlap is calculated as {}".format(min_overlap))
            else:
                min_overlap = self._get_min_overlap(masked_test_decoder_outputs, test_voter_results)
                gen_path = self._write(masked_test_decoder_outputs, test_voter_results)
                self.logger.info("{}th generated test data is saved at {}".format(num_data, gen_path))
                self.logger.info("Min overlap is {} and computing took {}".format(min_overlap, duration))
                num_data += 1

            del masked_test_decoder_outputs
            del test_voter_results

        if return_overlap:
            return min_overlap
        else:
            return gen_path
