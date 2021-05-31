import itertools

import numpy as np

import torch

from wavepy.loss import WMSELoss

class Evaluator(object):

    def __init__(self, loss=WMSELoss(), valSI=None,
                 precision='single', config=None, isize=1):
        if precision == 'single':
            torch.set_default_tensor_type('torch.FloatTensor')
            self.dtype = torch.float32
        else:
            torch.set_default_tensor_type('torch.DoubleTensor')
            self.dtype = torch.float64

        self.loss = loss
        self.inputSI = valSI
        self.sampling_rate = 4096.
        self.config = config
        self.isize = isize

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

    def _dat_processing(self, outputs, targets):
        outputs = outputs.cpu()
        tlist = []
        for t in targets:
            tlist.append(t.cpu().numpy().ravel())
        return outputs, tlist

    def _get_min_overlap(self, decoder_output, targets, cutoffs,
                         target_lengths, target_rlengths, verbose=False):
        batch_size = decoder_output.shape[1]

        decoder_output, targets = self._dat_processing(decoder_output, targets)

        overlap_bowl = []
        for i in range(batch_size):
            cutoff = int(cutoffs[i])*self.isize
            trlenth = int(target_rlengths[i].data.item())

            twave = np.array(targets[i]).ravel()[self.isize:self.isize+trlenth]
            temp_gwave = np.array(decoder_output[:, i])
            temp_gwave = temp_gwave.ravel()
            gwave = temp_gwave[:trlenth]

            # mlen = min(len(twave), len(gwave))
            mlen = len(gwave)
            if mlen == 0:
                return 0
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

    def _get_go_stop_prob(self, lengths, targets):
        sup = np.float(5)
        ans = torch.zeros(targets.size(0), targets.size(1), 1)
        for i, L in enumerate(lengths):
            L = int(L)
            x = torch.linspace(1, L-1, L)
            y = 1.0 - 0.5*(x/L)**sup
            ans[i, 1:L+1] = y.view(-1, 1)
        if torch.cuda.is_available():
            return ans.cuda()
        else:
            return ans

    def evaluate(self, seq2seq, inputs, targets, target_lengths, target_rlengths, verbose=False):
        seq2seq.eval()
        seq2seq.train(False)
        loss = self.loss
        loss.reset()

        batch_size = targets.size(0)

        with torch.no_grad():
            val_decoder_output, val_voter_results = seq2seq(inputs, targets, teacher_forcing_ratio=0)
        masking_weight = self._get_masking_weights(val_decoder_output, target_lengths, batch_size)
        masked_val_decoder_outputs = masking_weight*val_decoder_output
        vmasking_weight = self._get_masking_weights(val_decoder_output, target_lengths, batch_size, vote=True)
        masked_voter_outputs = vmasking_weight*val_voter_results[0]

        vote_answer = self._get_go_stop_prob(target_lengths, targets)

        min_overlap = self._get_min_overlap(masked_val_decoder_outputs.contiguous(), targets, val_voter_results[1], target_lengths, target_rlengths, verbose=verbose)

        for b in range(batch_size):
            tlen = int(target_lengths[b])
            rtlen = int(target_rlengths[b])
            _o = masked_val_decoder_outputs[:, b][:tlen].reshape(-1).contiguous()
            _o = _o[:rtlen]
            _t = targets[b][1:tlen+1].reshape(-1)
            _t = _t[:rtlen]

            _vo = masked_voter_outputs[:, b][:tlen].reshape(-1).contiguous()
            _vt = vote_answer[b][1:tlen+1].reshape(-1)

            loss.eval_batch(_o, _t, vote=False) # only see errors for waveforms
            loss.eval_batch(_vo, _vt, vote=True) # only see errors for waveforms

        loss_val = loss.get_loss()

        return min_overlap, loss_val, [masked_val_decoder_outputs, val_voter_results]
