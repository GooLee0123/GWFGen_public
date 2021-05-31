import os
import math
import time
import logging
import smtplib
import itertools

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from wavepy.loss import WMSELoss
from wavepy.utils import Checkpoint
from wavepy.utils import Plotter
from networks.optim import Optimizer
from networks.evaluator import Evaluator


class SupervisedTrainer(object):

    def __init__(self, Gloss=WMSELoss(), checkpoint_every=100, isize=100,
                 interim_plot=False, chckpt_dir='checkpoints', valSI=None,
                 alpha=5, print_every=10, config=None, out_token=None,
                 vote_size=1):
        self.interim_plot = interim_plot
        self.Gloss = Gloss
        self.evaluator = Evaluator(loss=self.Gloss, valSI=valSI, config=config, isize=isize)
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every
        self.config = config
        self.out_token = out_token
        self.alpha = alpha
        self.isize = isize
        self.vote_size = vote_size

        if self.config['precision'] == 'single':
            torch.set_default_tensor_type('torch.FloatTensor')
            self.dtype = torch.float32
        else:
            torch.set_default_tensor_type('torch.DoubleTensor')
            self.dtype = torch.float64

        if not os.path.isabs(chckpt_dir):
            chckpt_dir = os.path.join(os.getcwd(), chckpt_dir)
        self.chckpt_dir = chckpt_dir
        if not os.path.exists(self.chckpt_dir):
            os.makedirs(self.chckpt_dir)
        self.batch_train_epoch = config['batch_train_epoch']
        self.logger = logging.getLogger(__name__)

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
            dtype=self.dtype)
        idxs = idxs.view(sizes)

        for b in range(batch_size):
            weight[:, b] = torch.where(idxs[:, b] >= lengths[b].cpu(),
                                       idxs[:, b].cpu(), weight[:, b].cpu())
            weight[:, b] = torch.where(idxs[:, b] < lengths[b].cpu(),
                                       weight[:, b].cpu(), pad[:, b].cpu())

        return weight

    def _get_go_stop_prob(self, lengths, targets):
        sup = np.float(self.alpha)
        ans = torch.zeros(targets.size(0), targets.size(1), self.vote_size)
        for i, L in enumerate(lengths):
            L = int(L)
            x = torch.linspace(1, L-1, L)
            y = 1.0 - 0.5*(x/L)**sup
            ans[i, 1:L+1] = y.view(-1, 1)
        if torch.cuda.is_available():
            return ans.cuda()
        else:
            return ans

    def _train_batch(self, input_variable, target_variable, 
                     tgt_length, tgt_rlength, G, teacher_forcing_ratio,
                     epoch, batch_step):
        G.train(True)
        Gloss = self.Gloss

        vote_answer = self._get_go_stop_prob(tgt_length,
                                             target_variable)
        #self.logger.info(str(vote_answer.detach().cpu().numpy().shape))
        #self.logger.info(str(tgt_length))
        decoder_outputs, voter_outputs = G(
                            input_variable,
                            target_variable,
                            vote_answer,
                            teacher_forcing_ratio)
        #############################################

        voter_cutoff = voter_outputs[1]
        voter_outputs = voter_outputs[0]

        # Get masking weights
        batch_size = decoder_outputs.size(1)
        masking_weight = self._get_masking_weights(
                                        decoder_outputs,
                                        tgt_length,
                                        batch_size)
        vmasking_weight = self._get_masking_weights(
                                        decoder_outputs,
                                        tgt_length,
                                        batch_size,
                                        vote=True)

        # Get masked generator outputs
        masked_decoder_outputs = masking_weight*decoder_outputs
        masked_voter_outputs = vmasking_weight*voter_outputs

        Gloss.reset()
        # for step, step_output in enumerate(G_outputs):
        for b in range(batch_size):
            tlen = int(tgt_length[b])
            rtlen = int(tgt_rlength[b])
            _o = masked_decoder_outputs[:, b][:tlen].reshape(-1).contiguous()
            _o = _o[:rtlen]
            _t = target_variable[b][1:tlen+1].reshape(-1)
            _t = _t[:rtlen]
            _vo = masked_voter_outputs[:, b][:tlen].reshape(-1).contiguous()
            _vt = vote_answer[b][1:tlen+1].reshape(-1)

            Gloss.eval_batch(_o, _t, vote=False)
            Gloss.eval_batch(_vo, _vt, vote=True)
        # else:
        #     for step, step_outputs in enumerate(zip(
        #                             masked_decoder_outputs,
        #                             masked_voter_outputs)):
        #         Gloss.eval_batch(step_outputs[0].contiguous(),
        #                         target_variable[:, step+1])
        #         Gloss.eval_batch(step_outputs[1].contiguous(),
        #                         vote_answer[:, step+1])

        # for name, param in G.named_parameters():
        #     if param.requires_grad:
        #         print name, param.data

        G.zero_grad()
        Gloss.backward()
        self.Goptimizer.step()
        Gloss_val = Gloss.get_loss()

        return Gloss_val

    def _train_epochs(self, G, n_epochs, start_epoch,
                      start_step, train_data=None,
                      train_target=None, val_data=None,
                      val_target=None, wval_data=None,
                      wval_target=None, test_data=None,
                      test_target=None, teacher_forcing_ratio=0,
                      max_val_overlap=-1e10):
        log = self.logger
        self.n_epochs = n_epochs
        scheduler_bool = True

        G_print_loss_total = 0  # Reset every print_every for generator
        G_epoch_loss_total = 0  # Reset every epoch for generator
        G_epoch_val_loss_total = 0
        G_epcoh_val_ovlp_total = 0

        step_elapsed = 0

        test_overlap = 0

        total_steps = self.batch_train_epoch * n_epochs
        step = start_step
        ckpt_save = False

        stime = time.time()
        global_min_test_overlap = 0
        global_min_val_overlap = 0
        for epoch in range(start_epoch, n_epochs):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))

            epoch_loss_list = [[], []]
            epoch_ovlp_list = []

            loss_list = [[],[]]
            ovlp_list = []

            lr_list = []

            G.train(True)
            val_loss_denorm = 0
            for batch_step in range(self.batch_train_epoch):
                step += 1
                step_elapsed += 1

                # repeatable
                batch_inp_data, batch_padded, batch_inp_len = next(train_data)
                batch_tgt_data, batch_tgt_len, batch_tgt_rlen = next(train_target)

                inp_last = torch.tensor([batch_padded.squeeze()[li, lj-2] for li, lj in enumerate(batch_inp_len)]).view(-1)
                batch_tgt_data[:, 0] = inp_last.view(-1, 1)

                Gloss = self._train_batch(batch_inp_data, 
                                          batch_tgt_data,
                                          batch_tgt_len,
                                          batch_tgt_rlen,
                                          G, 
                                          teacher_forcing_ratio, 
                                          epoch, 
                                          batch_step)

                # Record average loss
                G_print_loss_total += Gloss
                G_epoch_loss_total += Gloss

                if step % self.print_every == 0 and \
                    step_elapsed >= self.print_every:
                    if step % (100*self.print_every) == 0 and max_val_overlap >= 0.99 and self.config['precision'] == 'double':
                        batch_val_inp_data, batch_val_padded, batch_val_inp_len = next(wval_data)
                        batch_val_tgt_data, batch_val_tgt_len, batch_val_tgt_rlen = next(wval_target)
                        val_inp_last = torch.tensor([batch_val_padded.squeeze()[li, lj-2] for li, lj in enumerate(batch_val_inp_len)]).view(-1)
                        batch_val_tgt_data[:, 0] = val_inp_last.view(-1, 1)
                    else:
                        batch_val_inp_data, batch_val_padded, batch_val_inp_len = next(val_data)
                        batch_val_tgt_data, batch_val_tgt_len, batch_val_tgt_rlen = next(val_target)
                        val_inp_last = torch.tensor([batch_val_padded.squeeze()[li, lj-2] for li, lj in enumerate(batch_val_inp_len)]).view(-1)
                        batch_val_tgt_data[:, 0] = val_inp_last.view(-1, 1)

                    G_print_loss_avg = G_print_loss_total / self.print_every
                    G_print_loss_total = 0

                    log_msg = 'Step: %d/%d, Progress: %d%%, ' % (
                            batch_step,
                            self.batch_train_epoch,
                            float(step)  / total_steps * 100) \
                            + 'Generator Train %s: %.8f' % (
                            self.Gloss.name,
                            G_print_loss_avg)

                    val_overlap, val_loss, eval_results = \
                        self.evaluator.evaluate(G, batch_val_inp_data,
                                                batch_val_tgt_data,
                                                batch_val_tgt_len,
                                                batch_val_tgt_rlen)
                    if global_min_val_overlap < val_overlap:
                        global_min_val_overlap = val_overlap

                    if val_overlap > 0.98:
                        batch_test_inp_data, batch_test_padded, batch_test_inp_len = next(test_data)
                        batch_test_tgt_data, batch_test_tgt_len, batch_test_tgt_rlen = next(test_target)

                        test_inp_last = torch.tensor([batch_test_padded.squeeze()[li, lj-2] for li, lj in enumerate(batch_test_inp_len)]).view(-1)
                        batch_test_tgt_data[:, 0] = test_inp_last.view(-1, 1)

                        test_overlap, test_loss, eval_results = \
                            self.evaluator.evaluate(G, batch_test_inp_data,
                                                    batch_test_tgt_data,
                                                    batch_test_tgt_len,
                                                    batch_test_tgt_rlen,
                                                    verbose=True)

                        if global_min_test_overlap < test_overlap:
                            global_min_test_overlap = test_overlap
                            check = True
                            self.logger.info("min test overlap: %.8f" % global_min_test_overlap)

                        if test_overlap > 0.98 and check:
                            Checkpoint(G=G,
                                    Goptimizer=self.Goptimizer,
                                    epoch=epoch, step=step,
                                    config=self.config,
                                    overlap=max_val_overlap).save()
                            log.info("Minimum overlap of validation is %.4f" % val_overlap)
                            log.info("Minimum overlap of test is %.4f" % test_overlap)
                            time_elapsed = time.time() - stime
                            hours, rem = divmod(time_elapsed, 3600)
                            minutes, seconds = divmod(rem, 60)
                            check = False
                            # try:
                            #     tfile = self.config['path'].split('/')[-1]
                            #     msize = self.config['hidden_size']

                            #     content1 = "Reaching to the desired accuracy,\n" 
                            #     content2 = "terminate training for the file named "
                            #     content3 = "%s\nwith model size %s at epoch %s\n\n" % (tfile, msize, epoch)
                            #     content4 = "##### TRAINING RESULTS #####\n\n"
                            #     content5 = "Minimum overlap for validation: %.4f \n" % val_overlap
                            #     content6 = "Minimum overlap for test: %.4f \n" % test_overlap
                            #     content7 = "Elapsed time for training: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
                            #     content = content1+content2+content3+content4+content5+content6+content7

                            #     mail = smtplib.SMTP('smtp.gmail.com', 587)
                            #     mail.ehlo()
                            #     mail.starttls() # next commands are encrypted
                            #     mail.login('ljg4471@gmail.com', 'Zovmffj447!')
                            #     mail.sendmail('ljg4471@gmail.com', 'ljg4471@gmail.com', content)
                            #     mail.close()
                            # except:
                            #     pass

                            # log.info("Reaching to desired accuracy, terminate training")
                            # log.info("Elapsed time for training: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

                            # quit()

                    G_epoch_val_loss_total += val_loss
                    G_epcoh_val_ovlp_total += val_overlap
                    val_loss_denorm += 1
                    #if max_val_overlap <= val_overlap:
                    #    max_val_overlap = val_overlap
                    #if val_overlap > 0.993:
                    #    ckpt_save = True

                    if self.interim_plot:
                        plotter = Plotter(self.config, epoch, batch_step)
                        plotter.plot_waves(
                            batch_val_inp_data, 
                            batch_val_tgt_data, 
                            eval_results[0], 
                            eval_results[1][0],
                            batch_val_inp_len,
                            batch_val_tgt_len,
                            eval_results[1][1],
                            batch_size=batch_val_tgt_data.size(0),
                            batch_plot=False,
                            random_index=False)

                    log.info("Cutoff indexes %s" % eval_results[1][1])
                    log.info("Target lengths: %s" % batch_val_tgt_len)

                    del eval_results

                    for param in self.Goptimizer.param_groups():
                        clr = param['lr']

                    log_msg += ", validation %s: %.8f, min overlap: %.8f, learning rate: %.8f" % \
                                (self.Gloss.name, val_loss, val_overlap, clr)
                    log_msg += "\n global min val overlap: %.5f" % global_min_val_overlap
                    log.info(log_msg)

                    loss_list[0].append(G_print_loss_avg)
                    loss_list[1].append(val_loss)
                    ovlp_list.append(val_overlap)
                    lr_list.append(clr)

                # Checkpoint
                # if step % self.checkpoint_every == 0 or \
                    # ckpt_save:
                if ckpt_save:
                    Checkpoint(G=G,
                            Goptimizer=self.Goptimizer,
                            epoch=epoch, step=step,
                            config=self.config,
                            overlap=max_val_overlap).save()
                    ckpt_save = False

            G_epoch_loss_avg = G_epoch_loss_total / min(
                                        self.batch_train_epoch, 
                                        (step - start_step))
            G_epoch_loss_total = 0

            G_epoch_val_loss_avg = G_epoch_val_loss_total/(val_loss_denorm)
            G_epoch_val_ovlp_avg = G_epcoh_val_ovlp_total/(val_loss_denorm)

            G_epoch_val_loss_total = 0
            G_epcoh_val_ovlp_total = 0

            epoch_loss_list[0].append(G_epoch_loss_avg)
            epoch_loss_list[1].append(G_epoch_val_loss_avg)
            epoch_ovlp_list.append(G_epoch_val_ovlp_avg)

            if scheduler_bool and G_epoch_val_ovlp_avg > self.config['high_overlap']:
                Gscheduler = ReduceLROnPlateau(self.Goptimizer.optimizer,
                                min_lr=self.config['gen_min_lr_h'])
                self.Goptimizer.set_scheduler(Gscheduler)
                scheduler_bool = False

            if epoch >= self.config['lr_decay_epoch']:
                self.Goptimizer.update(1./G_epoch_val_ovlp_avg, epoch)

            log_msg = "Finished epoch %d: Train %s: %.8f, Vaildation %s: %.8f, Avg validation overlap: %.4f" % \
                        (epoch, self.Gloss.name, G_epoch_loss_avg, 
                        self.Gloss.name, G_epoch_val_loss_avg,
                        G_epoch_val_ovlp_avg)
            log.info(log_msg)

            with open('./'+self.out_token+'_train_val_loss', 'a') as fvl:
                np.savetxt(fvl, np.array(loss_list).T, fmt='%.10f')
            with open('./'+self.out_token+'_epoch_loss', 'a') as fel:
                np.savetxt(fel, np.array(epoch_loss_list).T, fmt='%.10f')
            with open('./'+self.out_token+'_learning_rate', 'a') as flr:
                np.savetxt(flr, np.array(lr_list).T, fmt='%.10f')
            with open('./'+self.out_token+'_overlap', 'a') as fov:
                np.savetxt(fov, np.array(ovlp_list).T, fmt='%.10f')
            with open('./'+self.out_token+'_epoch_overlap', 'a') as fev:
                np.savetxt(fev, np.array(epoch_ovlp_list).T, fmt='%.10f')

    def train(self, G, I_train, I_val, WI_val, T_train, 
            T_val, WT_val, I_test, T_test,
            num_epochs=100, resume=False,
            Goptimizer=None, teacher_forcing_ratio=0):

        if self.config['resume']:
            latest_checkpoint_path = \
                Checkpoint.get_latest_checkpoint(self.config)
            resume_checkpoint, overlap = \
                Checkpoint.load(G, 
                            Goptimizer, 
                            latest_checkpoint_path,
                            self.config)

            self.logger.info(
                "Resume with checkpoints: %s" % latest_checkpoint_path)

            G = resume_checkpoint.G
            self.Goptimizer = resume_checkpoint.Goptimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.Goptimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.Goptimizer.optimizer = \
                resume_optim.__class__(G.parameters(), **defaults)

            for g in Goptimizer.param_groups():
                g['lr'] = self.config['gen_base_lr']
            # Resetting scheduler parameters
            if self.config['scheduler']:
                Gscheduler = ReduceLROnPlateau(self.Goptimizer.optimizer, 
                                patience=self.config['scheduler_patience'], 
                                factor=self.config['scheduler_factor'],
                                min_lr=self.config['gen_min_lr'])
                self.Goptimizer.set_scheduler(Gscheduler)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            if Goptimizer is None:
                Goptimizer = Optimizer(
                    optim.Adam(G.parameters(), 
                        lr=self.config['gen_base_lr'],
                        betas=(self.config['gen_beta1'], 0.999), 
                        weight_decay=5e-5), 
                    max_grad_norm=5)
            self.Goptimizer = Goptimizer
            overlap = -1e10

        self.logger.info("Generator Optimizer: %s, Generator Scheduler: %s" %
            (self.Goptimizer.optimizer, self.Goptimizer.scheduler))

        self._train_epochs(G, num_epochs, start_epoch, step,
                        train_data=I_train, train_target=T_train,
                        val_data=I_val, val_target=T_val,
                        wval_data=WI_val, wval_target=WT_val,
                        test_data=I_test, test_target=T_test,
                        teacher_forcing_ratio= \
                            self.config['teacher_forcing_ratio'],
                        max_val_overlap=overlap)
        return G
