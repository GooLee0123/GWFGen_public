import math

import torch
import torch.nn as nn
import numpy as np


class Loss(object):

    def __init__(self, name, criterion):
        self.name = name
        self.criterion = criterion
        if not issubclass(type(self.criterion), nn.modules.loss._Loss):
            raise ValueError("Criterion has to be a subclass of torch.nn._Loss")
        # accumulated loss
        self.acc_loss = 0
        # normalization term
        self.norm_term = 0

    def reset(self):
        """ Reset the accumulated loss. """
        self.acc_loss = 0
        self.norm_term = 0

    def get_loss(self):
        """ Get the loss.

        This method defines how to calculate the averaged loss given the
        accumulated loss and the normalization term.  Override to define your
        own logic.

        Returns:
            loss (float): value of the loss.
        """
        raise NotImplementedError

    def eval_batch(self, outputs, target):
        """ Evaluate and accumulate loss given outputs and expected results.

        This method is called after each batch with the batch outputs and
        the target (expected) results.  The loss and normalization term are
        accumulated in this method.  Override it to define your own accumulation
        method.

        Args:
            outputs (torch.Tensor): outputs of a batch.
            target (torch.Tensor): expected output of a batch.
        """
        raise NotImplementedError

    def cuda(self):
        self.criterion.cuda()

    def backward(self, retain_graph=False):
        self.acc_loss.backward(retain_graph=retain_graph)


class WMSELoss(Loss):
    _NAME = "Avg WMSELoss"

    def __init__(self, reduction=False):
        self.reduction = reduction

        if reduction:
            reduction = 'mean'
        else:
            reduction = 'sum'

        super(WMSELoss, self).__init__(
            self._NAME, nn.MSELoss(reduction=reduction))

    def get_loss(self):
        # total loss for all batches

        loss = self.acc_loss.item()
        if self.reduction:
            # average loss per batch
            loss /= self.norm_term
        return loss

    def eval_batch(self, outputs, target, vote=False, norm_update=True):
        if vote:
            self.acc_loss += self.criterion(outputs, target)
        else:
            onorm = torch.sqrt(torch.sum(outputs**2.))
            tnorm = torch.sqrt(torch.sum(target**2.))
            norm = onorm*tnorm

            tovlp = -torch.sum(outputs*target)/norm
            self.acc_loss += tovlp

        if norm_update:
            self.norm_term += 1


class FMSELoss(Loss):
    _NAME = "Avg FMSELoss"

    def __init__(self, reduction=False):
        self.reduction = reduction

        if reduction:
            reduction = 'mean'
        else:
            reduction = 'sum'

        super(FMSELoss, self).__init__(
            self._NAME, nn.MSELoss(reduction=reduction))

    def get_loss(self):
        # total loss for all batches

        loss = self.acc_loss.item()
        if self.reduction:
            # average loss per batch
            loss /= self.norm_term
        return loss

    def eval_batch(self, outputs, target, vote=False, norm_update=True):
        assert len(target) == len(outputs)
        if vote:
            self.acc_loss += self.criterion(outputs, target)
        else:
            mlen = len(target)

            t_ff = torch.rfft(target, 1, onesided=False)[:mlen//2]
            g_ff = torch.rfft(outputs, 1, onesided=False)[:mlen//2]
            df = 4096. / mlen

            t_norm_term = torch.sqrt(
                torch.trapz(t_ff[:, 0]**2 + t_ff[:, 1]**2, dx=df))
            g_norm_term = torch.sqrt(
                torch.trapz(g_ff[:, 0]**2 + g_ff[:, 1]**2, dx=df))
            norm_term = t_norm_term*g_norm_term

            rgtf = g_ff[:, 0]*t_ff[:, 0] + g_ff[:, 1]*t_ff[:, 1]
            igtf = -(g_ff[:, 0]*t_ff[:, 1] - g_ff[:, 1]*t_ff[:, 0])
            gtf = torch.cat((rgtf.view(-1, 1), igtf.view(-1, 1)), 1)
            ungtf = torch.sum(gtf[:-1]+gtf[1:], axis=0)*df/2.
            tovlp = -torch.norm(ungtf, 2)/norm_term

            self.acc_loss += tovlp

        if norm_update:
            self.norm_term += 1
