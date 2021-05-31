#! /usr/bin/python
import ConfigParser
import argparse
import logging
import math
import sys
import os

import pandas as pd
import numpy as np

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from wavepy.waves import DatPrep
from wavepy.loss import WMSELoss, FMSELoss
from wavepy.utils import Checkpoint

from networks.optim import Optimizer
from networks.trainer import SupervisedTrainer
from networks.seq2seq import EncoderRNN, DecoderRNN, VoterRNN, Seq2seq
from networks.evaluator import Predictor

def config_processing(opts):
    optdict = vars(opts)
    
    for key in optdict.keys():
        try:
            if '.' in optdict[key]:
                optdict[key] = float(optdict[key])
            else:
                optdict[key] = int(optdict[key])
        except:
            pass

    return optdict


def batch_generation(obj, trainD, valD, trainIA, valIA, 
                     dtype, bsize, valbsize, key, trainL=None, 
                     rtrainL=None, valL=None, rvalL=None, isize=1):
    logging.info("start to get data generator")

    batch_gen_train = obj.pad_batch_by_order(trainD, trainL, rtrainL, bsize, 
                                            isize, dtype, trainIA,
                                            key, train=True)
    batch_gen_val = obj.pad_batch_by_order(valD, valL, rvalL, valbsize, isize, 
                                            dtype, valIA, key)

    return batch_gen_train, batch_gen_val


def train_data_preparation(trainf, valf, optdict, key, dtype=torch.float32,
                           trainSI=None, valSI=None, wvalSI=None,
                           trainIA=None, valIA=None, wvalIA=None):
    if dtype == torch.float32:
        valbsize = 3000
    else:
        valbsize = 32
    datprep = DatPrep(optdict['path'])
    wdatprep = DatPrep(optdict['path'])
    bsize = optdict['batch_size']
    trainD, valD = datprep.load_split(trainf, valf)
    trainD, valD = datprep.tolist_split()
    trainD, valD = datprep.tofloat_split()
    wvalD = wdatprep.load(valf)
    wvalD = wdatprep.tolist()
    wvalD = wdatprep.tofloat()

    if key == 'target':
        trainD, valD = datprep.nonzero_split()
        wvalD = wdatprep.nonzero()

    if optdict['data_process'] == 'scale':
        trainD, valD = datprep.scale_split(sfactor=optdict['sfactor'])
        wvalD = datprep.scale(sfactor=optdict['sfactor'])
    elif optdict['data_process'] == 'normalize':
        trainD, valD = datprep.normalize_split(strain='hp')
        wvalD = wdatprep.normalize(strain='hp')

    if key == 'input':
        trainD, valD, trainSI, valSI = datprep.sort_by_len_split(key)
        wvalD, wvalSI = wdatprep.sort_by_len(key)
    else:
        trainD, valD = datprep.sort_by_len_split(key, trainSI, valSI)
        wvalD = wdatprep.sort_by_len(key, wvalSI)

    trainL, valL, rtrainL, rvalL = datprep.get_input_len_split(optdict['inp_size'])
    wvalL, rwvalL = wdatprep.get_input_len(optdict['inp_size'])

    if key == 'input':
        trainIA = datprep.get_iter_arr(trainL, bsize, True)
        valIA = datprep.get_iter_arr(valL, valbsize)
        wvalIA = wdatprep.get_iter_arr(wvalL, 3000)

    DBG_train, DBG_val = batch_generation(datprep, trainD, valD,
                                          trainIA, valIA, dtype, bsize,
                                          valbsize, key, trainL=trainL,
                                          rtrainL=rtrainL, valL=valL,
                                          rvalL=rvalL,
                                          isize=optdict['inp_size'])

    WDBG_val = wdatprep.pad_batch_by_order(wvalD, wvalL, rwvalL, 3000,
                                        optdict['inp_size'],
                                        dtype, wvalIA, key)

    total_dat_num = len(trainD)

    if key == 'input':
        return DBG_train, DBG_val, WDBG_val, trainSI, valSI, wvalSI, trainIA, valIA, wvalIA
    else:
        return DBG_train, DBG_val, WDBG_val, total_dat_num

conf_parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    add_help=False
    )
conf_parser.add_argument('-c', '--conf_file', 
                        default='./config_files/config.cfg',
                        help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()

Noption = 6
Keys = ["Computing", "Networks", "Training", "Waves", "Plots", "Logs"]
OptionDict = [{}]*Noption

if args.conf_file:
    config = ConfigParser.SafeConfigParser()
    config.read([args.conf_file])
    for i in xrange(Noption):
        OptionDict[i].update(dict(config.items(Keys[i])))

parser = argparse.ArgumentParser(parents=[conf_parser])

for i in xrange(Noption):
    parser.set_defaults(**OptionDict[i])

parser.add_argument('--train', default=False, type=bool,
                    dest='train', metavar="int",
                    help="the name of checkpoint to load, usually \
                    an encoded time string, (1 for load checkpoint)")
parser.add_argument('--log-level', default='info', type=str,
                    dest='log_level', metavar="info",
                    help="Logging level")

opts = parser.parse_args(remaining_argv)

optdict = config_processing(opts)

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging,
                    opts.log_level.upper()))
logging.info(optdict)

precision = optdict['precision']

if torch.cuda.is_available():
    logging.info("Cuda is available")
    torch.cuda.set_device(optdict['device'])
#####################################
# Seq2Seq for generator of GAN
hidden_size = optdict['hidden_size']
bidirectional = bool(optdict['bidirectional'])
resume = bool(optdict['resume'])

encoder_hs = hidden_size
decoder_hs = hidden_size

if bidirectional:
    logging.info("Use bidirectional seq2seq networks")
    decoder_hs *= 2
    bidirec_token = 'bidirectional'
else:
    logging.info("Use unidirectional seq2seq networks")
    bidirec_token = 'unidirectional'

encoder = EncoderRNN(optdict['inp_size'], encoder_hs,
                     num_stacked_layer=optdict['num_stacked_layer'],
                     input_dropout_p=optdict['input_drop_out'],
                     dropout_p=optdict['drop_out'],
                     bidirectional=bidirectional,
                     rnn_cell=optdict['rnn_cell'].lower(),
                     precision=precision)
decoder = DecoderRNN(optdict['out_size'], decoder_hs,
                     num_stacked_layer=optdict['num_stacked_layer'],
                     input_dropout_p=optdict['input_drop_out'],
                     dropout_p=optdict['drop_out'],
                     bidirectional=bidirectional,
                     rnn_cell=optdict['rnn_cell'].lower(),
                     start_token=optdict['start_token'],
                     precision=precision,
                     transfer=optdict['transfer'])
voter = VoterRNN(optdict['vote_size'], decoder_hs,
                 num_stacked_layer=optdict['num_stacked_layer'],
                 input_dropout_p=optdict['input_drop_out'],
                 dropout_p=optdict['drop_out'],
                 bidirectional=bidirectional,
                 rnn_cell=optdict['rnn_cell'].lower(),
                 start_token=optdict['start_token'],
                 precision=precision,
                 isize=optdict['inp_size'],
                 transfer=optdict['transfer'])

# Prepare loss
if optdict['loss'] == 'WMSE':
    Gloss = WMSELoss(reduction=True)
elif optdict['loss'] == 'FMSE':
    Gloss = FMSELoss(reduction=True)
else:
    raise NotImplementedError()

G = Seq2seq(encoder, decoder, voter, precision)
if torch.cuda.is_available():
    G.cuda()
    Gloss.cuda()

Goptimizer = Optimizer(
                torch.optim.Adam(
                    G.parameters(),
                    lr=optdict['gen_base_lr'],
                    betas=(optdict['gen_beta1'], 0.999),
                    weight_decay=5e-5),
                max_grad_norm=5)

if optdict['scheduler']:
    Gscheduler = ReduceLROnPlateau(Goptimizer.optimizer,
                                   patience=optdict['scheduler_patience'],
                                   factor=optdict['scheduler_factor'],
                                   min_lr=optdict['gen_min_lr'])
    Goptimizer.set_scheduler(Gscheduler)

precision = optdict['precision']
if precision == 'single':
    torch.set_default_tensor_type('torch.FloatTensor')
    dtype = torch.float32
else:
    torch.set_default_tensor_type('torch.DoubleTensor')
    dtype = torch.float64

# preapare sample plot dir for test #
outdir_rnn_token = optdict['rnn_cell'].upper()
outdir_hidden_token = 'n'+str(hidden_size)
outdir_batch_token = 'BS'+str(optdict['batch_size'])

# G_outdir_lr = str("%.7f" % optdict['gen_base_lr']).split('.')
# G_outdir_lr_token = G_outdir_lr[0]+G_outdir_lr[1]

outdir_tfr = str(optdict['teacher_forcing_ratio']).split('.')
outdir_tfr_token = outdir_tfr[0]+outdir_tfr[1]

data_token = optdict['path'].split('_')[-1]
if not 'bias' in data_token:
    data_token = 'bias_'+data_token

alpha_token = str(optdict['alpha'])
isize_token = 'Isize_'+str(optdict['inp_size'])

outdir_train_sample = optdict['strain']+'_'+outdir_rnn_token+'_' \
                        +bidirec_token+'_' \
                        +str(optdict['num_stacked_layer']) \
                        +outdir_hidden_token+'_' \
                        +outdir_batch_token+'_tfr' \
                        +outdir_tfr_token+'_'\
                        +data_token+'_alpha'\
                        +alpha_token+'_'\
                        +isize_token

if optdict['start_token'] == 'input':
	outdir_train_sample += '_ST_input'

if opts.train:
    # Data Preparation
    traini = os.path.join(optdict['path'], optdict['ftrain'].\
                        replace('strain', optdict['strain']))
    vali = os.path.join(optdict['path'], optdict['fval'].\
                        replace('strain', optdict['strain']))

    traint = traini.replace('input', 'target')
    valt = vali.replace('input', 'target')

    # IDBG: Input Data Batch Generator [0: for data, 1: for lengths]
    # TDBG: Target Data Batch Generator [0: for data, 1: for lengths]
    IDBG_train, IDBG_val, WIDBG_val, trainSI, valSI, wvalSI, trainIA, valIA, wvalIA = \
        train_data_preparation(traini, vali, optdict, 'input')
    TDBG_train, TDBG_val, WTDBG_val, total_dat_num= \
        train_data_preparation(traint, valt, optdict, 'target', dtype=dtype,
                        trainSI=trainSI, valSI=valSI, wvalSI=wvalSI,
                        trainIA=trainIA, valIA=valIA, wvalIA=wvalIA)

    tbatch_size = 3000
    testi = os.path.join(optdict['path'], optdict['ftest'] \
                    .replace('strain', optdict['strain']))
    testt = testi.replace('input', 'target')
    idatprep = DatPrep(optdict['path'])
    tdatprep = DatPrep(optdict['path'])
    itest = idatprep.load(testi)
    ttest = tdatprep.load(testt)
    itest = idatprep.tolist()
    ttest = tdatprep.tolist()
    itest = idatprep.tofloat()
    ttest = tdatprep.tofloat()
    ttest = tdatprep.nonzero()
    itest = idatprep.normalize()
    if tbatch_size != 1:
        itest, testSI = idatprep.sort_by_len('input')
        ttest = tdatprep.sort_by_len('target', testSI)
        batch_iter = 1
    else:
        testSI = None
        batch_iter = len(itest)
    itestL, ritestL = idatprep.get_input_len(optdict['inp_size'])
    ttestL, rttestL = tdatprep.get_input_len(optdict['inp_size'])
    testIA = idatprep.get_iter_arr(itestL, tbatch_size, shuffle=False)
    IDBG_itest = idatprep.pad_batch_by_order(itest, itestL, ritestL, tbatch_size,
                                        optdict['inp_size'],
                                        dtype, testIA, 'input',
                                        train=False)
    IDBG_ttest = tdatprep.pad_batch_by_order(ttest, ttestL, rttestL, tbatch_size,
                                        optdict['inp_size'],
                                        dtype, testIA, 'target',
                                        train=False)

    optdict['batch_train_epoch'] = int(
                                math.ceil(
                                total_dat_num/optdict['batch_size']))
    # train
    t = SupervisedTrainer(Gloss=Gloss, valSI=valSI,
                        checkpoint_every=optdict['checkpoint_every'],
                        isize=optdict['inp_size'],
                        interim_plot=optdict['interim_plot'],
                        print_every=optdict['print_every'],
                        chckpt_dir=optdict['chckpt_dir'],
                        alpha=optdict['alpha'],
                        config=optdict,
                        out_token=outdir_train_sample)

    GAN_seq2seq = t.train(G, IDBG_train, IDBG_val, WIDBG_val,
                        TDBG_train, TDBG_val, WTDBG_val,
                        IDBG_itest, IDBG_ttest,
                        num_epochs=optdict['batch_iteration'],
                        resume=optdict['resume'],
                        Goptimizer=Goptimizer,
                        teacher_forcing_ratio= \
                        optdict['teacher_forcing_ratio'])
else:
    G.eval()
    tbatch_size = optdict['batch_size']
    if optdict['search'].lower() == 'every':
        testi = os.path.join(optdict['path'], optdict['ftest'] \
                        .replace('strain', optdict['strain']))
    elif optdict['search'].lower() == 'latest':
        testi = os.path.join(optdict['path'], optdict['ftest'] \
                        .replace('strain', optdict['strain']))
    testt = testi.replace('input', 'target')
    idatprep = DatPrep(optdict['path'])
    tdatprep = DatPrep(optdict['path'])
    itest = idatprep.load(testi)
    ttest = tdatprep.load(testt)
    itest = idatprep.tolist()
    ttest = tdatprep.tolist()
    itest = idatprep.tofloat()
    ttest = tdatprep.tofloat()
    ttest = tdatprep.nonzero()
    itest = idatprep.normalize()
    unsorted_ttest = tdatprep.normalize()
    if tbatch_size != 1:
        itest, testSI = idatprep.sort_by_len('input')
        ttest = tdatprep.sort_by_len('target', testSI)
        batch_iter = 1
    else:
        testSI = None
        batch_iter = len(itest)
    itestL, ritestL = idatprep.get_input_len(optdict['inp_size'])
    ttestL, rttestL = tdatprep.get_input_len(optdict['inp_size'])
    testIA = idatprep.get_iter_arr(itestL, tbatch_size, shuffle=False)
    IDBG_itest = idatprep.pad_batch_by_order(itest, itestL, ritestL, tbatch_size,
                                        optdict['inp_size'],
                                        dtype, testIA, 'input',
                                        train=False,
                                        repeatable=False)
    IDBG_ttest = tdatprep.pad_batch_by_order(ttest, ttestL, ttestL, tbatch_size,
                                        optdict['inp_size'],
                                        dtype, testIA, 'target',
                                        train=False,
                                        repeatable=False)

    if optdict['search'].lower() == 'latest':
        latest_checkpoint_path = Checkpoint.get_latest_checkpoint(optdict)
        logging.info("load checkpoint from {}"\
                    .format(latest_checkpoint_path))
        checkpoint, overlap = Checkpoint.load(G, 
                                Goptimizer, 
                                latest_checkpoint_path, 
                                optdict)
        G = checkpoint.G

        # for name, param in G.named_parameters():
        #     if 'weight_hh_l3' in name:
        #         print name, param.data

        predictor = Predictor(G, IDBG_itest, IDBG_ttest, tbatch_size, optdict, testSI,
                             unsorted_target=unsorted_ttest, precision=precision, isize=optdict['inp_size'])
        gen_path = predictor.predict(batch_iter)
    elif optdict['search'].lower() == 'every':
        every_checkpoint_path = Checkpoint.get_every_checkpoint(optdict)

        min_max_overlap_time = None
        min_max_overlap = 0
        inputs, _ = next(IDBG_itest)
        targets, target_lengths = next(IDBG_ttest)

        overlap_bowl = []
        for checkpoint_path in every_checkpoint_path:
            logging.info("load checkpoint from {}"\
                    .format(checkpoint_path))
            checkpoint, overlap = Checkpoint.load(G, 
                                Goptimizer, 
                                checkpoint_path, 
                                optdict)
            G = checkpoint.G

            predictor = Predictor(G, inputs, targets, tbatch_size, optdict, testSI, 
                                target_lengths=target_lengths, unsorted_target=unsorted_ttest, 
                                precision=precision)
            temp_min_overlap = predictor.predict(batch_iter, return_overlap=True)
            overlap_bowl.append(temp_min_overlap)

            if temp_min_overlap > min_max_overlap:
                min_max_overlap = temp_min_overlap
                min_max_overlap_time = checkpoint_path

            logging.info("Current maximum value of overlap is {}".format(min_max_overlap))

        fcheckpoint, foverlap = Checkpoint.load(G,
                                Goptimizer,
                                min_max_overlap_time,
                                optdict)
        G = fcheckpoint.G
        fpredictor = Predictor(G, inputs, targets, tbatch_size, optdict, testSI, unsorted_ttest)
        gen_path = fpredictor.predict(batch_iter)
        new_path = min_max_overlap_time.replace(min_max_overlap_time.split('/')[-1], 'max_overlap_checkpoint')
        ckpt_path = min_max_overlap_time.replace(min_max_overlap_time.split('/')[-1], '')
        if min_max_overlap_time != new_path:
        	try:
        		os.system('rm -r {}'.format(new_path))
        	except:
        		pass
        	os.system('mv -v {} {}'.format(min_max_overlap_time, new_path))
        if optdict['remove_checkpoints']:
	        os.system('rm -r {}201*'.format(ckpt_path))
        np.savetxt(outdir_train_sample+'_overlaps', np.array(overlap_bowl).reshape(-1, 1))
    else:
        raise ValueError("Search option should be either 'latest' or 'every'")
