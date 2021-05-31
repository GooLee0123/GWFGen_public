import argparse
import copy
import logging
import os
import time
import math

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries
from pycbc.filter import matched_filter, highpass
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.waveform import get_td_waveform
from scipy.signal import hilbert

from networks.evaluator import Predictor
from networks.optim import Optimizer
from networks.seq2seq import DecoderRNN, EncoderRNN, Seq2seq, VoterRNN
from wavepy.utils import Checkpoint

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str, dest='data',
                    default='H-H1_LOSC_4_V1-1126293504-4096.hdf5')
parser.add_argument('-f', '--format', type=str, dest='format',
                    default='hdf5.losc')

opts = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level='INFO')

# GLOBAL #
duration = 32.
f_lower = 20.
f_upper = 1024.
rseed = 35

date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

alpha = 5
dtype = 'bias_40to100'
network = 'GRU_4n256_TFR05_'+dtype+'_alpha_'+str(alpha)
dpath = './data/IMR_'+dtype
outdn = './Pseudo_PE_test'
if '40to100' in dtype:
    outdn += '_40to100'
    gf_lower = 15.
    # targ_m1 = [90., 60.]
    # targ_m2 = [70., 50.]
    targ_m1 = [80., 65.]
    targ_m2 = [65., 55.]
    # dfact = [18., 13.]
    dfact = [16., 14.]

    min_mass = 40.
    max_mass = 100.
    far_minterval = 0.5
    vic_minterval = 0.1
    mtol = 4

    device = 1
else:
    gf_lower = 40.
    # targ_m1 = [35., 30., 25., 35.6]
    # targ_m2 = [20., 15., 15., 30.6]
    # dfact = [8., 7., 6., 4.3]
    targ_m1 = [30., 25.]
    targ_m2 = [25., 20.]
    dfact = [7., 6.]

    min_mass = 10.
    max_mass = 40.
    far_minterval = 0.25
    vic_minterval = 0.1
    mtol = 2

    device = 1

if not os.path.exists(outdn):
    os.makedirs(outdn)


def read_hdata():
    hdata = TimeSeries.read(opts.data, format=opts.format)
    return hdata.highpass(15.0).to_pycbc().crop(2, 2)


def hdata_rand_selection(hdata, dur=16):
    dt = hdata.delta_t
    sample_rate = 1/dt
    idx_interval = int(dur*sample_rate)
    ulim = len(hdata) - idx_interval
    np.random.seed(rseed)
    start_idx = np.random.randint(ulim)

    return hdata[start_idx-idx_interval/2:start_idx+idx_interval/2]


def inject_signal(noise, signal):
    temp_noise = copy.deepcopy(noise)

    len_sig = len(signal)
    inj_idx = len(temp_noise)//2

    # merging_time = (merg_idx+inj_idx)/4096.

    temp_noise[inj_idx:inj_idx+len_sig] += signal

    return temp_noise


def get_ISCO_index(signal, sampling_rate, total_mass, min_freq, t):
    analytic_signal = hilbert(signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) /
                               (2.0*np.pi)*sampling_rate)
    FISCO = 4400./total_mass
    ISCOS = np.argwhere(instantaneous_frequency >= FISCO)
    thr = gf_lower*8 if gf_lower == 15 else gf_lower*2
    IISCO = ISCOS[ISCOS > thr][0]+1

    # Merger_index = np.argwhere(t > 0.0)[0][0]

    # assert Merger_index 
    return IISCO


def def_optdict():
    optdict = {}
    optdict['bidirectional'] = None
    optdict['strain'] = 'hp'
    optdict['rnn_cell'] = 'gru'
    optdict['path'] = dpath
    optdict['num_stacked_layer'] = 4
    optdict['hidden_size'] = 256
    optdict['teacher_forcing_ratio'] = 0.5
    optdict['start_token'] = 'zero'

    optdict['inp_size'] = 1
    optdict['out_size'] = 1
    optdict['vote_size'] = 1
    optdict['input_drop_out'] = 0.
    optdict['drop_out'] = 0.
    optdict['alpha'] = 5

    optdict['gen_base_lr'] = 0.00002
    optdict['gen_beta1'] = 0.5

    global isize
    isize = optdict['inp_size']

    return optdict


def load_dl_model():
    optdict = def_optdict()

    latest_checkpoint_path = Checkpoint.get_latest_checkpoint(optdict)

    encoder = EncoderRNN(optdict['inp_size'], optdict['hidden_size'],
                         num_stacked_layer=optdict['num_stacked_layer'],
                         input_dropout_p=optdict['input_drop_out'],
                         dropout_p=optdict['drop_out'],
                         bidirectional=False,
                         rnn_cell=optdict['rnn_cell'].lower())
    decoder = DecoderRNN(optdict['out_size'], optdict['hidden_size'],
                         num_stacked_layer=optdict['num_stacked_layer'],
                         input_dropout_p=optdict['input_drop_out'],
                         dropout_p=optdict['drop_out'],
                         bidirectional=False,
                         rnn_cell=optdict['rnn_cell'].lower())
    voter = VoterRNN(optdict['vote_size'], optdict['hidden_size'],
                     num_stacked_layer=optdict['num_stacked_layer'],
                     input_dropout_p=optdict['input_drop_out'],
                     dropout_p=optdict['drop_out'],
                     bidirectional=False,
                     rnn_cell=optdict['rnn_cell'].lower(),
                     isize=optdict['inp_size'])
    G = Seq2seq(encoder, decoder, voter)
    G.eval()
    G.cuda()

    Goptimizer = Optimizer(torch.optim.Adam(
                           G.parameters(),
                           lr=optdict['gen_base_lr'],
                           betas=(optdict['gen_beta1'], 0.999),
                           weight_decay=5e-5),
                           max_grad_norm=5)

    logging.info("load checkpoint from {}"
                 .format(latest_checkpoint_path))

    checkpoint, ovlp = Checkpoint.load(G, Goptimizer,
                                       latest_checkpoint_path,
                                       optdict, device=device)
    G = checkpoint.G

    return G


def get_template(newsamp, index, seq2seq=None, key='DL'):
    hp, _ = get_td_waveform(approximant='SEOBNRv4_opt',
                            mass1=newsamp[0],
                            mass2=newsamp[1],
                            distance=100,
                            inclination=1.043,
                            delta_t=1/4096.,
                            f_lower=gf_lower)
    hp_ISCO = get_ISCO_index(hp, 4096., sum(newsamp), 15.,
                             hp.sample_times.numpy())

    mfn = os.path.join(dpath, 'maximum_amplitude')
    # logging.info("Read maximum amplitude from %s" % mfn)
    denorm = np.genfromtxt(mfn)[0]
    hp = hp/denorm

    org_hpi = hp[:hp_ISCO]
    hpt = hp[hp_ISCO-1:]
    hpt = hpt[np.nonzero(hpt)]

    def _roundup(x, th):
        return int(math.ceil(x/float(th))*th)

    if key == 'EOB':
        return np.array(hp) * denorm / dfact[index]
    elif key == 'DL':
        hpi = np.array(org_hpi, dtype=np.float32)
        len_hpi = len(hpi)
        plen = _roundup(len_hpi, 1)
        hpi = np.pad(hpi, (0, plen-len_hpi), 'constant', constant_values=(0.))

        hpi = torch.from_numpy(hpi)
        hpi = torch.nn.utils.rnn.pad_sequence([hpi], batch_first=True)
        hpi = hpi.view(hpi.size(0), -1, 1)
        hpi = torch.nn.utils.rnn.pack_padded_sequence(hpi, [len_hpi],
                                                      batch_first=True)

        hpt = torch.tensor(hpt, dtype=torch.float32).view(-1, len(hpt), 1)

        hpi = hpi.cuda()
        hpt = hpt.cuda()

        seq2seq.eval()
        with torch.no_grad():
            decoder_output, voter_output = seq2seq(hpi, hpt,
                                                   teacher_forcing_ratio=0)

        hpi = hpi.cpu()
        hpt = hpt.cpu()
        decoder_output = decoder_output.cpu()

        decoder_output = np.array(decoder_output).ravel()
        voter_output = voter_output[1]

        dec_out = decoder_output[:int(voter_output.item())]

        return np.hstack((np.array(org_hpi[:-1]), dec_out)) * denorm / dfact[index]


def Compute_SNR(masses, seq2seq, strain, psd, index, best_fit=False):
    masses = np.array(masses)
    template = get_template(masses, index, seq2seq)/10

    waveform = TimeSeries(template, dt=1/4096., dtype=np.float64)
    waveform = waveform.to_pycbc()
    waveform.resize(len(strain))
    waveform = waveform.cyclic_time_shift(waveform.start_time)
    snr = matched_filter(waveform, strain, psd=psd,
                         low_frequency_cutoff=f_lower)
    snr = snr.crop(4, 4)

    peak = abs(snr).numpy().argmax()
    psnr = snr[peak]

    if best_fit:
        return snr
    return np.abs(psnr)


def save_snr_timeseries(amass, snr, index):
    tsnr = snr.sample_times
    outputs = np.vstack((amass, np.vstack((tsnr, np.abs(snr))).T))
    snr_sn = os.path.join(outdn, 'bestfit_snr_%s' % index)
    np.savetxt(snr_sn, outputs)
    logging.info("SNR timeseries is saved at %s" % snr_sn)


def Pseudo_PE(seq2seq, strain, psd, tm1, tm2, index):
    eps = 0.01
    m1_range1 = np.arange(min_mass, tm1-mtol, far_minterval)
    m1_range2 = np.arange(tm1-mtol, tm1+mtol+eps, vic_minterval)
    m1_range3 = np.arange(tm1+mtol+far_minterval, max_mass+eps, far_minterval)
    m1_range = np.hstack((m1_range1, m1_range2, m1_range3))
    for i, m1 in enumerate(m1_range):
        if abs(tm1-m1) < mtol:
            m2s_1 = np.arange(min_mass, tm2-mtol, far_minterval)
            m2s_2 = np.arange(tm2-mtol, tm2+mtol+eps, vic_minterval)
            m2s_3 = np.arange(tm2+mtol+far_minterval, m1, far_minterval)
            m2s = np.hstack((m2s_1, m2s_2, m2s_3))
        else:
            m2s = np.arange(min_mass, m1+eps, far_minterval)
        m1s = np.zeros(len(m2s))+m1
        temp_mass_grid = np.vstack((m1s, m2s)).T
        if i == 0:
            mass_grid = temp_mass_grid
        else:
            mass_grid = np.vstack((mass_grid, temp_mass_grid))

    len_mgrid = len(mass_grid)
    expected_tcomp = len_mgrid/3600.
    logging.info("Expected computing time: %.3f hours" % expected_tcomp)

    psnrs = []
    lm = len(mass_grid)
    for i, masses in enumerate(mass_grid):
        psnr = Compute_SNR(masses, seq2seq, strain, psd, index)
        psnrs.append(psnr)

        if i % int(lm/10) == 0 and i != 0:
            percentage = i // int(lm/10) * 10
            logging.info("%s%% done" % percentage)

    psnrs = np.array(psnrs)
    out_PPE = np.hstack((mass_grid, psnrs.reshape(-1, 1)))

    answer = [tm1, tm2]
    best_fit = mass_grid[psnrs.argmax()]
    asnr = Compute_SNR(best_fit, seq2seq, strain, psd, index, best_fit=True)
    save_snr_timeseries(answer, asnr, index)

    pe_sn = os.path.join(outdn, "mass_SNR_%s" % index)
    np.savetxt(pe_sn, out_PPE)
    logging.info("Pesudo-PE results are saved at %s" % pe_sn)


def main():
    hdata = read_hdata()
    noise = hdata_rand_selection(hdata, dur=duration)
    for i in range(len(targ_m1)):
        signal = get_template(np.array([targ_m1[i], targ_m2[i]]), i, key='EOB')
        logging.info("target masses are %s %s" % (targ_m1[i], targ_m2[i]))
        strain = inject_signal(noise, signal)

        WelchTime = 4
        psd = strain.psd(WelchTime)
        psd = interpolate(psd, strain.delta_f)
        psd = inverse_spectrum_truncation(psd, WelchTime*strain.sample_rate,
                                          low_frequency_cutoff=15)

        global torch
        import torch

        torch.cuda.set_device(device)

        seq2seq = load_dl_model()

        Pseudo_PE(seq2seq, strain, psd, targ_m1[i], targ_m2[i], i)


if __name__ == '__main__':
    main()
