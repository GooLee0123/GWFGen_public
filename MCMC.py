import argparse
import logging
import time
import os

import torch
import theano
import matplotlib
matplotlib.use('Agg')

import pymc3 as pm
import numpy as np
import pandas as pd
import theano.tensor as tt
import matplotlib.pyplot as plt
import pymc3.distributions.transforms as tr

from matplotlib import cm as CM
from theano.compile.ops import as_op
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.optimize import minimize
from scipy.misc import logsumexp
from scipy.signal import hilbert

import pylab

from pycbc import filter
from pycbc.vetoes import power_chisq
from pycbc.waveform import get_td_waveform
from pycbc.filter import sigma, matched_filter
from pycbc.psd import interpolate, inverse_spectrum_truncation

from gwpy.timeseries import TimeSeries

from wavepy.utils import Checkpoint
from wavepy.waves import DatPrep

from networks.optim import Optimizer
from networks.seq2seq import EncoderRNN, DecoderRNN, VoterRNN, Seq2seq
from networks.evaluator import Predictor

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str, dest='data',
                    default='H-H1_LOSC_4_V1-1126432768-4096.hdf5')
parser.add_argument('-f', '--format', type=str, dest='format',
                    default='hdf5.losc')
parser.add_argument('-t', '--targets', type=str, dest='targets',
                    default='./data/IMR_PhenomV2/')
parser.add_argument('-w', '--waveforms', type=str, dest='waveforms',
                    default='./generated/GRU_4n256_TFR05_PhenomV2_alpha_5/')
parser.add_argument('-s', '--save', type=str, dest='save',
                    default='./MatchedFilteringResults')
parser.add_argument('-m', '--mcmc', type=str, dest='mcmc',
                    default='./MCMC')
parser.add_argument('-g', '--gradient-descent', type=bool, dest='gd',
                    default=False, help="Gradient descent option. When given, MCMC is not performed.")

opts = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level='INFO')

## GLOBAL ##
save_every = 10
idx_tol = 10

max_sigma = 4.
min_sigma = 0.1
f_lower = 10.
f_upper = 1024.
targ_m1 = 60.
targ_m2 = 50.
lm1 = 57.
um1 = 58.
lm2 = 50.
um2 = 51.
ltotmass_threshold = 100.
utotmass_threshold = 110.
min_mass = 40.
max_mass = 100.
rseed = 30.
dfact = 7.

date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
mcmc_save_dir = os.path.join(opts.mcmc, date_time)
if not os.path.exists(mcmc_save_dir):
    os.makedirs(mcmc_save_dir)

alpha = 5
data_set = 'test'
dtype = 'bias_PhenomV2'
network = 'GRU_4n256_TFR05_'+dtype+'_alpha_'+str(alpha)
dpath = './data/IMR_'+dtype

whitening = False
slicing = False
rsignal = False
plot_white = False
cud = torch.cuda.is_available()

if cud:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def read_hdata():
    hdata = TimeSeries.read(opts.data, format=opts.format)
    return hdata.highpass(f_lower)

def read_target(dset):
    fname = 'IMR_hp_'+dset+'_target.dat'
    datprep = DatPrep(opts.targets)
    targets = datprep.load(os.path.join(opts.targets, fname))
    targets = datprep.tolist()
    targets = datprep.tofloat()
    targets = datprep.nonzero()

    return targets

def read_waveforms():
    fname = 'test_data_prediction'
    datprep = DatPrep(opts.targets)
    waveforms = datprep.load(os.path.join(opts.waveforms, fname))
    waveforms = datprep.tolist()
    waveforms = datprep.tofloat()

    fdenorm = 'maximum_amplitude'
    denorm_path = os.path.join(os.path.join(opts.targets, fdenorm))
    denorm_term = np.genfromtxt(denorm_path)[0]

    for i in xrange(len(waveforms)):
        waveforms[i] = (pd.Series(waveforms[i])*denorm_term).tolist()

    return waveforms

def read_mass(dset):
    fname = os.path.join(opts.targets, 'mass_info_'+dset+'.dat')
    masses = np.genfromtxt(fname).T

    return masses

# dimension of duration: s
def hdata_rand_selection(hdata, duration):
    dt = hdata.dt.value
    sample_rate = 1/dt
    idx_interval = int(duration*sample_rate)
    ulim = len(hdata) - idx_interval
    # np.random.seed(rseed)
    start_idx = np.random.randint(ulim)

    return hdata[start_idx:start_idx+idx_interval]

def random_signal(targets):
    tmass = read_mass(data_set)
    tm1s, tm2s = tmass[0], tmass[1]
    # np.random.seed(rseed)
    while True:
        rand_idx = np.random.randint(len(targets))
        tm1, tm2 = tm1s[rand_idx], tm2s[rand_idx]
        if tm1+tm2 <= utotmass_threshold and \
            tm1+tm2 >= ltotmass_threshold and \
            tm1 <= um1 and lm1 <= tm1 and \
            tm2 <= um2 and lm2 <= tm2:
            break
    return np.array(targets[rand_idx])/dfact, tm1, tm2

def inject_signal(noise, signal):
    idx_ulim = len(noise)//2-len(signal)
    inj_idx = len(noise)//2

    len_sig = len(signal)
    noise[inj_idx:inj_idx+len_sig] += signal

    merg_idx = signal.argmax()
    merging_time = (merg_idx+inj_idx)/4096.

    global len_sig
    global merging_time

    return noise.to_pycbc()

def get_ISCO_index(signal, sampling_rate, total_mass, min_freq, t):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase)/(2.0*np.pi)*sampling_rate)
    FISCO = 4400./total_mass
    ISCOS = np.argwhere(instantaneous_frequency >= FISCO)
    IISCO = ISCOS[ISCOS>min_freq*8][0]+1
  
    Merger_index = np.argwhere(t > 0.0)[0][0]

    if Merger_index < IISCO:
        logging.info("IISCO Being smaller than ISCO time index return merger index")
    return min(IISCO, Merger_index)

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
                    rnn_cell=optdict['rnn_cell'].lower())
    G = Seq2seq(encoder, decoder, voter)
    if cud:
        G.cuda()

    Goptimizer = Optimizer(
                torch.optim.Adam(
                    G.parameters(), 
                    lr=optdict['gen_base_lr'],
                    betas=(optdict['gen_beta1'], 0.999), 
                    weight_decay=5e-5), 
                max_grad_norm=5)

    logging.info("load checkpoint from {}"\
                    .format(latest_checkpoint_path))

    checkpoint = Checkpoint.load(G,
                                Goptimizer,
                                latest_checkpoint_path,
                                optdict)
    G = checkpoint.G

    return G

def get_template(newsamp, seq2seq=None, key='DL'):
    hp, _ = get_td_waveform(approximant='SEOBNRv4_opt',
                            mass1=newsamp[0],
                            mass2=newsamp[1],
                            distance=100,
                            inclination=1.043,
                            delta_t=1/4096.,
                            f_lower=15.)
    hp_ISCO = get_ISCO_index(hp, 4096., sum(newsamp), 15.,
                         hp.sample_times.numpy())

    denorm = np.genfromtxt(os.path.join(dpath, 'maximum_amplitude'))[0]
    hp = hp/denorm
    
    org_hpi = hp[:hp_ISCO]
    hpt = hp[hp_ISCO-1:]
    hpt = hpt[np.nonzero(hpt)]
    temp_hpt = hpt

    if key == 'EOB':
        # return np.array(hp) * denorm / dfact
        return np.array(hpt) * denorm / dfact
    elif key == 'DL':
        len_hpi = len(org_hpi)
        hpi = torch.tensor(org_hpi, dtype=torch.float32)
        hpi = torch.nn.utils.rnn.pad_sequence([hpi], batch_first=True)
        hpi = hpi.unsqueeze(-1)
        hpi = torch.nn.utils.rnn.pack_padded_sequence(hpi, [len_hpi], batch_first=True)

        hpt = torch.tensor(hpt, dtype=torch.float32).view(-1, len(hpt), 1)

        if cud:
            hpi.cuda()
            hpt.cuda()

        with torch.no_grad():
            decoder_output, voter_output = seq2seq(hpi, hpt, teacher_forcing_ratio=0)

        decoder_output = np.array(decoder_output).ravel()
        voter_output = voter_output[1]

        dec_out = decoder_output[:int(voter_output.item())]

        peak_idx = dec_out.argmax()
        merg_idx = np.array(hpt).argmax()
        if abs(merg_idx-peak_idx) < idx_tol:
            merg_idx = peak_idx
            # logging.info("merging time and peak time is comparable in tolerable range")
        # return np.hstack((np.array(org_hpi[:-1]), dec_out)) * denorm / dfact, merg_idx+len(org_hpi)-1
        return dec_out * denorm / dfact, merg_idx

def align_template(strain, template, peak_time):
    aligned = template.cyclic_time_shift(merging_time)
    aligned.start_time = strain.start_time

    return aligned

def whiten_timeseries(strain, psd, template=None, key=True):
    # white_data = strain
    white_data = (strain.to_frequencyseries()/psd**0.5).to_timeseries()
    white_data = white_data.highpass_fir(30, 512).lowpass_fir(300, 512)

    if key:
        tapered = template.highpass_fir(30, 512, remove_corrupted=False)
        # white_template = tapered
        white_template = (tapered.to_frequencyseries() / psd**0.5).to_timeseries()
        white_template = white_template.highpass_fir(30, 512).lowpass_fir(300, 512)

        return white_data, white_template
    return white_data

def slice_data(peak_time, strain, template=None, key=True):
    slice_time = peak_time+strain.start_time

    sliced_strain = strain.time_slice(slice_time, slice_time+len_sig/4096.)
    if key:
        sliced_template = template.time_slice(slice_time, slice_time+len_sig/4096.)
        return sliced_strain, sliced_template
    return sliced_strain

hdata = read_hdata()
targets = read_target(data_set)

noise = hdata_rand_selection(hdata, 16)
signal = get_template(np.array([targ_m1, targ_m2]), key='EOB')
tm1, tm2 = targ_m1, targ_m2

logging.info("target masses are %s %s" %(tm1, tm2))

strain = inject_signal(noise, signal)

WelchTime = 4
psd = strain.psd(WelchTime)
psd = interpolate(psd, strain.delta_f)
psd = inverse_spectrum_truncation(psd, WelchTime*strain.sample_rate,
                                low_frequency_cutoff=15)

# whitened_strain = whiten_timeseries(strain, psd, key=False)
# whitened_strain = slice_data(8., whitened_strain, key=False)

sigma = np.std(strain)

pymc_seq2seq = load_dl_model()

def waveform_processing(theta, strain=strain, seq2seq=pymc_seq2seq):
    m1, m2 = theta[1], theta[0]
    sample = np.array([m1, m2])
    template, merg_idx = get_template(sample, seq2seq)
    waveform = TimeSeries(template, dt=1/4096., dtype=np.float64)
    waveform = waveform.to_pycbc()
    waveform.resize(len(strain))
    waveform = waveform.cyclic_time_shift(-merg_idx/4096.)

    peak_time = 8.
    aligned = align_template(strain, waveform, peak_time)

    # peak_time -= .25
    # _strain, aligned = whiten_timeseries(strain, psd, template=aligned)
    # _strain, aligned = slice_data(peak_time, wstrain, template=waligned)

    return aligned.numpy()

@as_op(itypes=[tt.dvector], otypes=[tt.dvector])
def gw_model(theta):
    print theta[0], theta[1]
    return waveform_processing(theta)

model = pm.Model()
savedir = os.path.join("./MCMC_traces", date_time)

with model:
    masses = pm.Uniform('masses', lower=40, upper=100, shape=2, transform=tr.Ordered(),
                     testval=[targ_m2, targ_m1])

    aligned = pm.Deterministic('Wave', gw_model(masses))

    like = pm.Normal('like', mu=aligned, sd=sigma, observed=strain.numpy())

    # logging.info("Launch MAP")
    # start = pm.find_MAP()
    logging.info("Launch MCMC")
    # trace = pm.sample(10, start=start, progressbar=True, njobs=1)
    step = pm.Metropolis()
    trace = pm.sample(10000, step=step, progressbar=False,
            njobs=1, tune=1000, thin=1, chains=1, trace=[masses])

    pm.save_trace(trace, directory=savedir)

    pm.traceplot(trace)
    plt.savefig(os.path.join(savedir, "traceplot"))
