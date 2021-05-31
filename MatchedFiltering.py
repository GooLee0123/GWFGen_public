import argparse
import copy
import logging
import os
import time
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
from gwpy.timeseries import TimeSeries
from matplotlib import cm as CM
from pycbc import filter
from pycbc.filter import matched_filter, overlap, overlap_cplx, sigma, highpass
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.vetoes import power_chisq
from pycbc.waveform import get_td_waveform
from scipy.interpolate import griddata
from scipy.misc import logsumexp
from scipy.optimize import minimize
from scipy.signal import hilbert
from scipy.special import iv

from networks.evaluator import Predictor
from networks.optim import Optimizer
from networks.seq2seq import DecoderRNN, EncoderRNN, Seq2seq, VoterRNN
from wavepy.utils import Checkpoint
from wavepy.waves import DatPrep

from pycbc.catalog import Merger

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str, dest='data',
                    default='H-H1_LOSC_4_V1-1126293504-4096.hdf5')
parser.add_argument('-f', '--format', type=str, dest='format',
                    default='hdf5.losc')
parser.add_argument('-t', '--targets', type=str, dest='targets',
                    default='./data/IMR_bias/')
parser.add_argument('-w', '--waveforms', type=str, dest='waveforms',
                    default='./generated/GRU_4n256_TFR05_bias_alpha_5_Isize_1/')
parser.add_argument('-s', '--save', type=str, dest='save',
                    default='./MatchedFilteringResults')
parser.add_argument('-m', '--mcmc', type=str, dest='mcmc',
                    default='./MCMC_DL_test')
parser.add_argument('-g', '--gradient-descent', type=bool, dest='gd',
                    default=False, help="Gradient descent option. When given, MCMC is not performed.")

opts = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level='INFO')

## GLOBAL ##
save_every = 10
idx_tol = 10

temperature = 1. ##
gf_lower = 40. ##             15 for 40to100
targ_m1 = 35.6 ##
targ_m2 = 30.6 ##
min_mass = 10. ##
max_mass = 40. ##
ini_sigmaprop = 1. ##         standard deviation of the proposal density
max_sigmaprop = ini_sigmaprop*4.
min_sigmaprop = ini_sigmaprop/2.
dfact = 4.3  ##

duration = 32.
f_lower = 15. 
f_upper = 1024.
rseed = 35

date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
mcmc_save_dir = os.path.join(opts.mcmc, date_time)
if not os.path.exists(mcmc_save_dir):
    os.makedirs(mcmc_save_dir)

alpha = 5
data_set = 'test'
dtype = 'bias'
network = 'GRU_4n256_TFR05_'+dtype+'_alpha_'+str(alpha)
dpath = './data/IMR_'+dtype

whitening = False
slicing = False
plot_white = False

def read_hdata():
    hdata = TimeSeries.read(opts.data, format=opts.format)
    return hdata.highpass(15.0).to_pycbc().crop(2, 2)

# dimension of duration: s
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

    merg_idx = signal.argmax()

    inj_idx = len(temp_noise)//2 - merg_idx

    merging_time = (merg_idx+inj_idx)/4096.

    len_sig = len(signal)
    temp_noise[inj_idx:inj_idx+len_sig] += signal

    global len_sig
    global merging_time

    return temp_noise

#Propose a new step (using a symmetric Gaussian proposal density centered at the old sample)
def ProposedStep(oldsamp, sigmaprop, D):
    # np.random.seed(rseed)
    newsamp = oldsamp + np.random.normal(0., sigmaprop, D)
    return newsamp

def priorrange_update(m1_priorrange, m1_old):
    priorrange = np.array([[m1_priorrange[0], m1_priorrange[1]],
                    [m1_priorrange[0], m1_old]])
    return priorrange

def get_ISCO_index(signal, sampling_rate, total_mass, min_freq, t):
    analytic_signal = hilbert(signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase)/(2.0*np.pi)*sampling_rate)
    FISCO = 4400./total_mass
    ISCOS = np.argwhere(instantaneous_frequency >= FISCO)
    IISCO = ISCOS[ISCOS>120][0]+1
  
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

    checkpoint, ovlp = Checkpoint.load(G,
                                Goptimizer,
                                latest_checkpoint_path,
                                optdict)
    G = checkpoint.G
    G.eval()

    return G

def get_template(newsamp, seq2seq=None, key='DL'):
    stime = time.time()
    hp, _ = get_td_waveform(approximant='SEOBNRv4_opt',
                            mass1=newsamp[0],
                            mass2=newsamp[1],
                            distance=100,
                            inclination=1.043,
                            delta_t=1/4096.,
                            f_lower=gf_lower)
    etime = time.time()
    dur = etime-stime
    logging.info("EOB took %ss" % dur)
    hp_ISCO = get_ISCO_index(hp, 4096., sum(newsamp), 15.,
                         hp.sample_times.numpy())

    denorm = np.genfromtxt(os.path.join(dpath, 'maximum_amplitude'))[0]
    hp = hp/denorm

    org_hpi = hp[:hp_ISCO]
    hpt = hp[hp_ISCO-1:]
    hpt = hpt[np.nonzero(hpt)]

    def _roundup(x, th):
        return int(math.ceil(x/float(th))*th)

    if key == 'EOB':
        # return np.array(hpt) * denorm / dfact
        return np.array(hp) * denorm / dfact
        # return hp/dfact
    elif key == 'DL':
        hpi = np.array(org_hpi, dtype=np.float32)
        len_hpi = len(hpi)
        plen = _roundup(len_hpi, isize)
        hpi = np.pad(hpi, (0, plen-len_hpi), 'constant', constant_values=(0.))

        hpi = torch.from_numpy(hpi)
        hpi = torch.nn.utils.rnn.pad_sequence([hpi], batch_first=True)
        hpi = hpi.view(hpi.size(0), -1, isize)
        hpi = torch.nn.utils.rnn.pack_padded_sequence(hpi, [int(len_hpi/isize)+1], batch_first=True)

        hpt = torch.tensor(hpt, dtype=torch.float32).view(-1, len(hpt), 1)

        hpi = hpi.cuda()
        hpt = hpt.cuda()

        stime = time.time()
        with torch.no_grad():
            decoder_output, voter_output = seq2seq(hpi, hpt, teacher_forcing_ratio=0)
        etime = time.time()
        dur = etime-stime
        logging.info("DL took %ss" % dur)

        hpi = hpi.cpu()
        hpt = hpt.cpu()
        decoder_output = decoder_output.cpu()

        decoder_output = np.array(decoder_output).ravel()
        voter_output = voter_output[1]

        dec_out = decoder_output[:int(voter_output.item())]

        # merg_idx = dec_out.argmax()
        merg_idx = np.array(hp).argmax()

        return np.hstack((np.array(org_hpi[:-1]), dec_out)) * denorm / dfact, merg_idx+len(org_hpi)-1
        # return hp * denorm / dfact, merg_idx
        # return hp/dfact, np.array(hp).argmax()
        # return dec_out*denorm/dfact, merg_idx

def plot_whitened_data(time, strain, aligned, psd, key=False, whiten=False, peak_time=32.):
    stime = time+strain.start_time
    if whiten:
        white_data = (strain.to_frequencyseries()/psd**0.5).to_timeseries()

        tapered = aligned.highpass_fir(30, 512, remove_corrupted=False)
        # white_template = tapered
        white_template = (tapered.to_frequencyseries() / psd**0.5).to_timeseries()

        white_data = white_data.highpass_fir(30, 512).lowpass_fir(300, 512)
        white_template = white_template.highpass_fir(30, 512).lowpass_fir(300, 512)
    else:
        white_data = strain
        white_template = aligned
    # print time
    plt.figure()
    plt.cla()
    plt.subplot(211)
    plt.plot(np.linspace(0, 2048., len(white_data)//2+1), white_data.to_frequencyseries(), label="Data")
    plt.plot(np.linspace(0, 2048., len(white_template)//2+1), white_template.to_frequencyseries(), label="Template")
    plt.subplot(212)
    plt.plot(white_data, label="Data")
    plt.plot(white_template, label="Template")
    # plt.plot(aligned.crop(0.25, 0.25)*1e23, label="Aligned")
    plt.xlim((peak_time-.5)*4096, (peak_time+.1)*4096)
    plt.vlines(peak_time*4096, -100, 100)
    plt.vlines((merging_time-float(strain.start_time))*4096, -100, 100, 'r')
    # print peak_time
    if not os.path.exists(opts.save):
        os.makedirs(opts.save)
    if key:
        print "save at aligned_template_and_strain1"
        plt.savefig(os.path.join(opts.save, "aligned_template_and_strain1"))
    else:
        print "save at aligned_template_and_strain"
        plt.savefig(os.path.join(opts.save, "aligned_template_and_strain"))

def plot_subtracted_f_plot(time, noise, strain, fit):
    subtracted = strain - fit
    time = float(strain.start_time + time)
    i = 0
    for data, title in [(noise, 'Before injection'),
    					(strain, 'After injection'),
                        (subtracted, 'After subtraction')]:
        t, f, p = data.whiten(4, 4).qtransform(.001,
                                                logfsteps=100,
                                                qrange=(8, 8),
                                                frange=(20, 512))

        plt.figure(figsize=[15, 7])
        plt.title(title)
        plt.pcolormesh(t, f, p**0.5, vmin=1, vmax=6)
        plt.yscale('log')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        # plt.xlim(time - 2, time + 1)

        plt.savefig(os.path.join(opts.save, "subtracted_f_plot_%s"%i))
        i += 1

def align_template(strain, template, peak_time, psd, snrp):
    aligned = template.cyclic_time_shift(peak_time)
    aligned /= sigma(aligned, psd=psd, low_frequency_cutoff=20.0)
    aligned.start_time = strain.start_time
    aligned = (aligned.to_frequencyseries() * snrp).to_timeseries()
    return aligned

def whiten_timeseries(strain, template, psd):
    # white_data = strain
    white_data = (strain.to_frequencyseries()/psd**0.5).to_timeseries()

    tapered = template.highpass_fir(30, 512, remove_corrupted=False)
    # white_template = tapered
    white_template = (tapered.to_frequencyseries() / psd**0.5).to_timeseries()

    white_data = white_data.highpass_fir(30, 512).lowpass_fir(300, 512)
    white_template = white_template.highpass_fir(30, 512).lowpass_fir(300, 512)

    return white_data, white_template

def loglikelihood(h, d, strain, psd, option=0):
    hf = h.to_frequencyseries()
    df = d.to_frequencyseries()
    if option:
        N = int(df.duration/df.delta_t)
        kmin, kmax = filter.get_cutoff_indices(f_lower, f_upper, df.delta_f, (N - 1) * 2)

        nf = df-hf

        nn = sum((nf[kmin:kmax].conj()*nf[kmin:kmax]).real()/psd[kmin:kmax]) * 4.0 * nf.delta_f

        logL = -0.5*nn/temperature
    else:
        N = int(df.duration/df.delta_t)

        kmin, kmax = filter.get_cutoff_indices(f_lower, f_upper, df.delta_f, (N - 1) * 2)

        hh = sum(4.0 * df.delta_f * (hf[kmin:kmax].conj()*hf[kmin:kmax]).real()/psd[kmin:kmax])
        dh = sum(4.0 * df.delta_f * (df[kmin:kmax].conj()*hf[kmin:kmax]).real()/psd[kmin:kmax])
        logL = -0.5*(hh - 2.0*dh)

    return logL

def slice_data(peak_time, strain, template):
    slice_time = peak_time+strain.start_time

    sliced_strain = strain.time_slice(slice_time, slice_time+len_sig/4096.)
    sliced_template = template.time_slice(slice_time, slice_time+len_sig/4096.)

    return sliced_strain, sliced_template

def preprocessing(newsamp, seq2seq, strain, psd, peak_time):
    new_template, merg_idx = get_template(newsamp, seq2seq)

    waveform = TimeSeries(new_template, dt=1/4096., dtype=np.float64)
    waveform = waveform.to_pycbc()
    waveform.resize(len(strain))
    waveform = waveform.cyclic_time_shift(waveform.start_time)

    snr = matched_filter(waveform, strain, psd=psd, low_frequency_cutoff=20)
    lensnr1 = len(snr)
    snr = snr.crop(4, 4)
    lensnr2 = len(snr)
    len_diff = lensnr1-lensnr2
    idx_rcv = len_diff/2

    peak = abs(snr).numpy().argmax()
    psnr = snr[peak]

    peak_time = snr.sample_times[peak]-float(strain.start_time)

    # align strain and template
    aligned = align_template(strain, waveform, peak_time, psd, psnr)

    # whiten and slice data
    if whitening and slicing:
        peak_time -= 0.25
        wstrain, waligned = whiten_timeseries(strain, aligned, psd)
        wstrain, waligned = slice_data(peak_time, wstrain, waligned)
        # strain = wstrain
    elif whitening:
        wstrain, waligned = whiten_timeseries(strain, aligned, psd)
    elif slicing:
        wstrain, waligned = slice_data(peak_time, strain, aligned)
        # strain = wstrain
    else:
        wstrain, waligned = strain, aligned

    return peak_time, strain, wstrain, waligned, aligned

def HastingsRatio(newsamp, oldsamp, priorrange, PDF, psd, strain, seq2seq, ologl, peak_time):
    if not ((np.array([p1 - p2 for p1, p2 in zip(newsamp, np.transpose(priorrange)[:][0])])>=0).all() \
            and (np.array([p2 - p1 for p1, p2 in zip(newsamp,np.transpose(priorrange)[:][1])])>=0).all()):
        acc = False
        return acc, oldsamp, ologl # make sure the samples are in the desired range

    pdf = [ologl]

    peak_time, strain, post_strain, post_template, aligned = preprocessing(newsamp, seq2seq, strain, psd, peak_time)

    # whiten and plot data
    if plot_white:
        plot_whitened_data(peak_time, strain, aligned, psd, whiten=True, peak_time=peak_time)

    # append new pdf
    pdf.append(PDF(post_template, post_strain, strain, psd, option=0))

    logging.info("new pdf: %.5f, old pdf: %.5f" % (pdf[1], pdf[0]))

    if pdf[1] >= pdf[0]:
        logging.info("accepted")

        acc = True

        return acc, newsamp, pdf[1]
    else:
        prob = np.exp(pdf[1]-pdf[0])

        logging.info("rejected")
        logging.info("accept probability: %s" % prob)

        acc = np.random.choice([True,False],p=[prob, 1.-prob])
        if acc == True:
            logging.info("but accepted")
            pdf = pdf[1]
        else:
            pdf = pdf[0]

        return acc, acc*newsamp + (1. - acc)*oldsamp, pdf

def GD_likelihood(masses, seq2seq, strain, psd, saving=False):
    masses = np.array(masses)
    template, merg_idx = get_template(masses, seq2seq)

    waveform = TimeSeries(template, dt=1/4096., dtype=np.float64)
    waveform = waveform.to_pycbc()
    waveform.resize(len(strain))
    waveform = waveform.cyclic_time_shift(waveform.start_time)
    snr = matched_filter(waveform, strain, psd=psd, low_frequency_cutoff=20)
    lensnr1 = len(snr)
    snr = snr.crop(4, 4)
    lensnr2 = len(snr)
    len_diff = lensnr1-lensnr2
    idx_rcv = len_diff/2

    peak = abs(snr).numpy().argmax()
    peak_time = snr.sample_times[peak]-float(strain.start_time)
    psnr = snr[peak]

    if saving:
        outputs = np.vstack((snr.sample_times, abs(snr))).T
        np.savetxt('snr', outputs)
        nbins = 26
        chisq = power_chisq(waveform, strain, nbins, psd, low_frequency_cutoff=20)
        chisq = chisq.crop(4, 4)

        # peak_time = 8.

        fpeaksnr = os.path.join(mcmc_save_dir, "SNR_peak")
        with open(fpeaksnr, 'w') as f:
            f.write(str(abs(snr).numpy()[peak].item()))

        plt.figure()
        #plt.subplot(211)
        plt.plot(snr.sample_times, abs(snr))
        plt.ylabel("Signal-to-noise ratio")
        plt.xlabel("Time (s)")
        # plt.xlim(float(peak_time+snr.start_time-4.15), float(peak_time+snr.start_time-3.85))
        #plt.subplot(212)
        #plt.plot(chisq.sample_times, chisq)
        #plt.xlim(float(peak_time+chisq.start_time-4.15), float(peak_time+chisq.start_time-3.85))
        #plt.ylabel("$chi^2_r$")
        #plt.xlabel("Time (s)")
        plt.savefig(os.path.join(mcmc_save_dir, "SNR_Chisq"))

    # align
    aligned = align_template(strain, waveform, peak_time, psd, psnr)

    # whiten and slice data
    if whitening and slicing:
        peak_time -= 0.25
        wstrain, waligned = whiten_timeseries(strain, aligned, psd)
        wstrain, waligned = slice_data(peak_time, wstrain, waligned)
    elif whitening:
        wstrain, waligned = whiten_timeseries(strain, aligned, psd)
    elif slicing:
        wstrain, waligned = slice_data(peak_time, strain, aligned)
    else:
        wstrain, waligned = strain, aligned

    if saving and plot_white:
        plot_whitened_data(peak_time, strain, aligned, psd, key=True, whiten=True, peak_time=peak_time)

    # print peak_time
    logL = loglikelihood(waligned, wstrain, strain, psd, option=0)

    return logL, peak_time

def MCMC(seq2seq, PDF, D, Nsteps, m1_priorrange, sigmaprop,
        strain, psd, mcmc_save_dir):
    # Answer
    answer = np.array([targ_m1, targ_m2])
    ans_logl, peak_time = GD_likelihood(answer, seq2seq, strain, psd, saving=True)
    quit()
    logging.info("Answer log likelihood: %s" % ans_logl)

    # Draw a random starting point
    mbound = 1
    m1_old = np.random.uniform(max(targ_m1-mbound, min_mass), min(targ_m1+mbound, max_mass))
    m2_old = np.random.uniform(max(targ_m2-mbound, min_mass), min(targ_m2+mbound, m1_old))
    oldsamp = np.array([m1_old, m2_old])
    priorrange = priorrange_update(m1_priorrange, m1_old)

    logl, _ = GD_likelihood(oldsamp, seq2seq, strain, psd)

    samples = [oldsamp]
    temp_samples = [np.hstack((oldsamp, logl))]

    tmass = [targ_m1, targ_m2]

    count = 0
    for i in xrange(Nsteps):
        newsamp = ProposedStep(oldsamp, sigmaprop, D)

        priorrange = priorrange_update(m1_priorrange, newsamp[0])

        acc, newsamp, logl = HastingsRatio(newsamp,
                                           oldsamp, 
                                           priorrange,
                                           PDF,
                                           psd,
                                           strain, 
                                           seq2seq, 
                                           logl,
                                           peak_time)

        if acc: count += 1
        samples.append(newsamp)
        temp_samples.append(np.hstack((newsamp, logl)))
        oldsamp = newsamp
        _AR = 1.*count/(i+1.)

        if _AR < 0.15 and sigmaprop > min_sigmaprop:
            sigmaprop -= 0.25
        elif _AR > 0.15 and sigmaprop < max_sigmaprop:
            sigmaprop += 0.25

        logging.info("%sth MCMC step is done."% i)
        logging.info("Sigma: %.2f, AR: %.2f, Acc: %s, New Samples: %.2f, %.2f"\
                    % (sigmaprop, _AR, acc, newsamp[0], newsamp[1]))
        logging.info("Targets: %.2f, %.2f" % (targ_m1, targ_m2))

        if i % save_every == 0:
            with open(os.path.join(mcmc_save_dir, "MCMC_samples"), 'a') as f:
                np.savetxt(f, np.array(temp_samples) )
            temp_samples = []
 
    AR = 1.*count/Nsteps
    return np.array(samples), AR

def MCMC_workflow(seq2seq, strain, psd):
    Nsteps = 300000 # length of the chain
    D = 2 # Dimension of the problem (m1, m2)
    m1_priorrange = np.array([min_mass, max_mass]) # Prior range of masses of primary black hole

    np.savetxt(os.path.join(mcmc_save_dir, "MCMC_target_mass"), np.array([targ_m1, targ_m2]))

    samples, AR = MCMC(seq2seq, loglikelihood, D, Nsteps, m1_priorrange, 
                        ini_sigmaprop, strain, psd, mcmc_save_dir)
    logging.info("Accpetance ratio: {}".format(AR))
    
    plt.figure()
    plt.subplot(211)
    plt.plot(samples[:, 0])
    plt.xlabel("iteration")
    plt.ylabel("m1")
    plt.subplot(212)
    plt.plot(samples[:, 1])
    plt.xlabel("iteration")
    plt.ylabel("m2")
    plt.savefig(os.path.join(mcmc_save_dir, "MCMC_iteration"))

def workflow(hdata):
    noise = hdata_rand_selection(hdata, dur=duration)
    signal = get_template(np.array([targ_m1, targ_m2]), key='EOB')
    logging.info("target masses are %s %s" % (targ_m1, targ_m2))
    strain = inject_signal(noise, signal)

    # merger = Merger('GW150914')
    # strain = merger.strain('H1')
    # strain = highpass(strain, 15.0).crop(2, 2)
    # merging_time = merger.time

    # global merging_time

    WelchTime = 4
    psd = strain.psd(WelchTime)
    psd = interpolate(psd, strain.delta_f)
    psd = inverse_spectrum_truncation(psd, WelchTime*strain.sample_rate, low_frequency_cutoff=15)
    fs = psd.delta_f * np.arange(psd.data.size)
    plt.loglog(fs, psd)
    plt.xlim(f_lower, f_upper)
    plt.savefig('psd')

    global torch
    import torch
    seq2seq = load_dl_model()
    # seq2seq = None

    MCMC_workflow(seq2seq, strain, psd)

def main():
    hdata = read_hdata()
    workflow(hdata)

if __name__ == '__main__':
    main()
