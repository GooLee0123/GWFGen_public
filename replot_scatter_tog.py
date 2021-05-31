import logging
import os

import numpy as np
import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib.ticker import NullFormatter
from wavepy.waves import DatPrep

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)

alpha = 5
dtype = 'bias'
dtype2 = 'bias_40to100'

network = 'GRU_4n128_TFR05_'+dtype+'_alpha_'+str(alpha)
network2 = 'GRU_4n128_TFR05_'+dtype2+'_alpha_'+str(alpha)

dpath = './data/IMR_'+dtype
dpath2 = './data/IMR_'+dtype2

finput = dpath+'/IMR_hp_test_input.dat'
ftarget = dpath+'/IMR_hp_test_target.dat'
fgenetd = './generated/'+network+'/test_data_prediction'

finput2 = dpath2+'/IMR_hp_test_input.dat'
ftarget2 = dpath2+'/IMR_hp_test_target.dat'
fgenetd2 = './generated/'+network2+'/test_data_prediction'

idatprep = DatPrep(dpath)
inputs = idatprep.load(finput)
inputs = idatprep.tolist()
inputs = idatprep.tofloat()
inputs = idatprep.normalize()

tdatprep = DatPrep(dpath)
target = tdatprep.load(ftarget)
target = tdatprep.tolist()
target = tdatprep.tofloat()
target = tdatprep.nonzero()
target = tdatprep.normalize()
targetL = tdatprep.get_input_len()

gdatprep = DatPrep(dpath)
generated = gdatprep.load(fgenetd)
generated = gdatprep.tolist()
generated = gdatprep.tofloat(blank=False)

idatprep2 = DatPrep(dpath2)
inputs2 = idatprep2.load(finput2)
inputs2 = idatprep2.tolist()
inputs2 = idatprep2.tofloat()
inputs2 = idatprep2.normalize()

tdatprep2 = DatPrep(dpath2)
target2 = tdatprep2.load(ftarget2)
target2 = tdatprep2.tolist()
target2 = tdatprep2.tofloat()
target2 = tdatprep2.nonzero()
target2 = tdatprep2.normalize()
targetL2 = tdatprep2.get_input_len()

gdatprep2 = DatPrep(dpath2)
generated2 = gdatprep2.load(fgenetd2)
generated2 = gdatprep2.tolist()
generated2 = gdatprep2.tofloat(blank=False)

sampling_rate = 4096.

redir = './Result_Plots_'+network
overlap = np.zeros(len(target))
overlap2 = np.zeros(len(target2))
tlength = np.zeros(len(target))
tlength2 = np.zeros(len(target2))
if not os.path.exists(redir):
    os.makedirs(redir)

duplicate = 1
extra = 10

best_num = 3
wrst_num = 1

for i, waves in enumerate(zip(target, generated)):
    iwaves = np.array(inputs[i])
    twaves = np.array(waves[0])
    gwaves = np.array(waves[1])
    
    ilen = 101
    tlen = len(twaves)
    glen = len(gwaves)
    mlen = min(tlen, glen)

    inp_x = np.arange(ilen)
    tgt_x = np.arange(ilen-duplicate, ilen+tlen-duplicate)
    gnt_x = np.arange(ilen-duplicate, ilen+glen-duplicate)

    idup = iwaves[-(duplicate+extra):]
    gdup = gwaves[:(duplicate+extra)]
    min_dup = min(min(idup), min(gdup))
    max_dup = max(max(idup), max(gdup))

    tw = twaves[:mlen]
    gw = gwaves[:mlen]
    df = sampling_rate / mlen

    t_ff = np.fft.fft(tw)[:mlen//2]
    g_ff = np.fft.fft(gw)[:mlen//2]

    t_norm_term = np.sqrt(np.trapz(t_ff*np.conjugate(t_ff), dx=df))
    g_norm_term = np.sqrt(np.trapz(g_ff*np.conjugate(g_ff), dx=df))
    norm_term = t_norm_term*g_norm_term

    temp_overlap = abs(np.trapz(g_ff*np.conjugate(t_ff), dx=df)/norm_term)

    tlength[i] = tlen
    overlap[i] = temp_overlap

for i, waves in enumerate(zip(target2, generated2)):
    iwaves = np.array(inputs2[i])
    twaves = np.array(waves[0])
    gwaves = np.array(waves[1])
    
    ilen = 101
    tlen = len(twaves)
    glen = len(gwaves)
    mlen = min(tlen, glen)

    inp_x = np.arange(ilen)
    tgt_x = np.arange(ilen-duplicate, ilen+tlen-duplicate)
    gnt_x = np.arange(ilen-duplicate, ilen+glen-duplicate)

    idup = iwaves[-(duplicate+extra):]
    gdup = gwaves[:(duplicate+extra)]
    min_dup = min(min(idup), min(gdup))
    max_dup = max(max(idup), max(gdup))

    tw = twaves[:mlen]
    gw = gwaves[:mlen]
    df = sampling_rate / mlen

    t_ff = np.fft.fft(tw)[:mlen//2]
    g_ff = np.fft.fft(gw)[:mlen//2]

    t_norm_term = np.sqrt(np.trapz(t_ff*np.conjugate(t_ff), dx=df))
    g_norm_term = np.sqrt(np.trapz(g_ff*np.conjugate(g_ff), dx=df))
    norm_term = t_norm_term*g_norm_term

    temp_overlap = abs(np.trapz(g_ff*np.conjugate(t_ff), dx=df)/norm_term)

    tlength2[i] = tlen
    overlap2[i] = temp_overlap

nullfmt = NullFormatter()

left, width = 0.130, 0.630
bottom, height = 0.175, 0.675
bottom_h = left_h = left + width + 0.01

rect_scatter = [left, bottom, width, height]
rect_histy = [left_h, bottom, 0.2, height]

plt.figure(1, figsize=(8.5, 4))
axScatter = plt.axes(rect_scatter)
axHisty = plt.axes(rect_histy)

axScatter.scatter(tlength, overlap, marker='+', color='r', alpha=0.3, label=r"$[10M_{\odot}, 40M_{\odot}]$")
axScatter.scatter(tlength2, overlap2, marker='x', color='b', alpha=0.3, label=r"$[40M_{\odot}, 100M_{\odot}]$")
axScatter.hlines(0.99, min(tlength)-20, max(tlength2)+20, linestyles='dashed')
axScatter.set_xlabel(r"$L_{t}$", fontsize=20)
axScatter.set_ylabel("Overlap", fontsize=20)
axScatter.tick_params(labelsize=15)
axScatter.set_xlim(min(tlength)-20, max(tlength2)+20)
axScatter.set_ylim(0.98, 1.0)
axScatter.legend(loc='lower left')

ol_bin = np.linspace(0.99, 1, 41)
axHisty.yaxis.set_major_formatter(nullfmt)
axHisty.hist(overlap, ol_bin, orientation='horizontal',
    color='r', alpha=1, histtype='step', ls='dashed',
    label=r"$[10M_{\odot}, 40M_{\odot}]$")
axHisty.hist(overlap2, ol_bin, orientation='horizontal',
    color='b', alpha=1, histtype='step', 
    label=r"$[40M_{\odot}, 100M_{\odot}]$")
axHisty.hlines(0.99, 0.1, 1e4, linestyles='dashed')
axHisty.set_xscale('log')
axHisty.tick_params(labelsize=15)
axHisty.set_xlabel("Occurrence", fontsize=20)
axHisty.set_xlim(0.2, 1e3)
axHisty.set_ylim(axScatter.get_ylim())
axHisty.legend(loc='lower center')

plt.tight_layout()
plt.savefig(os.path.join(redir, 'overlap_num_scatter'))
plt.close()
logging.info("Plot is saved at {}".format(os.path.join(redir, 'overlap_num_scatter.png')))

plt.hist(overlap, ol_bin, color='r', alpha=1, histtype='step',
    lw=2, ls='dashed', label=r"$[10M_{\odot}, 40M_{\odot}]$")
plt.hist(overlap2, ol_bin, color='b', alpha=1, histtype='step',
    lw=2, label=r"$[40M_{\odot}, 100M_{\odot}]$")
plt.vlines(0.99, 0.1, 1e4, linestyles='dashed')
plt.yscale('log')
plt.ylabel("N", fontsize=20)
plt.xlabel("Overlap", fontsize=20)
plt.tick_params(labelsize=15)
plt.xlim(0.98, 1.0)
plt.ylim(0.2, 1e3)
plt.legend(loc='upper left', fontsize='x-large')
plt.tight_layout()
plt.savefig('new_num_scatter.png')


