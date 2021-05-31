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
isize = 1
dtype = 'bias_40to100'
network = 'GRU_4n256_TFR05_'+dtype+'_alpha_'+str(alpha)+'_Isize_'+str(isize)
dpath = './data/IMR_'+dtype

finput = dpath+'/IMR_hp_test_input.dat'
ftarget = dpath+'/IMR_hp_test_target.dat'
fgenetd = './generated/'+network+'/test_data_prediction'

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
targetL, targetrL = tdatprep.get_input_len(isize)

gdatprep = DatPrep(dpath)
generated = gdatprep.load(fgenetd)
generated = gdatprep.tolist()
generated = gdatprep.tofloat(blank=False)

sampling_rate = 4096.

redir = './Result_Plots_'+network
overlap = np.zeros(len(target))
cos_sim = np.zeros(len(target))
len_dff = np.zeros(len(target))
tlength = np.zeros(len(target))
discont = np.zeros(len(target))
rdiscont = np.zeros(len(target))
if not os.path.exists(redir):
    os.makedirs(redir)

duplicate = 1
extra = 10

best_num = 1
wrst_num = 1
best_idx = np.zeros(best_num)
best_ovl = np.zeros(best_num)
best_iw = [[]]*best_num
best_il = [[]]*best_num
best_tw = [[]]*best_num
best_gw = [[]]*best_num
best_rd = [[]]*best_num
best_inp_x = [[]]*best_num
best_tgt_x = [[]]*best_num
best_gen_x = [[]]*best_num
best_rsd_x = [[]]*best_num
wrst_idx = np.zeros(wrst_num)
wrst_ovl = np.zeros(wrst_num)+np.inf
wrst_iw = [[]]*wrst_num
wrst_il = [[]]*wrst_num
wrst_tw = [[]]*wrst_num
wrst_gw = [[]]*wrst_num
wrst_rd = [[]]*wrst_num
wrst_inp_x = [[]]*wrst_num
wrst_tgt_x = [[]]*wrst_num
wrst_gen_x = [[]]*wrst_num
wrst_rsd_x = [[]]*wrst_num

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

    for j in xrange(best_num):
        if temp_overlap > best_ovl[j]:
            best_idx[j] = i
            best_ovl[j] = temp_overlap
            best_iw[j] = iwaves[-ilen:]
            best_il[j] = len(iwaves)
            best_tw[j] = twaves
            best_gw[j] = gwaves
            best_inp_x[j] = inp_x
            best_tgt_x[j] = tgt_x
            best_gen_x[j] = gnt_x
            pad_len = max(tlen, glen)
            if tlen > glen:
                padded_gen = np.pad(gwaves, (0, pad_len-glen), 'constant')
            else:
                padded_gen = gwaves[:tlen]
            best_rd[j] = (twaves - padded_gen)
            best_rsd_x[j] = np.arange(0, pad_len)
            break
    for k in xrange(wrst_num):
        if temp_overlap < wrst_ovl[k]:
            wrst_idx[k] = i
            wrst_ovl[k] = temp_overlap
            wrst_iw[k] = iwaves[-ilen:]
            wrst_il[k] = len(iwaves)
            wrst_tw[k] = twaves
            wrst_gw[k] = gwaves
            wrst_inp_x[k] = inp_x
            wrst_tgt_x[k] = tgt_x
            wrst_gen_x[k] = gnt_x
            pad_len = max(tlen, glen)
            if tlen > glen:
                padded_gen = np.pad(gwaves, (0, pad_len-glen), 'constant')
            else:
                padded_gen = gwaves[:tlen]
            wrst_rd[k] = (twaves - padded_gen)
            wrst_rsd_x[k] = np.arange(0, pad_len)
            break

    tlength[i] = tlen
    overlap[i] = temp_overlap
    # cos_sim[i] = metrics.pairwise.cosine_similarity(tw.reshape(1,-1), 
    #                                             gw.reshape(1, -1))
    len_dff[i] = tlen - glen
    discont[i] = iwaves[-1]-gwaves[0]
    rdiscont[i] = discont[i]/iwaves[-1]

ovl_sn = 'Overlap_40to100' if '40to100' in dtype else 'Overlap'
np.savetxt(ovl_sn, overlap)
quit()

nullfmt = NullFormatter()

left, width = 0.130, 0.630
bottom, height = 0.175, 0.675
bottom_h = left_h = left + width + 0.01

rect_scatter = [left, bottom, width, height]
rect_histy = [left_h, bottom, 0.2, height]

plt.figure(1, figsize=(8.5, 4))
axScatter = plt.axes(rect_scatter)
axHisty = plt.axes(rect_histy)

axScatter.scatter(tlength, overlap, marker='.', color='r', alpha=0.5)
axScatter.hlines(0.99, min(tlength)-20, max(tlength)+20, linestyles='dashed')
axScatter.set_xlabel(r"$L_{t}$", fontsize=20)
axScatter.set_ylabel("Overlap", fontsize=20)
axScatter.tick_params(labelsize=15)
axScatter.set_xlim(min(tlength)-20, max(tlength)+20)
axScatter.set_ylim(0.98, 1.0)

ol_bin = np.linspace(0.99, 1, 41)
axHisty.yaxis.set_major_formatter(nullfmt)
axHisty.hist(overlap, ol_bin, orientation='horizontal',
    color='r', alpha=0.5, linewidth=0)
axHisty.hlines(0.99, 0.1, 1e4, linestyles='dashed')
axHisty.set_xscale('log')
axHisty.tick_params(labelsize=15)
axHisty.set_xlabel("Occurrence", fontsize=20)
axHisty.set_xlim(0.2, 1e3)
axHisty.set_ylim(axScatter.get_ylim())

plt.tight_layout()
plt.savefig(os.path.join(redir, 'overlap_num_scatter'))
plt.close()
logging.info("Plot is saved at {}".format(os.path.join(redir, 'overlap_num_scatter.png')))

plt.cla()
fig, ax1 = plt.subplots()
ax1.scatter(tlength, overlap, marker='.', color='r', alpha=0.5)
ax1.set_xlabel(r"$L_{t}$", fontsize=20)
ax1.set_ylabel("Overlap", fontsize=20, color='r')
ax1.axes.tick_params(axis='both', labelsize=15)
ax1.set_ylim(0, 1)

ax2 = ax1.twinx()
ax2.hist(targetrL, 20, color='b', alpha=0.5)
ax2.set_ylabel("Occurrence", color='b', fontsize=20)
ax2.axes.tick_params(axis='both', labelsize=15)

fig.tight_layout()
plt.savefig(os.path.join(redir, 'overlap_scatter_with_tlen_hist'))
plt.close()
logging.info("Plot is saved at {}".format(os.path.join(redir, 'overlap_scatter_with_tlen_hist.png')))

ol_bin = np.linspace(0.9, 1, 40)
plt.cla()
plt.hist(overlap, ol_bin, color='r', alpha=0.5)
plt.xlabel("Overlap", fontsize=20)
plt.yscale('log')
plt.ylabel("Occurrence", fontsize=20)
plt.vlines(0.99, 0.8, 2000, linestyles='dashed')
plt.xlim(0.9, 1.0)
plt.ylim(0.8, 2000)
plt.tick_params(labelsize=15)
plt.tight_layout()
plt.savefig(os.path.join(redir, 'overlap'))
plt.close()
logging.info("Plot is saved at {}".format(os.path.join(redir, 'overlap.png')))

# cs_bin = np.linspace(0,1,100)
# plt.cla()
# plt.hist(cos_sim, cs_bin, color='r', alpha=0.5)
# plt.xlabel("Cosine similarity", fontsize=20)
# plt.yscale('log')
# plt.ylim(0.8, 300)
# plt.ylabel("Numbers", fontsize=20)
# plt.tight_layout()
# plt.savefig(os.path.join(redir, 'cosine_similarity'))
# plt.close()
# logging.info("Plot is saved at {}".format(os.path.join(redir, 'cosine_similarity.png')))

ld_bin = np.linspace(-10,10,21)
plt.cla()
plt.hist(len_dff, ld_bin, color='r', alpha=0.5)
plt.yscale('log')
plt.ylim(0.8, 800)
plt.xlabel(r"$L_{t} - L_{g}$", fontsize=20)
plt.ylabel("Occurrence", fontsize=20)
plt.tick_params(labelsize=15)
plt.tight_layout()
plt.savefig(os.path.join(redir, 'length_difference'))
plt.close()
logging.info("Plot is saved at {}".format(os.path.join(redir, 'length_difference.png')))

ndup = duplicate+extra
idup_x = range(ndup)
gdup_x = range(extra, ndup+extra)

for i in xrange(best_num):
    idup = best_iw[i][-ndup:]
    gdup = best_gw[i][:ndup]
    min_oup = min(min(best_iw[i]), min(best_tw[i]), min(best_gw[i]))-1e-2
    max_oup = max(max(best_iw[i]), max(best_tw[i]), max(best_gw[i]))+1e-2
    min_dup = min(min(idup), min(gdup))-1e-2
    max_dup = max(max(idup), max(gdup))+1e-2

    fig, ax = plt.subplots(2, 1, figsize=(7, 5))

    ax[0].plot(best_inp_x[i]+best_il[i], best_iw[i], 'g', ls='dashed', label="test: input")
    ax[0].plot(best_tgt_x[i]+best_il[i], best_tw[i], 'b', ls='solid', marker='.', label="test: target")
    ax[0].plot(best_gen_x[i]+best_il[i], best_gw[i], 'r', label="test: generated")
    ax[0].tick_params(labelsize=15)
    ax[0].set_ylim(min_oup, max_oup)

    leg = ax[0].legend(ncol=3, loc='upper center', handlelength=2, bbox_to_anchor=(0.5, 1.3))
    leg_lines = leg.get_lines()
    leg_texts = leg.get_texts()
    plt.setp(leg_lines, linewidth=1.2)
    plt.setp(leg_texts, fontsize=15)

    ax[1].plot(best_inp_x[i][-ndup:]+best_il[i], idup, 'g', ls='--')
    ax[1].plot(best_gen_x[i][:ndup]+best_il[i], gdup, 'r')
    ax[1].set_ylim(min_dup, max_dup)
    ax[1].tick_params(labelsize=15)
    ax[1].set_xlabel("L", fontsize=20)

    for a in ax:
        a.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        a.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

        a.set_yticks(a.get_yticks()[1:-1])
        a.set_xticks(a.get_xticks()[1:-1])

    ax2 = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, 
        grid_color='w', grid_alpha=0, grid_linewidth=0)
    plt.ylabel("Normalized h(t)", fontsize=20, labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(redir, "best"+str(i)))
    plt.close()
    logging.info("Plot is saved at {}".format(os.path.join(redir, 'best%s.png'%str(i))))

for j in xrange(wrst_num):
    idup = wrst_iw[j][-ndup:]
    gdup = wrst_gw[j][:ndup]
    #print wrst_iw[j]
    min_oup = min(min(wrst_iw[j]), min(wrst_tw[j]), min(wrst_gw[j]))-1e-2
    max_oup = max(max(wrst_iw[j]), max(wrst_tw[j]), max(wrst_gw[j]))+1e-2
    min_dup = min(min(idup), min(gdup))-1e-2
    max_dup = max(max(idup), max(gdup))+1e-2

    plt.cla()
    fig, ax = plt.subplots(2, 1, figsize=(7, 5))

    ax[0].plot(wrst_inp_x[j]+wrst_il[j], wrst_iw[j], 'g', ls='dashed', label="test: input")
    ax[0].plot(wrst_tgt_x[j]+wrst_il[j], wrst_tw[j], 'b', ls='solid', marker='.', label="test: target")
    ax[0].plot(wrst_gen_x[j]+wrst_il[j], wrst_gw[j], 'r', label="test: generated")
    ax[0].tick_params(labelsize=15)
    ax[0].set_ylim(min_oup, max_oup)

    leg = ax[0].legend(ncol=3, loc='upper center', handlelength=2, bbox_to_anchor=(0.5, 1.3))
    leg_lines = leg.get_lines()
    leg_texts = leg.get_texts()
    plt.setp(leg_lines, linewidth=1.2)
    plt.setp(leg_texts, fontsize=15)

    ax[1].plot(wrst_inp_x[j][-ndup:]+wrst_il[j], idup, 'g', ls='--')
    ax[1].plot(wrst_gen_x[j][:ndup]+wrst_il[j], gdup, 'r')
    ax[1].set_ylim(min_dup, max_dup)
    ax[1].tick_params(labelsize=15)
    ax[1].set_xlabel("L", fontsize=20)

    for a in ax:
        a.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        a.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

        a.set_yticks(a.get_yticks()[1:-1])
        a.set_xticks(a.get_xticks()[1:-1])

    ax2 = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False,
        grid_color='w', grid_alpha=0, grid_linewidth=0)
    plt.ylabel("Normalized h(t)", fontsize=20, labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(redir, "wrst"+str(j)))
    plt.close()
    logging.info("Plot is saved at {}".format(os.path.join(redir, 'wrst%s.png'%str(j))))

logging.info("Minimum overlap: {}, maximum overlap: {} and mean overlap: {}".format(min(overlap), max(overlap), np.mean(overlap)))
print np.sort(overlap)
uu = np.sum(overlap >= 0.99)
print len(overlap)-uu
print float(uu)/len(overlap)*100

mask = overlap < 0.99
masses = np.genfromtxt('./data/IMR_'+dtype+'/mass_info_test.dat').T
m1, m2 = masses[0], masses[1]
moverlap = overlap[mask]
soverlap = np.argsort(moverlap)
print m1[mask][soverlap]
print m2[mask][soverlap]

print max(len_dff)
lmax = np.argmax(len_dff)
print len(target[lmax])
print max(len_dff)/float(len(target[lmax]))
print max(len_dff/float(tlen))

print discont[int(best_idx[0])]
print discont[int(wrst_idx[0])]
print max(discont)
print max(rdiscont)
print np.argmax(discont)
print np.argmax(rdiscont)

sdismax_arg = np.argsort(abs(discont))
dismax = int(sdismax_arg[-1])
dismax2 = int(sdismax_arg[-2])

print overlap[dismax]
print overlap[dismax2]

dindx = [dismax, dismax2]

i = 0
for d in dindx:
    ilen = 101
    tlen = len(target[d])
    glen = len(generated[d])
    mlen = min(tlen, glen)

    inp_x = np.arange(ilen)
    tgt_x = np.arange(ilen-duplicate, ilen+tlen-duplicate)
    gnt_x = np.arange(ilen-duplicate, ilen+glen-duplicate)

    idup = inputs[d][-(duplicate+extra):]
    gdup = generated[d][:(duplicate+extra)]

    min_oup = min(min(inputs[d]), min(target[d]), min(generated[d]))-1e-2
    max_oup = max(max(inputs[d]), max(target[d]), max(generated[d]))+1e-2
    min_dup = min(min(idup), min(gdup))-1e-2
    max_dup = max(max(idup), max(gdup))+1e-2

    fig, ax = plt.subplots(2, 1, figsize=(7, 5))

    ax[0].plot(inp_x+len(inputs[d]), inputs[d][-ilen:], 'g', ls='dashed', label="test: input")
    ax[0].plot(tgt_x+len(inputs[d]), target[d], 'b', ls='solid', marker='.', label="test: target")
    ax[0].plot(gnt_x+len(inputs[d]), generated[d], 'r', label="test: generated")
    ax[0].tick_params(labelsize=15)
    ax[0].set_ylim(min_oup, max_oup)

    leg = ax[0].legend(ncol=3, loc='upper center', handlelength=2, bbox_to_anchor=(0.5, 1.3))
    leg_lines = leg.get_lines()
    leg_texts = leg.get_texts()
    plt.setp(leg_lines, linewidth=1.2)
    plt.setp(leg_texts, fontsize=15)

    ax[1].plot(inp_x[-ndup:]+len(inputs[d]), idup, 'g', ls='--')
    ax[1].plot(gnt_x[:ndup]+len(inputs[d]), gdup, 'r')
    ax[1].set_ylim(min_dup, max_dup)
    ax[1].tick_params(labelsize=15)
    ax[1].set_xlabel("L", fontsize=20)

    plt.savefig(os.path.join(redir, "wrst_discontinuity"))

    for a in ax:
        a.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        a.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

        a.set_yticks(a.get_yticks()[1:-1])
        a.set_xticks(a.get_xticks()[1:-1])

    ax2 = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, 
        grid_color='w', grid_alpha=0, grid_linewidth=0)
    plt.ylabel("Normalized h(t)", fontsize=20, labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(redir, "wrst_discont"+str(i)))
    plt.close()
    logging.info("Plot is saved at {}".format(os.path.join(redir, 'wrst_discont%s.png'%str(i))))
    i += 1

