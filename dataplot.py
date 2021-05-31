import sys
import os
import csv
import logging

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.mlab as ml
import matplotlib.colors as colors

from matplotlib import ticker
import scipy.interpolate as interp
from scipy.stats import gaussian_kde
from scipy.ndimage.filters import gaussian_filter
from kde_methods import kde_histogram
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)


def kde_scipy(x, x_grid):
    kde = gaussian_kde(x)
    return kde.evaluate(x_grid)


def kde_sklearn(x, x_grid):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(x.reshape(-1, 1))
    return np.exp(kde.score_samples(x_grid.reshape(-1, 1)))


def gmm(x, x_grid):
    gm = GaussianMixture(n_components=1, random_state=0).fit(x.reshape(-1, 1))
    return np.exp(gm.score_samples(x_grid.reshape(-1, 1)))


dset = sys.argv[1]

rseed = 123
np.random.seed(rseed)

sampling_rate = 4096

outdn = './DataPlot'
if not os.path.exists(outdn):
    os.makedirs(outdn)
dn = './data/IMR_bias'
if dset == '40to100':
    dn += '_'+dset

mass_outfn = 'MassScatter'
strain_outfn = 'StrainExample'
tlen_outfn = 'target_length_PDF'
tlen_ovl_scatter_hist_outfn = 'tlen_overlap_scatter_histogram'
gofunc_outfn = 'GO_function'
waveform_outfn = 'Waveform'

if '40to100' in dn:
    min_mass = 40
    max_mass = 100
    mass_outfn += '_40to100'
elif '10to100' in dn:
    min_mass = 10
    max_mass = 100
    mass_outfn += '_10to100'
else:
    min_mass = 10
    max_mass = 40

train_mfn = os.path.join(dn, 'mass_info_training.dat')
val_mfn = os.path.join(dn, 'mass_info_validation.dat')
# ptest_mfn = os.path.join(dn, 'BackUp', 'mass_info_test.dat')
test_mfn = os.path.join(dn, 'mass_info_test.dat')

cmass = lambda x, y: (x*y)**(0.6) * (x+y)**(-0.2)
train_masses = np.genfromtxt(train_mfn).T
val_masses = np.genfromtxt(val_mfn).T
# ptest_masses = np.genfromtxt(ptest_mfn).T
test_masses = np.genfromtxt(test_mfn).T

train_m1, train_m2 = train_masses[0], train_masses[1]
val_m1, val_m2 = val_masses[0], val_masses[1]
# ptest_m1, ptest_m2 = ptest_masses[0], ptest_masses[1]
test_m1, test_m2 = test_masses[0], test_masses[1]

train_cmass = cmass(train_m1, train_m2)
val_cmass = cmass(val_m1, val_m2)
# ptest_cmass = cmass(ptest_m1, ptest_m2)
test_cmass = cmass(test_m1, test_m2)

cmass_min = min(train_cmass.min(), val_cmass.min(), test_cmass.min())
cmass_max = max(train_cmass.max(), val_cmass.max(), test_cmass.max())
# cmass_min = min(train_cmass.min(), val_cmass.min(), ptest_cmass.min(), test_cmass.min())
# cmass_max = max(train_cmass.max(), val_cmass.max(), ptest_cmass.max(), test_cmass.max())

# xrange = [10, 40]
# x = np.random.uniform(xrange[0], xrange[1], 10000)
# y = np.array([np.random.uniform(xrange[0], u, 1) for u in x]).ravel()
# xx, yy = np.meshgrid(x, y)

cm = plt.cm.get_cmap('jet')

with plt.style.context(['science', 'ieee', 'high-vis']):
    L = 10
    x = np.arange(0, L, 0.01)
    go_func = lambda a: 1.0 - 0.5*(x/L)**a

    goprob = np.ones_like(x)
    goprob = np.hstack((goprob, 0))
    xp = np.hstack((x, 10))
    a = [1, 3, 5, 20]
    color = [0, 0.3, 0.5, 0.7][::-1]
    thickness = [1, 1.3, 1.6, 2]

    fig, ax = plt.subplots()
    ax.plot(xp, goprob, 'r', linestyle='dashed')
    for i, alp in enumerate(a):
        ax.plot(xp, np.hstack((go_func(alp), 0)),
                color=str(color[i]), label=r"$\alpha = %s$" % alp,
                linestyle='solid', linewidth=thickness[i])
    ax.hlines(0.5, 0, 11, linestyle='dotted', linewidth=1, color='b')
    xt = np.array([0, 2, 4, 6, 8, 10])
    xticks = [1, 2, 3, 4, '...', r'$\cal{T}$']
    # ax.set_xticks(xt, xticks)
    plt.xticks(xt, xticks)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xlim(0, 11)
    ax.legend()
    ax.set_ylabel(r'$\cal{G}$')
    ax.set_xlabel('Computing step, '+r'$\tau$')
    # fig.tight_layout()

    gofunc_sn = os.path.join(outdn, gofunc_outfn)
    fig.savefig(gofunc_sn)
    fig.savefig(gofunc_sn+'.pdf', format='pdf', dpi=300)
    plt.close('all')
    logging.info('GO function is saved at %s' % gofunc_sn)

with plt.style.context(['science', 'ieee', 'high-vis']):
    if '40to100' in dn:
        dticks = [40, 60, 80, 100]
        lpad = 0.4
    elif '10to100' in dn:
        dticks = [10, 40, 70, 100]
        lpad = 0.4
    else:
        dticks = [10, 20, 30, 40]
        lpad = None
    fig, ax = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    sp = ax[0].scatter(train_m1, train_m2,
                       c=train_cmass, marker='.',
                       s=5, vmin=cmass_min, vmax=cmass_max,
                       linewidth=0.3, edgecolor=None)
    ax[0].scatter(val_m1, val_m2,
                  c=val_cmass, marker='.',
                  s=5, vmin=cmass_min, vmax=cmass_max,
                  linewidth=0.3, edgecolor=None)
    # ax[0].scatter(ptest_m1, ptest_m2,
    #               c=ptest_cmass, marker='.',
    #               s=5, vmin=cmass_min, vmax=cmass_max,
    #               linewidth=0.3, edgecolor=None)
    ax[0].set_xlabel(r"$m_{1} \, [M_{\odot}]$", fontsize=13)
    ax[0].set_ylabel(r"$m_{2} \, [M_{\odot}]$", labelpad=lpad, fontsize=13)
    ax[0].set_xlim(min_mass-2, max_mass+2)
    ax[0].set_ylim(min_mass-2, max_mass+2)
    ax[0].set_xticks(dticks)
    ax[0].set_yticks(dticks)
    # ax[0].set(aspect='equal')

    for tick in ax[0].xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax[0].yaxis.get_major_ticks():
        tick.label.set_fontsize(12)

    sp2 = ax[1].scatter(test_m1, test_m2,
                        c=test_cmass, marker='.',
                        s=15, vmin=cmass_min, vmax=cmass_max,
                        linewidth=0.3, edgecolor=None)
    ax[1].set_xlabel(r"$m_{1} \, [M_{\odot}]$", fontsize=13)
    ax[1].set_xlim(min_mass-2, max_mass+2)
    ax[1].set_ylim(min_mass-2, max_mass+2)
    ax[1].set_xticks(dticks)
    ax[1].set_yticks(dticks)
    # ax[1].set(aspect='equal')

    for tick in ax[1].xaxis.get_major_ticks():
        tick.label.set_fontsize(12)

    fig.subplots_adjust(left=0.2,
                        right=0.8,
                        bottom=0.1,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.3)
    cbar_ax = fig.add_axes([0.81, 0.1, 0.015, 0.8])
    cbar = fig.colorbar(sp, cax=cbar_ax)
    cbar.set_label(r"$M_{ch} \, [M_{\odot}]$", fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    mass_sn = os.path.join(outdn, mass_outfn)
    plt.savefig(mass_sn)
    plt.savefig(mass_sn+'.pdf', format='pdf', dpi=300)
    plt.close('all')
    logging.info("Mass scattergram is saved at %s" % mass_sn)


snr_dn = 'Pseudo_PE_final_40to100' if '40to100' in dn else 'Pseudo_PE_final'
snr_fn = sorted([os.path.join(snr_dn, fn) for fn in os.listdir(snr_dn)
                if 'bestfit_snr' in fn])
for sfn in snr_fn:
    snr_outfn = sfn[-5:]
    if '40to100' in dn:
        snr_outfn += '_40to100'

    tsnr = np.genfromtxt(sfn)[1:].T
    tsample = tsnr[0]
    tcenter = tsample[len(tsample)//2]
    tsample -= tcenter
    snr = tsnr[1]
    with plt.style.context(['science', 'ieee', 'high-vis', 'grid']):
        fig, ax = plt.subplots()
        ax.plot(tsample, snr, color='b')
        plt.axvline(0, color='r',
                    linestyle='dashed',
                    linewidth=0.5,
                    label='Injection time')
        ax.set_ylabel("SNR")
        ax.set_xlabel("Time [s]")
        ax.set_ylim(0, 20)
        ax.set_xlim(-11, 11)
        ax.legend()

        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

        snr_sn = os.path.join(outdn, snr_outfn)
        fig.savefig(snr_sn)
        fig.savefig(snr_sn+'.pdf', format='pdf', dpi=300)
        plt.close('all')
        logging.info("SNR timeseries is saved at %s" % snr_sn)


def marginalization(masses, probs, axis=0):
    mgrids = np.sort(np.unique(masses[:, axis]))

    marg = []
    for m in mgrids:
        partial_idx = masses[:, axis] == m
        marg.append(np.mean(probs[partial_idx]))

    marg = np.array(marg)/np.sum(marg)

    marg_density = marg/np.trapz(marg, x=mgrids)

    return mgrids, marg_density


def conf_interval(mass, mpdensity):
    mode_idx = np.argmax(mpdensity)
    m_mode = mass[mode_idx]

    m_left = mass[:mode_idx][::-1]
    m_right = mass[mode_idx:]
    mpd_left = mpdensity[:mode_idx][::-1]
    mpd_right = mpdensity[mode_idx:]

    llim = len(m_left)
    rlim = len(m_right)

    i = 0
    lthr, rthr = 0.34, 0.34
    lprob, rprob = 0., 0.
    lproc, rproc = True, True
    while True:
        if lproc:
            lw = m_left[i]-m_left[i+1]
            lh = (mpd_left[i]+mpd_left[i+1])/2.
            lprob += lw*lh
        if rproc:
            rw = m_right[i+1]-m_right[i]
            rh = (mpd_right[i+1]+mpd_right[i])/2.
            rprob += rw*rh

        if lproc and i >= llim-2 or lprob >= lthr:
            lid = i
            lproc = False

        if rproc and i >= rlim-2 or rprob >= rthr:
            rid = i
            rproc = False

        if not any([lproc, rproc]):
            break

        i += 1

    linterval = m_left[lid]
    rinterval = m_right[rid]

    mlinterval = np.abs(np.diff(m_left))
    mrinterval = np.diff(m_right)

    lpaverage = (mpd_left[:-1]+mpd_left[1:])/2.
    rpaverage = (mpd_right[:-1]+mpd_right[1:])/2.

    lpcumsum = np.sum(mlinterval*lpaverage)

    intervals = [mlinterval[::-1], mrinterval[::-1]]
    mps = [lpaverage[::-1], rpaverage[::-1]]

    drt_idx = 0 if lpcumsum >= 0.5 else 1
    i = 0
    cprob = 0.
    while True:
        cprob += intervals[drt_idx][i]*mps[drt_idx][i]
        if cprob >= 0.5:
            median_idx = i
            break
        i += 1
    mx = [m_left[::-1], m_right[::-1]]
    m_median = mx[drt_idx][median_idx]

    m_left = mass[:median_idx][::-1]
    m_right = mass[median_idx:]
    mpd_left = mpdensity[:median_idx][::-1]
    mpd_right = mpdensity[median_idx:]

    llim = len(m_left)
    rlim = len(m_right)

    i = 0
    lthr, rthr = 0.34, 0.34
    lprob, rprob = 0., 0.
    lproc, rproc = True, True
    while True:
        if lproc:
            lw = m_left[i]-m_left[i+1]
            lh = (mpd_left[i]+mpd_left[i+1])/2.
            lprob += lw*lh
        if rproc:
            rw = m_right[i+1]-m_right[i]
            rh = (mpd_right[i+1]+mpd_right[i])/2.
            rprob += rw*rh

        if lproc and i >= llim-2 or lprob >= lthr:
            lid = i
            lproc = False

        if rproc and i >= rlim-2 or rprob >= rthr:
            rid = i
            rproc = False

        if not any([lproc, rproc]):
            break

        i += 1

    med_linterval = m_left[lid]
    med_rinterval = m_right[rid]

    return m_median, [med_linterval, med_rinterval], m_mode, [linterval, rinterval]


ppe_dn = 'Pseudo_PE_final_40to100' if '40to100' in dn else 'Pseudo_PE_final'
ppe_fn = sorted([os.path.join(ppe_dn, fn) for fn in os.listdir(ppe_dn) if 'mass_SNR' in fn])
best_fit = []
for i, pfn in enumerate(ppe_fn):
    ppe_outfn = os.path.join(outdn, 'Pseudo_PE_%s' % i)
    if '40to100' in dn:
        ppe_outfn += '_40to100'
    data = np.genfromtxt(pfn)
    m1, m2, psnr = data.T

    masses = np.vstack((m1, m2)).T

    # Pseudo-PE
    eps = 0.01
    m1_range = np.arange(min_mass, max_mass, 0.1)
    for j, rm1 in enumerate(m1_range):
        m2s = np.arange(min_mass, rm1+eps, 0.1)
        m1s = np.zeros(len(m2s))+rm1
        temp_mass_grid = np.vstack((m1s, m2s)).T
        if j == 0:
            mass_grid = temp_mass_grid
        else:
            mass_grid = np.vstack((mass_grid, temp_mass_grid))

    assert np.all(mass_grid[:, 0] >= mass_grid[:, 1])

    # sampling test
    nsample = 1000000
    temperature = 3
    temped_psnr = psnr**temperature
    sprob = temped_psnr/np.sum(temped_psnr)
    mlen = len(masses)
    sample_idx = np.random.choice(np.arange(mlen), nsample, p=sprob)
    sampled = masses[sample_idx]
    m1_samp, m2_samp = sampled[:, 0], sampled[:, 1]
    m1_samp_uni = np.sort(np.unique(mass_grid[:, 0]))
    m2_samp_uni = np.sort(np.unique(mass_grid[:, 1]))
    # m1_samp_marg = kde_scipy(m1_samp, m1_samp_uni)
    # m2_samp_marg = kde_scipy(m2_samp, m2_samp_uni)
    # m1_samp_marg = kde_sklearn(m1_samp, m1_samp_uni)
    # m2_samp_marg = kde_sklearn(m2_samp, m2_samp_uni)
    m1_samp_marg = gmm(m1_samp, m1_samp_uni)
    m2_samp_marg = gmm(m2_samp, m2_samp_uni)

    m1_asample = np.random.choice(m1_samp_uni, nsample, p=m1_samp_marg/np.sum(m1_samp_marg))
    m2_asample = np.random.choice(m2_samp_uni, nsample, p=m2_samp_marg/np.sum(m2_samp_marg))

    m1_med = np.percentile(m1_asample, 50)
    m2_med = np.percentile(m2_asample, 50)

    m1_confint = np.percentile(m1_asample, (5, 95))
    m2_confint = np.percentile(m2_asample, (5, 95))

    sfn = os.path.join(ppe_dn, 'bestfit_'+pfn[-5:].lower())
    tm1, tm2 = np.genfromtxt(sfn)[0]

    fig, ax = plt.subplots()
    ax.hist(m1_samp, 50, density=True, color='r', histtype='step')
    ax.hist(m2_samp, 50, density=True, color='b', histtype='step')
    ax.plot(m1_samp_uni, m1_samp_marg, color='r', linestyle='dashed')
    ax.plot(m2_samp_uni, m2_samp_marg, color='b', linestyle='dashed')
    temp_fn = os.path.join(outdn, 'test%s' % i)
    if '40to100' in dn:
        temp_fn += '_40to100'
    fig.savefig(temp_fn)

    outfn = os.path.join(outdn, "ParamSearch%s" % i)
    if '40to100' in dn:
        outfn += '_40to100'
    outfn += '.csv'
    m1c = np.array(m1_confint)-m1_med
    m2c = np.array(m2_confint)-m2_med
    dat = [tm1, tm2, m1_med, m1c[0], m1c[1], m2_med, m2c[0], m2c[1]]
    strdat = ['%.2f' % d for d in dat]
    best_fit.append(strdat)
    # with open(outfn, 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow([tm1, tm2])
    #     writer.writerow([m1_med, np.array(m1_confint)-m1_med])
    #     writer.writerow([m2_med, np.array(m2_confint)-m2_med])
    #     writer.writerow([cmass(tm1, tm2), cmass(m1_med, m2_med)])

    # logging.info("Parameter search result is saved at %s" % outfn)
    # quit()

    ##################

    # npsnr = interp.griddata(masses, psnr, mass_grid)
    # npsnr = np.array(npsnr).ravel()

    # # del nan
    # nan_idx = np.where(np.isnan(npsnr))
    # mass_grid = np.delete(mass_grid, nan_idx, 0)
    # npsnr = np.delete(npsnr, nan_idx, 0)

    # params = mass_grid.T
    # dlen = len(mass_grid)
    # probs = npsnr/np.sum(npsnr)

    # m1_unique, m1_marg = marginalization(mass_grid, probs, axis=0)
    # m2_unique, m2_marg = marginalization(mass_grid, probs, axis=1)

    # quit()

    # m1_med, m1_med_cint, m1_mode, m1_mode_cint = conf_interval(m1_unique, m1_marg)
    # m2_med, m2_med_cint, m2_mode, m2_mode_cint = conf_interval(m2_unique, m2_marg)

    # snr_argmax = np.argmax(psnr)
    # bm1 = m1[snr_argmax]
    # bm2 = m2[snr_argmax]
    # bsnr = psnr[snr_argmax]
    # best_fit.append([tm1, tm2, m1_med, m2_med, ])

    # sfn = os.path.join(ppe_dn, 'bestfit_'+pfn[-5:].lower())
    # tm1, tm2 = np.genfromtxt(sfn)[0]

    # print(tm1)
    # print(m1_med, m1_med-m1_med_cint[0], m1_med_cint[1]-m1_med)
    # print(m1_mode, m1_mode-m1_mode_cint[0], m1_mode_cint[1]-m1_mode)
    # print(tm2)
    # print(m2_med, m2_med-m2_med_cint[0], m2_med_cint[1]-m2_med)
    # print(m2_mode, m2_mode-m2_mode_cint[0], m2_mode_cint[1]-m2_mode)

    chirp_am = cmass(tm1, tm2)
    chirp_am_level = [chirp_am-2, chirp_am-1, chirp_am]
    chirp_m = cmass(m1, m2)
    with plt.style.context(['science', 'ieee', 'high-vis']):
        fig, ax = plt.subplots()

        sm1 = np.hstack((m1, m2))
        sm2 = np.hstack((m2, m1))
        spsnr = np.hstack((psnr, psnr))

        cfp = ax.tricontourf(sm1, sm2, spsnr, 256,
                             vmin=(psnr.min()+psnr.max())/2.,
                             vmax=psnr.max())
        schirp_m = np.hstack((chirp_m, chirp_m))
        cp = ax.tricontour(sm1, sm2, schirp_m, [chirp_am])
        for c in cp.collections:
            c.set_linestyles('--')
            c.set_linewidths(1)
            c.set_color('k')
        for c in cfp.collections:
            c.set_edgecolor("face")
        ax.scatter(tm1, tm2, marker='*', color='r', label="Injection", s=60)
        ax.scatter(m1_med, m2_med, marker='P', color='b', label="Best-fit", s=60)

        ax.set_xlabel(r"$m_{1}$")
        ax.set_ylabel(r"$m_{2}$")
        ax.legend(loc='upper left', frameon=True)

        cbar = fig.colorbar(cfp,
                            format='%.1f')
        cbar.set_label("SNR")

        if '40to100' in dn:
            ax.set_xlim(40, 100)
            ax.set_ylim(40, 100)
        else:
            ax.set_xlim(10, 40)
            ax.set_ylim(10, 40)

        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

        # print(ax.get_position())
        fig.savefig(ppe_outfn)
        fig.savefig(ppe_outfn+'.pdf', format='pdf', dpi=300)
        plt.close('all')
        logging.info("Pseudo-PE plot is saved at %s" % ppe_outfn)

bestfit_fn = os.path.join(outdn, 'ParamSearch')
if '40to100' in dn:
    bestfit_fn += '_40to100'
header = ["tm1", "tm2", "m1", "m1_lcint", "m1_ucint", "m2", "m2_lcint", "m2_ucint"]
with open(bestfit_fn, 'w') as outf:
    writer = csv.writer(outf)
    writer.writerow(header)
    for out in best_fit:
        writer.writerow(out)
logging.info("Best fit configurations are saved at %s" % bestfit_fn)

if '40to100' not in dn:
    train_hpifn = os.path.join(dn, 'IMR_hp_training_input.dat')
    train_hptfn = os.path.join(dn, 'IMR_hp_training_target.dat')

    with open(train_hpifn, 'r') as f:
        hpi = f.readlines()
    with open(train_hptfn, 'r') as f:
        hpt = f.readlines()

    cmass_lbound = 8
    cmass_ubound = 20
    while True:
        rand_idx = np.random.randint(0, len(hpt), 2)
        rand_m1, rand_m2 = train_m1[rand_idx], train_m2[rand_idx]
        rand_cmass = cmass(rand_m1, rand_m2)

        if all(rand_cmass <= cmass_ubound) and all(rand_cmass >= cmass_lbound):
            logging.info("Cmass: %s" % str(rand_cmass))
            break

    rand_hpi = [hpi[i] for i in rand_idx]
    rand_hpt = [hpt[i] for i in rand_idx]
    for i, (ip, tg) in enumerate(zip(rand_hpi, rand_hpt)):
        rand_hpi[i] = np.array(ip.split(), dtype=np.float32)
        temp_hpt = np.array(tg.split(), dtype=np.float32)
        rand_hpt[i] = temp_hpt[np.nonzero(temp_hpt)[0]]

    with plt.style.context(['science', 'ieee', 'high-vis']):
        overlap = 1

        minp_x = np.arange(0, len(rand_hpi[0]))/sampling_rate
        mtgt_x = (len(rand_hpi[0]) - overlap +
                  np.arange(0, len(rand_hpt[0])))/sampling_rate

        mant_x = mtgt_x[-1]+0.01
        mtxt_x = mtgt_x[-1]+0.18

        fig, ax = plt.subplots()
        ax.axes.yaxis.set_visible(False)
        ax.plot(minp_x, rand_hpi[0],
                color='g', label="inputs(I)",
                linestyle='dashed')
        ax.plot(mtgt_x, rand_hpt[0],
                color='b', label="targets(M, R)",
                linestyle='solid')
        ax.annotate(r'$M_{ch}=%.2fM_{\odot}$' % rand_cmass[0],
                    xy=(mant_x, 0), xytext=(mtxt_x, -1e-21), fontsize=6,
                    arrowprops=dict(arrowstyle='->', facecolor='black'),
                    va='top', ha='right')

        uinp_x = np.arange(0, len(rand_hpi[1]))/sampling_rate
        utgt_x = (len(rand_hpi[1]) - overlap +
                  np.arange(0, len(rand_hpt[1])))/sampling_rate
        uant_x = utgt_x[-1]+0.01
        utxt_x = utgt_x[-1]+0.14
        ax.plot(uinp_x, rand_hpi[1]+5e-21,
                color='g', linestyle='dashed')
        ax.plot(utgt_x, rand_hpt[1]+5e-21,
                color='b', linestyle='solid')
        ax.annotate(r'$M_{ch}=%.2fM_{\odot}$' % rand_cmass[1],
                    xy=(uant_x, 5e-21), xytext=(utxt_x, 2e-21), fontsize=6,
                    arrowprops=dict(arrowstyle='->', facecolor='black'),
                    va='center', ha='right')

        ax.set_xlim(0, 0.6)
        ax.set_ylim(-2.5e-21, 1e-20)
        ax.set_xlabel("Time")
        ax.legend()

        strain_sn = os.path.join(outdn, strain_outfn)
        fig.savefig(strain_sn)
        fig.savefig(strain_sn+'.pdf', format='pdf', dpi=300)
        plt.close('all')
        logging.info("Strain examples are saved at %s" % strain_sn)

    dn_40to100 = dn + '_40to100'
    train_hptfn_40to100 = os.path.join(dn_40to100, 'IMR_hp_training_target.dat')

    with open(train_hptfn_40to100, 'r') as f:
        hpt_40to100 = f.readlines()

    len_hpt = []
    for t in hpt:
        temp_t = np.array(t.split(), dtype=np.float32)
        len_hpt.append(len(temp_t[np.nonzero(temp_t)]))
    len_hpt = np.array(len_hpt)

    len_hpt_40to100 = []
    for t in hpt_40to100:
        temp_t = np.array(t.split(), dtype=np.float32)
        len_hpt_40to100.append(len(temp_t[np.nonzero(temp_t)]))
    len_hpt_40to100 = np.array(len_hpt_40to100)

    with plt.style.context(['science', 'ieee', 'high-vis']):
        fig, ax = plt.subplots()

        lgrid = np.linspace(len_hpt.min(), len_hpt.max(), 100)
        lpdf = gmm(len_hpt, lgrid)

        lgrid_40to100 = np.linspace(len_hpt_40to100.min(), len_hpt_40to100.max(), 100)
        lpdf_40to100 = gmm(len_hpt_40to100, lgrid_40to100)
        ax.plot(lgrid, lpdf,
                color='r',
                linewidth=1,
                linestyle='solid',
                label="Dataset-1")
        ax.plot(lgrid_40to100, lpdf_40to100,
                color='b',
                linewidth=0.5,
                linestyle='solid',
                label="Dataset-2")

        # ax.hist(len_hpt, 30,
        #         histtype='step', color='r',
        #         linewidth=1, density=True,
        #         linestyle='solid',
        #         label='dataset-1')
        # ax.hist(len_hpt_40to100, 30,
        #         histtype='step', color='b',
        #         linewidth=0.5, density=True,
        #         linestyle='solid',
        #         label='dataset-2')

        ax.set_xlabel(r"$L_{t}$")
        ax.set_ylabel("Number density")
        ax.legend()

        tlen_sn = os.path.join(outdn, tlen_outfn)
        fig.savefig(tlen_sn)
        fig.savefig(tlen_sn+'.pdf', format='pdf', dpi=300)
        plt.close('all')
        logging.info("Target length PDF is saved at %s" % tlen_sn)

    # quit()

    dn_40to100 = dn + '_40to100'
    test_hptfn = os.path.join(dn, 'IMR_hp_test_target.dat')
    test_hptfn_40to100 = os.path.join(dn_40to100, 'IMR_hp_test_target.dat')

    with open(test_hptfn, 'r') as f:
        hpt = f.readlines()
    with open(test_hptfn_40to100, 'r') as f:
        hpt_40to100 = f.readlines()

    len_hpt = []
    for t in hpt:
        temp_t = np.array(t.split(), dtype=np.float32)
        len_hpt.append(len(temp_t[np.nonzero(temp_t)]))
    len_hpt = np.array(len_hpt)

    len_hpt_40to100 = []
    for t in hpt_40to100:
        temp_t = np.array(t.split(), dtype=np.float32)
        len_hpt_40to100.append(len(temp_t[np.nonzero(temp_t)]))
    len_hpt_40to100 = np.array(len_hpt_40to100)

    ovl = np.genfromtxt('Overlap')
    ovl_40to100 = np.genfromtxt('Overlap_40to100')

    grid_min = min(len_hpt.min(), len_hpt_40to100.min())
    grid_max = max(len_hpt.max(), len_hpt_40to100.max())

    with plt.style.context(['science', 'ieee', 'high-vis']):
        toshs_fn = tlen_ovl_scatter_hist_outfn
        lengths = [len_hpt, len_hpt_40to100]
        ovls = [ovl, ovl_40to100]
        fig, ax = plt.subplots(1, 2,
                               figsize=(7, 3),
                               gridspec_kw={'width_ratios': [1, 1]})
        for i in range(2):
            hp = ax[i].hexbin(lengths[i], ovls[i], gridsize=20,
                              norm=colors.LogNorm(vmin=1e-1, vmax=30))
            if i == 0:
                ax[i].set_ylabel("Overlap")
            else:
                ax[i].axes.yaxis.set_visible(False)
            ax[i].set_xlabel(r"$L_{t}$")
            ax[i].margins(0) 
            ax[i].autoscale()
            ax[i].set_ylim(0.990, 1.0)

        fig.subplots_adjust(left=0.1, right=0.8, bottom=0.1,
                            top=0.9, hspace=0.1, wspace=0.01)
        cbar_ax = fig.add_axes([0.81, 0.1, 0.015, 0.81])
        cbar = fig.colorbar(hp, cax=cbar_ax)
        cbar.set_label("Density")

        tlen_ovl_scatter_hist_sn = os.path.join(outdn, toshs_fn)
        fig.savefig(tlen_ovl_scatter_hist_sn)
        fig.savefig(tlen_ovl_scatter_hist_sn+'.pdf', format='pdf', dpi=300)
        logging.info("Overlap vs target-length and histogram is saved at %s" %
                     tlen_ovl_scatter_hist_sn)
        plt.close('all')

        # x1 = np.vstack([len_hpt, ovl]).reshape(2, -1)
        # x2 = np.vstack([len_hpt_40to100, ovl_40to100]).reshape(2, -1)
        # c1 = kde_scipy(x1, x1)
        # c2 = kde_scipy(x2, x2)

        # sp1 = ax.scatter(len_hpt, ovl, c=c1,
        #                  norm=colors.LogNorm(vmin=1e-3),
        #                  label="dataset-1")
        # sp2 = ax.scatter(len_hpt_40to100, ovl_40to100, c=c2,
        #                  norm=colors.LogNorm(vmin=1e-3),
        #                  label="dataset-2")
        # ax.hlines(0.99, grid_min-20, grid_max+20, linestyles='dashed')
        # ax.set_xlabel(r"$L_{t}$")
        # ax.set_ylabel("Overlap")
        # ax.set_xlim(grid_min-20, grid_max+20)
        # ax.set_ylim(0.985, 1)
        # ax.legend(loc='lower left')

        # ol_bin = np.linspace(0.99, 1, 41)
        # ax[1].axes.yaxis.set_visible(False)
        # ax[1].hist(ovl, ol_bin,
        #            orientation='horizontal',
        #            color='r', density=True,
        #            linewidth=1, histtype='step',
        #            linestyle='solid', label="dataset-1")
        # ax[1].hist(ovl_40to100, ol_bin,
        #            orientation='horizontal',
        #            color='b', density=True,
        #            linewidth=0.5, histtype='step',
        #            linestyle='solid', label="dataset-2")
        # ax[1].hlines(0.99, 0.1, 1e4, linestyles='dashed')
        # ax[1].set_xscale('log')
        # ax[1].set_xlabel("Number density")
        # ax[1].set_xlim(0.2, 1e4)
        # ax[1].set_ylim(ax[0].get_ylim())
        # ax[1].legend(loc='lower left')

        # fig.subplots_adjust(wspace=0)
        # cbar_ax = fig.add_axes([0.81, 0.1, 0.015, 0.8])
        # cbar = fig.colorbar(sp2, cax=cbar_ax)

        # tlen_ovl_scatter_hist_sn = os.path.join(outdn,
        #                                         tlen_ovl_scatter_hist_outfn)
        # fig.savefig(tlen_ovl_scatter_hist_sn)
        # fig.savefig(tlen_ovl_scatter_hist_sn+'.pdf', format='pdf', dpi=300)
        # logging.info("Overlap vs target-length and histogram is saved at %s" %
        #              tlen_ovl_scatter_hist_sn)

    max_amp = np.genfromtxt('./data/IMR_bias/maximum_amplitude')[0]
    max_amp_40to100 = np.genfromtxt('./data/IMR_bias_40to100/maximum_amplitude')[0]
    max_amps = [max_amp, max_amp_40to100]

    dn_40to100 = dn+'_40to100'
    test_hpifn = os.path.join(dn, 'IMR_hp_test_input.dat')
    test_hpifn_40to100 = os.path.join(dn_40to100, 'IMR_hp_test_input.dat')
    gen_hptfn = 'generated/GRU_4n256_TFR05_bias_alpha_5_Isize_1/test_data_prediction'
    gen_hptfn_40to100 = 'generated/GRU_4n256_TFR05_bias_40to100_alpha_5_Isize_1/test_data_prediction'

    with open(test_hpifn, 'r') as f:
        hpi = f.readlines()
    with open(test_hpifn_40to100, 'r') as f:
        hpi_40to100 = f.readlines()
    with open(gen_hptfn, 'r') as f:
        ghpt = f.readlines()
    with open(gen_hptfn_40to100, 'r') as f:
        ghpt_40to100 = f.readlines()

    best_idx = np.argsort(ovl)[-1]
    wrst_idx = np.argsort(ovl)[0]
    # wrst_idx = 919
    best_idx_40to100 = np.argsort(ovl_40to100)[-1]
    wrst_idx_40to100 = np.argsort(ovl_40to100)[0]
    ids = [best_idx, wrst_idx, best_idx_40to100, wrst_idx_40to100]
    prefixs = ['best', 'worst', 'best_40to100', 'worst_40to100']
    hps = [hpi, hpt, hpi_40to100, hpt_40to100]
    ghpts = [ghpt, ghpt_40to100]
    with plt.style.context(['science', 'ieee', 'high-vis']):
        for i, idx in enumerate(ids):
            temp_hpi = np.array(hps[i//2*2][ids[i]].split(), dtype=np.float32)/max_amps[i//2]
            temp_hpt = np.array(hps[i//2*2+1][ids[i]].split(), dtype=np.float32)
            temp_hpt = temp_hpt[np.nonzero(temp_hpt)[0]]/max_amps[i//2]
            temp_ghpt = np.array(ghpts[i//2][ids[i]].split(), dtype=np.float32)

            hpi_x = np.arange(1, len(temp_hpi)+1)
            hpt_x = np.arange(len(temp_hpi), len(temp_hpi)+len(temp_hpt))
            ghpt_x = np.arange(len(temp_hpi), len(temp_hpi)+len(temp_ghpt))

            x_llim = hpi_x[-1]-100
            x_ulim = max(ghpt_x[-1], hpt_x[-1])+10

            fig, ax = plt.subplots(2, 1)
            ax[0].plot(hpi_x, temp_hpi, 'g', linewidth=0.5,
                       ls='dashed', label="input")
            ax[0].plot(hpt_x, temp_hpt, 'b',
                       ls='solid', marker='.',
                       markersize=1.5, linewidth=0.5,
                       label="target")
            ax[0].plot(ghpt_x, temp_ghpt, 'r',
                       ls='solid', linewidth=0.5,
                       label="generated")
            ax[0].set_xlim(x_llim, x_ulim)
            ax[0].set_yticks([], minor=True)
            ax[0].set_xticks([], minor=True)

            leg = ax[0].legend(ncol=3, loc='upper center',
                               handlelength=1.5, bbox_to_anchor=(0.5, 1.3))
            leg_lines = leg.get_lines()
            leg_texts = leg.get_texts()
            # plt.setp(leg_lines, linewidth=1.2)
            # plt.setp(leg_texts, fontsize=15)

            ax[1].plot(hpi_x[-10:], temp_hpi[-10:], 'g',
                       ls='dashed', linewidth=0.5)
            ax[1].plot(ghpt_x[:10], temp_ghpt[:10], 'r',
                       ls='solid', linewidth=0.5)
            ax[1].set_xlim(hpi_x[-10]-1, ghpt_x[10]+1)
            ax[1].set_xlabel("L")
            ax[1].set_yticks([], minor=True)
            ax[1].set_xticks([], minor=True)

            # for a in ax:
            #     a.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            #     a.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

            #     a.set_yticks(a.get_yticks()[1:-1])
            #     a.set_xticks(a.get_xticks()[1:-1])

            fig.subplots_adjust(left=0.1, right=0.9,
                                bottom=0.1, top=0.9,
                                hspace=0.3)

            ax2 = fig.add_subplot(111, frameon=False)
            plt.tick_params(axis='both', which='both', labelcolor='none',
                            top=False, bottom=False, left=False, right=False,
                            grid_color='w', grid_alpha=0, grid_linewidth=0)
            plt.ylabel("Normalized h(t)", labelpad=11)
            # plt.tight_layout()

            waveform_sn = os.path.join(outdn, waveform_outfn+'_'+prefixs[i])
            fig.savefig(waveform_sn)
            fig.savefig(waveform_sn+'.pdf', format='pdf', dpi=300)
            plt.close('all')
            logging.info('Waveform example is saved at %s' % waveform_sn)
