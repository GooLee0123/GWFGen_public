import os
import random
import ConfigParser
import argparse
import logging

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import skewnorm
from lmfit.models import SkewedGaussianModel

from pycbc.waveform import get_td_waveform

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)

def getWFparams():
    conf_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False)
    conf_parser.add_argument('-c', '--conf_file', default='params.cfg',
                            help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()

    Noption = 1
    Keys = ["Parameters"]
    OptionDict = [{}]*Noption

    if args.conf_file:
        config = ConfigParser.SafeConfigParser()
        config.read([args.conf_file])
        for i in range(Noption):
            OptionDict[i].update(dict(config.items(Keys[i])))

    parser = argparse.ArgumentParser(parents=[conf_parser])

    for i in range(Noption):
        parser.set_defaults(**OptionDict[i])

    args = parser.parse_args(remaining_argv)

    optsdict = vars(args)

    for key in optsdict.keys():
        try:
            optsdict[key] = float(optsdict[key])
        except:
            pass

    return optsdict

def getoutputFname(outdir):
    hp_input_trfname = os.path.join(outdir, 'IMR_hp_training_input.dat')
    hp_target_trfname = os.path.join(outdir, 'IMR_hp_training_target.dat')
    hc_input_trfname = os.path.join(outdir, 'IMR_hc_training_input.dat')
    hc_target_trfname = os.path.join(outdir, 'IMR_hc_training_target.dat')
    ms_input_trfname = os.path.join(outdir, 'mass_info_training.dat')
    hp_input_valfname = hp_input_trfname.replace('_training','_validation')
    hp_target_valfname = hp_target_trfname.replace('_training','_validation')    
    hc_input_valfname = hc_input_trfname.replace('_training','_validation')
    hc_target_valfname = hc_target_trfname.replace('_training','_validation')
    ms_input_valfname = ms_input_trfname.replace('_training','_validation')
    hp_input_tefname = hp_input_trfname.replace('_training','_test')
    hp_target_tefname = hp_target_trfname.replace('_training','_test')
    hc_input_tefname = hc_input_trfname.replace('_training','_test')
    hc_target_tefname = hc_target_trfname.replace('_training','_test')
    ms_input_tefname = ms_input_trfname.replace('_training','_test')

    return [hp_input_trfname, hp_target_trfname, hc_input_trfname,
            hc_target_trfname, ms_input_trfname, hp_input_valfname,
            hp_target_valfname, hc_input_valfname, hc_target_valfname,
            ms_input_valfname, hp_input_tefname, hp_target_tefname,
            hc_input_tefname, hc_target_tefname, ms_input_tefname]

def generateWFdata(m1, m2, optsdict):
    hp, hc = get_td_waveform(approximant=optsdict['approximant'],
                            mass1=m1, mass2=m2,
                            spin1x=optsdict['spin1x'],
                            spin1y=optsdict['spin1y'],
                            spin1z=optsdict['spin1z'],
                            spin2x=optsdict['spin2x'],
                            spin2y=optsdict['spin2y'],
                            spin2z=optsdict['spin2z'],
                            distance=optsdict['distance'],
                            inclination=optsdict['inclination'],
                            delta_t=1./optsdict['samplerate'],
                            f_lower=optsdict['f_min'])

    return hp.sample_times.numpy(), hp.numpy(), hc.numpy()

def get_ISCO_index(signal, sampling_rate, total_mass, min_freq, t):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase)/(2.0*np.pi)*sampling_rate)
    FISCO = 4400./total_mass
    ISCOS = np.argwhere(instantaneous_frequency >= FISCO)
    IISCO = ISCOS[ISCOS>min_freq*4][0]+1
    Merger_index = np.argwhere(t > 0.0)[0][0]

    #plt.cla()
    #plt.figure()
    #plt.plot(instantaneous_frequency)
    #plt.vlines(ISCOS[ISCOS>min_freq*8][0]+1, 0, 500, color='k')
    #plt.savefig('hilbert_frequency_test')
    #plt.close()

    if Merger_index < IISCO:
        logging.info("Being smaller than ISCO time index return merger index")
    return min(IISCO, Merger_index)

def MonteCarloSearch(repeat, min_mass, max_mass, chirp_func):
    repeat = 1000000

    logging.info("Perform monte carlo simulation with data grid with size of {}".format(repeat))
    logging.info("Exaggerated mass range of grid is from {} to {}*2".format(min_mass, max_mass))

    m1_fit = np.zeros(repeat)
    m2_fit = np.zeros(repeat)
    for i in xrange(repeat):
        m1_fit[i] = random.uniform(min_mass, max_mass*2)
        m2_fit[i] = random.uniform(min_mass, m1_fit[i])
    cmass_fit = chirp_func(m1_fit, m2_fit)

    points = np.histogram(cmass_fit, 1000)

    x = np.array([(points[1][i]+points[1][i+1])/2. for i in xrange(len(points[0]))])
    y = points[0]/float(max(points[0]))

    return x, y, cmass_fit

def Skew_Norm_Fit(x, y):
    logging.info("Fit probability distribution of chirp mass to skew normal distribution")
    model = SkewedGaussianModel()
    params = model.make_params(amplitude=30, center=10, sigma=20, gamma=20)
    result = model.fit(y, params, x=x)

    amplitude = result.params['amplitude']
    sigma = result.params['sigma']
    center = result.params['center']
    gamma = result.params['gamma']

    skew_pdf = lambda cm: amplitude*skewnorm.pdf(cm, gamma, loc=center, scale=sigma)

    plt.close()

    return skew_pdf

def Rejection_Method(Q, cmass, cmass_func, grid_num, min_mass, max_mass):
    logging.info("Perform rejection method to make chirp mass distribution uniform")
    m1_arr = np.zeros(grid_num)
    m2_arr = np.zeros(grid_num)

    min_cm = min(cmass)
    max_cm = max(cmass)
    
    P = 1./(max_cm-min_cm)*0.5 # Uniform distribution

    nstore = 0
    while True:
        mm1 = random.uniform(min_mass, max_mass*2)
        mm2 = random.uniform(min_mass, mm1)
        temp_cmass = cmass_func(mm1, mm2)
        temp_Q = Q(temp_cmass)
        temp_u = random.uniform(0, temp_Q)
        if temp_u < P:
            m1_arr[nstore] = mm1
            m2_arr[nstore] = mm2
            nstore += 1
            if nstore % 1000 == 0:
                logging.info("{}th data is stored".format(nstore))
        if nstore == grid_num:
            break

    return m1_arr, m2_arr

def DownSampling(m1_arr, m2_arr, cmass_func, max_mass, ndata):
    logging.info("Start post processing to cut off the mass grid below {}".format(max_mass))
    fit_to_mrange_mask = m1_arr < max_mass
    m1_fit_arr = m1_arr[fit_to_mrange_mask]
    m2_fit_arr = m2_arr[fit_to_mrange_mask]
    cmass = cmass_func(m1_fit_arr, m2_fit_arr)
    nbin = 100
    histogram = np.histogram(cmass, nbin)
    binning = histogram[1]

    m1 = np.array([])
    m2 = np.array([])

    if ndata%nbin != 0:
        raise ValueError("ndata should be divided by nbin")
    n_per_bin = ndata/nbin

    for n in xrange(nbin):
        upper_mask = cmass >= binning[n]
        lower_mask = cmass < binning[n+1]
        down_mask = map(bool, upper_mask*lower_mask)
        m1 = np.append(m1, m1_fit_arr[down_mask][:n_per_bin])
        m2 = np.append(m2, m2_fit_arr[down_mask][:n_per_bin])

    return m1, m2

def main():
    optsdict = getWFparams()
    outputFs = getoutputFname(optsdict['outdir'])

    if not os.path.exists(optsdict['outdir']):
        os.makedirs(optsdict['outdir'])

    logging.info(optsdict)

    training = 0 # Data amount of training file
    validation = 0 # Data amount of validation file
    test = 100 # Data amount of test file

    min_mass = float(optsdict['min_mass'])
    max_mass = float(optsdict['max_mass'])

    chirp_mass = lambda mm1, mm2: (mm1*mm2)**0.6/(mm1+mm2)**0.2

    doption = optsdict['doption']
    if doption == 'bias':
        ndata = 100
        massratio = np.arange(1.1, 2.5, 0.03) # m1/m2
        num_mr = len(massratio)
        m1_list = np.linspace(min_mass, max_mass, ndata)
    else:
        num_mr = 1
        repeat = 1000000
        grid_num = 10000
        ndata = 1000
        x, y, cmass = MonteCarloSearch(repeat, min_mass, max_mass, chirp_mass)
        skew_pdf = Skew_Norm_Fit(x, y)
        m1_, m2_ = Rejection_Method(skew_pdf, cmass, chirp_mass, grid_num, min_mass, max_mass)
        cmass_test = chirp_mass(m1_, m2_)
        m1_list, m2_list = DownSampling(m1_, m2_, chirp_mass, max_mass, ndata)
        ndata = len(m1_list)

    nth_data = 1
    max_hp_amp = 0
    max_hc_amp = 0
    for i in xrange(num_mr):
        idx = np.arange(ndata)
        np.random.shuffle(idx)
        tr_idx = idx[:int(np.round(ndata*training/100.))]
        va_idx = idx[int(np.round(ndata*training/100.)):\
            int(np.round(ndata*(training+validation)/100.))]
        te_idx = idx[int(np.round(ndata*(training+validation)/100.)):]        

        if doption == 'bias':
            dnorm = num_mr*ndata/2.
            logging.info("Start to generate mass ratio %.1f data" % massratio[i])
        for j in xrange(ndata):
            m1 = m1_list[j]

            if doption == 'bias':
                if j == 0:
                    mr = massratio[i]
                    m2 = min_mass
                    m1 = m2*mr
                else:
                    mr = massratio[i]
                    m2 = m1/mr
            else:
                m2 = m2_list[j]
                mr = m1/m2

            totmass = m1+m2
            chirpmass = (m1*m2)**(3./5.)/(totmass)**(1./5.)
            t, hp, hc = generateWFdata(m1, m2, optsdict)

            if m1 >= min_mass and m2 >= min_mass:
                if j in tr_idx or j == 0 or j == ndata-1:
                    hp_ip_var = outputFs[0]
                    hp_tg_var = outputFs[1]
                    hc_ip_var = outputFs[2]
                    hc_tg_var = outputFs[3]
                    ms_if_var = outputFs[4]
                elif j in va_idx:
                    hp_ip_var = outputFs[5]
                    hp_tg_var = outputFs[6]
                    hc_ip_var = outputFs[7]
                    hc_tg_var = outputFs[8]
                    ms_if_var = outputFs[9]
                else:
                    hp_ip_var = outputFs[10]
                    hp_tg_var = outputFs[11]
                    hc_ip_var = outputFs[12]
                    hc_tg_var = outputFs[13]
                    ms_if_var = outputFs[14]

                fpi = open(hp_ip_var, 'a')
                fpt = open(hp_tg_var, 'a')
                fci = open(hc_ip_var, 'a')
                fct = open(hc_tg_var, 'a')
                fcm = open(ms_if_var, 'a')

                hp_ISCO = get_ISCO_index(hp, optsdict['samplerate'], totmass, optsdict['f_min'], t)
                hc_ISCO = get_ISCO_index(hc, optsdict['samplerate'], totmass, optsdict['f_min'], t)

                hpi = hp[:hp_ISCO]
                hci = hc[:hc_ISCO]
                hpt = hp[hp_ISCO-1:]
                hct = hc[hc_ISCO-1:]

                abs_hp = abs(hp)
                abs_hc = abs(hc)
                temp_max_hp = max(abs_hp)
                temp_max_hc = max(abs_hc)
                if temp_max_hp > max_hp_amp:
                    max_hp_amp = temp_max_hp
                if temp_max_hc > max_hc_amp:
                    max_hc_amp = temp_max_hc

                if j%400 == 0 and j != 0:
                    if doption == 'bias': 
                        per = float(nth_data)/dnorm*100
                    else:
                        per = float(j+1)/ndata*100
                    logging.info("Generated GW waveforms of m1:{}, m2:{} binary".format(m1, m2))
                    logging.info("Maximum strain amplitude hp:{}, hc:{}".format(max_hp_amp, max_hc_amp))
                    logging.info("%2.2f/100%% is done" % per)
                nth_data += 1

                for pi in hpi:
                    fpi.write(' %.16e' % pi)
                for ci in hci:
                    fci.write(' %.16e' % ci)
                for pt in hpt:
                    fpt.write(' %.16e' % pt)
                for ct in hct:
                    fct.write(' %.16e' % ct)
                fcm.write('%.16f %.16f %.2f %.16e' % (m1, m2, mr, chirpmass))
                fpi.write('\n')
                fci.write('\n')
                fpt.write('\n')
                fct.write('\n')
                fcm.write('\n')
    
    with open(os.path.join(optsdict['outdir'],'maximum_amplitude'), 'w') as f:
        f.write('%.16e %.16e' % (max_hp_amp, max_hc_amp))

if __name__ == '__main__':
    main()
