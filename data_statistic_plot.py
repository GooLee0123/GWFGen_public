import os
import logging

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

from wavepy.waves import DatPrep

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)

sample_rate = 4096.

dtype = 'bias'
fdir = './data/IMR_'+dtype
save_dir = './Plots/'+dtype

if not os.path.exists(save_dir):
	os.makedirs(save_dir)

train_file = fdir+'/IMR_hp_training_input.dat'
val_file = fdir+'/IMR_hp_validation_input.dat'
test_file = fdir+'/IMR_hp_test_input.dat'

datprep = DatPrep(fdir)

train, val = datprep.load_split(train_file, val_file)
train, val = datprep.tolist_split()
train, val = datprep.tofloat_split()
test = datprep.load(test_file)
test = datprep.tolist()
test = datprep.tofloat()

i1 = train[0]
i2 = val[-1]
i3 = test[int(len(test)/2)]

train_tgt = fdir+'/IMR_hp_training_target.dat'
val_tgt = fdir+'/IMR_hp_validation_target.dat'
test_tgt = fdir+'/IMR_hp_test_target.dat'

traint, valt = datprep.load_split(train_tgt, val_tgt)
traint, valt = datprep.tolist_split()
traint, valt = datprep.tofloat_split()
traint, valt = datprep.nonzero_split()
testt = datprep.load(test_tgt)
testt = datprep.tolist()
testt = datprep.tofloat()
testt = datprep.nonzero()

t1 = traint[0]
t2 = valt[-1]
t3 = testt[int(len(testt)/2)]

train_m = fdir+'/mass_info_training.dat'
val_m = fdir+'/mass_info_validation.dat'
test_m = fdir+'/mass_info_test.dat'

train_mass = np.genfromtxt(train_m).T
val_mass = np.genfromtxt(val_m).T
test_mass = np.genfromtxt(test_m).T

m1 = train_mass[3][0]
m2 = val_mass[3][-1]
m3 = test_mass[3][len(test)/2]

for i in range(len(traint)):
	train[i] = len(train[i])
	traint[i] = len(traint[i])

for j in range(len(valt)):
	val[j] = len(val[j])
	valt[j] = len(valt[j])

for k in range(len(testt)):
	test[k] = len(test[k])
	testt[k] = len(testt[k])

inputs = train + val + test
targets = traint + valt + testt
mass = np.hstack((train_mass, val_mass, test_mass))

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(8,3))
ax1.hist(inputs, 20)
ax1.grid(which='both', alpha=0.5, ls='dashed')
ax1.set_ylabel("Number")
ax1.set_xlabel("Data size of I waveform")
ax2.hist(targets, 20)
ax2.grid(which='both', alpha=0.5, ls='dashed')
ax2.set_xlabel("Data size of M and R waveform")
ax3.hist(mass[3], 20)
ax3.grid(which='both', alpha=0.5, ls='dashed')
ax3.set_xlabel("Chirp mass")
plt.tight_layout()
f.subplots_adjust(wspace=0)
plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)
plt.savefig(save_dir+"/number_distribution")
logging.info("Plot is saved at %s.png" % (save_dir+"/number_distribution"))

overlap=1

minp_x = np.arange(0,len(i1))/sample_rate
mtgt_x = (len(i1)-overlap+np.arange(0,len(t1)))/sample_rate
plt.figure()
plt.gca().axes.get_yaxis().set_ticks([])
plt.grid(which='both', axis='x', alpha=0.5, ls='--')
plt.plot(minp_x, i1, color='g',
		label="inputs(I)")
plt.plot(mtgt_x, t1, 
		color='b', label="targets(M, R)")
plt.annotate(r'$\mathcal{M}=%.2fM_{\odot}$'%m1, xy=(0.6, 0),
		xytext=(0.6, -0.5e-20),
		arrowprops=dict(facecolor='black', shrink=0.05))

uinp_x = np.arange(0,len(i2))/sample_rate
utgt_x = (len(i2)-overlap+np.arange(0,len(t2)))/sample_rate
uant_x = utgt_x[-1]+0.01
utxt_x = utgt_x[-1]+0.11
plt.plot(uinp_x,
		np.array(i2)+1e-20, color='g')
plt.plot(utgt_x, 
		np.array(t2)+1e-20, color='b')
plt.annotate(r'$\mathcal{M}=%.2fM_{\odot}$'%m2, xy=(uant_x, 1e-20),
		xytext=(utxt_x, 1e-20),
		arrowprops=dict(facecolor='black', shrink=0.05))

binp_x = np.arange(0,len(i3))/sample_rate
btgt_x = (len(i3)-overlap+np.arange(0,len(t3)))/sample_rate
bant_x = btgt_x[-1]+0.01
btxt_x = btgt_x[-1]+0.11
plt.plot(binp_x,
		np.array(i3)-1e-20, color='g')
plt.plot(btgt_x,
		np.array(t3)-1e-20, color='b')
plt.annotate(r'$\mathcal{M}=%.2fM_{\odot}$'%m3, xy=(bant_x, -1e-20),
		xytext=(btxt_x, -1e-20),
		arrowprops=dict(facecolor='black', shrink=0.05))
plt.xlabel("Time")
plt.legend()
plt.savefig(save_dir+"/Strain_inp_tgt")
logging.info("Plot is saved at %s.png" % (save_dir+"/Strain_inp_tgt"))

plt.figure()
plt.grid(which='both', alpha=0.5, ls='--')
plt.scatter(mass[3], inputs, color='g', marker='.', label="input")
plt.scatter(mass[3], targets, color='b', marker='.', label="targets")
plt.yscale('log')
plt.ylabel("log(Data size)")
plt.xlabel("Chirp Mass")
plt.legend()
plt.savefig(save_dir+"/CM_vs_N")
logging.info("Plot is saved at %s.png" % (save_dir+"/CM_vs_N"))

plt.figure()
plt.grid(which='both', alpha=0.5, ls='--')
plt.scatter(mass[0], inputs, color='g', marker='.', label="input")
plt.scatter(mass[0], targets, color='b', marker='.', label="targets")
plt.yscale('log')
plt.ylabel("log(Data size)")
plt.xlabel("Mass1")
plt.legend()
plt.savefig(save_dir+"/M1_vs_N")
logging.info("Plot is saved at %s.png" % (save_dir+"/M1_vs_N"))

plt.figure()
plt.grid(which='both', alpha=0.5, ls='--')
plt.scatter(mass[1], inputs, color='g', marker='.', label="input")
plt.scatter(mass[1], targets, color='b', marker='.', label="targets")
plt.yscale('log')
plt.ylabel("log(Data size)")
plt.xlabel("Mass2")
plt.legend()
plt.savefig(save_dir+"/M2_vs_N")
logging.info("Plot is saved at %s.png" % (save_dir+"/M2_vs_N"))

plt.figure()
plt.grid(which='both', alpha=0.5, ls='--')
plt.scatter(mass[0], mass[1], marker='.')
plt.xlabel("Mass1")
plt.ylabel("Mass2")
plt.savefig(save_dir+"/M1_vs_M2")
logging.info("Plot is saved at %s.png" % (save_dir+"/M1_vs_M2"))
