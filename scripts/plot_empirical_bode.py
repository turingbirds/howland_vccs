#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.font_manager as fm
# mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# mpl.rcParams["text.usetex"] = True
# mpl.rcParams["text.latex.unicode"] = True
#mpl.rcParams.update({'mathtext.default':  'cmr10' })
#mpl.rcParams.update({'mathtext.it':  'cmmi' })
#mpl.rcParams['axes.unicode_minus'] = False
#mpl.rcParams['font.family'] = 'cmr10'
mpl.rcParams['font.family'] = 'CMU Serif'

mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.serif"] = [mpl.rcParams['font.family']] + mpl.rcParams["font.serif"]
mpl.rcParams['axes.labelsize'] = 20.
mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.linewidth'] = 0.5
# mpl.rcParams['grid.dashes'] = (5, 5)
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['lines.dashed_pattern'] = [3, 3]
mpl.rcParams['lines.dashdot_pattern'] = [3, 5, 1, 5]
mpl.rcParams['lines.dotted_pattern'] = [1, 3]
mpl.rcParams['lines.scale_dashes'] = False
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16

# mpl.rcParams.update({'mathtext.default':  'cmr10' })
# mpl.rcParams.update({'mathtext.it':  'cmmi' })
mpl.use("PDF")
# mpl.rcParams['font.family'] = 'Computer Modern'

import matplotlib.ticker as plticker
from matplotlib.ticker import FormatStrFormatter

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.stats


#
# numerical data (simulated transfer function)
#

freqs_sim = np.loadtxt("../data/transfer_func_sim/sim_bode_freqs.txt")
amp_se_sim = np.abs(np.loadtxt("../data/transfer_func_sim/sim_bode_transfer_func_single-ended_[C_L=0.0]_[C_FB=6.600000E-12].txt", dtype=np.complex))
phase_se_sim = np.angle(np.loadtxt("../data/transfer_func_sim/sim_bode_transfer_func_single-ended_[C_L=0.0]_[C_FB=6.600000E-12].txt", dtype=np.complex))
amp_diff_sim = np.abs(np.loadtxt("../data/transfer_func_sim/sim_bode_transfer_func_differential_[C_L=0.0]_[C_FB=3.300000E-12].txt", dtype=np.complex))
phase_diff_sim = np.angle(np.loadtxt("../data/transfer_func_sim/sim_bode_transfer_func_differential_[C_L=0.0]_[C_FB=3.300000E-12].txt", dtype=np.complex))



#
# data from LT1211 with CFB = 3.3, 6.6 pF
# measured as voltage directly on load
#

freqs	=		np.array([1E3,
				          10E3,20E3,30E3,40E3,50E3,60E3,70E3,80E3,90E3,
				          100E3,200E3,300E3,400E3,500E3,600E3,700E3,800E3,900E3,
				          1E6])	#	[Hz]
amp_se = 		np.array([20.2,
					      20, 19.4, 18.6, 17, 15.8, 14.6, 13.6,12.4,11.6,
					      10.7,6.1,4.1,3.1,2.4,2.,1.7,1.5,1.3,
					      1.2
					      ])/20.2*20
phase_se	=	np.array([0,
					      10, 19, 29, 34, 40, 48,52,59,62,
					      62,84,90,100,104,112,113,119,123,
					      129
					      ])
amp_diff	=	np.array([20.4,
                          20.4, 20., 18.8, 18.8, 18.4, 17.8,17.2,16.4,15.8,
                          15.2, 10,7,5.4,4.4,3.6,3.1,2.6,2.4,
                          2.1])/20.4*20
phase_diff	=	np.array([0.,
					      6, 16, 20, 22, 27,33,35,38,40,
					      45,70,81,91,100,103,106,113,113,
					      115])

load_R = 10E3
amp_se /= load_R		# convert from VL to IL
amp_diff /= load_R		# convert from VL to IL

# gain_se = 1/250.*load_R		# for voltage-based measurement
# gain_diff = 1/250.*load_R		# for voltage-based 
gain_se = 1/250.		# for current-based measurement
gain_diff = 1/250.		# for current-based measurement

amp_se /= amp_se[0] / gain_se
amp_diff /= amp_diff[0] / gain_diff

def parallel(x, y):
	return 1 / (1 / x + 1 / y)



def get_H_se_analytic(freqs, CFB):
	R1 = 10250.
	R2 = 10250.
	R3 = 10250.
	R4A = 10000.
	R4B = 250.
	RL = 10E3
	_R2 = parallel(R2, 1/(1j*2*np.pi*freqs*CFB))

	return .5 * (_R2 * (R3 + 2 * R4A) + R1 * (R4A + R4B)) / (_R2 * R3 * RL - R1 * (R3 * R4B + R4B * RL + R4A * (R4B + RL)))


def get_H_diff_analytic(freqs, CFB):
	R1 = 10250.
	R3 = 10250.
	R2A = 10E3
	R4A = 10E3
	R2B = 250.
	R4B = 250.
	RL = 10E3
	_R1 = parallel(R1, 1/(2*np.pi*freqs*1j*CFB))
	_R3 = parallel(R3, 1/(2*np.pi*freqs*1j*CFB))
	return (_R1 * (R2A + _R3) + R4A * (R2A + _R3) - R2B * R4B) / (RL * ((R2A + R2B) * (R4A + R4B) - _R1 * _R3) + R2A * (R4B * (_R1 + R4A) + R2B * (R4A + R4B)) + R2B * R4A * (_R3 + R4B))


freqs_analytic = np.linspace(freqs[0], freqs[-1], 1000)
H_se_analytic = get_H_se_analytic(freqs_analytic, CFB=6.6E-12)
amp_se_analytic = np.abs(H_se_analytic)
phase_se_analytic = np.pi - np.angle(H_se_analytic)
phase_se_analytic *= 360. / (2 * np.pi)

H_diff_analytic = get_H_diff_analytic(freqs_analytic, CFB=3.3E-12)
amp_diff_analytic = np.abs(H_diff_analytic)
phase_diff_analytic = np.pi - np.angle(H_diff_analytic)
phase_diff_analytic *= 360. / (2 * np.pi) - 180


def log_interp(zz, xx, yy):
	"""interpolation between points on a log-log axis (wrapper for scipy interp1d)"""
	assert np.all(np.diff(xx) > 0)
	logz = np.log10(zz)
	logx = np.log10(xx)
	logy = np.log10(yy)
	interp = sp.interpolate.interp1d(logx, logy, kind="linear")
	interp = np.power(10., interp(logz))
	return interp


def find_x(y, xx, yy, epsilon=1E-6, debug=False):
	if debug:
		print("xx = " + str(xx))
		print("yy = " + str(yy))
		print("y = " + str(y))
	if np.array(y).size == 1:
		y = np.array([y])
	x = np.zeros_like(y)
	for i, _y in enumerate(y):
		if debug:
			print("\t_y = " + str(_y))
		if _y < np.amin(yy) or _y > np.amax(yy):
			x[i] = np.nan
		elif _y == yy[0]:
			x[i] = xx[0]
		elif _y == yy[-1]:
			x[i] = xx[-1]
		else:
			x_guess = np.mean(xx)
			y_guess = np.mean(yy)
			step = np.diff(xx) / 2.
			while np.abs(y[i] - y_guess) > epsilon:
				y_guess = log_interp(x_guess, xx, yy)

				if debug:
					print("\t\tCurrent x_guess = " + str(x_guess) + " y_guess = " + str(y_guess))

				if y_guess > _y:
					if debug:
						print("\t\ty_guess > _y")
					if yy[-1] > yy[0]:
						x_guess -= step
					else:
						x_guess += step
				else:
					if debug:
						print("\t\ty_guess <= _y")
					if yy[-1] > yy[0]:
						x_guess += step
					else:
						x_guess -= step
				step /= 2.
				if debug:
					print("\t\tstep = " + str(step))

			x[i] = x_guess
	
	return x


if __name__ == "__main__":

	# fig, ax = plt.subplots(2, 1, figsize=(8, 5))
	fig = plt.figure(figsize=(8, 5)) 
	gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[3, 1])
	ax = plt.subplot(gs[0])
	ax2 = plt.subplot(gs[1])
	ax.set_xlim(np.amin(freqs), np.amax(freqs))
	ax2.set_xlim(np.amin(freqs), np.amax(freqs))
	ax.set_ylim(1E-1, 1.01E1)
	ax2.set_xscale('log') 

	ax.plot(freqs_analytic, 1E3 * amp_se_analytic, linestyle="--", marker=None, color="#f36e00", label="Single-ended (analytic)", linewidth=2, zorder=99)#, hatch="-", linewidth=10.0)
	ax.plot(freqs_analytic, 1E3 * amp_diff_analytic, linestyle="--", marker=None, color="#4671d5", label="Differential (analytic)", linewidth=2, zorder=99)#, hatch="-", linewidth=10.0)
	ax2.plot(freqs_analytic, phase_se_analytic, linestyle="--", color="#f36e00", label="Single-ended (analytic)", linewidth=2, zorder=99, alpha=1)#, hatch="-", linewidth=10.0)
	ax2.plot(freqs_analytic, phase_diff_analytic, linestyle="--", color="#4671d5", label="Differential (analytic)", linewidth=2, zorder=99, alpha=1)#, edgecolor="#ffffff", hatch="-", linewidth=10.0)

	ax.plot(freqs_sim, 1E3 * amp_se_sim, marker=None, color="#f36e00", label="Single-ended (numeric)", linewidth=2, zorder=99)#, hatch="-", linewidth=10.0)
	ax.plot(freqs_sim, 1E3 * amp_diff_sim, marker=None, color="#4671d5", label="Differential (numeric)", linewidth=2, zorder=99)#, hatch="-", linewidth=10.0)
	ax2.plot(freqs_sim, phase_se_sim, linestyle="-", color="#f36e00", label="Single-ended (numeric)", linewidth=2, zorder=99, alpha=1)#, hatch="-", linewidth=10.0)
	ax2.plot(freqs_sim, phase_diff_sim, linestyle="-", color="#4671d5", label="Differential (numeric)", linewidth=2, zorder=99, alpha=1)#, edgecolor="#ffffff", hatch="-", linewidth=10.0)

	ax.plot(freqs, 1E3 * amp_se, marker="o", markersize=1.2*6, color="#f36e00", label="Single ended (empirical)", linestyle=":", linewidth=2, zorder=99, markeredgewidth=.5, markeredgecolor="black")#, hatch="-", linewidth=10.0)
	ax2.plot(freqs, phase_se, marker="o", markersize=1.2*6, color="#f36e00", label="Single-ended (empirical)", linestyle=":", linewidth=2, zorder=99, alpha=1, markeredgewidth=.5, markeredgecolor="black")#, hatch="-", linewidth=10.0)

	ax.plot(freqs, 1E3 * amp_diff, marker="D", markersize=1.2*5.5, color="#4671d5", label="Differential (empirical)", linestyle=":", linewidth=2, zorder=99, markeredgewidth=.5, markeredgecolor="black")#, edgecolor="#ffffff", hatch="-", linewidth=10.0)
	ax2.plot(freqs, phase_diff, marker="D", markersize=1.2*5.5, color="#4671d5", label="Differential (empirical)", linestyle=":", linewidth=2, zorder=99, alpha=1, markeredgewidth=.5, markeredgecolor="black")#, edgecolor="#ffffff", hatch="-", linewidth=10.0)
	ax.set_yscale('log')
	ax.set_xscale('log')

	amp_diff_0 = amp_diff[0]
	amp_se_0 = amp_se[0]
	amp_diff_0_db = 10 * np.log10(amp_diff_0)
	amp_se_0_db = 10 * np.log10(amp_se_0)

	amp_diff_at_3_db = 10. ** ((amp_diff_0_db - 3.) / 10.)
	amp_se_at_3_db = 10. ** ((amp_se_0_db - 3.) / 10.)


	idx_lo = np.argmin((amp_se_at_3_db - amp_se)**2)
	if amp_se[idx_lo] < amp_se_at_3_db or idx_lo == amp_se.size - 1:
		idx_lo -= 1
	freq_se_3_db = find_x(amp_se_at_3_db, np.array([freqs[idx_lo], freqs[idx_lo+1]]), np.array([amp_se[idx_lo], amp_se[idx_lo+1]]))
	print("Single-ended -3 dB point at f = " + str(freq_se_3_db) + " Hz")

	idx_lo = np.argmin((amp_diff_at_3_db - amp_diff)**2)
	if amp_se[idx_lo] < amp_diff_at_3_db:
		idx_lo -= 1
	freq_diff_3_db = find_x(amp_diff_at_3_db, np.array([freqs[idx_lo], freqs[idx_lo+1]]), np.array([amp_diff[idx_lo], amp_diff[idx_lo+1]]))
	print("Differential -3 dB point at f = " + str(freq_diff_3_db) + " Hz")

	ax.plot(np.squeeze(np.tile(freq_se_3_db, [1, 2])), [ax.get_ylim()[0], 1E3 * amp_se_at_3_db], linestyle="--", color="#f36e00", linewidth=2, zorder=99)#, edgecolor="#ffffff", hatch="-", linewidth=10.0)
	ax2.plot(np.squeeze(np.tile(freq_se_3_db, [1, 2])), ax2.get_ylim(), linestyle="--", color="#f36e00", linewidth=2, zorder=99)#, edgecolor="#ffffff", hatch="-", linewidth=10.0)
	ax.plot(freq_se_3_db, 1E3 * amp_se_at_3_db, marker="x", markersize=1.5*5.5, color="#f36e00", linewidth=2, markeredgewidth=2, zorder=99, alpha=.8)#, edgecolor="#ffffff", hatch="-", linewidth=10.0)

	ax.plot(np.squeeze(np.tile(freq_diff_3_db, [2,1 ])), [ax.get_ylim()[0], 1E3 * amp_diff_at_3_db], linestyle="--", color="#4671d5", linewidth=2, zorder=99)#, edgecolor="#ffffff", hatch="-", linewidth=10.0)
	ax2.plot(np.squeeze(np.tile(freq_diff_3_db, [2,1 ])), ax2.get_ylim(), linestyle="--", color="#4671d5", linewidth=2, zorder=99)#, edgecolor="#ffffff", hatch="-", linewidth=10.0)
	ax.plot([freq_diff_3_db], [1E3 * amp_diff_at_3_db], marker="x", markersize=1.5*5.5, color="#4671d5", linewidth=20, markeredgewidth=2, zorder=99, alpha=.8)#, edgecolor="#ffffff", hatch="-", linewidth=10.0)

	ax.plot([ax.get_xlim()[0], freq_diff_3_db], [1E3 * amp_diff_at_3_db, 1E3 * amp_diff_at_3_db], linestyle="--", marker=None, color="#999999", linewidth=2, zorder=90, alpha=.8)#, edgecolor="#ffffff", hatch="-", linewidth=10.0)

	# ax.loglog(freqs, r_se, marker="o", color="#f36e00", label="Single ended", linewidth=2)#, hatch="-", linewidth=10.0)
	ax2.set_xlabel("Frequency [Hz]", size=18.)
	ax.set_ylabel("Gain [mA/V]", size=18.)
	ax2.set_ylabel(r"Phase", size=18.)

	# fig.suptitle('VCCS output resistance')
	# ax[0].set_xlim(ax[0].get_xlim()[0], np.amax(bin_edges)/1E6/2.)
	# ax[1].set_xlim(ax[0].get_xlim()[0], 2*ax[0].get_xlim()[1])
	# ax.set_xlim(2E3, 5E4)
	fig.canvas.draw()
	ax.set_xticklabels([])

	for _ax in [ax, ax2]:
		#_ax.grid(b=True, which='both', zorder=0, color="#999999", linestyle=(0, (6, 6)))
		#_ax.grid(b=True, which='both')
		_ax.grid(b=True, which='major')
		_ax.grid(b=True, which='minor', color="#cccccc")

	ax2.set_ylim(0, 121)
	loc = plticker.MultipleLocator(base=60) # this locator puts ticks at regular intervals
	ax2.yaxis.set_major_locator(loc)
	ax2.set_yticklabels([str(int(i)) + r"${}^{\circ}$" for i in ax2.get_yticks()])

	loc = plticker.NullLocator()
	ax2.yaxis.set_minor_locator(loc)
	ax.minorticks_on()
	ax2.minorticks_on()

	for _ax in [ax, ax2]:
		l, b, w, h = _ax.get_position().bounds
		_ax.set_position([l, b + .05, w, h])

	leg = ax.legend(loc="lower left", handlelength=1.8, fontsize=14., fancybox=True, framealpha=1)

	# remove errorbars from legend
	# get handles
	handles, labels = ax.get_legend_handles_labels()
	# remove the errorbars
	# handles = [h[0] for h in handles]
	h = handles
	# use them in the legend
	ax.legend(handles, labels, loc="lower left", handlelength=1.8, fontsize=14., fancybox=True, framealpha=1)

	fig.savefig("/tmp/bode_empirical.png")
	fig.savefig("/tmp/bode_empirical.pdf")
	plt.close(fig)
