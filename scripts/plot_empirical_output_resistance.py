#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run as:

.. code-block:: sh

   python3 -i plot_empirical_output_resistance.py -- `find ../data/Z_hat_out_measurements/data_ref -name "*.txt" | sort -n`
"""

import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
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

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter

import glob
import numpy as np
import os
import re
import scipy.stats
import scipy.optimize
import sys
import matplotlib.pyplot as plt

from vccs_measurement import Z_o_from_ILa_ILb


#
#   parameters
#

RLa = 10E3
RLb = 11E3


#
#   flags for what to run
#   set ref_ckt = True when plotting the reference circuit data, False otherwise
#

ref_ckt = False
do_curve_fitting = False
if not ref_ckt:
	do_curve_fitting = True
plot_theoretical_curves = False
if ref_ckt:
	plot_theoretical_curves = True
do_plot_sensitivity = False
plot_curves_tolerance = False


#
# 	read empirical data from text files
#

R_out = {}
freqs = {}
Z_o = {}

if len(sys.argv) > 1:
    for fn in sys.argv[1:]:
        circuit_name = os.path.splitext(os.path.basename(fn))[0]
        freqs[circuit_name] = []
        Z_o[circuit_name] = {}
        file = open(fn, "r")
        try:
            raw_a = None
            raw_b = None
            for i, l in enumerate(file.readlines()):
                print(l)
                if l[0] == "%":
                    freq = float(l.split()[3])
                if l.split()[0] == "raw_a":
                    raw_a = l.split()[1]
                    ILa = float(raw_a)
                if l.split()[0] == "raw_b":
                    raw_b = l.split()[1]
                    ILb = float(raw_b)
                if (raw_a is not None) and (raw_b is not None):
                    if not freq in Z_o[circuit_name].keys():
                        Z_o[circuit_name][freq] = []
                    Z_o[circuit_name][freq].append(Z_o_from_ILa_ILb(RLa, RLb, ILa, ILb))
                    raw_a = None
                    raw_b = None
        finally:
            file.close()


#
# 	compute means and standard deviations
#

n_outlier_skip = 2

R_out_mean = {}
R_out_std = {}

min_freq = np.inf
max_freq = -np.inf
R_out_means = {}
R_out_stds = {}
print("* Statistics:")

for circuit_name in Z_o.keys():
	freqs[circuit_name] = []
	R_out_means[circuit_name] = []
	R_out_stds[circuit_name] = []
	print("  circuit: " + circuit_name)
	for freq in np.sort(list(Z_o[circuit_name].keys())):
		print("  frequency: " + "{0:E}".format(freq))

		if not circuit_name in R_out_mean.keys():
			R_out_mean[circuit_name] = {}
			R_out_std[circuit_name] = {}

		#
		# 	outlier removal
		#

		_R_out = Z_o[circuit_name][freq]
		if len(_R_out) > n_outlier_skip*2:
			_R_out = np.sort(np.abs(_R_out))[n_outlier_skip:-n_outlier_skip]

		R_out_mean[circuit_name][freq] = np.mean(np.abs(_R_out))
		R_out_std[circuit_name][freq] = np.std(np.abs(_R_out))

		print("  ---> mean = " + str(R_out_mean[circuit_name][freq]))
		print("  ---> std = " + str(R_out_std[circuit_name][freq]))

		#
		# 	store in a format more suitable to matplotlib's taste
		#

		min_freq = min(min_freq, freq)
		max_freq = max(max_freq, freq)

		freqs[circuit_name].append(freq)
		R_out_means[circuit_name].append(R_out_mean[circuit_name][freq])
		R_out_stds[circuit_name].append(R_out_std[circuit_name][freq])



#
# 	plotting
#

colours = {
	"1MR_10nF": "#a379c9",
	"1MR_100pF" : "#ffeb3a",
	"100kR_10nF" : "#94fce7",
	"100kR_100pF" : "#9cfc97",

	"single-ended" : "#f36e00",
	"differential" : "#4671d5"
}

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
plt.subplots_adjust(bottom=.18, top=.95)

def synthetic_Z_o(Ro, Co, freqs, method=None, RLa_delta=0., RLb_delta=0.):
	assert method in ["voltage", "current", None]

	if method is None:
		assert RLa_delta == 0 and RLb_delta == 0	# method does not matter for perfect RLa and RLb

	RLa = 10E3
	RLb = 11E3
	I_L = 1E-3

	if method == "voltage":
		V_La_expected = Mag_V_L(I_L, RLa + RLa_delta, Ro, 2*np.pi*freqs, Co)
		V_Lb_expected = Mag_V_L(I_L, RLb + RLb_delta, Ro, 2*np.pi*freqs, Co)
		R_out_expected = Z_o_from_VLa_VLb(RLa, RLb, V_La_expected, V_Lb_expected)
	else:
		I_La_expected = Mag_I_L(I_L, RLa + RLa_delta, Ro, 2*np.pi*freqs, Co)
		I_Lb_expected = Mag_I_L(I_L, RLb + RLb_delta, Ro, 2*np.pi*freqs, Co)
		R_out_expected = Z_o_from_ILa_ILb(RLa, RLb, I_La_expected, I_Lb_expected)

	return np.abs(R_out_expected)


def plot_sensitivity(freq, max_delta=10., method="voltage", fn_snip=""):

	assert method in ["voltage", "current"]
	fn_snip = "_[method=" + method + "]" + fn_snip

	fig, ax = plt.subplots(1, 1, figsize=(8, 4))
	plt.subplots_adjust(bottom=.18, top=.95)
	cm = LinearSegmentedColormap.from_list('twilight', [[0x00/255, 0x99/255, 0], [0x00/255, 0xcc/255, 0], [0x99/255, 0xaa/255, 0x66/255], [0xff/255, 0xff/255, 0x00/255]])
	norm = mpl.colors.LogNorm(vmin=np.amin(freq), vmax=np.amax(freq))

	for i in range(100):
		pos_delta = scipy.stats.uniform.rvs(-max_delta, 2*max_delta)
		neg_delta = scipy.stats.uniform.rvs(-max_delta, 2*max_delta)

		R_out_expected = synthetic_Z_o(Ro=10E6, Co=1E-9, freqs=freq, method=method)
		R_out_actual = synthetic_Z_o(Ro=10E6, Co=1E-9, freqs=freq, RLa_delta=pos_delta, RLb_delta=neg_delta, method=method)

		ax.scatter(R_out_expected, R_out_actual, c=freq, norm=norm, s=20, cmap=cm, linewidth=2, zorder=2, label="1 M$\Omega$, 10 nF")

	ax.plot((np.amin(R_out_expected), np.amax(R_out_expected)), (np.amin(R_out_expected), np.amax(R_out_expected)), color="black", linewidth=2, zorder=10, linestyle="--")

	ax_colorbar = fig.add_axes([.925, .65, .025, .2])	# [left, bottom, width, height]
	ax_colorbar.spines['top'].set_visible(False)
	ax_colorbar.spines['right'].set_visible(False)
	ax_colorbar.spines['bottom'].set_visible(False)
	ax_colorbar.spines['left'].set_visible(False)

	cb1 = mpl.colorbar.ColorbarBase(ax_colorbar, cmap=cm, norm=norm, orientation='vertical')
	cb1.set_label('$f$')
	ax_colorbar.xaxis.set_minor_locator(mpl.ticker.NullLocator())
	ax_colorbar.yaxis.set_minor_locator(mpl.ticker.NullLocator())

	ax.set_xlabel("expected $\hat{Z}_{out}$")
	ax.set_ylabel("actual $\hat{Z}_{out}$")

	ax.set_xscale("log")
	ax.set_yscale("log")

	ax.spines['left'].set_linewidth(1.5)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_linewidth(1.5)
	ax.xaxis.set_ticks_position("bottom")
	ax.yaxis.set_ticks_position("left")
	ax.grid(b=True, which='both', zorder=1, color="#999999")

	ax.yaxis.set_major_locator(mpl.ticker.LogLocator())
	ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
	ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(16)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(16)

	im_fn = "/tmp/sensitivity" + fn_snip + ".png"
	print("* Saving " + im_fn)
	fig.savefig(im_fn)
	#fig.savefig("/tmp/sensitivity" + fn_snip + ".pdf")
	plt.close(fig)

#################

if do_plot_sensitivity:
	freq = np.logspace(np.log10(min_freq/10.), np.log10(max_freq), 100)
	plot_sensitivity(freq, method="voltage")
	plot_sensitivity(freq, method="current")

#################

if do_curve_fitting:
	for i, circuit_name in enumerate(R_out_means.keys()):
		if circuit_name.startswith("SE"):
			color="#f36e00"
		elif circuit_name.startswith("diff"):
			color="#4671d5"
		else:
			color = colours[circuit_name]

	freqs_sim = np.logspace(np.log10(min_freq), np.log10(max_freq), 2000)

	# curve fitting for single-ended

	def f(x, *params):
		R_o_est, C_o_est = x
		circuit_name = params[0]
		Z_o = synthetic_Z_o(Ro=R_o_est, Co=C_o_est, freqs=np.array(freqs[circuit_name]))
		SSE = 0.
		for i, Z_o_empirical in enumerate(R_out_means[circuit_name]):
			SSE += (np.log10(Z_o[i]) - np.log10(Z_o_empirical))**2
		return SSE


	for circuit_name in R_out_means.keys():

		if circuit_name == "SE":
			R_o_est = 1.1E7
			C_o_est = .2E-9
		else:
			R_o_est = 1E8
			C_o_est = .1E-9

		res = scipy.optimize.minimize(f, (R_o_est, C_o_est), options={'disp': True}, method='Nelder-Mead', tol=1e-6, args=(circuit_name))

		R_o_est = res["x"][0]
		C_o_est = res["x"][1]

		print("For circuit " + circuit_name + ": R_o_est = " + str(R_o_est/1E6) + " MR, C_o_est = " + str(C_o_est*1E9) + " nF")

		if circuit_name == "SE":
			Z_o = synthetic_Z_o(Ro=R_o_est, Co=C_o_est, freqs=freqs_sim)
			ax.plot(freqs_sim, np.abs(Z_o), color=colours["single-ended"], linewidth=2, zorder=98, label="Single-ended")
		else:
			Z_o = synthetic_Z_o(Ro=R_o_est, Co=C_o_est, freqs=freqs_sim)
			ax.plot(freqs_sim, np.abs(Z_o), color=colours["differential"], linewidth=2, zorder=98, label="Differential")

		leg = ax.legend(loc="upper right", handlelength=1.8, fontsize=16. , fancybox=True, framealpha=1)

#################

if plot_curves_tolerance:
	freqs_sim = np.logspace(np.log10(min_freq/10.), np.log10(10 * max_freq), 2000)
	R_out_expected_100k = synthetic_Z_o(Ro=1000E3, Co=10E-9, freqs=freqs_sim, RLa_delta=-10., RLb_delta=15., method="current")
	ax.plot(freqs_sim, np.abs(R_out_expected_100k), color=colours["100kR_10nF"], linewidth=2, zorder=98, label="current")

	R_out_expected_100k = synthetic_Z_o(Ro=1000E3, Co=10E-9, freqs=freqs_sim, RLa_delta=-10., RLb_delta=15., method="voltage")
	ax.plot(freqs_sim, np.abs(R_out_expected_100k), color=colours["100kR_10nF"], linewidth=2, zorder=98, label="voltage")

#################

if plot_theoretical_curves:
	freqs_sim = np.logspace(np.log10(min_freq/10.), np.log10(10 * max_freq), 2000)

	R_out_expected_100k = synthetic_Z_o(Ro=100E3, Co=100E-12, freqs=freqs_sim)
	ax.plot(freqs_sim, R_out_expected_100k, color=colours["100kR_100pF"], linewidth=2, zorder=98, label="100kR_100pF")

	R_out_expected_100k = synthetic_Z_o(Ro=100E3, Co=10E-9, freqs=freqs_sim)
	ax.plot(freqs_sim, np.abs(R_out_expected_100k), color=colours["100kR_10nF"], linewidth=2, zorder=98, label="100kR_10nF")

	R_out_expected_1M = synthetic_Z_o(Ro=1E6, Co=100E-12, freqs=freqs_sim)
	ax.plot(freqs_sim, R_out_expected_1M, color=colours["1MR_100pF"], linewidth=2, zorder=98, label="1MR_100pF")

	R_out_expected_1M = synthetic_Z_o(Ro=1E6, Co=10E-9, freqs=freqs_sim)
	ax.plot(freqs_sim, R_out_expected_1M, color=colours["1MR_10nF"], linewidth=2, zorder=98, label="1MR_10nF")

	ax.set_ylim(1E3, 1.5E6)
	# ax.set_ylim(1E3, 1.3E6)
	ax.set_xlim(1E2-2E1, 1E5+2E4)

#################

for i, circuit_name in enumerate(R_out_means.keys()):
	if circuit_name.startswith("SE"):
		color="#f36e00"
		marker="o"
	elif circuit_name.startswith("diff"):
		color="#4671d5"
		marker="D"
	else:
		color = colours[circuit_name]
		if circuit_name == "1MR_100pF":
			marker="^"
		elif circuit_name == "1MR_10nF":
			marker="v"
		elif circuit_name == "100kR_100pF":
			marker="^"
		elif circuit_name == "100kR_10nF":
			marker="v"

	ax.plot(freqs[circuit_name], R_out_means[circuit_name], color=color, linewidth=2, zorder=98, linestyle="none", alpha=.6)#, hatch="-", linewidth=10.0)
	ax.errorbar(freqs[circuit_name], R_out_means[circuit_name], yerr=R_out_stds[circuit_name], marker=marker, markersize=1.2*6, color=color, label=circuit_name, linewidth=2, zorder=99, markeredgecolor="k", markeredgewidth=.5, linestyle="none")#, hatch="-", linewidth=10.0)

ax.set_yscale('log')
ax.set_xscale('log')

ax.set_ylabel("$\hat{Z}_{out}$")
ax.set_xlabel("Frequency [Hz]")

ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.grid(b=True, which='both', zorder=0, color="#999999")

for tick in ax.xaxis.get_major_ticks():
	tick.label.set_fontsize(16)

for tick in ax.yaxis.get_major_ticks():
	tick.label.set_fontsize(16)

if plot_theoretical_curves:
	handles_orig, labels_orig = ax.get_legend_handles_labels()

	ax.plot([-99], [0], color=colours["1MR_100pF"], linewidth=2, zorder=98, label="1 M$\Omega$, 100 pF", marker="^", linestyle="-", markeredgecolor="k", markeredgewidth=.5)
	ax.plot([-99], [0], color=colours["1MR_10nF"], linewidth=2, zorder=98, label="1 M$\Omega$, 10 nF", marker="v", linestyle="-", markeredgecolor="k", markeredgewidth=.5)
	ax.plot([-99], [0], color=colours["100kR_100pF"], linewidth=2, zorder=98, label="100 k$\Omega$, 100 pF", marker="^", linestyle="-", markeredgecolor="k", markeredgewidth=.5)
	ax.plot([-99], [0], color=colours["100kR_10nF"], linewidth=2, zorder=98, label="100 k$\Omega$, 10 nF", marker="v", linestyle="-", markeredgecolor="k", markeredgewidth=.5)

	leg = ax.legend(loc="lower left", handlelength=1.8, fontsize=16. , fancybox=True, framealpha=1)

	# remove errorbars from legend
	# get handles
	handles, labels = ax.get_legend_handles_labels()
	# remove the errorbars
	# handles = [h[0] for h in handles]

	handles = [h for h in handles if not h in handles_orig]#handles[-4:]
	labels = [h for h in labels if not h in labels_orig]#handles[-4:]

	# use them in the legend
	ax.legend(handles, labels, loc="lower left", handlelength=1.8, fontsize=16. , fancybox=True, framealpha=1)
	# leg.get_frame().set_alpha(0.5)

if do_curve_fitting:
	"""empirical results legend"""
	handles_orig, labels_orig = ax.get_legend_handles_labels()
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	ax.plot([-99], [1E-99], color=colours["differential"], linewidth=2, zorder=98, label="Differential", marker="D", linestyle="-", markeredgewidth=.5, markeredgecolor="k")
	ax.plot([-99], [1E-99], color=colours["single-ended"], linewidth=2, zorder=98, label="Single-ended", marker="o", linestyle="-", markeredgewidth=.5, markeredgecolor="k")
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

	leg = ax.legend(loc="upper right", handlelength=1.8, fontsize=16. , fancybox=True, framealpha=1)

	# remove errorbars from legend
	# get handles
	handles, labels = ax.get_legend_handles_labels()
	# remove the errorbars
	new_handles_idx = [i for i, h in enumerate(handles) if not h in handles_orig]
	handles = [handles[i] for i in new_handles_idx]
	labels = [labels[i] for i in new_handles_idx]
	# use them in the legend
	ax.legend(handles, labels, loc="upper right", handlelength=1.8, fontsize=16. , fancybox=True, framealpha=1)

im_fn = "/tmp/output_resistance_empirical.png"
print("* Saving " + im_fn)
fig.savefig(im_fn)
fig.savefig("/tmp/output_resistance_empirical_ref.pdf")
plt.close(fig)

