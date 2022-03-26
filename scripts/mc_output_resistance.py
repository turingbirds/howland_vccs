#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run Monte Carlo simulation on resistance mismatch to obtain a distribution for VCCS output resistance"""

import matplotlib as mpl
import matplotlib.font_manager as fm
# mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rcParams['font.family'] = 'cm'
mpl.rcParams["text.usetex"] = True
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 18

#mpl.rcParams["text.latex.unicode"] = True
# mpl.use("PDF")
# mpl.rcParams['font.family'] = 'Computer Modern qwefwqef'
mpl.rcParams['grid.color'] = '#CCCCCC'
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['lines.dashed_pattern'] = [3, 3]
mpl.rcParams['lines.dashdot_pattern'] = [3, 5, 1, 5]
mpl.rcParams['lines.dotted_pattern'] = [1, 3]
mpl.rcParams['lines.scale_dashes'] = False


prop = fm.FontProperties(fname='/tmp/Serif/cmunrm.ttf')

import matplotlib.ticker as plticker
from matplotlib.ticker import FormatStrFormatter

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


def parallel(x, y):
	return 1 / (1 / x + 1 / y)


def sample_resistance(mean, tol=0.1):
	"""
	Parameters
	----------
	tol : float > 0
		tolerance rating in %
	"""
	return scipy.stats.uniform.rvs(loc=mean - tol/100.*mean, scale=2*tol/100.*mean)		# tolerance is taken as std deviation for Gaussian distribution over value
	#return scipy.stats.norm.rvs(loc=mean, scale=tol/100.*mean)		# tolerance is taken as std deviation for Gaussian distribution over value


def make_R_out_hist(R_out, n_bins=30, MAX_R_OUT=5E6):
	# XXX: was 2E6
	n_samples = R_out.size

	l = 0.
	h = 5*np.median(R_out)#MAX_R_OUT
	hist, bin_edges = np.histogram(R_out, bins=n_bins, range=(l, h))
	hist = hist.astype(float) / float(n_samples)

	return hist, bin_edges


def plot_R_out_hist(R_out, fname_snip, title_snip, tol):

	n_samples = R_out.size

	hist, bin_edges = make_R_out_hist(R_out)

	fig = plt.figure(figsize=(8., 3.))
	ax = fig.add_axes([.1, .1, .8, .8])
	ax.bar(bin_edges[:-1]/1E6, hist, width=(bin_edges[1] - bin_edges[0])/1E6)
	fig.suptitle(title_snip + " improved Howland $R_{out}$ (resistor tol = " + str(np.round(tol, 2)) + "\%, min $R_{out}$ = " + "{:+.1E}".format(np.amin(np.abs(R_out)))[1:] + ", \#samples = " + "{0:E}".format(n_samples) + ")")
	ax.set_xlabel(r"$R_{out}~[\mathrm{M}\Omega]$")
	ax.set_ylabel("$P(R_{out})$")
	ax.set_xlim(ax.get_xlim()[0], np.amax(bin_edges)/10E6)
	fig.savefig("/tmp/mc_output_resistance_" + fname_snip + ".png")
	fig.savefig("/tmp/mc_output_resistance_" + fname_snip + ".pdf")
	plt.close(fig)


def plot_R_out_hist_dual(bin_edges, hist_SE, hist_sym, gain, bar_linewidth=.5, title_snip=""):

	fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
	bar_width = (bin_edges[1] - bin_edges[0]) / 1E6 / 2.3
	b = ax.bar(bin_edges[:-1]/1E6 + bar_width / 2., hist_sym, width=bar_width, color="#4671d5", label=u"Symmetric ($\\div$2)")#, edgecolor="#ffffff", hatch="-", linewidth=10.0)
	for _b in b:
		_b.set_linewidth(0.)
	b = ax.bar(bin_edges[:-1]/1E6 - bar_width /2., hist_SE, width=bar_width, color="#f36e00", edgecolor="#ffffff", label="Single ended")#, hatch="-", linewidth=10.0)
	for _b in b:
		_b.set_linewidth(0.)

	ax.set_xlabel(r"$|R_{out}|~[\mathrm{M}\Omega]$") #labelpad=55., 
	ax.set_ylabel("$P(|R_{out}|)$")
	ax.spines['left'].set_linewidth(1.5)
	ax.set_xlim(0., np.amax(bin_edges)/1E6)

	_ax_spine_spacing = .1

	ax.spines['top'].set_position(('axes', -_ax_spine_spacing))
	# ax2.spines['bottom'].set_position(('axes', -_ax_spine_spacing))

	loc = plticker.MultipleLocator(base=.02) # this locator puts ticks at regular intervals
	ax.yaxis.set_major_locator(loc)
	# ax2.yaxis.set_major_locator(loc)
	ax.tick_params(axis='y', which='major', pad=8, width=2., length=4)
	ax.spines['right'].set_visible(False)
	ax.spines["top"].set_visible(False)
	ax.spines["bottom"].set_linewidth(1.5)
	ax.spines["bottom"].set_color("black")
	ax.xaxis.set_ticks_position("none")
	ax.yaxis.set_ticks_position("none")
	ax.tick_params(axis='x', which='minor', pad=8, width=2., length=4, color="black")
	ax.tick_params(axis='x', which='major', pad=8, width=2., length=5., color="black")
	ax.xaxis.set_major_formatter(FormatStrFormatter(r"$%.1f$"))
	ax.grid(b=True, which='both')
	leg = ax.legend(loc="upper right", handlelength=.78, fontsize=16.)
	if title_snip:
		fig.suptitle(title_snip)
	plt.tight_layout()
	for ext in ["png", "pdf"]:
		fig.savefig("/tmp/mc_output_resistance_[gain=" + "{:.2f}".format(gain) + "." + ext)
	plt.close(fig)


#
#   PARAMETERS
#

tol = .1		# [%]
n_samples = int(1E5)
perfect_trim = False
A_OL = np.inf
A_OL_is_infty = np.isinf(A_OL)
n_trials = 10

gain_multiplier_list = [.03125, .0625, .125, .25, .5, 1, 2, 4, 8, 16]
#gain_multiplier_list = [.0625, 1, 8.]
#gain_multiplier_list = [1.]
n_gain_multiplier = len(gain_multiplier_list)
gain_list_SE = np.nan * np.ones((n_gain_multiplier))
gain_list_SE_alt = np.nan * np.ones((n_gain_multiplier))
gain_list_sym = np.nan * np.ones((n_gain_multiplier))
gain_list_FD = np.nan * np.ones((n_gain_multiplier))
median_R_out_SE = np.nan * np.ones((n_gain_multiplier, n_trials))
median_R_out_SE_alt = np.nan * np.ones((n_gain_multiplier, n_trials))
median_R_out_sym = np.nan * np.ones((n_gain_multiplier, n_trials))
median_R_out_fd = np.nan * np.ones((n_gain_multiplier, n_trials))

for trial_idx in range(n_trials):
	for gain_idx, _gain_multiplier in enumerate(gain_multiplier_list):
		# for 4 mA/V
		_LO = 250.
		_HI = 10E3

		### SINGLE ENDED

		R_out_SE = np.empty((n_samples, ))

		for i in range(n_samples):
			R_1 = sample_resistance(_gain_multiplier*_LO + _gain_multiplier*_HI, tol=tol)
			R_2_mean = _LO + _HI
			R_2 = sample_resistance(R_2_mean, tol=tol)
			R_3_mean = _gain_multiplier*_LO + _gain_multiplier*_HI
			R_3 = sample_resistance(R_3_mean, tol=tol)
			R_4a = sample_resistance(_HI, tol=tol)
			R_4b_mean = _LO
			R_4b = sample_resistance(R_4b_mean, tol=tol)

			if perfect_trim:
				R_2 = R_4a + R_4b
			
			alpha = (R_1 + R_2) * (R_3 + R_4a) * R_4b
			beta = R_1 * (R_3 + R_4a) * R_4b
			gamma = (R_1 + R_2) * (R_3 + R_4a + R_4b)
			delta = R_1 * (R_4a + R_4b) - R_2 * R_3

			if A_OL_is_infty:
				R_out_SE[i] = beta / delta
			else:
				R_out_SE[i] = (alpha + A_OL * beta) / (gamma + A_OL * delta)

			R_out_SE[i] = np.abs(R_out_SE[i])

		gain_SE = R_2_mean / (R_3_mean * R_4b_mean)
		print("SE gain = " + str(gain_SE))


		### SINGLE ENDED (alternate method)

		R_out_SE_alt = np.empty((n_samples, ))

		for i in range(n_samples):
			R_1 = sample_resistance(_gain_multiplier*_LO + _HI, tol=tol)
			R_2_mean = _gain_multiplier*_LO + _HI
			R_2 = sample_resistance(R_2_mean, tol=tol)
			R_3_mean = _gain_multiplier*_LO + _HI
			R_3 = sample_resistance(R_3_mean, tol=tol)
			R_4a = sample_resistance(_HI, tol=tol)
			R_4b_mean = _gain_multiplier*_LO
			R_4b = sample_resistance(R_4b_mean, tol=tol)

			if perfect_trim:
				R_2 = R_4a + R_4b
			
			alpha = (R_1 + R_2) * (R_3 + R_4a) * R_4b
			beta = R_1 * (R_3 + R_4a) * R_4b
			gamma = (R_1 + R_2) * (R_3 + R_4a + R_4b)
			delta = R_1 * (R_4a + R_4b) - R_2 * R_3

			if A_OL_is_infty:
				R_out_SE_alt[i] = beta / delta
			else:
				R_out_SE_alt[i] = (alpha + A_OL * beta) / (gamma + A_OL * delta)

			R_out_SE_alt[i] = np.abs(R_out_SE_alt[i])

		gain_SE_alt = R_2_mean / (R_3_mean * R_4b_mean)
		print("SE gain (alt) = " + str(gain_SE_alt))
		#np.testing.assert_allclose(gain_SE, gain_SE_alt)
		
		
		### SYMMETRIC DIFFERENTIAL

		R_out_SYM = np.empty((n_samples, ))

		for i in range(n_samples):
			R_1 = sample_resistance(_gain_multiplier*_LO + _gain_multiplier*_HI, tol=tol)
			R_3 = sample_resistance(_gain_multiplier*_LO + _gain_multiplier*_HI, tol=tol)
			R_2a = sample_resistance(_gain_multiplier*_HI, tol=tol)
			R_4a = sample_resistance(_gain_multiplier*_HI, tol=tol)
			R_2b_mean = _gain_multiplier*_LO
			R_2b = sample_resistance(R_2b_mean, tol=tol)
			R_4b = sample_resistance(_gain_multiplier*_LO, tol=tol)

			if perfect_trim:
				R_1 = R_2a + R_2b
				R_3 = R_4a + R_4b

			alpha = R_1 * R_2b * (R_2a + R_3) + R_2b * (R_2a + R_3) * R_4a + R_1 * R_2a * R_4b + R_1 * R_2b * R_4b + R_1 * R_3 * R_4b + R_2b * R_3 * R_4b + R_2b * R_4a * R_4b + R_3 * R_4a * R_4b + R_2a * (R_2b + R_4a) * R_4b
			beta = R_2b * (R_2a + R_3) * R_4a + R_4b * R_1 * R_2a + R_4b * R_4a * R_2b + R_2a * (R_2b + R_4a) * R_4b
			gamma = -(R_2a + R_2b + R_3) * (R_1 + R_4a + R_4b)
			delta = R_1 * R_3 - (R_2a + R_2b) * (R_4a + R_4b)


			# R_out_SYM[i] = ((R_2a + R_2b) * (R_4a + R_4b) - R_1 * R_3) / (R_2a * R_4a * R_4b + R_2a * R_4a * R_2b + R_4a * R_2b * (R_4b + R_1) + R_2a * R_4b * (R_2b + R_3))

			if A_OL_is_infty:
				R_out_SYM[i] = beta / delta
			else:
				R_out_SYM[i] = (alpha + A_OL * beta) / (gamma + A_OL * delta)

			R_out_SYM[i] = np.abs(R_out_SYM[i])

		gain_sym = 1 / R_2b_mean
		print("sym gain = " + str(gain_sym))



		### FULLY DIFF OPAMP

		R_out_FD = np.empty((n_samples, ))

		for i in range(n_samples):

			# XXX: N.B.: when adjusting gain via R_1 and R_3 alone, no tradeoff between gain and Z_out observed!!

			R_1 = sample_resistance(_HI * _gain_multiplier, tol=tol)
			R_3 = sample_resistance(_HI * _gain_multiplier, tol=tol)
			
			R_2 = sample_resistance(_HI + _LO, tol=tol) # R2 = R4a + R4b
			R_5 = sample_resistance(_HI + _LO, tol=tol) # R5 = R6a + R6b
			
			R_4a = sample_resistance(_HI, tol=tol)
			R_6a = sample_resistance(_HI, tol=tol)

			R_4b = sample_resistance(_LO, tol=tol)
			R_6b = sample_resistance(_LO, tol=tol)

			#R_1 = sample_resistance(_HI / 2., tol=tol)
			#R_3 = sample_resistance(_HI / 2., tol=tol)
			
			#R_2 = sample_resistance(_HI + _LO, tol=tol) # R2 = R4a + R4b
			#R_5 = sample_resistance(_HI + _LO, tol=tol) # R5 = R6a + R6b
			
			#R_4a = sample_resistance(_HI, tol=tol)
			#R_6a = sample_resistance(_HI, tol=tol)

			#R_4b = sample_resistance(_LO, tol=tol)
			#R_6b = sample_resistance(_LO, tol=tol)
			
			if perfect_trim:
				R_1 = R_2a + R_2b		# uncomment <-> assuming perfect trimming
				R_3 = R_4a + R_4b		# uncomment <-> assuming perfect trimming

			assert A_OL_is_infty

			R_out_FD[i] = -(R_2 * R_3 * (R_4a * R_4b * (R_6a + R_6b)) + R_6a * ((R_4a - R_5) * R_6b + R_4b * (R_5 + R_6b))) \
				          + R_1 * (R_2 * R_4b * R_5 * (R_3 + R_4a) \
							       - R_2 * (R_4a * R_5 + R_3 * (R_4a - R_4b + R_5)) * R_6b \
								   - R_4b * (2 * R_3 * R_4a + (R_3 + R_4a) * R_5) * (R_6a + R_6b) \
								   - R_5 * R_6a * R_6b * (R_4a + R_4b) \
								   - R_3 * (R_4b * R_5 * R_6a - R_4b * R_5 * R_6b + 2 * R_4a * R_6a * R_6b + 2 * R_4b * R_6a * R_6b) \
								   - R_2 * R_3 * ((R_4a - R_5) * R_6b + R_4b * (R_5 + R_6b)))

			R_out_FD[i] /= ((R_2 * R_3 * (R_4a + R_4b - R_5) * (R_6a + R_6b) + R_1 * (-(R_2 * (2 * R_3 + R_4a + R_4b) * R_5) + (R_4a + R_4b) * (2 * R_3 + R_5) * (R_6a + R_6b))))

			R_out_FD[i] = np.abs(R_out_FD[i])


		gain_FD = R_2 / (2 * R_3 * R_4b)
		print("FD gain = " + str(gain_FD))

		print('Number of samples: ' + str(n_samples))
		print('Min R_out, SE achieved: ' + str(np.amin(np.abs(R_out_SE))))
		print('Min R_out, SE (alt) achieved: ' + str(np.amin(np.abs(R_out_SE_alt))))
		print('Min R_out, SYM achieved: ' + str(np.amin(np.abs(R_out_SYM))))
		print('Min R_out, FD achieved: ' + str(np.amin(np.abs(R_out_FD))))

		#if trial_idx == 0:
		#plot_R_out_hist(R_out_SE, 'single_ended', title_snip="Single ended", tol=tol)
		#plot_R_out_hist(R_out_SYM, 'differential', title_snip="Differential", tol=tol)

		hist_SE, bin_edges_SE = make_R_out_hist(R_out_SE)
		hist_SE_alt, bin_edges_SE_alt = make_R_out_hist(R_out_SE_alt)
		hist_fd, bin_edges_FD = make_R_out_hist(R_out_FD / 2.)#), bin_edges)
		hist_sym, bin_edges_sym = make_R_out_hist(R_out_SYM / 2.)#, bin_edges)

		#np.testing.assert_allclose(gain_SE, gain_sym, atol=1E-3)
		#gain_list[gain_idx] = np.mean([gain_SE, gain_sym])
		gain_list_SE[gain_idx] = gain_SE
		gain_list_SE_alt[gain_idx] = gain_SE_alt
		gain_list_sym[gain_idx] = gain_sym
		gain_list_FD[gain_idx] = gain_FD
		
		# peak detection
		median_R_out_SE[gain_idx, trial_idx] = np.mean(bin_edges_SE[np.argmax(hist_SE):np.argmax(hist_SE)+2])
		median_R_out_SE_alt[gain_idx, trial_idx] = np.mean(bin_edges_SE_alt[np.argmax(hist_SE_alt):np.argmax(hist_SE_alt)+2])
		median_R_out_sym[gain_idx, trial_idx] = 2 * np.mean(bin_edges_sym[np.argmax(hist_sym):np.argmax(hist_sym)+2])
		median_R_out_fd[gain_idx, trial_idx] = 2 * np.mean(bin_edges_FD[np.argmax(hist_fd):np.argmax(hist_fd)+2])

fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))

ax.errorbar(1E3*gain_list_SE, np.mean(median_R_out_SE, axis=1), np.std(median_R_out_SE, axis=1), marker=None, color="#f36e00", linewidth=2, zorder=99, markeredgecolor=None, capsize=4., capthick=1.5)
ax.plot(1E3*gain_list_SE, np.mean(median_R_out_SE, axis=1), linestyle='none', marker="o", markersize=1.2*6, color="#f36e00", label="Single ended", linewidth=2, zorder=999, markeredgecolor="k", markeredgewidth=1.)
ax.grid(b=True, which='both', zorder=0)

eb = ax.errorbar(1E3*gain_list_SE_alt, np.mean(median_R_out_SE_alt, axis=1), np.std(median_R_out_SE_alt, axis=1), marker=None, color="#f36e00", linewidth=2, zorder=99, markeredgecolor=None, capsize=4., capthick=1.5)
#eb[-1][0].set_linestyle('--') #eb[-1][0] is the LineCollection objects of the errorbar lines
eb.lines[0].set_linestyle('--')
ax.plot(1E3*gain_list_SE_alt, np.mean(median_R_out_SE_alt, axis=1), linestyle='none', marker="o", markersize=1.2*6, color="#f36e00", label="Single ended (alternate)", linewidth=2, zorder=999, markeredgecolor="k", markeredgewidth=1.)

ax.errorbar(1E3*gain_list_sym, np.mean(median_R_out_sym, axis=1), np.std(median_R_out_sym, axis=1), marker=None, color="#4671d5", linewidth=2, zorder=99, markeredgecolor=None, capsize=4., capthick=1.5)
ax.plot(1E3*gain_list_sym, np.mean(median_R_out_sym, axis=1), linestyle='none', marker="D", markersize=1.2*5.5, color="#4671d5", label="Symmetric", linewidth=2, zorder=999, markeredgecolor="k", markeredgewidth=1.)

ax.errorbar(1E3*gain_list_FD, np.mean(median_R_out_fd, axis=1), np.std(median_R_out_fd, axis=1), marker=None, color="#4671d5", linewidth=2, zorder=99, markeredgecolor=None, capsize=4., capthick=1.5)
ax.plot(1E3*gain_list_FD, np.mean(median_R_out_fd, axis=1), linestyle='none', marker="D", markersize=1.2*5.5, color="#4671d5", label="Differential", linewidth=2, zorder=999, markeredgecolor="k", markeredgewidth=1.)

ax.grid(b=True, which='both', zorder=0)
ax.set_xscale('log')
ax.set_yscale('log')
#ax.set_xticklabels(["{0:.0f}".format(1E3 * gain) for gain in gain_list])   # 1E3: convert A to mA
ax.legend(loc="upper right", handlelength=1.8, fontsize=16)#, bbox_to_anchor=(1.22, .95))#. , fancybox=True, framealpha=1
ax.set_xlabel("Gain [mA/V]")
ax.set_ylabel("Peak $|Z_{out}|$")
plt.tight_layout()
for ext in ["png", "pdf"]:
	fig.savefig("/tmp/mc_output_resistance_vs_gain." + ext)
plt.close(fig)
