#!/usr/bin/env python
# -*- coding: utf-8 -*-


import matplotlib as mpl
mpl.use("Agg")
import matplotlib.cm
import matplotlib.font_manager as fm

def mpl_setdefaults():
	mpl.rc('font', **{'family': 'serif', 'serif': ['stix']})
	mpl.rcParams["text.usetex"] = True
	mpl.rcParams["text.latex.unicode"] = True

	mpl.rcParams['font.family'] = 'STIXGeneral'
	# mpl.rcParams['font.sans-serif'] = 'cmr10'
	# mpl.rcParams['font.serif'] = 'cmr10'
	# mpl.rcParams['font.cursive'] = 'cmmi10'
	mpl.rcParams['mathtext.fontset'] = 'stix'
	# mpl.rcParams['mathtext.rm'] = 'cmr10'
	# mpl.rcParams['mathtext.cal'] = 'cmr10'
	# mpl.rcParams['mathtext.it'] = 'cmmi10'
	# mpl.rcParams['mathtext.bf'] = 'cmr10'
	mpl.rcParams["axes.labelsize"] = 20.
	mpl.rcParams["xtick.labelsize"] = 16.
	mpl.rcParams["ytick.labelsize"] = 16.
	mpl.rcParams["legend.fontsize"] = 14.

mpl_setdefaults()

# prop = fm.FontProperties(fname='/tmp/Serif/cmunrm.ttf')

from matplotlib import collections
import matplotlib.ticker as plticker
from matplotlib.ticker import FormatStrFormatter

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.stats

import collections
#from collections Iterable



def brighten(rgb):
	assert 0 <= factor <= 1
	rgb = np.ones(3) - rgb
	return np.ones(3) - (rgb * factor)
	

def darken(rgb, factor):
	assert 0 <= factor <= 1
	return rgb * factor


def get_dual_linear_colour_map(n_sweeps, reversed=False):
	n_circuits = 2
	col_se = np.array([.95, .43, .0])
	col_se1 = brightnessAdjust(col_se, 1.)
	col_se2 = brightnessAdjust(col_se, .6)
	col_diff = np.array([.275, .443, .835])
	col_diff1 = brightnessAdjust(col_diff, 1.)
	col_diff2 = brightnessAdjust(col_diff, .6)
	cm_se = mpl.colors.LinearSegmentedColormap.from_list("se_cm", [col_se2, col_se1])
	cm_diff = mpl.colors.LinearSegmentedColormap.from_list("diff_cm", [col_diff2, col_diff1])
	
	colours = np.empty((n_circuits, n_sweeps), dtype=np.object)
	if n_sweeps == 1:
		colours[0, 0] = cm_se(0.)
		colours[1, 0] = cm_diff(0.)
	else:
		for sweep_idx in range(n_sweeps):
			colours[0, n_sweeps - sweep_idx - 1] = cm_se(sweep_idx / float(n_sweeps - 1))
			colours[1, n_sweeps - sweep_idx - 1] = cm_diff(sweep_idx / float(n_sweeps - 1))

	# colours[1, ...] = colours[0, ...]

	if reversed:
		colours = colours[:, ::-1]
	
	return colours


def log_interp(zz, xx, yy):
	"""interpolation between points on a log-log axis (wrapper for scipy interp1d)"""
	assert np.all(np.diff(xx) > 0)

	logz = np.log10(zz)
	logx = np.log10(xx)
	logy = np.log10(yy)

	interp = sp.interpolate.interp1d(logx, logy, kind="linear")
	interp = np.power(10., interp(logz))

	return interp


def find_x(y, xx, yy, epsilon=1E-6):
	"""binary search: given a series of pairs (xx, yy) where xx = x1, x2, ... xN and yy = f(xx) find x such that f(x) = y"""
	if np.array(y).size == 1:
		y = np.array([y])
	x = np.zeros_like(y)
	for i, _y in enumerate(y):
		if _y < np.amin(yy) or _y > np.amax(yy):
			x[i] = np.nan
		elif _y == yy[0]:
			x[i] = xx[0]
		elif _y == yy[-1]:
			x[i] = xx[-1]
		else:
			def f(x):
				return log_interp(x, xx, yy) - _y
			x = sp.optimize.bisect(f, xx[0], xx[-1], xtol=epsilon)
	return x

	
def create_2d_colour_map(dim1, dim2):

	assert dim1 <= 3, "more than 3 not yet implemented"

	col_se = np.array([.95, .43, .0])
	col_se1 = brightnessAdjust(col_se, 1.)
	col_se2 = brightnessAdjust(col_se, .6)
	cm_se = mpl.colors.LinearSegmentedColormap.from_list("se_cm", [col_se2, col_se1])

	col_diff = np.array([.275, .443, .835])
	col_diff1 = brightnessAdjust(col_diff, 1.)
	col_diff2 = brightnessAdjust(col_diff, .6)
	cm_diff = mpl.colors.LinearSegmentedColormap.from_list("diff_cm", [col_diff2, col_diff1])
	
	col_diff_cs = np.array([.275, .835, .443])
	col_diff_cs2 = brightnessAdjust(col_diff_cs, 1.)
	col_diff_cs1 = brightnessAdjust(col_diff_cs, .6)
	cm_diff_cs = mpl.colors.LinearSegmentedColormap.from_list("diff_cs_cm", [col_diff_cs2, col_diff_cs1])

	colour_maps = [cm_se, cm_diff, cm_diff_cs]
	
	colours = np.empty((dim1, dim2, 3), dtype=np.float)
	for sweep_idx in range(dim2):
		for cm_idx in range(dim1):
			colours[cm_idx, sweep_idx, :] = colour_maps[cm_idx](sweep_idx / float(max(1, dim2 - 1)))[:3]

	return colour_maps, colours



def plot_bode(fn, f, mag, ang, ckt_ids=None, labels=None, colours=None, markers=None, figsize=(8, 5), ylim_mag=None, ylim_ang=None, colourmap=mpl.cm.jet, mag_ax_ylabel="Magnitude", ang_ax_ylabel="Phase", title=None, log_scale=20., markers_f=None, markers_mag=None, markers_ang=None, marker_h_colour=np.array([0., 0., 0.]), marker_colours=np.array([0., 0., 0.]), markers_size=40., intersect_markers="o", log_y_axis=True, ang_tick_multiples=60.):

	"""
	Parameters
	----------
	mag : numpy array, shape (n_circuits, n_freqs) or (n_circuits, n_sweeps, n_freqs)
	
	"""

	assert np.all(mag.shape == ang.shape), "Magnitude and angle should be in the same shape"

	fig = plt.figure(figsize=figsize)
	gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
	ax = plt.subplot(gs[0])
	ax2 = plt.subplot(gs[1])
	ax.set_xlim(np.amin(f), np.amax(f))
	ax2.set_xlim(np.amin(f), np.amax(f))
	if ylim_mag is None:
		ylim_mag = [np.amin(mag), np.amax(mag)]
	if ylim_ang is None:
		ylim_ang = [np.amin(ang), np.amax(ang)]
	ax.set_ylim(ylim_mag)
	ax2.set_ylim(ylim_ang)
	if len(mag.shape) == 1:
		mag = mag[np.newaxis, np.newaxis, :]
		ang = ang[np.newaxis, np.newaxis, :]
	elif len(mag.shape) == 2:
		mag = mag[:, np.newaxis, :]
		ang = ang[:, np.newaxis, :]
	n_circuits, n_sweeps, n_freqs = mag.shape
	if colours is None:
		_, colours = create_2d_colour_map(n_circuits, n_sweeps)

	for circuit_idx in range(n_circuits):
		for sweep_idx in range(n_sweeps):
			if labels is None:
				_label = None
			else:
				if type(labels) is np.ndarray and len(labels.shape) == 2:
					_label = labels[circuit_idx, sweep_idx]
				else:
					assert type(labels) is list \
					 or (type(labels) is np.ndarray and len(labels.shape) == 1)
					_label = labels[sweep_idx]
			
			if markers is None:
				_marker = None
			else:
				if len(_markers.shape) == 2:
					_marker = markers[circuit_idx, sweep_idx]
				else:
					assert len(_labels.shape) == 1
					_marker = markers[sweep_idx]

			ax.plot(f, mag[circuit_idx, sweep_idx, :], linestyle="-", marker=_marker, markersize=1.2*5.5, color=colours[circuit_idx, sweep_idx, :], label=_label, linewidth=2, zorder=99)#, edgecolor="#ffffff", hatch="-", linewidth=10.0)
			ax2.plot(f, ang[circuit_idx, sweep_idx, :], marker=_marker, markersize=1.2*5.5, linestyle="-", color=colours[circuit_idx, sweep_idx, :], label=_label, linewidth=2, zorder=99, alpha=1)#, edgecolor="#ffffff", hatch="-", linewidth=10.0)

	if log_y_axis:
		ax.set_yscale('log')
	ax.set_xscale('log')
	ax2.set_xscale('log') 
	
	loc = plticker.MultipleLocator(base=ang_tick_multiples) # this locator puts ticks at regular intervals
	ax2.yaxis.set_major_locator(loc)

	if not markers_f is None:
		if not marker_colours is None:
			if type(marker_colours) is np.ndarray and len(marker_colours.shape) == 1:
				marker_colours = np.tile(marker_colours, (len(markers_f), 1))
			marker_colours = np.array(marker_colours)
		if not intersect_markers is None:
			if not type(intersect_markers) in [list, np.ndarray]:
				intersect_markers = np.tile(intersect_markers, len(markers_f))
			else:
				assert len(intersect_markers) == len(markers_f)
			if not type(markers_size) in [list, np.ndarray]:
				markers_size = np.tile(markers_size, len(markers_f))
			else:
				assert len(markers_size) == len(markers_f)
		if not markers_mag is None:
			assert len(markers_f) == len(markers_mag)
		if not markers_ang is None:
			assert len(markers_f) == len(markers_ang)
		for i, _f in enumerate(markers_f):
			if not markers_mag is None:
				ax.scatter(_f, markers_mag[i], marker=intersect_markers[i], s=markers_size[i], zorder=999, facecolor=marker_colours[i, :])
				ax.plot([_f, _f], [np.amin(ax.get_ylim()), markers_mag[i]], linestyle="--", linewidth=2., color=marker_colours[i, :])		# vertical marker line
				ax.plot([np.amin(ax.get_xlim()), _f], [markers_mag[i], markers_mag[i]], linestyle="--", color=marker_h_colour, linewidth=2)		# horizontal black marker line
				ax2.plot([_f, _f], [markers_ang[i], ax2.get_ylim()[1]], linestyle="--", linewidth=2., color=marker_colours[i, :])		# vertical marker line
			if not markers_ang is None:
				ax2.scatter(_f, markers_ang[i], marker=intersect_markers[i], s=markers_size[i], zorder=999, facecolor=marker_colours[i, :])
				ax2.plot([0, _f], [markers_ang[i], markers_ang[i]], linestyle="--", linewidth=2., color=marker_colours[i, :])		# horizontal coloured marker line

				
	fig.canvas.draw()
	ax.set_xticklabels([])

	for _ax in [ax, ax2]:
		_ax.grid(b=True, zorder=0)  # , which='both'
		_ax.spines['left'].set_linewidth(1.5)
#		_ax.spines['right'].set_visible(False)
#		_ax.spines['top'].set_visible(False)
		_ax.spines['bottom'].set_linewidth(1.5)
		_ax.xaxis.set_ticks_position("bottom")
		_ax.yaxis.set_ticks_position("left")

		for tick in _ax.xaxis.get_major_ticks():
			tick.label.set_fontsize(16)
		for tick in _ax.yaxis.get_major_ticks():
			tick.label.set_fontsize(16)

	lbl = []
	for t in ax.get_yticks():
		lbl.append(log_scale*np.log10(t))
	ax.set_yticklabels(lbl)
		
	ax.set_ylabel(mag_ax_ylabel)
	ax2.set_ylabel(ang_ax_ylabel)
	ax2.set_xlabel("Frequency [Hz]")

	leg = ax.legend(loc="upper right", handlelength=1.8, fontsize=16., fancybox=True, framealpha=1)

	if not title is None:
		fig.suptitle(title)
	
	fig.savefig(fn)
	plt.close(fig)

	
def brightnessAdjust(rgb, value):
	hsv = mpl.colors.rgb_to_hsv(rgb)
	hsv[2] = value
	rgb = mpl.colors.hsv_to_rgb(hsv)
	return rgb

	
if __name__ == "__main__":
	
	freqs	=		np.array([1E3,	2E3,	3E3,	4E3,	5E3,	10E3,	20E3,	30E3,	40E3,	50E3,	100E3,	200E3,	300E3,	400E3,	500E3,	1E6])	#	[Hz]
	amp_se	=		np.array([20.6,	20.4,	20.3,	20.5,	20.3,	20.0,	18.1,	16.8,	15.5,	13.5,	7.9,	4.2,	2.7,	2.0,	1.5,	0.7])
	phase_se	=	np.array([172,	172,	172,	172,	174,	160,	153,	134,	127,	127,	102,	90,		75,		73,		72,		36])
	amp_diff	=	np.array([20.4,	20.9,	20.5,	21.1,	20.9,	20.9,	20.4,	20.,	19.1,	17.9,	13.,	7.6,	5.1,	3.84,	3.1,	1.4])
	phase_diff	=	np.array([7,	10,		10,		16.,	13,		16,		21.,	23.,	30.,	38,		55,		82.,	88.,	92.,	95.,	115])

	phase_se -= phase_se[0]
	phase_diff -= phase_diff[0]
	# phase_se = -phase_se
	phase_diff = -phase_diff


	gain_se = 0.0101
	gain_diff = 0.01
	load_R = 10E3
	amp_se /= amp_se[0] / gain_se
	amp_diff /= amp_diff[0] / gain_diff
		
	f = np.logspace(3, 6, 10)
	plot_bode(f, mag, ang, label="Differential", colours=["#4671d5", "#f36e00"])


def plot_pm_vs_gbw(max_pm, max_pm_f, best_c_fb, c_fb_list, fn, circuit_names, c_load_list=None, gbw_list=None, figsize=(2.5, 6), MARKER_SIZE=5):
	def _plot_phase_margins(ax, data, circuit_name, legend=False, ylabel=None, log_y_axis=False, ylim=None, labels=None, plot_xlabels=True):
		marker, marker_size = get_marker_style(circuit_name, MARKER_SIZE)
		circuit_idx = circuit_names.index(circuit_name)
			
		n_circuits = len(circuit_names)
		_, colours = create_2d_colour_map(n_circuits, 1)

		linescollection = ax.semilogx(1E-6 * np.array(gbw_list), data[circuit_idx, :], marker=marker, markersize=marker_size, color=colours[circuit_idx, 0], linewidth=2.)

		ax.set_xlim(1E-6 * np.amin(gbw_list), 1E-6 * np.amax(gbw_list))
		if plot_xlabels:
			ax.set_xlabel("$GBW$ [MHz]")
			ax.set_xticklabels([])
		if not ylabel is None:
			ax.set_ylabel(ylabel)
		if log_y_axis:
			ax.set_yscale("log")
		if not ylim is None:
			ax.set_ylim(ylim)
		# ax.legend(linescollection_se + linescollection_diff, tuple(labels_se) + tuple(labels_diff), loc="best")
		if legend:
			ax.legend(linescollection, tuple(labels), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

		ax.grid(b=True, zorder=0)  # , which='both'
		ax.spines['left'].set_linewidth(1.5)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['bottom'].set_linewidth(1.5)
		ax.xaxis.set_ticks_position("bottom")
		ax.yaxis.set_ticks_position("left")

		return linescollection

	n_gbw_vals = len(gbw_list)	

	fig = plt.figure(figsize=figsize)
	gs = mpl.gridspec.GridSpec(3, 1)

	linescollection_se_pm = _plot_phase_margins(plt.subplot(gs[0]), data=max_pm, circuit_name="single-ended", legend=False, ylim=(0., 161.), plot_xlabels=False)
	linescollection_diff_pm = _plot_phase_margins(plt.subplot(gs[0]), data=max_pm, circuit_name="differential", legend=False, ylim=(0., 161.), plot_xlabels=False)
	linescollection_diff_cs_pm = _plot_phase_margins(plt.subplot(gs[0]), data=max_pm, circuit_name="differential-counter-steering", legend=False, ylabel="Best obtained\nphase margin [deg]", ylim=(0., 161.), plot_xlabels=False)
	
	linescollection_se_pm_f = _plot_phase_margins(plt.subplot(gs[1]), data=max_pm_f, circuit_name="single-ended", legend=False, ylim=(np.amin(max_pm_f), np.amax(max_pm_f)), log_y_axis=True, plot_xlabels=False)
	linescollection_diff_pm_f = _plot_phase_margins(plt.subplot(gs[1]), data=max_pm_f, circuit_name="differential", legend=False, ylim=(np.amin(max_pm_f), np.amax(max_pm_f)), log_y_axis=True, plot_xlabels=False)
	linescollection_diff_cs_pm_f = _plot_phase_margins(plt.subplot(gs[1]), data=max_pm_f, circuit_name="differential-counter-steering", legend=False, ylim=(np.amin(max_pm_f), np.amax(max_pm_f)), ylabel="Frequency of best\nphase margin [Hz]", log_y_axis=True, plot_xlabels=False)
	
	linescollection_se_pm = _plot_phase_margins(plt.subplot(gs[2]), data=1E12*best_c_fb, circuit_name="single-ended", legend=False, ylim=(np.amin(1E12*best_c_fb), np.amax(1E12*best_c_fb)), plot_xlabels=True, log_y_axis=True)
	linescollection_diff_pm = _plot_phase_margins(plt.subplot(gs[2]), data=1E12*best_c_fb, circuit_name="differential", legend=False, ylim=(np.amin(1E12*best_c_fb), np.amax(1E12*best_c_fb)), ylabel="Optimal\n$C_{FB}$ [pF]", plot_xlabels=True, log_y_axis=True)
	linescollection_diff_cs_pm = _plot_phase_margins(plt.subplot(gs[2]), data=1E12*best_c_fb, circuit_name="differential-counter-steering", legend=False, ylim=(np.amin(1E12*best_c_fb), np.amax(1E12*best_c_fb)), ylabel="Optimal\n$C_{FB}$ [pF]", plot_xlabels=True, log_y_axis=True)

	fig.savefig(fn)
	plt.close(fig)


def plot_pm_vs_gbw2(max_pm, max_pm_f, best_c_fb, c_fb_list, circuit_names, fn, c_load_list=None, gbw_list=None, labels_diff="Differential", labels_se="Single ended", figsize=(8, 6.5), MARKER_SIZE=5):

	def _plot_phase_margins(ax, data, colour, legend=False, ylabel=None, log_y_axis=False, ylim=None, labels=None, plot_xlabels=True, alpha=1., plot_ylabels=True):
		marker, marker_size = get_marker_style(circuit_name, MARKER_SIZE)
		
		linescollection = ax.semilogx(1E-6 * np.array(gbw_list), data, marker=marker, markersize=marker_size, color=colour, linewidth=2., alpha=alpha)#), markeredgecolor=colours[circuit_idx, c_load_idx])

		ax.set_xlim(1E-6 * np.amin(gbw_list), 1E-6 * np.amax(gbw_list))
		if plot_xlabels:
			ax.set_xlabel("$GBW$ [MHz]")
		else:
			ax.set_xticklabels([])
		ax.grid(True)
		if log_y_axis:
			ax.set_yscale("log")
		if not ylim is None:
			ax.set_ylim(ylim)
		# ax.legend(linescollection_se + linescollection_diff, tuple(labels_se) + tuple(labels_diff), loc="best")

		if plot_ylabels:
			ax.set_ylabel(ylabel)
		else:
			ax.set_yticklabels([])

		return linescollection

	n_c_load_vals = len(c_load_list)
	n_circuits = len(circuit_names)
	_, colours = create_2d_colour_map(n_circuits, n_c_load_vals)

	fig = plt.figure(figsize=figsize)
	gs = mpl.gridspec.GridSpec(3, 3)

	for circuit_idx, circuit_name in enumerate(circuit_names):
		linescollection = []
		labels = []

		for c_load_idx, C_L in enumerate(c_load_list):
		
			# faded background single-ended
			if not circuit_name == "single-ended":
				alpha = .25
				_circuit_idx = circuit_names.index("single-ended")
				_plot_phase_margins(plt.subplot(gs[0, circuit_idx]), data=max_pm[_circuit_idx, :, c_load_idx], colour=colours[_circuit_idx, c_load_idx], legend=True, ylim=(55.-1E-12, 160.+1E-12), plot_xlabels=False, ylabel="Best obtained\nphase margin [deg]", alpha=alpha, plot_ylabels=False)
				_plot_phase_margins(plt.subplot(gs[1, circuit_idx]), data=max_pm_f[_circuit_idx, :, c_load_idx], colour=colours[_circuit_idx, c_load_idx], legend=True, ylim=(np.amin(max_pm_f), np.amax(max_pm_f)), plot_xlabels=False, log_y_axis=True, ylabel="Frequency of best\nphase margin [Hz]", alpha=alpha, plot_ylabels=False)
				_plot_phase_margins(plt.subplot(gs[2, circuit_idx]), data=1E12*best_c_fb[_circuit_idx, :, c_load_idx], colour=colours[_circuit_idx, c_load_idx], legend=True, labels=["$C_L = " + str(c_load_list[i]) + "$" for i in range(n_c_load_vals)], ylabel="Optimal\n$C_{FB}$ [pF]", plot_xlabels=True, log_y_axis=True, alpha=alpha, plot_ylabels=False)#, ylim=(1.-1E-12, 100.+1E-12))

			alpha = 1.
			_plot_phase_margins(plt.subplot(gs[0, circuit_idx]), data=max_pm[circuit_idx, :, c_load_idx], colour=colours[circuit_idx, c_load_idx], legend=True, ylim=(55.-1E-12, 160.+1E-12), plot_xlabels=False, ylabel="Best obtained\nphase margin [deg]", alpha=alpha, plot_ylabels=circuit_idx == 0)
			_plot_phase_margins(plt.subplot(gs[1, circuit_idx]), data=max_pm_f[circuit_idx, :, c_load_idx], colour=colours[circuit_idx, c_load_idx], legend=True, ylim=(np.amin(max_pm_f), np.amax(max_pm_f)), plot_xlabels=False, log_y_axis=True, ylabel="Frequency of best\nphase margin [Hz]", alpha=alpha, plot_ylabels=circuit_idx == 0)
			linescollection += _plot_phase_margins(plt.subplot(gs[2, circuit_idx]), data=1E12*best_c_fb[circuit_idx, :, c_load_idx], colour=colours[circuit_idx, c_load_idx], legend=True, labels=["$C_L = " + str(c_load_list[i]) + "$" for i in range(n_c_load_vals)], ylabel="Optimal\n$C_{FB}$ [pF]", plot_xlabels=True, log_y_axis=True, alpha=alpha, plot_ylabels=circuit_idx == 0)#, ylim=(1.-1E-12, 100.+1E-12))
			labels += ["$C_L = " + str(C_L) + "$"]

		# plot legend on top row
		ax = plt.subplot(gs[circuit_idx, 0])
		ax.legend(linescollection, tuple(labels), bbox_to_anchor=(1.05, 1), loc=9, borderaxespad=0.)

	fig.subplots_adjust(right=.8)
	fig.savefig(fn)
	plt.close(fig)


def plot_phase_margins_vs_cfb(pm, c_fb_list, fn, circuit_names, c_load_list=None, figsize=(5, 6), MARKER_SIZE=5, ang_ylim=(59., 161.), title=""):

	def _plot_phase_margins(ax, circuit_idx, circuit_name, labels=[], legend=False):
		assert 0 <= circuit_idx <= 2
		marker, marker_size = get_marker_style(circuit_name, MARKER_SIZE)
		if circuit_idx == 1:
			other_marker = "o"
			other_marker_size = MARKER_SIZE
		elif circuit_idx == 2:
			other_marker = "s"
			other_marker_size = MARKER_SIZE
			ax.set_xlabel("$C_{FB}$ [pF]")
			ax.set_ylabel("Phase margin [deg]")
		else:
			other_marker = "d"
			other_marker_size = MARKER_SIZE * 1.25

		other_circuit_idx = 1 - circuit_idx

		n_circuits = len(circuit_names)
		_, colours = create_2d_colour_map(n_circuits, n_c_load_vals)
		
		linescollection = None
		for c_load_idx in range(n_c_load_vals):
			_l = ax.semilogx(1E12 * c_fb_list, pm[other_circuit_idx, :, c_load_idx], marker=other_marker, markersize=other_marker_size, color=colours[other_circuit_idx, c_load_idx], linewidth=2., alpha=.25)#), markeredgecolor=colours[other_circuit_idx, c_load_idx])

			_l = ax.semilogx(1E12 * c_fb_list, pm[circuit_idx, :, c_load_idx], marker=marker, markersize=marker_size, color=colours[circuit_idx, c_load_idx], linewidth=2.)#, markeredgecolor=colours[circuit_idx, c_load_idx])

			if linescollection is None:
				linescollection = _l
			else:
				linescollection += _l				

		if circuit_idx == 0:
			ax.set_xticklabels([])
				
		ax.set_xlim(1E12 * np.amin(c_fb_list), 1E12 * np.amax(c_fb_list))
		if not ang_ylim is None:
			ax.set_ylim(ang_ylim)
		ax.grid(True)
		if legend:
			leg = ax.legend(linescollection, tuple(labels), loc="best")
			leg.get_frame().set_alpha(.6)

		return linescollection


	n_circuits = len(circuit_names)
	n_c_fb_vals = len(c_fb_list)
	n_c_load_vals = len(c_load_list)
	# pm = np.inf * np.ones((n_circuits, n_c_fb_vals, n_c_load_vals))
	
	fig = plt.figure(figsize=figsize)
	gs = mpl.gridspec.GridSpec(n_circuits, 1)
	for circuit_idx, circuit_name in enumerate(circuit_names):
		_plot_phase_margins(plt.subplot(gs[circuit_idx]), circuit_idx=circuit_idx, circuit_name=circuit_name, labels=["$C_L = " + "{:.2E}".format(c_load_list[i]) + "$" + " (" + circuit_name + ")"for i in range(n_c_load_vals)], legend=True)
	fig.suptitle(title)
	fig.savefig(fn)
	plt.close(fig)


def get_marker_style(circuit_name, _marker_size=40.):
	if circuit_name == "se":
		circuit_name = "single-ended"
	if circuit_name == "sym":
		circuit_name = "differential"
	if circuit_name == "ctst":
		circuit_name = "differential-counter-steering"
	assert circuit_name in ["single-ended", "differential", "differential-counter-steering"]

	if circuit_name == "differential":
		return "d", _marker_size * 1.25
	if circuit_name == "differential-counter-steering":
		return "s", _marker_size
	assert circuit_name == "single-ended"
	return "o", _marker_size
