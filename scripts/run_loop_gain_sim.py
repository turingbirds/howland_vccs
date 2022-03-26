# -*- coding: utf-8 -*-

"""

Calculation of output impedance across frequency

- Set up ltspice simulation for transient analysis
- Use sinusoidal voltage source for :math:`V_{I,set}`, measure output current
- Set up parameter sweep across input frequency; modify total simulation time to match
- Do this for two values of output resistor (10k, 11k)
	
For each frequency:
-	Calculate amplitude ratio
-	Calculate phase

"""

from plot_funcs import plot_bode, brightnessAdjust, find_x, log_interp, create_2d_colour_map, plot_phase_margins_vs_cfb, plot_pm_vs_gbw, plot_pm_vs_gbw2, get_dual_linear_colour_map, get_marker_style
from spice_sim import *

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.signal
import copy
import os
import re
import sys
from guppy import hpy
from invoke import exec_binary
from si_prefix import si_format


##############################################################################

VARIABLE_NAME_IOUT_R = "I(Rl)"
VARIABLE_NAME_IOUT_C = "I(Cl)"
VARIABLE_NAME_VOUT = "V(vl)"
VARIABLE_NAME_VOUTP = "V(vlpos)"
VARIABLE_NAME_VOUTN = "V(vlneg)"
VARIABLE_NAME_FREQUENCIES = "frequency"

scad3_exe_path = "/cygdrive/c/Program Files (x86)/LTC/LTspiceIV/scad3.exe"

circuit_names = ["se", "sym", "ctst"]

circuit_path_loop_gain = {
	"se" : "/cygdrive/c/sim/single_ended_howland_middlebrook.net",
	"sym" : "/cygdrive/c/sim/diff_howland_middlebrook.net",
	"ctst" : "/cygdrive/c/sim/diff_howland_countersteering_middlebrook.net"
}

ltspice_results_path = "/cygdrive/c/sim/ckt_tmp.raw"
mod_circuit_path = "/cygdrive/c/sim/ckt_tmp.net"
mod_circuit_path_windows = r"c:\sim\ckt_tmp.net"		# XXX mixed path styles because using ltspice & CygWin under Windows

n_circuits = len(circuit_names)

	
##############################################################################

def parallel(R_1, R_2):
	return (R_1 * R_2) / (R_1 + R_2)


def brighten(rgb):
	assert 0 <= factor <= 1
	rgb = np.ones(3) - rgb
	return np.ones(3) - (rgb * factor)
	

def darken(rgb, factor):
	assert 0 <= factor <= 1
	return rgb * factor

	
##############################################################################


def run_phase_margin_sim(circuit_fn, C_FB, GBW, A_OL, R_L, C_L, fn_snip="", title_snip="", make_bode_plot=False, verbose=False):
	orig_spice = read_orig_spice(circuit_fn)

	new_vals = {
		".param AOL" : [(2, str(A_OL))],
		".param GBW" : [(2, str(GBW))],
		".param RLOAD" : [(2, "{0:E}".format(R_L))],
		".param CLOAD" : [(2, "{0:E}".format(C_L))]}
	new_vals[".param CFB"] = [(2, "{0:E}".format(C_FB))]

	print("===========================================")
	print("      CIRCUIT: " + str(os.path.basename(circuit_fn)))
	print("      GBW: " + si_format(GBW) + "Hz")
	print("      C_L: " + si_format(C_L) + "F")
	print("      C_FB: " + si_format(C_FB) + "F")
	print("===========================================")
	
	header_dict, variable_names, variable_values, n_timepoints = run_circuit(orig_spice, new_vals, scad3_exe_path, mod_circuit_path, ltspice_results_path, mod_circuit_path_windows, verbose=verbose)
	variable_idx_frequencies = find_variable_idx(variable_names, VARIABLE_NAME_FREQUENCIES)
	frequencies = np.real(variable_values[variable_idx_frequencies]); assert np.all(np.imag(variable_values[variable_idx_frequencies])) == 0

	I_VIF = variable_values[find_variable_idx(variable_names, "I(VIF)"), ...]
	I_VII = variable_values[find_variable_idx(variable_names, "I(VII)"), ...]
	V_VFM = variable_values[find_variable_idx(variable_names, "V(VFM)"), ...]
	V_VIM = variable_values[find_variable_idx(variable_names, "V(VIM)"), ...]

	loop_gain = ((I_VIF / I_VII) * (-V_VFM / V_VIM) - 1) / ((I_VIF / I_VII) + (-V_VFM / V_VIM) + 2)
	
	
	#
	# 	find phase margin
	#
	
	loop_gain_mag = np.absolute(loop_gain)
	loop_gain_phase = np.angle(loop_gain)
	loop_gain_phase = np.unwrap(loop_gain_phase)
	freq_idx_0_db = np.argmin((loop_gain_mag - 1.)**2)		# find the frequency index where gain = 1
	pm = -(-180. - 180. / np.pi * loop_gain_phase[freq_idx_0_db])

	print("\tloop gain = " + "{0:E}".format(np.absolute(loop_gain)[freq_idx_0_db]) + u" (≈ 1) at f = " + si_format(frequencies[freq_idx_0_db]) + "Hz, pm = " + str(int(pm)))

	#
	#	Bode plots of loop gain
	#

	if make_bode_plot:
		for fext in ["png", "pdf"]:
			_title = r"Loop gain (" + title_snip + "): PM = " + str(int(pm))
			fn = "/tmp/sim_bode_" + fn_snip + "." + fext
			print("* Writing Bode plot to: " + fn)
			plot_bode(fn, frequencies, loop_gain_mag, 180. / np.pi * loop_gain_phase, ylim_mag=[np.amin(loop_gain_mag), np.amax(loop_gain_mag)], title=_title, markers_f=[frequencies[freq_idx_0_db]])

	return frequencies, freq_idx_0_db, pm, loop_gain


##############################################################################

def run_phase_margin_vs_cload_sweep(c_load_list, c_fb_list, GBW, A_OL, R_L, fn_snip="", return_loop_gains=False, debug=False, debug_inner=False):
	n_c_fb_vals = len(c_fb_list)
	n_c_load_vals = len(c_load_list)
	frequencies = None
	pm = np.inf * np.ones((n_circuits, n_c_fb_vals, n_c_load_vals))
	f_0dB = np.inf * np.ones((n_circuits, n_c_fb_vals, n_c_load_vals))
	if return_loop_gains:
		loop_gains = None

	for circuit_idx, circuit_name in enumerate(circuit_names):
		for c_load_idx, C_L in enumerate(c_load_list):
			for c_fb_idx, C_FB in enumerate(c_fb_list):
				title = circuit_name + " ($C_L=" + "{0:E}".format(C_L) + "$, " + "$C_{FB}=" + "{0:E}".format(C_FB) + "$)"
				_fn_snip = fn_snip + "_" + circuit_name + "_[C_L=" + str(C_L) + "]_[C_FB=" + "{0:E}".format(C_FB) + "]"
	
				_frequencies, _pm_f_idx, _pm, _loop_gain = run_phase_margin_sim(circuit_path_loop_gain[circuit_name], C_FB, GBW, A_OL, R_L, C_L, fn_snip=_fn_snip, title_snip=title, make_bode_plot=debug_inner)

				if frequencies is None:
					frequencies = _frequencies
				else:
					np.testing.assert_almost_equal(frequencies, _frequencies)	# check for mismatch between returned frequencies between simulations

				if return_loop_gains and loop_gains is None:
					loop_gains = np.nan * np.ones((n_circuits, n_c_fb_vals, n_c_load_vals, len(frequencies)), dtype=np.complex)		# can only preallocate now that number of frequency points is known

				pm[circuit_idx, c_fb_idx, c_load_idx] = _pm
				f_0dB[circuit_idx, c_fb_idx, c_load_idx] = _frequencies[_pm_f_idx]
				if return_loop_gains:
					loop_gains[circuit_idx, c_fb_idx, c_load_idx, :] = _loop_gain

	if return_loop_gains:
		return frequencies, pm, f_0dB, loop_gains
	else:
		return frequencies, pm, f_0dB

	
##############################################################################

def transfer_func_SE(R_1, R_2, R_3, R_4a, R_4b, R_L, f, C_FB):
	Z_C_FB = 1 / (1j * 2 * np.pi * f * C_FB)
	R_2p = parallel(R_2, Z_C_FB)
	H = .5 * (R_2p * (R_3 + 2 * R_4a) + R_1 * (R_4a + R_4b)) / (R_2p * R_3 * R_L - R_1 * (R_3 * R_4b + R_4b * R_L + R_4a * (R_4b + R_L)))
	return H


def transfer_func_diff(R_1, R_2a, R_2b, R_3, R_4a, R_4b, R_L, f, C_FB):
	Z_C_FB = 1 / (1j * 2 * np.pi * f * C_FB)
	R_1p = parallel(R_1, Z_C_FB)
	R_3p = parallel(R_3, Z_C_FB)
	H = (R_1p * (R_2a + R_3p) + R_4a * (R_2a + R_3p) - R_2b * R_4b) / (-(R_1p * R_3p * R_L) + R_2a * (R_1p * R_4b + R_4a * R_4b + R_2b * (R_4a + R_4b) + R_4a * R_L + R_4b * R_L) + R_2b * (R_3p * R_4a + R_4b * R_L + R_4a * (R_4b + R_L)))
	return H


def c_fb_to_corner_freq(circuit_name, C_FB, transfer_func_args, f_0=1., tol=1E-6):
	"""find f such that H(f) = H(f_0) / sqrt(2)"""
	
	if circuit_name == "se":
		transfer_func = transfer_func_SE
	else:
		assert circuit_name == "sym" or circuit_name == "ctst"
		transfer_func = transfer_func_diff
	
	H_0 = transfer_func(f=f_0, C_FB=C_FB, **transfer_func_args)
	f_c_upper_bound = 1 / (2 * np.pi * transfer_func_args["R_1"] * C_FB)
	f_c_lower_bound = 0.
	success = False
	while not success:
		f = (f_c_upper_bound + f_c_lower_bound) / 2.
		H = transfer_func(f=f, C_FB=C_FB, **transfer_func_args)

		print("guess = " + str(f) + " Hz, H = " + str(H) + ", H_0 = " + str(H_0) + ", target = " + str(np.absolute(H_0)/np.sqrt(2.)))

		if np.abs(np.absolute(H_0 / H) - np.sqrt(2.)) < tol:
			success = True
			break

		if np.absolute(H_0 / H) > np.sqrt(2.):
			# guess for f is too high
			f_c_upper_bound = f
		else:
			# guess for f is too low
			f_c_lower_bound = f
		
	return f


if __name__ == "__main__":
	debug = True
	
	#
	# 	common parameters
	#
	
	A_OL = 1E6
	R_L = 10E3			# [Ohm]
	R_1_SE = 10250.		# [Ohm]
	R_2_SE = 10250.		# [Ohm]
	R_3_SE = 10250.		# [Ohm]
	R_4a_SE = 10E3		# [Ohm]
	R_4b_SE = 250.		# [Ohm]
	R_1_diff = 10250.		# [Ohm]
	R_3_diff = 10250.		# [Ohm]
	R_2a_diff = 10E3		# [Ohm]
	R_4a_diff = 10E3		# [Ohm]
	R_2b_diff = 250.		# [Ohm]
	R_4b_diff = 250.		# [Ohm]

	
	#
	# 	first plot: phase margin vs. GBW for smallest and largest load capacitance
	#

	def first_plot():
		c_fb_list = [1E-12, 5E-12, 50E-12]	# [F]
		GBW = 10E6		# [Hz]
		c_load_list = np.array([0., 1E-6])		# [F]
		
		n_c_load_vals = len(c_load_list)
		n_c_fb_vals = len(c_fb_list)
		n_c_load_vals = len(c_load_list)
		frequencies = None

		f, pm, f_0dB, loop_gains = run_phase_margin_vs_cload_sweep(c_load_list=c_load_list, c_fb_list=c_fb_list, GBW=GBW, A_OL=A_OL, R_L=R_L, fn_snip="_[GBW=" + "{0:E}".format(GBW) + "]", return_loop_gains=True, debug=debug)
		loop_gains_mag = np.absolute(loop_gains)
		loop_gains_phase = np.angle(loop_gains)
		loop_gains_phase = np.unwrap(loop_gains_phase, axis=-1)

		for c_load_idx, c_load in enumerate(c_load_list):
			for fext in ["png", "pdf"]:
				_mag = loop_gains_mag[:, :, c_load_idx, :]
				_ang = 180. / np.pi * loop_gains_phase[:, :, c_load_idx, :]
				labels = ["$C_{FB}=" + "{0:E}".format(c_fb) + "~$pF" for c_fb in c_fb_list]

				#
				# 	compute marker positions to indicate 0 dB points
				#

				_marker_size = 40.
				zero_db_freqs = []
				zero_db_angs = []
				zero_db_markers = []
				zero_db_markers_size = []
				zero_db_marker_colours = []
				for circuit_idx, circuit_name in enumerate(circuit_names):
					for c_fb_idx, c_fb in enumerate(c_fb_list):
						_freqs = find_x(1., f, _mag[circuit_idx, c_fb_idx, :])
						zero_db_freqs.append(_freqs)
						_freqs = np.array(_freqs)
						if len(_freqs.shape) == 0:
							_freqs = _freqs[np.newaxis]
						for _freq in _freqs:
							zero_db_angs.append(np.interp(_freq, f, _ang[circuit_idx, c_fb_idx, :]))
						marker, marker_size = get_marker_style(circuit_name, _marker_size=_marker_size)
						zero_db_markers.append(marker)
						zero_db_markers_size.append(marker_size)
						zero_db_marker_colours.append(create_2d_colour_map(n_circuits, n_c_fb_vals)[1][circuit_idx, c_fb_idx, :])
				zero_db_mags = np.ones_like(zero_db_freqs)

				plot_bode(fn="/tmp/sim_phase_margins_[GBW="+"{0:E}".format(GBW)+"]_[C_L=" + "{0:E}".format(1E6*c_load) + "uF]." + fext, f=f, mag=_mag, ang=_ang, ckt_ids=None, labels=labels, colours=None, markers=None, figsize=(8., 5.), ylim_mag=None, ylim_ang=(max(-181., np.amin(_ang)), np.amax(_ang)), colourmap=mpl.cm.jet, mag_ax_ylabel="Magnitude", ang_ax_ylabel="Phase", title="Bode plot, $C_L$ = " + "{0:E}".format(1E6*c_load) + " uF", log_scale=20., markers_f=zero_db_freqs, markers_mag=zero_db_mags, markers_ang=zero_db_angs, marker_colours=zero_db_marker_colours, intersect_markers=zero_db_markers, markers_size=zero_db_markers_size, log_y_axis=True)

				#plot_phase_margins(pm, c_fb_list=c_fb_list, c_load_list=c_load_list, fn="/tmp/sim_phase_margins_[GBW="+"{0:E}".format(GBW)+"]." + fext, labels_diff=["$C_L$ = " + "{:.2E}".format(c_load_list[i]) + ")" for i in range(n_c_load_vals)], labels_se=["Single-ended ($C_L$ = " + "{0:E}".format(c_load_list[i]) + ")" for i in range(n_c_load_vals)], figsize=(8., 6.), ang_ylim=(40., 161.), title="Phase margins (GBW = "+"{0:E}".format(GBW)+")", MARKER_SIZE=0)

	#first_plot()

	#
	#	second plot: PM vs C_FB for various values of C_L and constant GBW
	#
	# 	use a larger amount of values for C_L to show the trend
	#
	
	def second_plot():

		#
		# 	parameters
		#

		GBW = 10E6

		#c_fb_list = np.logspace(-12, -10, 10)	# [F]		was: 200 points
		#c_fb_list = np.array(list(c_fb_list[::2]))
		#c_fb_list = np.array([c_fb_list[0], c_fb_list[-1]])
		#c_fb_list = np.array([1E-12, 10E-12, 100E-12])	# [F]
		c_fb_list = np.logspace(-12, -10, 20)  # XXX: was: 40 points
	
		c_load_list = np.hstack([[0.], np.logspace(-11, -9, 0), [1E-6, 10E-6, 100E-6]])   # was: 9 points
		#c_load_list = np.array([1E-6])
		#c_load_list = np.array([0., 100E-12, 1E-6])
		

		#
		#
		#
		
		n_c_load_vals = len(c_load_list)

		f, pm, f_0dB = run_phase_margin_vs_cload_sweep(c_load_list=c_load_list, c_fb_list=c_fb_list, GBW=GBW, A_OL=A_OL, R_L=R_L, fn_snip="_[GBW=" + "{0:E}".format(GBW) + "]_[A_OL=" + "{0:E}".format(A_OL) + "]")

		for fext in ["png", "pdf"]:
			plot_phase_margins_vs_cfb(pm, circuit_names=circuit_names, c_fb_list=c_fb_list, c_load_list=c_load_list, fn="/tmp/sim_phase_margins_[GBW="+"{0:E}".format(GBW)+"]_[A_OL=" + "{0:E}".format(A_OL) + "]." + fext, figsize=(8., 6.), ang_ylim=(40., 161.), title="Phase margins (GBW = "+"{0:E}".format(GBW) + ")", MARKER_SIZE=0)

	second_plot()

	
	#
	#	third plot: best PM, frequency of best PM, and best C_FB for various C_L and GBW
	#

	def third_plot():
		
		#
		# 	parameters
		#
		
		#GBW_list = np.array([1E6, 2E6, 3E6, 4E6, 5E6, 10E6, 20E6, 30E6, 40E6, 50E6, 100E6])
		#GBW_list = np.array([10E6])	# [Hz]
		GBW_list = np.logspace(6, 8, 20)	# no fewer than 20

		#c_fb_list = np.array([1E-12, 10E-12, 100E-12])	# [F]
		#c_fb_list = np.linspace(1E-12, 12E-12, 100)#
		c_fb_list = np.logspace(-12, -10, 40)	# [F] XXX was 40

		#c_load_list = np.array([0.])#, 10E-12, 100E-12, 1E-9, 1E-6])	# [F]
		c_load_list = np.array([0., 10E-12, 100E-12, 1E-9, 1E-6])	# [F]

		
		#
		#
		#
		
		n_c_load_vals = len(c_load_list)
		best_pm = np.nan * np.ones((n_circuits, len(GBW_list), n_c_load_vals))
		best_pm_f = np.nan * np.ones((n_circuits, len(GBW_list), n_c_load_vals))
		best_c_fb = np.nan * np.ones((n_circuits, len(GBW_list), n_c_load_vals))
		for GBW_idx, GBW in enumerate(GBW_list):
			f, pm, f_0dB = run_phase_margin_vs_cload_sweep(c_load_list=c_load_list, c_fb_list=c_fb_list, GBW=GBW, A_OL=A_OL, R_L=R_L, fn_snip="_[GBW=" + "{0:E}".format(GBW) + "]_[A_OL=" + "{0:E}".format(A_OL) + "]")

			for fext in ["png", "pdf"]:
				plot_phase_margins_vs_cfb(pm, circuit_names=circuit_names, c_fb_list=c_fb_list, c_load_list=c_load_list, fn="/tmp/_sim_phase_margins_vs_C_L[GBW="+"{0:E}".format(GBW)+"]_[A_OL=" + "{0:E}".format(A_OL) + "]." + fext, figsize=(8., 6.), ang_ylim=(40., 161.), title="Phase margins (GBW = "+"{0:E}".format(GBW)+")", MARKER_SIZE=0)

			print("**** ANALYSIS ****")

			#kernel = sp.signal.gaussian(10, 5)
			#_pm_filt = sp.filter.convolve1d(pm, kernel, axis=1)

			best_c_fb_idx = np.argmax(pm, axis=1)	# argmax to find "best" C_FB
			for circuit_idx, circuit_name in enumerate(circuit_names):
				for c_load_idx, C_L in enumerate(c_load_list):
					best_pm[circuit_idx, GBW_idx, c_load_idx] = pm[circuit_idx, best_c_fb_idx[circuit_idx, c_load_idx], c_load_idx]
					best_pm_f[circuit_idx, GBW_idx, c_load_idx] = f_0dB[circuit_idx, best_c_fb_idx[circuit_idx, c_load_idx], c_load_idx]
					best_c_fb[circuit_idx, GBW_idx, c_load_idx] = c_fb_list[best_c_fb_idx[circuit_idx, c_load_idx]]
					print("\t[Circuit: " + circuit_name + "\tC_L = " + "{0:E}".format(C_L) + " F\tGBW = " + "{0:E}".format(GBW/1E6) + " MHz]\tBest achieved PM = " + str(int(best_pm[circuit_idx, GBW_idx, c_load_idx])) + " at frequency " + "{0:E}".format(best_pm_f[circuit_idx, GBW_idx, c_load_idx]) + " Hz with C_FB = " + si_format(c_fb_list[best_c_fb_idx[circuit_idx, c_load_idx]]) + "F")

		for fext in ["png", "pdf"]:
			for c_load_idx, C_L in enumerate(c_load_list):
				plot_pm_vs_gbw(best_pm[..., c_load_idx], best_pm_f[..., c_load_idx], best_c_fb[..., c_load_idx], gbw_list=GBW_list, circuit_names=circuit_names, c_fb_list=c_fb_list, c_load_list=c_load_list, fn="/tmp/_sim_phase_margins_vs_gbw_[C_L="+"{0:E}".format(C_L)+"]." + fext, figsize=(8., 10.))
				# , labels_diff=["$C_L$ = " + "{:.2E}".format(C_L)], labels_se=["Single-ended ($C_L$ = " + "{:.2E}".format(C_L) + ")"]

			plot_pm_vs_gbw2(best_pm, best_pm_f, best_c_fb, circuit_names=circuit_names, gbw_list=GBW_list, c_fb_list=c_fb_list, c_load_list=c_load_list, fn="/tmp/sim_phase_margins_vs_gbw_and_C_L." + fext, labels_diff=["$C_L$ = " + "{:.2E}".format(C_L)], labels_se=["Single-ended ($C_L$ = " + "{:.2E}".format(C_L) + ")"], figsize=(8., 8.), MARKER_SIZE=0)

	third_plot()

