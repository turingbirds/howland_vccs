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

from plot_funcs import plot_bode, brightnessAdjust
from spice_sim import *

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import re
import sys
from guppy import hpy
from invoke import exec_binary


##############################################################################

def plot_R_out_sweep_results(f, R_out, sweep_variable, sweep_variable_latex, sweep_variable_vals, circuit_names, marker_size=0., mag_ax_ylim=None, ang_ax_ylim=None, figsize=(8., 5.), fn="/tmp/R_out.png"):

	fig = plt.figure(figsize=figsize)
	gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[3, 1])
	ax = plt.subplot(gs[0])
	ax2 = plt.subplot(gs[1])

	labels = ["$" + sweep_variable_latex + "=10^{"+str(int(np.log10(sweep_variable_val))) + "}$" for sweep_variable_val in sweep_variable_vals]
	
	col_se = np.array([.95, .43, .0])
	col_se2 = brightnessAdjust(col_se, 1.)
	col_se1 = brightnessAdjust(col_se, .6)
	col_diff_cs = np.array([.275, .835, .443])
	col_diff_cs2 = brightnessAdjust(col_diff_cs, 1.)
	col_diff_cs1 = brightnessAdjust(col_diff_cs, .6)
	col_diff = np.array([.275, .443, .835])
	col_diff2 = brightnessAdjust(col_diff, 1.)
	col_diff1 = brightnessAdjust(col_diff, .6)
	cm_se = mpl.colors.LinearSegmentedColormap.from_list("se_cm", [col_se2, col_se1])
	cm_diff = mpl.colors.LinearSegmentedColormap.from_list("diff_cm", [col_diff2, col_diff1])
	cm_diff_cs = mpl.colors.LinearSegmentedColormap.from_list("diff_cs_cm", [col_diff_cs2, col_diff_cs1])

	min_R_out_ang = np.inf
	max_R_out_ang = -np.inf
	for sweep_variable_idx, sweep_variable_val in enumerate(sweep_variable_vals):
		for circuit_idx, circuit_name in enumerate(circuit_names):
			n_sweeps = len(sweep_variable_vals)
			if circuit_name == "single-ended":
				colours = cm_se(sweep_variable_idx / float(n_sweeps - 1))[:3]
			elif circuit_name == "differential-counter-steering":
				colours = cm_diff_cs(sweep_variable_idx / float(n_sweeps - 1))[:3]
			else:
				assert circuit_name == "symmetric"
				colours = cm_diff(sweep_variable_idx / float(n_sweeps - 1))[:3]

			ax.loglog(f[:len(R_out[circuit_idx, sweep_variable_idx, :])], np.abs(R_out[circuit_idx, sweep_variable_idx, :]), label="$" + sweep_variable_latex + " = 10^{"+str(int(np.log10(sweep_variable_val))) + "}$ (" + circuit_names[circuit_idx] + ")", color=colours, marker="o", markersize=marker_size, linewidth=2.)

			_R_out_ang = np.angle(R_out[circuit_idx, sweep_variable_idx, :])
			_R_out_ang = np.unwrap(_R_out_ang)
			_R_out_ang *= 180. / np.pi
			min_R_out_ang = min(min_R_out_ang, np.amin(_R_out_ang))
			max_R_out_ang = max(max_R_out_ang, np.amax(_R_out_ang))

			ax2.semilogx(f[:len(R_out[circuit_idx, sweep_variable_idx, :])], _R_out_ang, label="$" + sweep_variable_latex + " = 10^{"+str(int(np.log10(sweep_variable_val))) + "}$ (" + circuit_names[circuit_idx] + ")", color=colours, marker="o", markersize=marker_size, linewidth=2.)
	
	for _ax in [ax, ax2]:
		_ax.grid(True)
		_ax.set_xlim(np.amin(f), np.amax(f))

	#ax2.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=30.))
	#ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=10.))
	if not mag_ax_ylim is None:
		ax.set_ylim(*mag_ax_ylim)
	if not ang_ax_ylim is None:
		ax2.set_ylim(*ang_ax_ylim)
	else:
		ax2.set_ylim(min_R_out_ang, max_R_out_ang)
	ax.set_ylabel("$|R_{out}|$")
	ax2.set_ylabel(r"$\angle R_{out}$")
	ax2.set_xlabel("Frequency [Hz]")
	leg = ax.legend()
	leg.get_frame().set_alpha(.6)
	fig.savefig(fn)
	fig.savefig(fn + ".pdf")
	plt.close(fig)


def run_R_out_sweep(sweep_variable, sweep_variable_latex, plot_raw_data=False):

	#
	# 	parameters
	#	

	VARIABLE_NAME_IOUT_R = "I(RL)"
	VARIABLE_NAME_IOUT_C = "I(CL)"
	VARIABLE_NAME_VOUT = "V(vl)"
	VARIABLE_NAME_VOUTP = "V(vlpos)"
	VARIABLE_NAME_VOUTN = "V(vlneg)"
	VARIABLE_NAME_FREQUENCIES = "frequency"

	CIRCUIT_IDX_SE = 0
	CIRCUIT_IDX_SYM = 1
	CIRCUIT_IDX_CS = 2
	circuit_names = ["single-ended", "symmetric", "differential-counter-steering"]
	n_circuits = 3

	scad3_exe_path = "/cygdrive/c/Program Files (x86)/LTC/LTspiceIV/scad3.exe"

	circuit_path_R_out_sweep = {
		CIRCUIT_IDX_SE : "/cygdrive/c/sim/single_ended_howland.net",
		CIRCUIT_IDX_SYM : "/cygdrive/c/sim/diff_howland.net",
		CIRCUIT_IDX_CS : "/cygdrive/c/sim/diff_howland_countersteering.net"
	}

	ltspice_results_path = "/cygdrive/c/sim/ckt_tmp.raw"
	mod_circuit_path = "/cygdrive/c/sim/ckt_tmp.net"
	mod_circuit_path_windows = r"c:\sim\ckt_tmp.net"		# mixed path styles because using ltspice & CygWin under Windows


	assert sweep_variable in ["GBW", "A_OL"]
	if sweep_variable == "GBW":
		GBW_vals = [100E6, 10E6, 1E6]		# [Hz]
		A_OL = 100E3		# [1]
		sweep_variable_vals = GBW_vals
	elif sweep_variable == "A_OL":
		A_OL_vals = [1E6, 100E3, 10E3]		# [V/V]
		GBW = 10E6		# [Hz]
		sweep_variable_vals = A_OL_vals

	R_L_list = [10E3, 11E3]		# [Ω]
	assert len(R_L_list) == 2		# comparison between two load conditions
	n_circuits = len(circuit_path_R_out_sweep.keys())
	n_sweeps = len(sweep_variable_vals)
	R_out = None
	Delta_R_L = np.diff(R_L_list)[0]

	#
	#	memory profiling
	#

	hp = hpy()
	before = hp.heap()


	#
	#	main loop
	#

	for circuit_idx, circuit_name in enumerate(circuit_names):
		print("===========================================")
		print("      CIRCUIT: " + str(os.path.basename(circuit_path_R_out_sweep[circuit_idx])))
		print("===========================================")

		orig_spice = read_orig_spice(circuit_path_R_out_sweep[circuit_idx])

		frequencies = None
	
		for sweep_idx, sweep_variable_val in enumerate(sweep_variable_vals):
			if sweep_variable == "GBW":
				GBW = sweep_variable_val
			elif sweep_variable == "A_OL":
				A_OL = sweep_variable_val
			print("GBW = " + str(GBW))
			print("A_OL = " + str(A_OL))

			V_L = None
			I_L = None
			for R_L_idx, R_L in enumerate(R_L_list):
				new_vals = {
					".param AOL" : [(2, str(A_OL))],
					".param GBW" : [(2, str(GBW))],
					".param RLOAD" : [(2, "{0:E}".format(R_L))]
				}
				header_dict, variable_names, variable_values, n_timepoints = run_circuit(orig_spice, new_vals, scad3_exe_path, mod_circuit_path, ltspice_results_path, mod_circuit_path_windows, )

				variable_idx_frequencies = find_variable_idx(variable_names, VARIABLE_NAME_FREQUENCIES)
				_frequencies = variable_values[variable_idx_frequencies]

				if frequencies is None:
					frequencies = np.abs(_frequencies)		# abs() gets rid of complex part
				else:
					np.testing.assert_almost_equal(frequencies, _frequencies)	# check for mismatch between returned frequencies between simulations

				if V_L is None:
					# N.B. can only allocate now that we know the number of frequency points
					n_freq = frequencies.size
					V_L = np.zeros((2, n_freq), dtype=np.complex)
					I_L = np.zeros((2, n_freq), dtype=np.complex)

				if circuit_name == "single-ended":
					V_L[R_L_idx, :] = variable_values[find_variable_idx(variable_names, VARIABLE_NAME_VOUT), :]
				else:
					assert circuit_name in ["symmetric", "differential-counter-steering"]
					V_L[R_L_idx, :] = variable_values[find_variable_idx(variable_names, VARIABLE_NAME_VOUTP), :] - variable_values[find_variable_idx(variable_names, VARIABLE_NAME_VOUTN), :]

				_I_R_L = variable_values[find_variable_idx(variable_names, VARIABLE_NAME_IOUT_R), :]
				I_L[R_L_idx, :] += _I_R_L
				if find_variable_idx(variable_names, VARIABLE_NAME_IOUT_C) > 0:
					_I_C_L = variable_values[find_variable_idx(variable_names, VARIABLE_NAME_IOUT_C), :]
					if not np.any(np.isnan(_I_C_L)):
						I_L[R_L_idx, :] += _I_C_L


			if R_out is None:
				# N.B. can only allocate now that we know the number of frequency points
				n_freq = frequencies.size
				R_out = np.empty((n_circuits, n_sweeps, n_freq), dtype=np.complex)

			#R_out[circuit_idx, sweep_idx, ...] = (V_L[1, :] - V_L[0, :]) / (V_L[0, :] / R_L_list[0] - V_L[1, :] / R_L_list[1])
			#R_out[circuit_idx, sweep_idx, ...] = R_L_list[0] * R_L_list[1] * (V_L[1, :] - V_L[0, :]) / (V_L[0, :] * R_L_list[1] - V_L[1, :] * R_L_list[0])
			_R_out_voltage_based = R_L_list[0] * R_L_list[1] * (V_L[1, :] - V_L[0, :]) / (V_L[0, :] * R_L_list[1] - V_L[1, :] * R_L_list[0])
			_R_out_current_based = (I_L[1, :] * R_L_list[1] - I_L[0, :] * R_L_list[0]) / (I_L[0, :] - I_L[1, :])
			np.testing.assert_allclose(_R_out_voltage_based, _R_out_current_based)
			R_out[circuit_idx, sweep_idx, ...] = _R_out_current_based
			print("R_out = " + str(R_out[circuit_idx, sweep_idx, ...]))
		

	#
	# 	plotting
	#	

	fn = "/tmp/R_out_across_" + sweep_variable + ".png"
	print("Plotting to " + fn)
	plot_R_out_sweep_results(frequencies, R_out, sweep_variable=sweep_variable, sweep_variable_latex=sweep_variable_latex, sweep_variable_vals=sweep_variable_vals, circuit_names=circuit_names, fn=fn, mag_ax_ylim=(10**2-1E-9, 10**9+1E-9), ang_ax_ylim=(-150.-1E-9, 0.+1E-9))
	
	
	#
	#	export results to file
	#

	fn = "/tmp/freqs.txt"
	print("Saving magnitude data to " + fn)
	np.savetxt(fn, frequencies)
	
	for circuit_idx, circuit_name in enumerate(circuit_names):
		for sweep_variable_idx, sweep_variable_val in enumerate(sweep_variable_vals):
			fn = "/tmp/R_out_" + circuit_name + "_[" + sweep_variable + "=" + str(sweep_variable_val) + "].txt"
			print("Saving magnitude data to " + fn)
			np.savetxt(fn, np.abs(R_out[circuit_idx, sweep_variable_idx, :]))
		
	#
	#	memory profiling
	#

	after = hp.heap()
	leftover = after - before
	print(leftover)


if __name__ == "__main__":

	debug = True

	run_R_out_sweep("A_OL", sweep_variable_latex="A_{OL}")
	run_R_out_sweep("GBW", sweep_variable_latex="GBW")
