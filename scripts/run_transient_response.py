# -*- coding: utf-8 -*-

"""

simulate the transient response

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

def plot_transient_sim_results(timevec, I_L, sweep_variable, sweep_variable_latex, sweep_variable_vals, circuit_names, timevec_alt=None, I_L_alt=None, t_offset_alt=0., marker_size=0., mag_ax_ylim=None, ang_ax_ylim=None, figsize=(8., 4.), fn="/tmp/R_out.png", ax_locator_xwidth = .25):

	fig = plt.figure(figsize=figsize)
	gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[1, 5])
	gs.update(wspace=.05, hspace=.05)
	ax = plt.subplot(gs[1])	# main axis
	ax2 = plt.subplot(gs[0])	# (left) side axis

	col_se = np.array([.95, .43, .0])
	col_se2 = brightnessAdjust(col_se, 1.)
	col_se1 = brightnessAdjust(col_se, .6)
	col_diff = np.array([.275, .443, .835])
	col_diff2 = brightnessAdjust(col_diff, 1.)
	col_diff1 = brightnessAdjust(col_diff, .6)
	cm_se = mpl.colors.LinearSegmentedColormap.from_list("se_cm", [col_se2, col_se1])
	cm_diff = mpl.colors.LinearSegmentedColormap.from_list("diff_cm", [col_diff2, col_diff1])

	for sweep_variable_idx, sweep_variable_val in enumerate(sweep_variable_vals):
		for circuit_idx, circuit_name in enumerate(circuit_names):
			n_sweeps = len(sweep_variable_vals)
			if circuit_name == "single-ended":
				colours = cm_se(sweep_variable_idx / float(n_sweeps - 1))[:3]
			else:
				assert circuit_name == "differential"
				colours = cm_diff(sweep_variable_idx / float(n_sweeps - 1))[:3]

			ax.plot(1E6*timevec, 1E3*I_L[circuit_idx, sweep_variable_idx, :], label=sweep_variable_latex + " = " + str(sweep_variable_val) + " (" + circuit_names[circuit_idx] + ")", color=colours, marker="o", markersize=marker_size, linewidth=2.)

			if not timevec_alt is None:
				ax2.plot(1E6*timevec_alt, 1E3*I_L_alt[circuit_idx, sweep_variable_idx, :], label=sweep_variable_latex + " = " + str(sweep_variable_val) + " (" + circuit_names[circuit_idx] + ")", color=colours, marker="o", markersize=marker_size, linewidth=2.)
	
	ax.set_xlim(1E6*np.amin(timevec), 1E6*np.amax(timevec))
	ax2.set_xlim(1E6*(t_offset_alt + np.amin(timevec_alt)), 1E6*(np.amax(timevec_alt)))
	
	for _ax in [ax, ax2]:
		_ax.grid(True)
		_ax.set_ylim(1E3*min(np.amin(I_L), np.amin(I_L_alt)), 1E3*max(np.amax(I_L), np.amax(I_L_alt)))
	
	ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=ax_locator_xwidth))

	ax.set_ylabel("")
	ax.set_yticklabels([])
	ax2.set_ylabel(r"$I_L$ [mA]")
	ax.set_xlabel(r"Time [us]")
	leg = ax.legend()
	leg.get_frame().set_alpha(.6)
	fig.savefig(fn)
	fig.savefig(fn + ".pdf")
	plt.close(fig)


	
def run_transient_sim(plot_raw_data=False):

	#
	# 	parameters
	#

	VARIABLE_NAME_TIME = "time"
	VARIABLE_NAME_IOUT_R = "I(RL)"
	VARIABLE_NAME_IOUT_C = "I(CL)"
	
	CIRCUIT_IDX_SE = 0
	CIRCUIT_IDX_DIFF = 1
	circuit_names = ["single-ended", "differential"]
	n_circuits = 2

	scad3_exe_path = "/cygdrive/c/Program Files (x86)/LTC/LTspiceIV/scad3.exe"

	circuit_path = {
		"single-ended" : "/cygdrive/c/sim/single_ended_howland.net",
		"differential" : "/cygdrive/c/sim/diff_howland.net"
	}

	ltspice_results_path = "/cygdrive/c/sim/ckt_tmp.raw"
	mod_circuit_path = "/cygdrive/c/sim/ckt_tmp.net"
	mod_circuit_path_windows = r"c:\sim\ckt_tmp.net"		# XXX mixed path styles because using ltspice & CygWin under Windows

	C_L_vals = [1E-15, 100E-12, 1E-9, 10E-9, 1E-6]		# [F]
	C_FB = {"single-ended" : 9E-12, "differential" : 9E-12}	# optimal values for C_FB for each circuit [F]
	R_in_series = 10E3
	T_sim = 10E-6	# total simulation time [s]
	T_sim_alt = 1.4E-6	# total simulation time for alternate run [s]
	dt_max = 0.#1E-9	# maximum timestep [s]
	t_offset_alt = .9E-6
	
	#
	#	memory profiling
	#

	hp = hpy()
	before = hp.heap()


	#
	#	main loop: for RINSERIES > 0
	#

	I_L = None

	for circuit_idx, circuit_name in enumerate(circuit_names):
		print("===========================================")
		print("      CIRCUIT: " + str(os.path.basename(circuit_path[circuit_name])))
		print("===========================================")

		_C_FB = C_FB[circuit_name]
		
		orig_spice = read_orig_spice(circuit_path[circuit_name])

		for C_L_idx, C_L in enumerate(C_L_vals):
			print("\tC_L = " + str(C_L))
			I_L_avg = 0.
			Delta_I_L = 0.
			new_vals = {
				".ac" : [(0, ".tran"), (1, "0"), (2, "{0:E}".format(T_sim)), (3, "0"), (4, "{0:E}".format(dt_max))],
				".param CFB" : [(2, "{0:E}".format(_C_FB))],
				".param CLOAD" : [(2, "{0:E}".format(C_L))],
				"V1" : [(3, "PULSE(0.2475 0.25 1E-6 1E-12 1E-12 50E-6)"), (4, "")],
			}

			if circuit_name == "differential":
				new_vals[".param RINSERIES"] = [(2, "{0:E}".format(R_in_series))]

			header_dict, variable_names, variable_values, n_timepoints = run_circuit(orig_spice, new_vals, scad3_exe_path, mod_circuit_path, ltspice_results_path, mod_circuit_path_windows, )

			if I_L is None:
				# N.B. can only allocate now that we know the number of timepoints
				I_L = np.empty((2, len(C_L_vals), n_timepoints), dtype=np.float)
				timevec = variable_values[find_variable_idx(variable_names, VARIABLE_NAME_TIME), :].astype(np.float)
				assert len(timevec) == n_timepoints
			else:
				assert n_timepoints == I_L.shape[-1]

			variable_idx_Iout_C = find_variable_idx(variable_names, VARIABLE_NAME_IOUT_C)
			variable_idx_Iout_R = find_variable_idx(variable_names, VARIABLE_NAME_IOUT_R)
			I_L[circuit_idx, C_L_idx, :] = variable_values[variable_idx_Iout_R, :] + variable_values[variable_idx_Iout_C, :]
	

	#
	#	alternate loop: for RINSERIES = 0
	#

	I_L_alt = None

	for circuit_idx, circuit_name in enumerate(circuit_names):
		print("===========================================")
		print("      CIRCUIT: " + str(os.path.basename(circuit_path[circuit_name])))
		print("===========================================")

		_C_FB = C_FB[circuit_name]
				
		orig_spice = read_orig_spice(circuit_path[circuit_name])

		for C_L_idx, C_L in enumerate(C_L_vals):
			print("\tC_L = " + str(C_L))
			new_vals = {
				".ac" : [(0, ".tran"), (1, "0"), (2, "{0:E}".format(T_sim_alt)), (3, "0"), (4, "{0:E}".format(dt_max))],
				".param CFB" : [(2, "{0:E}".format(_C_FB))],
				".param CLOAD" : [(2, "{0:E}".format(C_L))],
				"V1" : [(3, "PULSE(0.2475 0.25 1E-6 1E-12 1E-12 50E-6)"), (4, "")],
			}
		
			header_dict, variable_names, variable_values, n_timepoints = run_circuit(orig_spice, new_vals, scad3_exe_path, mod_circuit_path, ltspice_results_path, mod_circuit_path_windows, )

			if I_L_alt is None:
				# N.B. can only allocate now that we know the number of timepoints
				I_L_alt = np.empty((2, len(C_L_vals), n_timepoints), dtype=np.float)
				timevec_alt = variable_values[find_variable_idx(variable_names, VARIABLE_NAME_TIME), :].astype(np.float)
				assert len(timevec_alt) == n_timepoints
			else:
				assert n_timepoints == I_L_alt.shape[-1]

			variable_idx_Iout_C = find_variable_idx(variable_names, VARIABLE_NAME_IOUT_C)
			variable_idx_Iout_R = find_variable_idx(variable_names, VARIABLE_NAME_IOUT_R)
			I_L_alt[circuit_idx, C_L_idx, :] = variable_values[variable_idx_Iout_R, :] + variable_values[variable_idx_Iout_C, :]
	


	#
	# 	plotting
	#

	plot_transient_sim_results(timevec, I_L, timevec_alt=timevec_alt, I_L_alt=I_L_alt, t_offset_alt=t_offset_alt, sweep_variable="$C_L$", sweep_variable_latex="$C_L$", sweep_variable_vals=C_L_vals, circuit_names=circuit_names, fn="/tmp/transient_response.png", mag_ax_ylim=(10**2-1E-9, 10**9+1E-9), ang_ax_ylim=(-150.-1E-9, 0.+1E-9))

	import pdb;pdb.set_trace()
	
	#
	#	memory profiling
	#

	after = hp.heap()
	leftover = after - before
	print(leftover)




if __name__ == "__main__":

	debug = True

	run_transient_sim()
