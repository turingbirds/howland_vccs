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

from plot_funcs import plot_bode, brightnessAdjust, find_x, log_interp, create_2d_colour_map, plot_phase_margins_vs_cfb, plot_pm_vs_gbw, plot_pm_vs_gbw2, get_dual_linear_colour_map
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

circuit_names = ["single-ended", "differential"]

circuit_path = {
	"single-ended" : "/cygdrive/c/sim/single_ended_howland.net",
	"differential" : "/cygdrive/c/sim/diff_howland.net"
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


def run_transfer_sim(circuit_fn, C_FB, GBW, A_OL, R_L, C_L, fn_snip="", title_snip="", make_bode_plot=False, verbose=False):
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

	I_L = variable_values[find_variable_idx(variable_names, "I(RL)"), ...]

	return frequencies, I_L


##############################################################################

if __name__ == "__main__":
	debug = True

	#
	# 	common parameters
	#

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
	# 	op-amp parameters
	#

	c_fb = {"single-ended" : 6.6E-12, \
	        "differential" : 3.3E-12 }	# [F]
	A_OL = 1E6
	GBW = 10E6		# [Hz]
	C_L = 0.		# [F]


	frequencies = None

	for circuit_idx, circuit_name in enumerate(circuit_names):
		_c_fb = c_fb[circuit_name]
		title = circuit_name + " ($C_L=" + "{0:E}".format(C_L) + "$, " + "$C_{FB}=" + "{0:E}".format(_c_fb) + "$)"
		_fn_snip = "transfer_func_" + circuit_name + "_[C_L=" + str(C_L) + "]_[C_FB=" + "{0:E}".format(_c_fb) + "]"

		_frequencies, _I_L = run_transfer_sim(circuit_path[circuit_name], _c_fb, GBW, A_OL, R_L, C_L, fn_snip=_fn_snip, title_snip=title)

		I_L_mag = np.abs(_I_L)
		I_L_phase = np.angle(_I_L)

		if frequencies is None:
			frequencies = _frequencies
		else:
			np.testing.assert_almost_equal(frequencies, _frequencies)	# check for mismatch between returned frequencies between simulations

		#
		# 	save to file
		#

		fn = "/tmp/sim_bode_freqs.txt"
		print("* Saving data to: " + fn)
		np.savetxt(fn, frequencies)

		fn = "/tmp/sim_bode_" + _fn_snip + ".txt"
		print("* Saving data to: " + fn)
		np.savetxt(fn, _I_L)


		#
		# 	plot
		#

		for fext in ["png", "pdf"]:
			fn = "/tmp/sim_bode_" + _fn_snip + "." + fext
			print("* Writing Bode plot to: " + fn)
			plot_bode(fn, frequencies, I_L_mag, 180. / np.pi * I_L_phase, title=title)
