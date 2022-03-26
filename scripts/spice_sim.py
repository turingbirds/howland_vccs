import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import re
import sys
from guppy import hpy
from invoke import exec_binary
from plot_funcs import plot_bode, brightnessAdjust


def c_fb_to_corner_freq(transfer_func, circuit_idx, C_FB, transfer_func_args, f_0=1., tol=1E-6):
	"""find f such that H(f) = H(f_0) / sqrt(2)"""
	
	H_0 = transfer_func(f=f_0, C_FB=C_FB, **transfer_func_args)
	f_c_upper_bound = 1 / (2 * np.pi * transfer_func_args["R_1"] * C_FB)
	f_c_lower_bound = 0.
	success = False
	while not success:
		f = (f_c_upper_bound + f_c_lower_bound) / 2.
		H = transfer_func(f=f, C_FB=C_FB, **transfer_func_args)

		print("guess = " + str(f) + " Hz, H = " + str(H) + ", H_0 = " + str(H_0) + ", target = " + str(H_0/np.sqrt(2.)))

		if np.abs(np.abs(H_0 / H) - np.sqrt(2.)) < tol:
			success = True
			break

		if np.abs(H_0 / H) > np.sqrt(2.):
			# guess for f is too high
			f_c_upper_bound = f
		else:
			# guess for f is too low
			f_c_lower_bound = f
		
	return f

def str_first_comp(s, substr):
	return s[:len(substr)] == substr


def lines_get_split(lines, key):
	for idx, l in enumerate(lines):
		if str_first_comp(l, key):
			#return re.split("[ \t\n\r]", l)
			return re.findall('\S+\([^\)]*\)|\S+', l)


def lines_first_replace(lines, oldl, newl, append_line_ending=True):
	for idx, l in enumerate(lines):
		if str_first_comp(l, oldl):
			
			line_ending = "\n"
			if re.match("\r\n", lines[idx]):
				line_ending = "\r\n"
		
			lines[idx] = newl
			if append_line_ending:
				lines[idx] += line_ending

			return


def read_ltspice_ascii_raw(fname, verbose=False, variable_type=np.complex):	# XXX: was: np.float
	print("* Reading ASCII ltspice output from file: " + fname)
	f = open(fname, 'r')
	try:
		lines = f.readlines()

		# find dimensions for preallocation
		n_variables = None
		n_timepoints = None
		header_dict = dict()
		current_section = 'header'
		timestep = -1
		variable_idx = 0
		for l in lines:
			l = l.strip("\t\n\r ")

			if str_first_comp(l, "Variables:"):
				assert not n_variables is None, "\"No. Variables\" entry missing from header"
				variable_idx = 0
				variable_names = n_variables * [None]
				current_section = 'variables'
				continue
			elif str_first_comp(l, "Values:"):
				assert not n_timepoints is None, "\"No. Points\" entry missing from header"
				variable_idx = 0
				print("\tVariable type: " + str(variable_type))
				variable_values = np.nan * np.ones((n_variables, n_timepoints), dtype=variable_type)
				current_section = 'values'
				continue

			if current_section == 'header':
				variable_name = l[:l.index(":")]
				variable_value = l[l.index(":")+1:]
				header_dict[variable_name] = variable_value
				if variable_name == "No. Points":
					n_timepoints = int(variable_value)
				elif variable_name == "No. Variables":
					n_variables = int(variable_value)
				elif variable_name == "Flags":
					if "complex" in l:
						variable_type = np.complex
				if verbose:
					print("Header item \"" + variable_name + "\" = " + variable_value)
			elif current_section == 'variables':
				# read variable names
				variable_names[variable_idx] = l.strip("\t\n\r ").split("\t")[1]
				if verbose:
					print("Variable " + str(variable_idx) + " has name: " + str(variable_names[variable_idx]))
			elif current_section == 'values':
				lsplit = l.split()	#re.split(r"[\,|\t|\;]+", l)
				if len(lsplit) > 1:
					variable_idx = 0
					timestep += 1
					val_str = lsplit[-1]
					assert timestep == int(lsplit[0])
				else:
					val_str = l
				
				if "," in lsplit[-1]:
					try:
						# complex number, consists of real and imaginary parts separated by a comma
						val_re, val_im = [float(s) for s in lsplit[-1].split(",")]
						val = np.complex(val_re, val_im)
					except:
						val = np.nan
				else:
					val = float(lsplit[-1])

				variable_values[variable_idx, timestep] = val

			variable_idx += 1

	finally:
		f.close()

	return header_dict, variable_names, variable_values, n_timepoints


def write_spice(lines, fname, verbose=False):
	if verbose:
		print("* Writing SPICE to file: " + fname)

	f = open(fname, "w")
	try:
		for l in lines:
			f.write(l)
	except:
		print("\tError writing SPICE to file: " + fname)
		exit(1)
	finally:
		f.close()

		
def find_variable_idx(variable_names, variable_name, case_sens=False):
	try:
		if case_sens:
			return next(idx for (idx, obj) in enumerate(variable_names) if obj == variable_name)
		else:
			return next(idx for (idx, obj) in enumerate(variable_names) if obj.upper() == variable_name.upper())
	except:
		e = sys.exc_info()[0]
		print("Error: variable by name \"" + variable_name + "\" not found")
		print(str(e))

	 
def read_orig_spice(fname):
	print("* Reading SPICE file: " + fname)
	f = None
	try:
		f = open(fname, "r")
		orig_spice = f.readlines()
	except:
		print("Error reading original SPICE file: " + fname)
		exit(1)
	finally:
		if not f is None:
			f.close()

	return orig_spice


def pad_or_truncate_list(l, n, default_value=0.):
    return l[:n] + (n - len(l)) * [default_value]


def run_circuit(orig_spice, new_vals, scad3_exe_path, mod_circuit_path, ltspice_results_path, mod_circuit_path_windows, verbose=False):
	mod_spice = copy.deepcopy(orig_spice)
	
	for variable_name, idx_val_pairs in new_vals.iteritems():
		s = lines_get_split(orig_spice, variable_name)
		if s is None:
			raise Exception("Variable by name \"" + variable_name + "\" not found")
		for variable_idx, variable_val_str in idx_val_pairs:
			if variable_idx >= len(s):
				s = pad_or_truncate_list(s, variable_idx + 1, default_value=0.)
			s[variable_idx] = variable_val_str
		lines_first_replace(mod_spice, variable_name, " ".join(s))
	write_spice(mod_spice, mod_circuit_path, verbose=verbose)
	cmd_line_list = [scad3_exe_path, "-ascii", "-b", mod_circuit_path_windows]
	exec_binary(cmd_line_list, verbose=verbose)
	header_dict, variable_names, variable_values, n_timepoints = read_ltspice_ascii_raw(ltspice_results_path, verbose=verbose)

	return header_dict, variable_names, variable_values, n_timepoints

