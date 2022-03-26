#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.font_manager as fm
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

import matplotlib.ticker as plticker
from matplotlib.ticker import FormatStrFormatter

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import os
import csv


def read_array_from_csv(fname, n_header_lines, num_channels=3):
	reader = csv.reader(open(fname, "r"), delimiter=",")
	x = list(reader)
	x = np.array(x)

	dt = float(x[1][5])		# [s]
	print("Detected dt = " + str(1E6*dt) + " us")

	n_rows = x.shape[0] - n_header_lines
	_x = np.empty((n_rows, num_channels))
	timevec = np.zeros(n_rows)
	for i in range(n_rows):
		# assert i == int(x[i + n_header_lines][0])
		if i > 0:
			_prev_timestep = int(x[i + n_header_lines - 1][0])
		else:
			_prev_timestep = -1
		_x[i, :] = np.array(x[i + n_header_lines][1:num_channels+1]).astype(np.float)
		timevec[i] = i * dt
		#if i > 0:
			#n_timesteps = int(x[i + n_header_lines][0]) - _prev_timestep
			#timevec[i] = timevec[i - 1] + n_timesteps * float(x[1][2 + num_channels])

	return timevec, _x


#######################################

base_dir = "../data/transients_measurements"
fname_se_transient_input = os.path.join(base_dir, "load_transient_se_100R_10kR.csv")
fname_diff_transient_input = os.path.join(base_dir, "load_transient_diff_100R_10kR.csv")

n_header_lines = 2

R_L = 10E3


#######################################


timevec_rl_se, data_se_transient_input_rl = read_array_from_csv(fname_se_transient_input, n_header_lines=n_header_lines)
timevec_rl_diff, data_diff_transient_input_rl = read_array_from_csv(fname_diff_transient_input, n_header_lines=n_header_lines)

timevec_rl_se *= 1E6		# [s] to [us]
timevec_rl_diff *= 1E6		# [s] to [us]

idx_end = np.argmin((timevec_rl_se - 800)**2) + 1
data_se_transient_input_rl = data_se_transient_input_rl[:idx_end, ...]
data_diff_transient_input_rl = data_diff_transient_input_rl[:idx_end, ...]
timevec_rl_se = timevec_rl_se[:idx_end]
timevec_rl_diff = timevec_rl_diff[:idx_end]

data_se_transient_input_rl[:, 0] /= 10		# oops due to probe 1x/10x setting
data_diff_transient_input_rl[:, 0] /= 10		# oops due to probe 1x/10x setting

#######################################


fig = plt.figure(figsize=(8., 6.))

gs = mpl.gridspec.GridSpec(3, 1,
                       height_ratios=[2, 2, 1]
                       )

ax_I_L = plt.subplot(gs[0, 0])
ax_V_L = plt.subplot(gs[1, 0])
ax_pulse = plt.subplot(gs[2, 0])

ax_V_L.plot(timevec_rl_diff, 2*data_diff_transient_input_rl[:, 2], alpha=1, linewidth=2, zorder=99, color="#4671d5")
ax_V_L.plot(timevec_rl_se, data_se_transient_input_rl[:, 2], alpha=1, linewidth=2, zorder=99, color="#f36e00")
ax_V_L.set_ylabel("$V_L$ [V]")
ax_V_L.set_xticklabels([])

ax_pulse.plot(timevec_rl_diff, data_diff_transient_input_rl[:, 1], linewidth=2, zorder=99, color="#a379c9")
ax_pulse.set_yticks([])
ax_pulse.set_ylabel("$V_{ctrl}$", size=18.)
ax_pulse.set_xlabel(u"Time [µs]", size=18.)

ax_I_L.plot(timevec_rl_se, data_se_transient_input_rl[:, 0] / .51, color="#f36e00", linewidth=2, zorder=99, label="SE")
ax_I_L.plot(timevec_rl_diff, -data_diff_transient_input_rl[:, 0] / .51, color="#4671d5", linewidth=2, zorder=99, label="diff")
ax_I_L.set_ylim(np.amin(data_se_transient_input_rl[:, 0] / .51), .1)
ax_I_L.set_ylabel("$\Delta I_L$ [mA]", size=18.)
ax_I_L.set_xticklabels([])

ax_pulse.grid(True, axis="x")
ax_I_L.set_xticklabels([])

for _ax in [ax_I_L, ax_V_L, ax_pulse]:
	_ax.set_xlim(0, np.amax(timevec_rl_se))
	_ax.grid(True)

	l, b, w, h = _ax.get_position().bounds
	_ax.set_position([l, b + .05, w, h])

leg = ax_I_L.legend(loc="lower left", handlelength=1.8, fontsize=16)#. , fancybox=True, framealpha=1)

fig.savefig("/tmp/load_transient.png")
fig.savefig("/tmp/load_transient.pdf")
plt.close(fig)
