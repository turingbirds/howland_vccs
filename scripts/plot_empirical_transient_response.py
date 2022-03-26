#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.font_manager as fm
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.ticker as plticker
from matplotlib.ticker import FormatStrFormatter

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import os
import csv


def read_array_from_csv(fname, n_header_lines, num_channels=4):
	reader = csv.reader(open(fname, "r"), delimiter=",")
	x = list(reader)
	x = np.array(x)

	_x = np.empty((x.size - n_header_lines, num_channels))
	timevec = np.zeros(x.size - n_header_lines)
	for i in range(x.size - n_header_lines):
		assert i == int(x[i + n_header_lines][0])
		_x[i, :] = np.array(x[i + n_header_lines][1:num_channels+1]).astype(np.float)
		if i > 0:
			timevec[i] = timevec[i - 1] + float(x[1][6])

	return timevec, _x


#######################################

base_dir = "../data/transients_measurements"
fname_se_transient_input = os.path.join(base_dir, "se_transient.csv")
fname_diff_transient_input = os.path.join(base_dir, "diff_transient.csv")

n_header_lines = 2

I_meas_gain = 510.

#######################################


timevec_rl_se, data_se_transient_input_rl = read_array_from_csv(fname_se_transient_input, n_header_lines=n_header_lines)
timevec_cl_se, data_se_transient_input_cl = read_array_from_csv(fname_diff_transient_input, n_header_lines=n_header_lines)

timevec_rl_diff, data_diff_transient_input_rl = read_array_from_csv(fname_se_transient_input, n_header_lines=n_header_lines)
timevec_cl_diff, data_diff_transient_input_cl = read_array_from_csv(fname_diff_transient_input, n_header_lines=n_header_lines)

timevec_rl_se *= 1E3		# [s] to [ms]
timevec_rl_diff *= 1E3		# [s] to [ms]
timevec_cl_se *= 1E3		# [s] to [ms]
timevec_cl_diff *= 1E3		# [s] to [ms]

idx_offset = np.argmax(data_se_transient_input_rl[:, 0] > 0)
data_se_transient_input_rl = data_se_transient_input_rl[idx_offset:, ...]
data_diff_transient_input_rl = data_diff_transient_input_rl[idx_offset:, ...]
timevec_rl_se = timevec_rl_se[idx_offset:]
timevec_rl_diff = timevec_rl_diff[idx_offset:]
timevec_rl_se -= timevec_rl_se[0]
timevec_rl_diff -= timevec_rl_diff[0]

idx_offset = np.argmax(data_se_transient_input_cl[:, 0] > 0)
data_se_transient_input_cl = data_se_transient_input_cl[idx_offset:, ...]
data_diff_transient_input_cl = data_diff_transient_input_cl[idx_offset:, ...]
timevec_cl_se = timevec_cl_se[idx_offset:]
timevec_cl_diff = timevec_cl_diff[idx_offset:]
timevec_cl_se -= timevec_cl_se[0]
timevec_cl_diff -= timevec_cl_diff[0]

#######################################


fig = plt.figure(figsize=(8., 5.))

gs = mpl.gridspec.GridSpec(3, 2,
                       width_ratios=[1, 1],
                       height_ratios=[2, 2, 1]
                       )

ax_se_rl_v = plt.subplot(gs[0, 0])
ax_se_rl_i = ax_se_rl_v.twinx()
ax_se_cl_v = plt.subplot(gs[0, 1])
ax_se_cl_i = ax_se_cl_v.twinx()
ax_diff_rl_v = plt.subplot(gs[1, 0])
ax_diff_rl_i = ax_diff_rl_v.twinx()
ax_diff_cl_v = plt.subplot(gs[1, 1])
ax_diff_cl_i = ax_diff_cl_v.twinx()
ax_rl_pulse = plt.subplot(gs[2, 0])
ax_cl_pulse = plt.subplot(gs[2, 1])

ax_rl_pulse.plot(timevec_rl_se, 1E3 * data_se_transient_input_rl[:, 0], alpha=.5, linewidth=2, zorder=99)
ax_rl_pulse.plot(timevec_rl_diff, 1E3 * data_diff_transient_input_rl[:, 0], alpha=.5, linewidth=2, zorder=99)
ax_rl_pulse.plot(timevec_rl_se, 1E3 * data_se_transient_input_rl[:, 3], alpha=.5, linewidth=2, zorder=99)
ax_rl_pulse.plot(timevec_rl_diff, 1E3 * data_diff_transient_input_rl[:, 3], alpha=.5, linewidth=2, zorder=99)
ax_rl_pulse.set_yticks([-50, 0, 50])
ax_rl_pulse.set_ylabel("$V_{in,diff}$ [mV]", size=18.)

ax_cl_pulse.plot(timevec_cl_se, 1E3 * data_se_transient_input_rl[:, 0], alpha=.5, linewidth=2, zorder=99)
ax_cl_pulse.plot(timevec_cl_diff, 1E3 * data_diff_transient_input_rl[:, 0], alpha=.5, linewidth=2, zorder=99)
ax_cl_pulse.plot(timevec_cl_se, 1E3 * data_se_transient_input_rl[:, 3], alpha=.5, linewidth=2, zorder=99)
ax_cl_pulse.plot(timevec_cl_diff, 1E3 * data_diff_transient_input_rl[:, 3], alpha=.5, linewidth=2, zorder=99)
ax_cl_pulse.set_yticks([-50, 0, 50])
ax_cl_pulse.set_yticklabels([])


ax_se_rl_v.set_title("Resistive load", size=20.)
ax_se_rl_v.scatter(timevec_rl_se, data_se_transient_input_rl[:, 1], label="SE", color="grey", marker=',', lw=0, s=4)
ax_se_rl_i.plot(timevec_rl_se, data_se_transient_input_rl[:, 2] / I_meas_gain * 1E3, color="#f36e00", linewidth=2, zorder=99)
ax_se_rl_i.set_ylim(-2.5,2.5)
ax_se_rl_v.set_xticklabels([])
# ax_se_rl_i.set_ylabel("$I_L$ [mA]", size=18.)
ax_se_rl_i.set_yticklabels([])
ax_se_rl_v.set_ylabel("$V_L$ [V]", size=18.)

ax_se_cl_v.set_title("Capacitive load", size=20.)
ax_se_cl_v.scatter(timevec_cl_se, data_se_transient_input_cl[:, 1], label="SE", color="grey", marker=',', lw=0, s=4)
ax_se_cl_i.plot(timevec_cl_se, data_se_transient_input_cl[:, 2] / I_meas_gain * 1E3, color="#f36e00", linewidth=2, zorder=99)
ax_se_cl_i.set_ylim(-2.5,2.5)
ax_se_cl_v.set_xticklabels([])
ax_se_cl_i.set_ylabel("$I_L$ [mA]", size=18.)
ax_se_cl_v.set_yticklabels([])
# ax_se_cl_v.set_ylabel("$V_L$ [V]", size=18.)

ax_diff_rl_v.scatter(timevec_rl_diff, data_diff_transient_input_rl[:, 1], label="diff", color="grey", marker=',', lw=0, s=4)
ax_diff_rl_i.plot(timevec_rl_diff, data_diff_transient_input_rl[:, 2] / I_meas_gain * 1E3, color="#4671d5", linewidth=2, zorder=99)
ax_diff_rl_i.set_ylim(-2.5,2.5)
ax_diff_rl_i.set_yticklabels([])
ax_diff_rl_v.set_xticklabels([])
# ax_diff_rl_i.set_ylabel("$I_L$ [mA]", size=18.)
ax_diff_rl_v.set_ylabel("$V_L$ [V]", size=18.)

# ax_diff_cl_v.set_title("Capacitive load")
ax_diff_cl_v.scatter(timevec_cl_diff, data_diff_transient_input_cl[:, 1], label="diff", color="grey", marker=',', lw=0, s=4)
ax_diff_cl_i.plot(timevec_cl_diff, data_diff_transient_input_cl[:, 2] / I_meas_gain * 1E3, color="#4671d5", linewidth=2, zorder=99)
ax_diff_cl_i.set_ylim(-2.5,2.5)
ax_diff_cl_v.set_xticklabels([])
ax_diff_cl_i.set_ylabel("$I_L$ [mA]", size=18.)
ax_diff_cl_v.set_yticklabels([])
# ax_diff_cl_v.set_ylabel("$V_L$ [V]", size=18.)

ax_rl_pulse.set_xlabel("Time [ms]", size=18.)
ax_cl_pulse.set_xlabel("Time [ms]", size=18.)

ax_rl_pulse.grid(True, axis="x")
for _ax in [ax_se_cl_v, ax_se_rl_v, ax_diff_cl_v, ax_diff_rl_v]:
	_ax.grid(True)

for _ax in [ax_se_cl_v, ax_se_rl_v, ax_diff_cl_v, ax_diff_rl_v, ax_rl_pulse, ax_cl_pulse]:
	_ax.set_xlim(.25, 1.75)
	_ax.set_xticks([.5, 1., 1.5])
	_ax.grid(True)

#plt.tight_layout()
fig.savefig("/tmp/transients.png")
fig.savefig("/tmp/transients.pdf")
plt.close(fig)
