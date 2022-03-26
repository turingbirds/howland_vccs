#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib as mpl
mpl.rcParams["font.family"] = "CMU Serif"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.serif"] = [mpl.rcParams["font.family"]] + mpl.rcParams["font.serif"]
mpl.rcParams["axes.labelsize"] = 20.
mpl.rcParams["grid.color"] = "k"
mpl.rcParams["grid.linestyle"] = "--"
mpl.rcParams["grid.linewidth"] = 0.5
# mpl.rcParams["grid.dashes"] = (5, 5)
mpl.rcParams["lines.linewidth"] = 1.0
mpl.rcParams["lines.dashed_pattern"] = [3, 3]
mpl.rcParams["lines.dashdot_pattern"] = [3, 5, 1, 5]
mpl.rcParams["lines.dotted_pattern"] = [1, 3]
mpl.rcParams["lines.scale_dashes"] = False
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


tempC = np.array([30., 40., 50., 60.])
I_ref = 1E-3
I_se = np.array([np.array([.49335, .49290, .49256, .49239])/510,
				 np.array([.49322, .49285, .49250, .49220])/510])
I_diff = np.array([np.array([.49480, .49416, .49360, .49284])/510,
		           np.array([.49475, .49428, .49350, .49272])/510,
		           np.array([.49385, .49320, .49259, .49161])/510])

# normalise to 100%
for i in range(I_diff.shape[0]):
	I_diff[i, :] /= I_diff[i, 0]
for i in range(I_se.shape[0]):
	I_se[i, :] /= I_se[i, 0]
	

I_diff_mean = np.mean(I_diff, axis=0)
I_diff_std = np.std(I_diff, axis=0)
I_se_mean = np.mean(I_se, axis=0)
I_se_std = np.std(I_se, axis=0)

fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))

ax.errorbar(tempC, 1E2 * I_diff_mean - 1E2, 1E2 * I_diff_std, marker="D", color="#4671d5", linewidth=2, zorder=99, label="Differential")
ax.errorbar(tempC, 1E2 * I_se_mean - 1E2, 1E2 * I_se_std, marker="o", color="#f36e00", linewidth=2, zorder=99, label="Single-ended")

ax.set_ylabel("Relative change in $I_L$ [%]", size=18.)
ax.set_xlabel(r"Temperature [$^{\circ}$C]", size=18.)

ax.spines["left"].set_linewidth(1.5)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_linewidth(1.5)
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.grid(b=True, which="both", zorder=0)

l, b, w, h = ax.get_position().bounds
ax.set_position([l, b + .1, w, h])

leg = ax.legend(loc="lower left", handlelength=1.8, fontsize=16. , fancybox=True, framealpha=1)

fig.savefig("/tmp/temperature_coeff.png")
fig.savefig("/tmp/temperature_coeff.pdf")
plt.close(fig)
