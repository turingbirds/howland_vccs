help = r"""Acquire Z_hat_out samples via FT232H controlling relay, GPIB HP3456B voltmeter, and GPIB HP33120A function generator.

Usage:

	gpib_acq.py CIRCUIT_NAME [OUT_FN]

"""

import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
from math import sin, cos, pi
from pylibftdi import BitBangDevice
import gpib

if len(sys.argv) < 2:
	print(help)
	sys.exit(0)


#
# 	GPIB
#

gpib_dev = gpib.find("hp3456a")
assert not gpib_dev is None

sg_gpib_dev = gpib.find("hp33120a")
assert not sg_gpib_dev is None

n_bytes = 100		# GPIB buffer size


#
# 	acquisition parameters
#

#mode = "voltage"
mode = "current"

rate = 1.

circuit_name = sys.argv[1]
print("Parsed command-line parameter: circuit = " + str(circuit_name))

fn = None		# ASCII output file (when used)

RLa = 10E3      # resistance for relay OFF [Ohm]
RLb = 11E3      # resistance for relay ON [Ohm]

gain = 1 / 510. # for current mode: gain from measured voltage [V] on the DMM, to load current [mA]

RELAY_RESISTIVE = 0   # relay channel (0 or 1)
RELAY_RESISTIVE_STATE_RLA = 0
RELAY_RESISTIVE_STATE_RLB = 1

def plot_Z_out_quasi(freqs, Z_out_quasi_means, Z_out_quasi_stds, data_fname_snip=''):
	colours = {
		"ONEMEGC": "#a379c9",#ffc857",#"#cf9b2a",#ffee93",#"#ffaa46",##f30e90",#ef4949",
		"ONEMEG" : "#ffeb3a",#"#cf9b2a",#"#8ae234",
		"HUNDREDKOHMC" : "#94fce7",#"#b047b4",#"#75507b",
		"HUNDREDKOHM" : "#9cfc97",#"#eeee33",#"#4671d5",#8a8a8a",

		"single-ended" : "#f36e00",
		"differential" : "#4671d5"
	}

	for i, circuit_name in enumerate(Z_out_quasi_means.keys()):
		if circuit_name == "SE":
			color="#f36e00"
			marker="o"
		elif circuit_name == "diff":
			color="#4671d5"
			marker="D"
		else:
			if circuit_name in colours.keys():
				color = colours[circuit_name]
			else:
				color = colours["differential"]

			if circuit_name == "ONEMEG":
				marker="^"
			elif circuit_name == "ONEMEGC":
				marker="v"
			elif circuit_name == "HUNDREDKOHM":
				marker="^"
			elif circuit_name == "HUNDREDKOHMC":
				marker="v"
		fig, ax = plt.subplots(1, 1, figsize=(8, 4))
		plt.subplots_adjust(bottom=.18, top=.95)

		ax.plot(freqs[circuit_name], Z_out_quasi_means[circuit_name], color=color, linewidth=2, zorder=98, linestyle="none", alpha=.6)#, hatch="-", linewidth=10.0)
		ax.errorbar(freqs[circuit_name], Z_out_quasi_means[circuit_name], yerr=Z_out_quasi_stds[circuit_name], marker=marker, markersize=1.2*6, color=color, label=circuit_name, linewidth=2, zorder=99, markeredgecolor="k", markeredgewidth=.5, linestyle="none")#, hatch="-", linewidth=10.0)
	# ax.errorbar(freqs, r_se2, yerr=np.zeros_like(r_se_std), marker="o", markersize=1.2*6, color="#f36e00", label="Single ended", linewidth=2, zorder=99, alpha=.2)#, hatch="-", linewidth=10.0)
	# ax.errorbar(freqs, r_diff2, yerr=np.zeros_like(r_diff_std), marker="D", markersize=1.2*5.5, color="#4671d5", label="Differential", linewidth=2, zorder=99, alpha=.3)#, edgecolor="#ffffff", hatch="-", linewidth=10.0)
	ax.set_yscale('log')
	ax.set_xscale('log')

	# ax.loglog(freqs, r_se, marker="o", color="#f36e00", label="Single ended", linewidth=2)#, hatch="-", linewidth=10.0)
	ax.set_ylabel("$\hat{Z}_{out}$")
	ax.set_xlabel("Frequency [Hz]")

	ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
	ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

	# fig.suptitle('VCCS output resistance')
	# ax[0].set_xlim(ax[0].get_xlim()[0], np.amax(bin_edges)/1E6/2.)
	# ax[1].set_xlim(ax[0].get_xlim()[0], 2*ax[0].get_xlim()[1])
	# ax.set_xlim(2E3, 5E4)
	# ax.set_xlim(min_freq, max_freq)
	ax.spines['left'].set_linewidth(1.5)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_linewidth(1.5)
	ax.xaxis.set_ticks_position("bottom")
	ax.yaxis.set_ticks_position("left")
	ax.grid(b=True, which='both', zorder=0, color="#999999")

	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(16)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(16)



def set_relay(bb, relay_id, state):
	print("set relay " + str(state))
	state = not (state > 0)		# reversed !!
	if state:
		bb.port |= 1 << relay_id
	else:
		bb.port &= 0xFF - (1 << relay_id)


def acq_abs_load_current(gpib_dev, gain):
	IL = gpib.read(gpib_dev, n_bytes)
	IL = float(IL) * gain
	print("|IL| = " + str(1E3 * IL) + " mA(RMS) = " + str(1E3 * IL * 2**.5) + " mA[pk]")
	return IL


def acq_abs_load_voltage(gpib_dev, circuit):
	VL = gpib.read(gpib_dev, n_bytes)
	VL = float(VL)
	if circuit == "differential":
		VL *= 2
	print("|VL| = " + str(VL) + " V(RMS)")
	return VL


def acq_abs_load_voltages(bb, gpib_dev, circuit):
	set_relay(bb, RELAY_RESISTIVE, RELAY_RESISTIVE_STATE_RLA)
	time.sleep(.5 / rate)
	VLa_abs = acq_abs_load_voltage(gpib_dev, circuit)

	set_relay(bb, RELAY_RESISTIVE, RELAY_RESISTIVE_STATE_RLB)
	time.sleep(.5 / rate)
	VLb_abs = acq_abs_load_voltage(gpib_dev, circuit)

	return VLa_abs, VLb_abs


def acq_abs_load_currents(bb, gpib_dev, circuit, gain):
	set_relay(bb, RELAY_RESISTIVE, RELAY_RESISTIVE_STATE_RLA)
	time.sleep(.5 / rate)
	ILa_abs = acq_abs_load_current(gpib_dev, gain=gain)

	set_relay(bb, RELAY_RESISTIVE, RELAY_RESISTIVE_STATE_RLB)
	time.sleep(.5 / rate)
	ILb_abs = acq_abs_load_current(gpib_dev, gain=gain)

	return ILa_abs, ILb_abs


def enter_phase(bb):
	set_relay(bb, RELAY_RESISTIVE, RELAY_RESISTIVE_STATE_RLA)
	time.sleep(.5 / rate)
	print("Please enter phase for R_L = R_La = " + str(RLa/1E3) + " kR in DEGREES:")
	anga = float(input()) / 360. * 2 * pi

	set_relay(bb, RELAY_RESISTIVE, RELAY_RESISTIVE_STATE_RLB)
	time.sleep(.5 / rate)
	print("Please enter phase for R_L = R_Lb = " + str(RLb/1E3) + " kR in DEGREES:")
	angb = float(input()) / 360. * 2 * pi
	return anga, angb


def complex_from_mag_ang(mag, ang):
	re = mag * cos(ang)
	im = mag * sin(ang)
	return re + 1j * im


def compute_RC_from_Z(z, w):
	R = (z.imag**2 + z.real**2) / z.real
	C = - z.imag / ((z.imag**2 + z.real**2) * w)
	return R, C


def Z_o_from_VLa_VLb(RLa, RLb, VLa, VLb):
	if np.abs(VLa * RLb - VLb * RLa) < 1E-12:
		return np.nan
	Z_o = (RLa * RLb * (VLb - VLa)) / (VLa * RLb - VLb * RLa)
	return Z_o

def Z_o_from_ILa_ILb(RLa, RLb, ILa, ILb):
	if np.abs(ILa - ILb) < 1E-12:
		return np.nan
	Z_o = (ILb * RLb - ILa * RLa) / (ILa - ILb)
	return Z_o


def set_sg_freq(f):
	global sg_gpib_dev
	
	gpib.write(sg_gpib_dev, "FUNC:SHAPE SIN")
	gpib.write(sg_gpib_dev, "VOLT 3.0")        # tune here for 1 mApeak output
	gpib.write(sg_gpib_dev, "FREQ " + "%f.5E".format(f))


def acquire_Z_o_quasi_datapoints(bb, gpib_dev, circuit):
	"""acquire samples for Z_o_quasi until the reading converges"""
	Z_o_quasi_datapoints = []
	terminate = False
	while not terminate:
		if mode == "current":
			global gain
			ILa_abs, ILb_abs = acq_abs_load_currents(bb, gpib_dev, circuit, gain)
			print("ILa_abs = " + str(ILa_abs) + " V")
			print("ILb_abs = " + str(ILb_abs) + " V")
			Z_o_quasi = Z_o_from_ILa_ILb(RLa, RLb, ILa_abs, ILb_abs)
		else:
			assert mode == "voltage"
			VLa_abs, VLb_abs = acq_abs_load_voltages(bb, gpib_dev, circuit)
			print("VLa_abs = " + str(VLa_abs) + " V")
			print("VLb_abs = " + str(VLb_abs) + " V")
			Z_o_quasi = Z_o_from_VLa_VLb(RLa, RLb, VLa_abs, VLb_abs)
		print("\tComputed Z_o_quasi = " + str(Z_o_quasi/1E6) + " MR")
		Z_o_quasi_datapoints.append(Z_o_quasi)
	
		if len(Z_o_quasi_datapoints) > 100 \
		or len(Z_o_quasi_datapoints) > 3 and (np.abs(np.std(Z_o_quasi_datapoints)) < 1E-9 \
											or np.abs(np.mean(Z_o_quasi_datapoints) / np.std(Z_o_quasi_datapoints)) < 1E-2):
			terminate = True

	return Z_o_quasi_datapoints


def acq_loop(gpib_dev, circuit, f=None):

	freqs = np.array([1E2, 1E3, 1E4, 1E5])
	Z_o_quasi_means = np.nan * np.ones_like(freqs)
	Z_o_quasi_stds = np.nan * np.ones_like(freqs)

	with BitBangDevice() as bb:
		for i, f in enumerate(freqs):
			set_sg_freq(f)
			Z_o_quasi_datapoints = acquire_Z_o_quasi_datapoints(bb, gpib_dev, circuit)
			Z_o_quasi_means[i] = np.mean(Z_o_quasi_datapoints)
			Z_o_quasi_stds[i] = np.std(Z_o_quasi_datapoints)
			plot_Z_out_quasi({circuit_name: freqs}, {circuit_name: Z_o_quasi_means}, {circuit_name: Z_o_quasi_stds})


def main():
	f = None
	if len(sys.argv) > 2:
		fn = sys.argv[2]
		print("Opening " + str(fn) + " for writing")
		f = open(fn, "w")

	acq_loop(gpib_dev, circuit_name, f=f)


if __name__ == '__main__':
	main()
