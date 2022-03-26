# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.ticker as plticker
# mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# mpl.rcParams["text.usetex"] = True
# mpl.rcParams["text.latex.unicode"] = True
#mpl.rcParams.update({'mathtext.default':  'cmr10' })
#mpl.rcParams.update({'mathtext.it':  'cmmi' })
#mpl.rcParams['axes.unicode_minus'] = False
#mpl.rcParams['font.family'] = 'cmr10'
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



colours = {
	"single-ended" : "#f36e00",
	"differential" : "#4671d5"
}

#import matplotlib.font_manager as fm
# mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# mpl.rcParams["text.usetex"] = True
# mpl.rcParams["text.latex.unicode"] = True
#mpl.rcParams.update({'mathtext.default':  'cmr10' })
#mpl.rcParams.update({'mathtext.it':  'cmmi' })
#mpl.rcParams['axes.unicode_minus'] = False
#mpl.rcParams['font.family'] = 'cmr10'
# mpl.use("PDF")
# mpl.rcParams['font.family'] = 'Computer Modern'


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scipy.fftpack
import scipy.signal
import wave, struct
import scipy
import scipy.signal
import scipy.io.wavfile
from scipy.signal import freqz


def get_min_length(fnames):
	min_length = float("inf")
	for fname in fnames:
		try:
			fs, data = scipy.io.wavfile.read(fname)
			min_length = min(min_length, np.amax(data.shape))
		except:
			pass
	print("min_length = " + str(min_length))
	return min_length



def plot_dBmA(xf, yf, fname, dpi=150.):
	
	fig, ax = plt.subplots(figsize=(12., 5.))
	ax.loglog(xf, yf, linewidth=2, c=colours["differential"])

	ax.set_ylabel("dBmA")
	ax.set_xlabel("f [Hz]")
	ax.grid(b=True, which="major")
	ax.grid(b=True, which='minor', color='#dddddd', linestyle='--')
	ax.set_ylim(np.amin(yf[1:]), np.amax(yf[1:]))
	ax.set_xlim(max(ax.get_xlim()[0], 1.), ax.get_xlim()[1])

	#ax.yaxis.set_major_formatter(FormatStrFormatter(r"$%d$"))
	#ax.set_yticklabels(['%d' % (20 * np.log10(y / 1E-3)) for y in ax.get_yticks()])

	fig.savefig("/tmp/fft_[" + fname + "].png", dpi=dpi)
	#fig.savefig("/tmp/fft_[" + fname + "].pdf")
	plt.close(fig)



def plot_PSD(xf, yf, scale, fname, dpi=150.):
	
	fig, ax = plt.subplots(figsize=(12., 5.))

	ax.yaxis.set_major_formatter(FormatStrFormatter(r"$%d$"))

	delta_f = xf[1] - xf[0]
	print("One frequency bin width: " + str(delta_f) + " Hz")
	psd = yf**2 / (delta_f * scale)
	ax.loglog(xf, psd , linewidth=2, c=colours["differential"])

	ax.set_ylabel("$A^2/Hz$ [dB]")
	ax.set_xlabel("f [Hz]")
	ax.grid(b=True, which="major")
	ax.grid(b=True, which='minor', color='#dddddd', linestyle='--')
	ax.set_ylim(np.amin(psd[1:]), np.amax(psd[1:]))
	ax.set_xlim(max(ax.get_xlim()[0], 1.), ax.get_xlim()[1])
	ax.set_yticklabels(['%d' % (10 * np.log10(y)) for y in ax.get_yticks()])

	fig.savefig("/tmp/fft_[" + fname + "].png", dpi=dpi)
	#fig.savefig("/tmp/fft_[" + fname + "].pdf")
	plt.close(fig)


def process_wav_recording(fname, max_len=float("inf"), dpi=150, debug_plot=False, remove_DC=True):
	fs, data = read_wav(fname)
	return process_recording(fs, data, fname_snip=fname, max_len=max_len, dpi=dpi, debug_plot=debug_plot, remove_DC=remove_DC)


def read_wav(fname):
	fs, data = scipy.io.wavfile.read(fname)
	print("Sample rate: " + str(fs))
	return fs, data


def process_recording(fs, data, fname_snip, max_len=float("inf"), dpi=150, debug_plot=False, remove_DC=True):
	"""try:
		waveFile = wave.open(fname, 'r')
		import pdb;pdb.set_trace()
		length = waveFile.getnframes()
		data = np.nan * np.ones((length, 2))
		for i in range(0, 2*length):
			waveData = waveFile.readframes(1)
			#data[i, :] = struct.unpack("<hh", waveData)
			data[i, 0] = struct.unpack("f", waveData)
			#print(int(data[0]))
			#import pdb;pdb.set_trace()

		#rate, data = waveFile.read(path)
		#left = data[:, 0], right = data[:, 1]

		if not np.isinf(max_len):
			data = data[:max_len, :]
			length = max_len

		sample_rate = waveFile.getframerate()	# [Hz]

		N = length
		# Number of samplepoints
		# sample spacing
		x = np.linspace(0.0, N/sample_rate, N)
		#y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)


		#for i in [0, 1]:
			#data[:, i] -= np.mean(data[:, i])
			#data[:, i] /= np.std(data[:, i])

		#y = data[:, 0] - data[:, 1]   # differential between the two channels
	except Exception as e:
		print("Error: Cannot open " + fname)
		print(e)
		return None, None"""


	N = data.shape[0]
	if len(data.shape) == 1:
		y = data
	else:
		#y = data[:, 0] - data[:, 1]   # differential between the two channels
		y = data[:, 0]		# pick one channel

	if not np.isinf(max_len):
		y = y[:max_len]

	raw_data = y.copy()

	N = len(y)
	x = np.linspace(0., N / fs, N)

	#window = np.hamming(N)
	window = scipy.signal.blackmanharris(N)
	y = y * window

	#y[:100] = 0.	# skip startup transient
	if remove_DC:
		y -= np.mean(y)

	if debug_plot:
		fig, ax = plt.subplots(figsize=(8., 5.))
		ax.plot(x, y)
		ax.grid(True)
		fig.savefig("/tmp/ts_["+fname_snip+"].png", dpi=dpi)
		plt.close(fig)

		fig, ax = plt.subplots(figsize=(8., 5.))
		N = len(x)
		ax.plot(x[N//2:N//2+1000], y[N//2:N//2+1000])
		ax.grid(True)
		fig.savefig("/tmp/ts_zoom_["+fname_snip+"].png", dpi=dpi)
		plt.close(fig)

		fig, ax = plt.subplots(figsize=(8., 5.))
		if len(data.shape) == 1:
			ax.plot(x, data)
		else:
			ax.plot(x, data[:,0])
			ax.plot(x, data[:,1])
		ax.grid(True)
		fig.savefig("/tmp/ts_raw_["+fname_snip+"].png")
		plt.close(fig)

		fig, ax = plt.subplots(figsize=(8., 5.))
		if len(data.shape) == 1:
			ax.plot(x[N//2:N//2+1000], data[N//2:N//2+1000])
		else:
			ax.plot(x[N//2:N//2+1000], data[N//2:N//2+1000,0])
			ax.plot(x[N//2:N//2+1000], data[N//2:N//2+1000,1])
		ax.grid(True)
		fig.savefig("/tmp/ts_raw_zoom_["+fname_snip+"].png", dpi=dpi)
		plt.close(fig)

	xf = np.linspace(0., fs/2., N//2)
	yf = scipy.fftpack.fft(y)
	yf = 2./N * np.abs(yf[:N//2])

	#f_lo = 1.	# [Hz]
	#f_hi = 20E3	# [Hz]
	#xf_idx_lo = np.argmin((xf - f_lo)**2)
	#xf_idx_hi = np.argmin((xf - f_hi)**2)

	#xf = xf[xf_idx_lo:xf_idx_hi]
	#yf = yf[xf_idx_lo:xf_idx_hi]

	plot_dBmA(xf, yf, fname_snip, dpi=dpi)

	scale = 2.  # correction factor for 4-term Blackman-Harris window
	#plot_PSD(xf, yf, scale, fname + "_psd", dpi=dpi)

	return x, y, raw_data, xf, yf


#def plot_fft_panels(xf, yf_se_0p1mA, yf_se_1mA, yf_diff_0p1mA, yf_diff_1mA, fname_snip, f_lo, f_hi, min_dBmA=-140., dpi=150):

	#yf_se_0p1mA *= .1E-3 / np.amax(yf_se_0p1mA)
	#yf_se_1mA *= 1E-3 / np.amax(yf_se_1mA)
	#yf_diff_0p1mA *= .1E-3 / np.amax(yf_diff_0p1mA)
	#yf_diff_1mA *= 1E-3 / np.amax(yf_diff_1mA)
	
	##yf_se_0p1mA = 20*np.log10(yf_se_0p1mA)
	##yf_se_1mA = 20*np.log10(yf_se_1mA)
	##yf_diff_0p1mA = 20*np.log10(yf_diff_0p1mA)
	##yf_diff_1mA = 20*np.log10(yf_diff_1mA)

	#fig, ax = plt.subplots(figsize=(8., 5.))
	#gs = mpl.gridspec.GridSpec(2, 2,
						#width_ratios=[1, 1],
						#height_ratios=[1, 1],
						#hspace=.1,
						#wspace=.1)

	#ax_se_0p1mA = plt.subplot(gs[0, 0])
	#ax_diff_0p1mA = plt.subplot(gs[0, 1])

	#ax_se_1mA = plt.subplot(gs[1, 0])
	#ax_diff_1mA = plt.subplot(gs[1, 1])

	#for data, ax, colour in [(yf_se_0p1mA, ax_se_0p1mA, colours["single-ended"]),
					 #(yf_se_1mA, ax_se_1mA, colours["single-ended"]),
	                 #(yf_diff_0p1mA, ax_diff_0p1mA, colours["differential"]),
				     #(yf_diff_1mA, ax_diff_1mA, colours["differential"])]:
		#if not data is None:
			#ax.loglog(xf, data, linewidth=2, c=colour)

	#ax_se_0p1mA.set_xticklabels([])
	#ax_diff_0p1mA.set_xticklabels([])
	#ax_diff_0p1mA.set_yticklabels([])
	#ax_diff_1mA.set_yticklabels([])

	##ampl_min_plot = np.amin([np.amin(d[xf_idx_lo:xf_idx_hi]) for d in [yf_scaled]])
	##ampl_max_plot = np.amax([np.amax(d[xf_idx_lo:xf_idx_hi]) for d in [yf_scaled]])

	#ampl_min = np.amin([yf_se_0p1mA[1:], yf_diff_0p1mA[1:]])
	#ampl_max = np.amax([yf_se_0p1mA[1:], yf_diff_0p1mA[1:]])
	#for ax in [ax_se_0p1mA, ax_diff_0p1mA]:
		#ax.set_ylim(ampl_min, ampl_max)

	#ampl_min = np.amin([yf_se_1mA[1:], yf_diff_1mA[1:]])
	#ampl_max = np.amax([yf_se_1mA[1:], yf_diff_1mA[1:]])
	#for ax in [ax_se_1mA, ax_diff_1mA]:
		#ax.set_ylim(ampl_min, ampl_max)

	#for ax in [ax_se_0p1mA, ax_diff_0p1mA, ax_se_1mA, ax_diff_1mA]:
		#ax.grid(b=True, axis="both", which="major")
		#ax.grid(b=True, axis="both", which='minor', color='#dddddd', linestyle='--')
		#ax.minorticks_on()
		#ax.set_xlim(f_lo, f_hi)

	#ax_se_0p1mA.set_ylabel("dBmA")
	#ax_se_1mA.set_ylabel("dBmA")

	#ax_se_1mA.set_xlabel("f [Hz]")
	#ax_diff_1mA.set_xlabel("f [Hz]")

	#for ax in [ax_se_0p1mA, ax_diff_0p1mA, ax_se_1mA, ax_diff_1mA]:
		#if 20*np.log10(ax.get_ylim()[0]) < min_dBmA:
			#ax.set_ylim(1E-3*10**(min_dBmA / 20.), ax.get_ylim()[1])

	#for ax in [ax_se_0p1mA, ax_se_1mA]:
		#ax.yaxis.set_major_formatter(FormatStrFormatter(r"$%d$"))
		#ax.set_yticklabels(['%d' % (20 * np.log10(y / 1E-3)) for y in ax.get_yticks()])

		##for i, tick, label in enumerate(zip(ax.get_yticks(), ax.get_yticklabels())):
			##print("Replacing label " + str(label.get_text()))
				##print("Replacing label " + str(ax.get_yticklabels()[i]) + " by " + str(20*np.log10(float(ax.get_yticks()[i]))))
				##ax.get_yticklabels()[i].set_text(str(20*np.log10(float(ax.get_yticks()[i]))))
	#fig.canvas.draw()


	#fig.savefig("/tmp/fft" + fname_snip + ".png", dpi=dpi)
	#fig.savefig("/tmp/fft" + fname_snip + ".pdf")
	#plt.tight_layout()
	#plt.close(fig)


def plot_fft_panels(xf, yf_se_1mA, yf_diff_1mA, fname_snip, f_lo, f_hi, min_dBmA=-140., dpi=150):

	yf_se_1mA *= 1E-3 / np.amax(yf_se_1mA)
	yf_diff_1mA *= 1E-3 / np.amax(yf_diff_1mA)
	
	fig, ax = plt.subplots(figsize=(8., 5.))
	gs = mpl.gridspec.GridSpec(1, 2,
						width_ratios=[1, 1],
						hspace=.1,
						wspace=.1)

	ax_se_1mA = plt.subplot(gs[0])
	ax_diff_1mA = plt.subplot(gs[1])

	for data, ax, colour in [(yf_se_1mA, ax_se_1mA, colours["single-ended"]),
				     (yf_diff_1mA, ax_diff_1mA, colours["differential"])]:
		if not data is None:
			ax.loglog(xf, data, linewidth=2, c=colour)

	ax_diff_1mA.set_yticklabels([])

	#ampl_min_plot = np.amin([np.amin(d[xf_idx_lo:xf_idx_hi]) for d in [yf_scaled]])
	#ampl_max_plot = np.amax([np.amax(d[xf_idx_lo:xf_idx_hi]) for d in [yf_scaled]])

	ampl_min = np.amin([yf_se_1mA[1:], yf_diff_1mA[1:]])
	ampl_max = np.amax([yf_se_1mA[1:], yf_diff_1mA[1:]])
	for ax in [ax_se_1mA, ax_diff_1mA]:
		ax.set_ylim(ampl_min, ampl_max)

		l, b, w, h = ax.get_position().bounds
		ax.set_position([l, b + .06, w, h])


	ax_se_1mA.set_ylabel("Amplitude [dBmA]")

	ax_se_1mA.set_xlabel("$f$ [Hz]")
	ax_diff_1mA.set_xlabel("$f$ [Hz]")

	for ax in [ax_se_1mA, ax_diff_1mA]:
		if 20*np.log10(ax.get_ylim()[0]) < min_dBmA:
			ax.set_ylim(1E-3*10**(min_dBmA / 20.), ax.get_ylim()[1])

	for ax in [ax_se_1mA, ax_diff_1mA]:
		ax.grid(b=True, axis="both", which="major")
		ax.grid(b=True, axis="both", which='minor', color='#dddddd', linestyle='--')
		ax.set_xlim(f_lo, f_hi)
		ax.minorticks_on()
		
		loc = plticker.LogLocator(base=10, subs=[1.], numticks=15) # numticks=15 is some awful hack
		ax.yaxis.set_major_locator(loc)

		loc = plticker.LogLocator(base=10, subs=[1.], numticks=15) # numticks=15 is some awful hack
		ax.xaxis.set_major_locator(loc)

		ax.xaxis.set_minor_formatter(FormatStrFormatter(""))
		ax.yaxis.set_minor_formatter(FormatStrFormatter(""))

		locmin = mpl.ticker.LogLocator(base=10., subs=np.linspace(0,1,10),numticks=15)
		ax.yaxis.set_minor_locator(locmin)

		locmin = mpl.ticker.LogLocator(base=10., subs=np.linspace(0,1,10),numticks=15)
		ax.xaxis.set_minor_locator(locmin)

	for ax in [ax_se_1mA]:
		ax.yaxis.set_major_formatter(FormatStrFormatter(r"$%d$"))
		ax.set_yticklabels(['%d' % (20 * np.log10(y / 1E-3)) for y in ax.get_yticks()])

		#for i, tick, label in enumerate(zip(ax.get_yticks(), ax.get_yticklabels())):
			#print("Replacing label " + str(label.get_text()))
				#print("Replacing label " + str(ax.get_yticklabels()[i]) + " by " + str(20*np.log10(float(ax.get_yticks()[i]))))
				#ax.get_yticklabels()[i].set_text(str(20*np.log10(float(ax.get_yticks()[i]))))
	#plt.tight_layout()
	fig.canvas.draw()

	#import pdb;pdb.set_trace()

	fig.savefig("/tmp/fft" + fname_snip + ".png", dpi=dpi)
	fig.savefig("/tmp/fft" + fname_snip + ".pdf")



#xf, yf = process_recording(fname='recording_voltage_sanity_check_diff.wav')
#xf, yf = process_recording(fname='output.wav')
#xf, yf = process_recording(fname='recording.wav')

#xf, yf = process_recording(fname='recording_0p1mA_se.wav')
#xf, yf = process_recording(fname='recording_1mA_se.wav')
#xf, yf = process_recording(fname='recording_0p1mA_diff.wav')
#xf, yf = process_recording(fname='recording_1mA_diff.wav')

#xf, yf = process_recording(fname='with_input_to_gnd.wav')
#xf, yf = process_recording(fname='loopback_behringer_no_signal.wav')
#xf, yf = process_recording(fname='loopback_u24xl_with_signal.wav')
#xf, yf = process_recording(fname='loopback_behringer_with_ignal.wav')
#xf, yf = process_recording(fname='1v_ref_signal_pauls.wav')
#xf, yf = process_recording(fname='1v_ref_signal_behringer.wav')
#xf, yf = process_recording(fname='dangling_wire_behringer.wav')


#xf, yf = process_recording(fname='recording_0p1mA_se.wav')
#xf, yf = process_recording(fname='loopback_via_vccs_no_signal.wav')
#xf, yf = process_recording(fname='loopback_via_vccs_no_signal_diff.wav')




fnames = {'recording_1mA_diff.wav' : 'recording_1mA_diff.wav',
		  'recording_0p1mA_diff.wav' : 'recording_0p1mA_diff.wav',
		  'recording_0p1mA_se.wav' : 'recording_0p1mA_se.wav',
		  'recording_1mA_se.wav' : 'recording_1mA_se.wav'}

f_lo = 10.    # [Hz]
f_hi = 20E3   # [Hz]



max_len = get_min_length(fnames.values())

xf = {}
yf = {}
raw_data = {}

for fname in fnames.values():
	_x, _y, _raw_data, _xf, _yf = process_wav_recording(fname=fname, max_len=max_len)
	xf[fname] = _xf
	yf[fname] = _yf
	raw_data[fname] = _raw_data

plot_fft_panels(xf['recording_0p1mA_se.wav'], yf['recording_1mA_se.wav'], yf['recording_1mA_diff.wav'], fname_snip="", f_lo=f_lo, f_hi=f_hi)
#plot_fft_panels(xf['recording_0p1mA_se.wav'], yf['recording_0p1mA_se.wav'], yf['recording_1mA_se.wav'], yf['recording_0p1mA_diff.wav'], yf['recording_1mA_diff.wav'], fname_snip="", f_lo=f_lo, f_hi=f_hi)



def butter_bandpass(lowcut, highcut, fs, order=3):
	b, a = scipy.signal.butter(order, [lowcut, highcut], fs=fs, btype='bandpass')
	#b, a = scipy.signal.iirnotch(100., Q=100, fs=fs)
	#b, a = scipy.signal.ellip(N=order, rp=1., rs=60., Wn=[low, high], btype='bandstop')
	return b, a

def notch_bandstop(f, fs, Q=100):
	b, a = scipy.signal.iirnotch(f, Q, fs=fs)
	return b, a


def butter_bandpass_filter(data, f, fs, order=3):
	b, a = notch_bandstop(f, fs)
	y = scipy.signal.filtfilt(b, a, data)
	return y


order = 3

fs = 1 / (_x[1] - _x[0])





fig, ax = plt.subplots(figsize=(8., 5.))

b, a = notch_bandstop(f=100, fs=fs, Q=100)
w, h = freqz(b, a, worN=2000)
ax.plot((fs * .5 / np.pi) * w, abs(h), label="order = %d" % order, alpha=.5)
ax.set_xlim(1., 1E3)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Gain')
ax.grid(True)
ax.legend(loc='best')
fig.savefig("/tmp/fft_filt.png", dpi=150)
plt.close(fig)





#
#	do the filtering
#

f_notch = 100.   # [Hz]

for fname in ['recording_1mA_diff.wav', 'recording_1mA_se.wav']:
	y = butter_bandpass_filter(raw_data[fname], f_notch, fs, order=order)
	y = butter_bandpass_filter(y, f_notch, fs, order=order)
	y = butter_bandpass_filter(y, f_notch, fs, order=order)

	raw_data[fname][:100000] = 0.
	y[:100000] = 0.

	b, a = butter_bandpass(10., 1E3, fs=fs)
	raw_data_filt = scipy.signal.filtfilt(b, a, raw_data[fname])
	y_filt = scipy.signal.filtfilt(b, a, y)

	process_recording(fs, y_filt , fname_snip="filt_[" + fname + "]", max_len=max_len, dpi=150, debug_plot=True)

	def calc_rms(y):
		return (np.sum(y**2) / len(y))**.5



	RMS_raw = calc_rms(raw_data_filt)
	RMS_filt = calc_rms(y_filt)
	#RMS_raw = calc_rms(raw_data[fname])
	#RMS_filt = calc_rms(y)
	print("THD+N = " + str(100 * RMS_filt / RMS_raw) + " %")






