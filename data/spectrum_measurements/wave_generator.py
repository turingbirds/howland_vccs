#!/usr/bin/python 

import wave
import struct
import numpy as np
import scipy
import scipy.signal
import scipy.io.wavfile


freq = 100. # [Hz]
volume = 1.
duration = 10 * 60.  # [s]
sample_rate = 48000  # [Hz]

print("Generating...")
timevec = np.arange(0., duration, 1/sample_rate)
n_samples = len(timevec)
audio = np.empty((n_samples, 2))
audio[:, 0] = volume * np.sin(2 * np.pi * freq * timevec)
audio[:, 1] = -volume * np.sin(2 * np.pi * freq * timevec)
#audio[:, 1] = audio[:, 0]

print("Saving...")
fname = "output.wav"
#fname = "output_cm.wav"
scipy.io.wavfile.write(filename=fname, rate=sample_rate, data=audio)


