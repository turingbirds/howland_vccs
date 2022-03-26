import pyaudio
import wave

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
fs = 44100  # Record at 44100 samples per second
seconds = 60.
filename = "recording.wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio
info = p.get_host_api_info_by_index(0)

numdevices = info.get('deviceCount')
for i in range(0, numdevices):
	if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
		print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

input_device_idx = 11
print("Recording from device: " + str(input_device_idx))

print('Recording')

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True,
                input_device_index=input_device_idx)

frames = []  # Initialize array to store frames

# Store data in chunks
for i in range(0, int(fs / chunk * seconds)):
    data = stream.read(chunk)
    frames.append(data)

# Stop and close the stream 
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('Finished recording')

# Save the recorded data as a WAV file
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()


