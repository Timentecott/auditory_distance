import librosa
import numpy as np
import scipy.signal
import soundfile as sf



left_IR_loc = r"C:\Users\tim_e\source\repos\auditory_distance\pyrirtool\recorded\newrir18\RIR1.wav"
right_IR_loc = r"C:\Users\tim_e\source\repos\auditory_distance\pyrirtool\recorded\newrir18\RIR2.wav"

dry_stim = r"C:\Users\tim_e\source\repos\auditory_distance\experiment_1\original_audios\ISTS\ISTS-V1.0_60s_24bit_3.wav"

y, sr = librosa.load(dry_stim, sr=44100)
ir_left, sr_ir = librosa.load(left_IR_loc, sr=44100)
ir_right, sr_ir = librosa.load(right_IR_loc, sr=44100)

outputfile = 'convolved_audio.wav'

#convolve left and right with IRs
convolved_audio_left = scipy.signal.fftconvolve(y, ir_left, mode='full')
convolved_audio_right = scipy.signal.fftconvolve(y, ir_right, mode='full')

#stack left and right channels
stereo_audio = np.vstack((convolved_audio_left, convolved_audio_right)).T

#enbiggen
stereo_audio = stereo_audio / np.max(np.abs(stereo_audio))

#save the stereo audio to a new file
sf.write(outputfile, stereo_audio, sr)

