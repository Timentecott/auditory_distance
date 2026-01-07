#10/12/25 learning how to use librosa
import librosa
import librosa.display
import matplotlib.pyplot as plt #for creating waveform plot
#load an audio file
#by default, librosa.load loads the audio as a mono signal at 22050 Hz sample rate. This can be overriden as follows@
#waveform, sample_rate = librosa.load(r'C:\Users\tim_e\Desktop\pilot\pilot_exp\resources\other_stimuli\blackbird_bbc.wav', sr=None, mono=False) 
waveform, sample_rate = librosa.load(r'C:\Users\tim_e\Desktop\pilot\pilot_exp\resources\other_stimuli\blackbird_bbc.wav') 
plt.figure(figsize=(10, 4))
#draw a waveform
librosa.display.waveshow(waveform, sr=sample_rate, color='blue')
#add title and labels
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
