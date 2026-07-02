import os
from scipy.io.wavfile import write as wavwrite
import numpy as np
import sounddevice as sd


#--------------------------
def record(testsignal,fs,inputChannels,outputChannels):

    sd.default.samplerate = fs
    sd.default.dtype = 'float32'
    print("Input channels:",  inputChannels)
    print("Output channels:", outputChannels)

    # Start the recording
    recorded = sd.playrec(testsignal, samplerate=fs, input_mapping = inputChannels,output_mapping = outputChannels)
    sd.wait()

    return recorded


#--------------------------
def saverecording(RIR, RIRtoSave, testsignal, recorded, fs):

        dirflag = False
        counter = 1
        dirname = 'recorded/newrir1'
        while dirflag == False:
            if os.path.exists(dirname):
                counter = counter + 1
                dirname = 'recorded/newrir' + str(counter)
            else:
                os.mkdir(dirname)
                dirflag = True

        # Saving the RIRs and the captured signals
        np.save(dirname+ '/RIR.npy',RIR)
        np.save(dirname+ '/RIRac.npy',RIRtoSave)
        wavwrite(dirname+ '/sigtest.wav',fs,testsignal)

        for idx in range(recorded.shape[1]):
            wavwrite(dirname+ '/sigrec' + str(idx+1) + '.wav',fs,recorded[:,idx])
            wavwrite(dirname+ '/RIR' + str(idx+1) + '.wav',fs,RIR[:,idx])

        # Create a cropped RIR that starts at the direct arrival and save it as RIRcrop.npy
        try:
            # Compute a robust detection of the first arrival across channels
            peak = np.max(np.abs(RIR))
            if peak <= 0:
                # fallback: no cropping if RIR is silent
                start_idx = 0
            else:
                thresh = peak * 0.05  # 5% of peak
                first_idxs = []
                for ch in range(RIR.shape[1]):
                    ch_abs = np.abs(RIR[:, ch])
                    above = np.where(ch_abs >= thresh)[0]
                    if above.size > 0:
                        first_idxs.append(above[0])
                    else:
                        # fallback to absolute max for this channel
                        first_idxs.append(int(np.argmax(ch_abs)))

                # take the earliest detected arrival across channels
                start_idx = int(np.min(first_idxs))

                # keep a small pre-roll (~5 ms) to preserve onset
                pre_samples = int(0.005 * fs)
                start_idx = max(0, start_idx - pre_samples)

            RIRcrop = RIR[start_idx:, :]
            np.save(dirname + '/RIRcrop.npy', RIRcrop)
            # also save in lastRecording for quick check
            np.save('recorded/lastRecording/RIRcrop.npy', RIRcrop)
        except Exception:
            # If anything goes wrong, skip cropping but continue saving
            pass

        # Save in the recorded/lastRecording for a quick check
        np.save('recorded/lastRecording/RIR.npy',RIR)
        np.save( 'recorded/lastRecording/RIRac.npy',RIRtoSave)
        wavwrite( 'recorded/lastRecording/sigtest.wav',fs,testsignal)
        for idx in range(recorded.shape[1]):
            wavwrite('sigrec' + str(idx+1) + '.wav',fs,recorded[:,idx])
            wavwrite(dirname+ '/RIR' + str(idx+1) + '.wav',fs,RIR[:,idx])


        print('Success! Recording saved in directory ' + dirname)
