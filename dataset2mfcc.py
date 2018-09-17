import os
import librosa
import numpy as np

from scipy.io import wavfile

digits = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

def wav2mono(signal):
    """Converts a wav signal to mono. If the signal is already mono, it is not modified.

    Parameters
    ----------
    signal: ndarray
        The wav sound signal

    Return:
    -------
    signal: ndarray
        A mono wav signal
    """
    signal = np.squeeze(signal)
    if len(signal.shape) <= 1:
        return signal
    axis = 0 if signal.shape[0] == 2 else 1
    return np.mean(signal, axis=axis)


def labelOf(folder):
    if folder in digits:
        return digits.index(folder)
    else: return 10

dataset_path = "./google_speech_commands/"

nb_coefficient = 32

if __name__ == "__main__":
    labels = os.listdir(dataset_path)
    folder_list = labels
    folder_list.remove('_background_noise_')
    # Counting the number of samples
    count = 0
    for folder in folder_list:
        for filename in os.listdir(dataset_path+folder):
            count = count+1

    # Creating the dataset
    print(count)
    X = np.zeros((count, nb_coefficient, 32))
    y = np.zeros(count)

    
    count = 0
    for folder in folder_list:
        for filename in os.listdir(dataset_path+folder):
            sampling_rate, signal = wavfile.read(dataset_path+folder+'/'+filename)
            signal = wav2mono(signal.astype(float)) # to handle stereophonic signals
            mfcc = librosa.feature.mfcc(signal, sr=sampling_rate, n_mfcc=nb_coefficient)


            X[count,0:mfcc.shape[0],0:mfcc.shape[1]] = mfcc
            y[count] = labelOf(folder)
            count = count+1
        print(folder,"done")
    
    np.save('X.npy', X)
    np.save('y.npy', y)