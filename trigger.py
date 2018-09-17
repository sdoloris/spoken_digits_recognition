import pyaudio
import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Net

RATE = 16000
CHUNK = 4096

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
              frames_per_buffer=CHUNK)

detection_state = False
recorded_samples = 0
record = np.array([])

print("Loading the model...")
net = Net()
net.load_state_dict(torch.load('./best_model_state.pt', map_location='cpu'))

print("You can talk !")
try:
    while True:
        data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
        #print(data.shape)
        peak= np.average(np.abs(data))*2
        bars = int(50*peak/2**16)
        if bars > 0 and not detection_state:
            detection_state = True
            print("Listening...")
        elif recorded_samples >= 0.7*RATE and detection_state:
            detection_state = False
            mfcc = librosa.feature.mfcc(record.astype(float), sr=RATE, n_mfcc=32)
            
            X = np.zeros((1,32,32))
            X[0,0:mfcc.shape[0],0:min(32,mfcc.shape[1])] = mfcc[:, 0:min(32,mfcc.shape[1])]

            X = torch.from_numpy(X).unsqueeze(0)
            X = X.float()

            with torch.no_grad():
                output = net(X)
                prediction = torch.max(output,1)[1].squeeze(0).item()
                print(prediction)
            recorded_samples = 0
        if detection_state:
            recorded_samples = recorded_samples + CHUNK
            record = np.concatenate((record, data))
        if not detection_state:
            record = data
        #print("%05d %s"%(peak,bars))
except KeyboardInterrupt:
    print("Recording ended")


stream.stop_stream()
stream.close()
p.terminate()