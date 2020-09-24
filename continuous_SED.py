import numpy as np
import librosa
import matplotlib.pyplot as plt
import noisereduce as nr
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import pyaudio

#Load segment audio classification model

model_path = r"Models/"
model_name = "audio_NN_New2020_05_17_20_59_11_acc_84.68"

# Model reconstruction from JSON file
with open(model_path + model_name + '.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights(model_path + model_name + '.h5')

# Replicate label encoder
lb = LabelEncoder()
lb.fit_transform(['Clapping', 'Falling', 'WashingHand','Sweeping','WatchingTV','other'])



# Plot audio with zoomed in y axis
def plotAudio(output):
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,10))
    plt.plot(output, color='blue')
    ax.set_xlim((0, len(output)))
    ax.margins(2, -0.1)
    plt.show()

# Plot audio
def plotAudio2(output):
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
    plt.plot(output, color='blue')
    ax.set_xlim((0, len(output)))
    plt.show()
# Splits a given long audio file on silent parts.
# Accepts audio numpy array audio_data, window length w and hop length h, threshold_level, tolerence
# threshold_level: Silence threshold
# set tolerence to prevent small silence parts from over splitting the audio.

def split_audio(audio_data, w, h, threshold_level, tolerence=10):
    split_map = []
    start = 0
    data = np.abs(audio_data)
    threshold = threshold_level*np.mean(data[:25000])
    inside_sound = False
    near = 0
    for i in range(0,len(data)-w, h):
        win_mean = np.mean(data[i:i+w])
        if(win_mean>threshold and not(inside_sound)):
            inside_sound = True
            start = i
        if(win_mean<=threshold and inside_sound and near>tolerence):
            inside_sound = False
            near = 0
            split_map.append([start, i])
        if(inside_sound and win_mean<=threshold):
            near += 1
    return split_map

def minMaxNormalize(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    return (arr-mn)/(mx-mn)

def predictSound(X):
    clip, index = librosa.effects.trim(X, top_db=20, frame_length=512, hop_length=64) # Empherically select top_db for every sample
    stfts = np.abs(librosa.stft(clip, n_fft=512, hop_length=256, win_length=512))
    stfts = np.mean(stfts,axis=1)
    stfts = minMaxNormalize(stfts)
    result = model.predict(np.array([stfts]))
    predictions = [np.argmax(y) for y in result]
    print(lb.inverse_transform([predictions[0]])[0])
    plotAudio2(clip)


CHUNKSIZE = 22050  # Record in chunks of 22050 samples
RATE = 22050 # Record at 22050 samples per second
sample_format = pyaudio.paFloat32  # 32bits per sample

# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=sample_format, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

# noise window
audio_data = stream.read(10000)
noise_sample = np.frombuffer(audio_data, dtype=np.float32)
print("Noise Sample")
plotAudio2(noise_sample)
loud_threshold = np.mean(np.abs(noise_sample)) * 10
print("Loud threshold", loud_threshold)
audio_buffer = [] # Initialize array to store frames
near = 0

while (True):
    # Read chunk and load it into numpy array.
    data = stream.read(CHUNKSIZE)
    current_window = np.frombuffer(data, dtype=np.float32) # Interpret a buffer as a 1-dimensional array.

    # Reduce noise real-time
    current_window = nr.reduce_noise(audio_clip=current_window, noise_clip=noise_sample, verbose=False)

    if (audio_buffer == []):
        audio_buffer = current_window
    else:
        if (np.mean(np.abs(current_window)) < loud_threshold):
            print("Inside silence reign")
            if (near < 10):
                audio_buffer = np.concatenate((audio_buffer, current_window))
                near += 1
            else:
                predictSound(np.array(audio_buffer))
                audio_buffer = []
                near
        else:
            print("Inside loud reign")
            near = 0
            audio_buffer = np.concatenate((audio_buffer, current_window))

# close stream
stream.stop_stream()
stream.close()
p.terminate()

