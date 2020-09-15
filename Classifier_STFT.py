
import os
import librosa as lb
import noisereduce as nr
import numpy as np
import pathlib


f_path = "/home/nikeboke/PycharmProjects/SEC/Dataset_audio"


def extract_STFT(file, name, activity, subject):
    # read audio data
    audio_data, sample_rate = lb.load(file)

    # noise reduction
    noisy_part = audio_data[0:25000]
    reduced_noise = nr.reduce_noise(audio_clip=audio_data, noise_clip=noisy_part, verbose=False)

    # trimming
    trimmed, index = lb.effects.trim(reduced_noise, top_db=20, frame_length=512, hop_length=64)

    # extract features
    stft = np.abs(lb.stft(trimmed, n_fft=512, hop_length=256, win_length=512))
    # save features
    np.save("STFT_features/stft_257_1/" + subject + "-" + name[:-4] + "_" + activity + ".npy", stft)


if __name__ == "__main__":
    activities = ['Calling', 'Clapping', 'Drinking', 'Eating', 'Entering',
                  'Exiting', 'Falling', 'LyingDown', 'OpeningPillContainer',
                  'PickingObject', 'Reading', 'SitStill', 'Sitting', 'Sleeping',
                  'StandUp', 'Sweeping', 'UseLaptop', 'UsingPhone', 'WakeUp', 'Walking',
                  'WashingHand', 'WatchingTV', 'WaterPouring', 'Writing','env1','env2','env3','env4','env5']

    subjects = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09',
                's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17']

    pathlib.Path("STFT_features/stft_257_1/").mkdir(parents=True, exist_ok=True)

    for activity in activities:
        for subject in subjects:
            path = "Dataset_audio/" + subject + "/" + activity
            for file in os.listdir(path):
                if file.endswith(".wav"):
                    extract_STFT(path + "/" + file,
                               file,
                               activity,
                               subject)
                    print(subject, activity, file)
