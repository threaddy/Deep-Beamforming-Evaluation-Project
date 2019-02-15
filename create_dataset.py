import numpy as np
import os
import wave
import math
from sklearn.model_selection import train_test_split
import random
import prepare_data as pp
import pickle

timit_dataset_folder = 'timit'
dnn1_dataset_folder = "data_train/dnn1_train/"
dnn2_dataset_folder = "data_train/dnn2_train/"
noise1_path = 'babble.wav'
mixed_snr = "snr_list.p"
dnn1_utterances = 4
dnn2_utterances = 6
fs = 16000

def load_data(data_directory):  # load the dataset, return a list of all found file
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    allfiles = []
    for d in directories:
        speaker_directory = os.path.join(data_directory, d)
        utterance_names = [os.path.join(speaker_directory, f)
                           for f in os.listdir(speaker_directory)
                           if f.endswith(".wav")]
        allfiles.append(utterance_names)
    return allfiles


def flatten(arr):
  for i in arr:
    if isinstance(i, list):
      yield from flatten(arr)
    else:
      yield i

def set_microphone_at_distance(clean_data, noise_data, framerate, distance):
    meter_snr = 10 ** (15 / 20)  # chosen snr at one meter distance (15 dB)
    clean_energy = 0
    noise_energy = 0
    sound_speed = 343

    frame_delay = int(math.ceil(framerate * distance / sound_speed))  # compute number of frame to delay

    delay_silence = np.zeros(frame_delay)
    clean_data = np.append(delay_silence, clean_data)           # add delay to attenuated speech

    shift = random.randint(0, abs(len(clean_data) - len(
        noise_data)))                                           # calculate random shift for noise in the possible range, to avoid "overflows"

    n = min(len(clean_data), len(noise_data))
    clean_data = clean_data[0:n]
    noise_data = noise_data[shift:(n + shift)]

    for t in clean_data:
        clean_energy = clean_energy + abs(t)

    for t in noise_data:
        noise_energy = noise_energy + abs(t)

    first_snr = clean_energy / noise_energy
    snr_ratio = meter_snr / first_snr
    new_noise_data = noise_data / snr_ratio                      # normalizing snr level at 15 db at one meter distance
    new_clean_data = clean_data / distance                       # attenuating clean speech at 1/distance rate

    mixed_data = (new_noise_data + new_clean_data)
    return mixed_data, new_noise_data



training_data = load_data(timit_dataset_folder)  # list of training files
# training_data = (np.asarray(training_data)).flatten(order = 'C')

dnn1_data, dnn2_data = train_test_split(training_data, test_size=0.5,
                                        random_state=13)  # split training data between two dnn
dnn1_data = (np.asarray(dnn1_data)).flatten()
dnn2_data = (np.asarray(dnn2_data)).flatten()
print(dnn1_data.shape)
(noise, _) = pp.read_audio(noise1_path)
pp.create_folder(dnn1_dataset_folder)
distance1_list = []
for n in range(dnn1_utterances):
    # rand_x = random.randint(0, 3)
    # rand_y = random.randint(0, 9)
    current_file = random.choice(dnn1_data)
    dist = random.uniform(1, 20)
    distance1_list.append(dist)
    (clean, _) = pp.read_audio(current_file)

    mixed, noise_new = set_microphone_at_distance(clean, noise, fs, dist)

    audio_path = os.path.join(dnn1_dataset_folder, "mix_%s" % os.path.basename(current_file))
    pp.write_audio(audio_path, mixed, fs)

    clean_path = os.path.join(dnn1_dataset_folder, "clean_%s" % os.path.basename(current_file))
    pp.write_audio(clean_path, clean, fs)

snr_file = open(os.path.join(dnn1_dataset_folder, mixed_snr), "wb")
pickle.dump(distance1_list, snr_file)
snr_file.close()



distance2_list = []
for n in range(dnn2_utterances):
    current_file = random.choice(dnn2_data)

    dist = random.uniform(1, 20)
    distance2_list.append(dist)
    (clean, _) = pp.read_audio(current_file)

    mixed, noise_new = set_microphone_at_distance(clean, noise, fs, dist)

    audio_path = os.path.join(dnn2_dataset_folder, "mix_%s" % os.path.basename(current_file))
    pp.write_audio(audio_path, mixed, fs)

    clean_path = os.path.join(dnn2_dataset_folder, "clean_%s" % os.path.basename(current_file))
    pp.write_audio(clean_path, clean, fs)

snr_file = open(os.path.join(dnn2_dataset_folder, mixed_snr), "wb")
pickle.dump(distance2_list, snr_file)
snr_file.close()


