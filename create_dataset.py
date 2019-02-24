import numpy as np
import os
import math
from sklearn.model_selection import train_test_split
import random
import string
import prepare_data as pp
import config_dnn1 as conf1
import pickle
import h5py
# from numba import vectorize
########################################################################################################################
# Parameters
########################################################################################################################
timit_dataset_folder = 'timit'
dnn1_dataset_folder = "data_train/dnn1_train/"
dnn2_dataset_folder = "data_train/dnn2_train/"
noise1_path = 'noise/babble.wav'
mixed_snr = "snr_list.p"
dnn1_utterances = 100
dnn2_utterances = 100
fs = 16000

data_file_dimension = conf1.data_file_dimension

save_single_files = True

########################################################################################################################
# Functions
########################################################################################################################

# @vectorize(["float64(float64, float64)"], target='cuda')
# def VectorAdd(a, b):
#     return a + b




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


    # mixed_data = VectorAdd(noise_data, clean_data/distance)     # attenuating clean speech at 1/distance rate and adding noise
    mixed_data = (new_noise_data + (clean_data / distance))     # attenuating clean speech at 1/distance rate and adding noise
    return mixed_data, new_noise_data, clean_data


########################################################################################################################
# GENERATE FILE
########################################################################################################################

training_data = load_data(timit_dataset_folder)  # load list of training files

dnn1_data, dnn2_data = train_test_split(training_data, test_size=0.5,
                                        random_state=13)  # split training data between two dnn

dnn1_data = (np.asarray(dnn1_data)).flatten()
dnn2_data = (np.asarray(dnn2_data)).flatten()
print(dnn1_data.shape)
(noise, _) = pp.read_audio(noise1_path)
pp.create_folder(dnn1_dataset_folder)


# GENERATE FILES FOR DNN1

distance1_list = []
mix_all = []
clean_all = []
i = 0
for n in range(dnn1_utterances):
    # rand_x = random.randint(0, 3)
    # rand_y = random.randint(0, 9)
    current_file = random.choice(dnn1_data)
    dist = random.uniform(1, 20)
    distance1_list.append(dist)
    (clean, _) = pp.read_audio(current_file)
    sr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))

    mixed, noise_new, clean_new = set_microphone_at_distance(clean, noise, fs, dist)

    # Print.
    if n % 10 == 0:
        print(n)

    if save_single_files:
        path_list = current_file.split(os.sep)
        audio_path = os.path.join(dnn1_dataset_folder,
                                  "mix_%s_%s_%s" % (path_list[2], sr, os.path.basename(current_file)))

        clean_path = os.path.join(dnn1_dataset_folder,
                                  "clean_%s_%s_%s" % (path_list[2], sr, os.path.basename(current_file)))
        pp.write_audio(audio_path, mixed, fs)
        pp.write_audio(clean_path, clean_new, fs)

    else:
        clean_all.append(clean_new)
        mix_all.append(mixed)

        clean_all_new = np.concatenate(clean_all, axis=0)
        mix_all_new = np.concatenate(mix_all, axis=0)

        if (n) % data_file_dimension == 0:
            i += 1
            # Write out data to .h5 file.
            out_path = os.path.join(dnn1_dataset_folder, "data_mix_%s.h5" % str(i))
            pp.create_folder(os.path.dirname(out_path))

            with h5py.File(out_path, 'w') as hf:
                hf.create_dataset('x', data=mix_all_new)
                hf.create_dataset('y', data=clean_all_new)

            clean_all = []
            mix_all = []

snr_file = open(os.path.join(dnn1_dataset_folder, mixed_snr), "wb")
pickle.dump(distance1_list, snr_file)
snr_file.close()


# GENERATE FILES FOR DNN2

i = 0
distance2_list = []
for n in range(dnn2_utterances):
    current_file = random.choice(dnn2_data)

    dist = random.uniform(1, 20)
    distance2_list.append(dist)
    (clean, _) = pp.read_audio(current_file)
    sr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))

    mixed, noise_new, clean_new = set_microphone_at_distance(clean, noise, fs, dist)

    # Print.
    if n % 10 == 0:
        print(n)

    if save_single_files:
        path_list = current_file.split(os.sep)
        audio_path = os.path.join(dnn2_dataset_folder,
                                  "mix_%s_%s_%s" % (path_list[2], sr, os.path.basename(current_file)))

        clean_path = os.path.join(dnn2_dataset_folder,
                                  "clean_%s_%s_%s" % (path_list[2], sr, os.path.basename(current_file)))
        pp.write_audio(audio_path, mixed, fs)
        pp.write_audio(clean_path, clean_new, fs)

    else:
        clean_all.append(clean_new)
        mix_all.append(mixed)

        clean_all_new = np.concatenate(clean_all, axis=0)
        mix_all_new = np.concatenate(mix_all, axis=0)

        if n % data_file_dimension == 0:
            i += 1
            # Write out data to .h5 file.
            out_path = os.path.join(dnn2_dataset_folder, "data_mix_%s.h5" % str(i))
            pp.create_folder(os.path.dirname(out_path))

            with h5py.File(out_path, 'w') as hf:
                hf.create_dataset('x', data=mix_all_new)
                hf.create_dataset('y', data=clean_all_new)

            clean_all = []
            mix_all = []


snr_file = open(os.path.join(dnn2_dataset_folder, mixed_snr), "wb")
pickle.dump(distance2_list, snr_file)
snr_file.close()


