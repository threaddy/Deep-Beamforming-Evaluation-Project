import numpy as np
import os
from sklearn.model_selection import train_test_split

########################################################################################################################
# Parameters
########################################################################################################################
timit_dataset_folder = 'timit'
noise1_path = 'noise/babble.wav'

fs = 16000

########################################################################################################################
# Functions
########################################################################################################################
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


########################################################################################################################
# GENERATE FILE LIST
########################################################################################################################


training_data = load_data(timit_dataset_folder)  # load list of training files

dnn1_data, dnn2_data = train_test_split(training_data, test_size=0.5,
                                        random_state=13)  # split training data between two dnn

dnn1_data = (np.asarray(dnn1_data)).flatten()
dnn2_data = (np.asarray(dnn2_data)).flatten()
print(dnn1_data.shape)
print(dnn2_data.shape)


if os.path.exists(os.path.join('dnn1', 'dnn1_files_list.txt')):
        os.remove(os.path.join('dnn1', 'dnn1_files_list.txt'))
f1 = open(os.path.join('dnn1', 'dnn1_files_list.txt'), 'w')
for line in dnn1_data:
    f1.write("%s\n" % line)


if os.path.exists(os.path.join('dnn2', 'dnn2_files_list.txt')):
        os.remove(os.path.join('dnn2', 'dnn2_files_list.txt'))
f1 = open(os.path.join('dnn2', 'dnn2_files_list.txt'), 'w')
for line in dnn2_data:
    f1.write("%s\n" % line)
