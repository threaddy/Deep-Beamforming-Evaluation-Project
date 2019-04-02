########################################################################################################################
# DNN1  Parameters
########################################################################################################################
import os

train_folder = os.path.join('dnn1', 'dnn1_train')
test_folder = os.path.join('dnn1', 'dnn1_test')
packed_feature_dir = os.path.join('dnn1', 'dnn1_packed_features')
data_train_dir = os.path.join("dnn1", "dnn1_packed_features", "train")
data_test_dir = os.path.join("dnn1", "dnn1_packed_features", "test")
logs = os.path.join('dnn1', 'logs')
model_dir = os.path.join("dnn1", "dnn1_packed_features", "models")
stats_dir = os.path.join("dnn1", "training_stats")



noise_path = 'noise/babble.wav'
fs = 16000

data_file_dimension = 512
create_new_database = True
save_single_files = False

training_number = 8192
test_number = 1024
n_files_to_save = 5
# h5_files_to_use = 8

sample_rate = 16000
n_window = 512      # windows size for FFT
n_overlap = 256     # overlap of window
n_concat = 7
n_hop = 3

lr = 0.008                                               #learning rate

epochs = 50
batch_size = 512

retrain = 0                                             #inset prev iteration model to retrain, 0 = no previous model

multi_gpu = 2                                           # number of gpus to use

iterations = 200000                                       # to select corresponding model in evaluation

# print("%d iterations / epoch" % int(tr_x.shape[0] / batch_size))
