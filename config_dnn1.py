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
stats_dir = os.path.join("dnn1", "dnn1_packed_features", "training_stats")



noise_path = 'noise/babble.wav'
fs = 16000

data_file_dimension = 10
create_new_database = True
save_single_files = True

training_number = 40
test_number = 20
n_files_to_save = 5

sample_rate = 16000
n_window = 512      # windows size for FFT
n_overlap = 256     # overlap of window
n_concat = 7
n_hop= 3

lr = 0.08                                               #learning rate
iterations = 1000
batch_size = 512

# print("%d iterations / epoch" % int(tr_x.shape[0] / batch_size))
