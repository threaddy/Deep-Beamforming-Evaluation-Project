
train_folder = "data_train/dnn1_train/"
packed_feature_dir = "data_train/dnn1_packed_features"

dnn2_train_folder = "data_train/dnn2_train/"         # just to test dnn1 with dnn2 files, too

data_file_dimension = 10
use_previous_files = True

sample_rate = 16000
n_window = 512      # windows size for FFT
n_overlap = 256     # overlap of window
n_concat = 7
n_hop= 3

lr = 0.08                                               #learning rate
iterations = 100
batch_size = 512

# print("%d iterations / epoch" % int(tr_x.shape[0] / batch_size))