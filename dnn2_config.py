########################################################################################################################
# DNN2  Parameters
########################################################################################################################
import os


train_folder = os.path.join('dnn2', 'dnn2_train')
test_folder = os.path.join('dnn2', 'dnn2_test')
packed_feature_dir = os.path.join('dnn2', 'dnn2_packed_features')

logs = os.path.join('dnn2', 'logs')
model_dir = os.path.join("dnn2", "dnn2_packed_features", "models")


noise_path = 'noise/babble.wav'
fs = 16000

create_new_database = True
save_single_files = False

training_number = 2048
test_number = 512
n_files_to_save = 5

sample_rate = 16000


lr = 0.08                                               #learning rate
epochs = 1000
batch_size = 32




