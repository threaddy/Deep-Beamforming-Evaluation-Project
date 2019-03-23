import numpy as np
import os
import pickle
import time
import h5py
import math
import random
import string

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import multi_gpu_model

from data_generator import DataGenerator
import prepare_data as pp
import dnn1_config as conf1
from sklearn import preprocessing

from keras.callbacks import TensorBoard
import tensorflow as tf


########################################################################################################################################
# FUNCTIONS
########################################################################################################################################
import GPUtil
from tensorflow.python.keras import backend as K

def get_gpu():
    import os
    import tensorflow as tf
    import GPUtil
    from tensorflow.python.keras import backend as K
    # Get the first available GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    try:

        DEVICE_ID = GPUtil.getAvailable(order='memory', limit=2, maxLoad=0.001, maxMemory=0.001)[0]

    except:
        print('GPU not compatible with NVIDIA-SMI')
        DEVICE_ID = 'Not Found'
    else:

        # print(DEVICE_ID)
        # DEVICE_ID = (1 + DEVICE_ID) % 2
        # print(DEVICE_ID)
        # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
        # sess = tf.Session(config=tf.ConfigProto())
        config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        # config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)

    finally:
        # Since all other GPUs are masked out, the first available GPU will now be identified as GPU:0
        # device = '/gpu:0'
        print('Device ID (unmasked): ' + str(DEVICE_ID))
        print('Device ID (masked): ' + str(0))

    return DEVICE_ID


def eval(model, gen, x, y):
    """Validation function.

    Args:
      model: keras model.
      gen: object, data generator.
      x: 3darray, input, (n_segs, n_concat, n_freq)
      y: 2darray, target, (n_segs, n_freq)
    """
    pred_all, y_all = [], []

    # Inference in mini batch.
    for (batch_x, batch_y, _) in gen.generate(xs=[x], ys=[y]):
        pred = model.predict(batch_x)
        pred_all.append(pred)
        y_all.append(batch_y)

    # Concatenate mini batch prediction.
    pred_all = np.concatenate(pred_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    # Compute loss.
    loss = pp.np_mean_absolute_error(y_all, pred_all)
    return loss


def compute_scaler(data_type):
    """Compute and write out scaler of data.
    """

    # Load data.
    t1 = time.time()
    data_folder = os.path.join("dnn1", "dnn1_packed_features", data_type)
    data_file_names = [os.path.join(data_folder, f)
                       for f in os.listdir(data_folder)
                       if f.endswith(".h5")]

    for hdf5_path in data_file_names:
        with h5py.File(hdf5_path, 'r') as hf:
            x = hf.get('x')
            x = np.array(x)  # (n_segs, n_concat, n_freq)


            # Compute scaler.
            (n_segs, n_concat, n_freq) = x.shape
            y2d = x.reshape((n_segs * n_concat, n_freq))
            scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(y2d)
            # print(scaler.mean_)
            # print(scaler.scale_)

            tfile = os.path.splitext(os.path.basename(hdf5_path))[0]

            # Write out scaler.
            out_path = os.path.join(conf1.packed_feature_dir, data_type, "%s_scaler.p" % tfile)
            pickle.dump(scaler, open(out_path, 'wb'))

    print("Save scaler to %s" % out_path)
    print("Compute scaler finished! %s s" % (time.time() - t1,))


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


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

    final_snr = (clean_energy / distance) / noise_energy  # calculating snr of attenuated speech

    return mixed_data, new_noise_data, clean_data, final_snr


########################################################################################################################
########################################################################################################################
# PREPARE DATASET
########################################################################################################################
#######################################################################################################################
def prepare_database():

    if conf1.retrain != 0:
        for file in os.listdir(conf1.data_train_dir):
            file_path = os.path.join(conf1.data_train_dir, file)
            os.remove(file_path)
        for file in os.listdir(conf1.data_test_dir):
            file_path = os.path.join(conf1.data_test_dir, file)
            os.remove(file_path)

    (noise, _) = pp.read_audio(conf1.noise_path)

    with open('dnn1/dnn1_files_list.txt') as f:
        dnn1_data = f.readlines()


    # generate train spectrograms
    mixed_all = []
    clean_all = []

    snr1_list = []
    mixed_avg = []

    i = 0
    while i < conf1.training_number:
        for n in range(conf1.data_file_dimension):
            i += 1
            current_file = (random.choice(dnn1_data)).rstrip()
            dist = random.randint(1, 20)
            (clean, _) = pp.read_audio(current_file)


            mixed, noise_new, clean_new, snr = set_microphone_at_distance(clean, noise, conf1.fs, dist)

            snr1_list.append(snr)
            mixed_avg.append(np.mean(mixed))


            if i % 10 == 0:
                print(n)

            if conf1.save_single_files and n < conf1.n_files_to_save:

                sr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))

                path_list = current_file.split(os.sep)
                mixed_name = "mix_%s_%s_%s" % (path_list[2], sr, os.path.basename(current_file))
                clean_name = "clean_%s_%s_%s" % (path_list[2], sr, os.path.basename(current_file))
                path_list = current_file.split(os.sep)

                mixed_path = os.path.join(conf1.train_folder, mixed_name)
                clean_path = os.path.join(conf1.train_folder, clean_name)

                pp.write_audio(mixed_path, mixed, conf1.fs)
                pp.write_audio(clean_path, clean_new, conf1.fs)


            clean_spec = pp.calc_sp(clean_new, mode='magnitude')
            mixed_spec = pp.calc_sp(mixed, mode='complex')

            clean_all.append(clean_spec)
            mixed_all.append(mixed_spec)

        print(len(clean_all), ',', len(mixed_all))
        num_tr = pp.pack_features(mixed_all, clean_all, 'train', str(int(time.time())))

    compute_scaler('train')




    # generate test spectrograms
    mixed_all = []
    clean_all = []

    snr1_list = []
    mixed_avg = []

    i = 0
    while i < conf1.test_number:
        for n in range(conf1.data_file_dimension):
            i += 1
            current_file = (random.choice(dnn1_data)).rstrip()
            dist = random.randint(1, 20)
            (clean, _) = pp.read_audio(current_file)


            mixed, noise_new, clean_new, snr = set_microphone_at_distance(clean, noise, conf1.fs, dist)

            snr1_list.append(snr)
            mixed_avg.append(np.mean(mixed))


            if i % 10 == 0:
                print(n)

            if conf1.save_single_files and n < conf1.n_files_to_save:

                sr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))

                path_list = current_file.split(os.sep)
                mixed_name = "mix_%s_%s_%s" % (path_list[2], sr, os.path.basename(current_file))
                clean_name = "clean_%s_%s_%s" % (path_list[2], sr, os.path.basename(current_file))
                path_list = current_file.split(os.sep)

                mixed_path = os.path.join(conf1.train_folder, mixed_name)
                clean_path = os.path.join(conf1.train_folder, clean_name)

                pp.write_audio(mixed_path, mixed, conf1.fs)
                pp.write_audio(clean_path, clean_new, conf1.fs)


            clean_spec = pp.calc_sp(clean_new, mode='magnitude')
            mixed_spec = pp.calc_sp(mixed, mode='complex')

            clean_all.append(clean_spec)
            mixed_all.append(mixed_spec)

        print(len(clean_all), ',', len(mixed_all))
        num_te = pp.pack_features(mixed_all, clean_all, 'test', str(int(time.time())))

    compute_scaler('test')


    return num_tr, num_te

########################################################################################################################
########################################################################################################################
# TRAIN
########################################################################################################################
########################################################################################################################
# get_gpu()

pp.create_folder(conf1.train_folder)
pp.create_folder(conf1.test_folder)
pp.create_folder(conf1.packed_feature_dir)
pp.create_folder(conf1.data_train_dir)
pp.create_folder(conf1.data_test_dir)
pp.create_folder(conf1.logs)
pp.create_folder(conf1.model_dir)
pp.create_folder(conf1.stats_dir)


t1 = time.time()

if conf1.create_new_database:
    num_tr, num_te = prepare_database()

h5_train_list = [f for f in sorted(os.listdir(conf1.data_train_dir))
                                       if f.endswith('.h5')]

h5_test_list = [f for f in os.listdir(conf1.data_test_dir)
                                       if f.endswith('.h5')]


tr_x = []
tr_y = []
te_x = []
te_y = []



for i in h5_test_list:
    te_x_t, te_y_t = pp.load_hdf5(os.path.join(conf1.data_test_dir, i))
    te_x.append(te_x_t)
    te_y.append(te_y_t)


te_x = np.concatenate(te_x, axis=0)
te_y = np.concatenate(te_y, axis=0)

#scale test data
scaler = pickle.load(open(os.path.join(conf1.packed_feature_dir, 'test', 'scaler.p'), 'rb'))
te_x = pp.scale_on_3d(te_x, scaler)
te_y = pp.scale_on_2d(te_y, scaler)
print("Scale data time: %s s" % (time.time() - t1,))

print("Load data time: %s s" % (time.time() - t1,))

# conf.batch_size = 512
# print("%d iterations / epoch" % int(tr_x.shape[0] / conf1.batch_size))




tr_x, tr_y = pp.load_hdf5(os.path.join(conf1.data_train_dir, h5_train_list[0]))


# Debug plot.
# if False:
#     plt.matshow(tr_x[0: 1000, 0, :].T, origin='lower', aspect='auto', cmap='jet')
#     plt.show()
#     pause


# DAB model
(_, n_concat, n_freq) = tr_x.shape
n_hid = 1024

model = Sequential()
model.add(Flatten(input_shape=(n_concat, n_freq)))
model.add(Dense(n_hid, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(n_hid, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(n_freq, activation='linear'))
model.summary()
if conf1.multi_gpu > 1:
    model = multi_gpu_model(model, gpus=conf1.multi_gpu)
model.compile(loss='mean_absolute_error',
              optimizer=SGD(lr=conf1.lr, momentum=0.9, decay=0.0015))


# # SEDNN alternative model
# (_, n_concat, n_freq) = tr_x.shape
# n_hid = 2048
#
# model = Sequential()
# model.add(Flatten(input_shape=(n_concat, n_freq)))
# model.add(Dense(n_hid, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(n_hid, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(n_hid, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(n_freq, activation='linear'))
# model.summary()
# if conf1.multi_gpu > 1:
#     model = multi_gpu_model(model, gpus=conf1.multi_gpu)
# model.compile(loss='mean_absolute_error',
#               optimizer=Adam(lr=conf1.lr))




# load previous model
if conf1.retrain != 0:
    model_path = os.path.join(conf1.model_dir, "md_%diters.h5" % conf1.retrain)
    model = load_model(model_path)


callback = TensorBoard(log_dir=conf1.logs)
callback.set_model(model)
train_names = ['train_loss', 'train_mae']
val_names = ['val_loss', 'val_mae']

# Data generator.
tr_gen = DataGenerator(batch_size=conf1.batch_size, gtype='train')
eval_te_gen = DataGenerator(batch_size=conf1.batch_size, gtype='test', te_max_iter=100)
eval_tr_gen = DataGenerator(batch_size=conf1.batch_size, gtype='test', te_max_iter=100)




# Train.
t1 = time.time()
iter = 0
epochs = 0
# while iter < conf1.iterations:
while epochs < conf1.epochs:
    random.shuffle(h5_train_list)
    print("Epoch number ---> %d" % epochs)
    for i in h5_train_list:
        tr_x, tr_y = pp.load_hdf5(os.path.join(conf1.data_train_dir, i))
        # print('current h5 file: %s' % i)

        # scale current data
        hfile = os.path.splitext(os.path.basename(i))[0]
        scaler = pickle.load(open(os.path.join(conf1.packed_feature_dir, 'train', '%s_scaler.p' % hfile), 'rb'))
        tr_x = pp.scale_on_3d(tr_x, scaler)
        # tr_y = np.concatenate(tr_y, axis=0)
        tr_y = pp.scale_on_2d(tr_y, scaler)


        for (batch_x, batch_y, epoch) in tr_gen.generate(xs=[tr_x], ys=[tr_y]):
            loss = model.train_on_batch(batch_x, batch_y, )
            iter += 1

            # Validate and save training stats.
            if iter % 10 == 0:
                tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
                te_loss = eval(model, eval_te_gen, te_x, te_y)
                print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))

                write_log(callback, train_names, [tr_loss, te_loss], iter)
                # Save out training stats.
                stat_dict = {'iter': iter,
                             'tr_loss': tr_loss,
                             'te_loss': te_loss, }
                stat_path = os.path.join(conf1.stats_dir, "%diters.p" % iter)
                pickle.dump(stat_dict, open(stat_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


            if iter % 1000 == 0:
                model_path = os.path.join(conf1.model_dir, "md_%diters.h5" % iter)
                model.save(model_path)
                print("Saved model to %s" % model_path)

    epochs += 1


# model.fit(tr_x, tr_y, epochs= 5)
#
# test_loss, test_acc = model.evaluate(tr_x, tr_y)



print("Training time: %s s" % (time.time() - t1,))
