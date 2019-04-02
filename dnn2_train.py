import numpy as np
import os
import time
import math
import random
import string
import prepare_data as pp
import dnn1_eval as dnn1
import dnn1_config as conf1
import dnn2_config as conf2


# from get_gpu import get_gpu
# if sys.platform == 'linux':
#     gpuID = get_gpu()
#     logging.info("GPU ID:" + str(gpuID))



from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import TensorBoard






########################################################################################################################################
# FUNCTIONS
########################################################################################################################################


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
    for (batch_x, batch_y) in gen.generate(xs=[x], ys=[y]):
        pred = model.predict(batch_x)
        pred_all.append(pred)
        y_all.append(batch_y)

    # Concatenate mini batch prediction.
    pred_all = np.concatenate(pred_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    # Compute loss.
    loss = pp.np_mean_absolute_error(y_all, pred_all)
    return loss



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
    new_noise_energy = noise_energy/snr_ratio                   # updating noisy energy after normalization

    # mixed_data = VectorAdd(noise_data, clean_data/distance)     # attenuating clean speech at 1/distance rate and adding noise
    mixed_data = (new_noise_data + (clean_data / distance))     # attenuating clean speech at 1/distance rate and adding noise

    final_s2nr = (clean_energy/distance)/(new_noise_energy + clean_energy/distance)            #calculating s2nr of attenuated speech


    return mixed_data, new_noise_data, clean_data, final_s2nr

########################################################################################################################
########################################################################################################################
# PREPARE DATASET
########################################################################################################################
########################################################################################################################

def prepare_database():

    (noise, _) = pp.read_audio(conf2.noise_path)

    with open(os.path.join('dnn2', 'dnn2_files_list.txt')) as f:
        dnn2_data = f.readlines()

    (model1, scaler1) = dnn1.load_dnn()

    # generate train mean values

    snr2_list = []
    mixed_avg = []
    clean_avg = []
    enh_avg = []


    for n in range(conf2.training_number):
        current_file = (random.choice(dnn2_data)).rstrip()
        dist = random.randint(1, 20)
        (clean, _) = pp.read_audio(current_file)


        mixed, noise_new, clean_new, s2nr = set_microphone_at_distance(clean, noise, conf2.fs, dist)


        (_, enh, _) = dnn1.predict_file(current_file, model1, scaler1)

        # s2nr = 1 / (1 + (1 / float(snr)))
        snr2_list.append(s2nr)

        mixed_avg.append(np.mean(mixed))
        clean_avg.append(np.mean(clean_new))
        enh_avg.append(np.mean(enh))

        sr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
        path_list = current_file.split(os.sep)
        mixed_name = "mix_%s_%s_%s" % (path_list[2], sr, os.path.basename(current_file))
        clean_name = "clean_%s_%s_%s" % (path_list[2], sr, os.path.basename(current_file))
        enh_name = "enh_%s_%s_%s" % (path_list[2], sr, os.path.basename(current_file))


        if n % 10 == 0:
            print(n)

        if conf2.save_single_files and n < conf1.n_files_to_save:


            mixed_path = os.path.join(conf2.train_folder, mixed_name)
            clean_path = os.path.join(conf2.train_folder, clean_name)
            enh_path = os.path.join(conf2.train_folder, enh_name)
            pp.write_audio(mixed_path, mixed, conf2.fs)
            pp.write_audio(clean_path, clean_new, conf2.fs)
            pp.write_audio(enh_path, enh, conf2.fs)

    if len(mixed_avg) != len(enh_avg):
        raise Exception('Number of mixed and enhanced audio must be the same')

    num_tr = len(mixed_avg)

    if os.path.exists(os.path.join(conf2.train_folder, 'train_data.txt')):
        os.remove(os.path.join(conf2.train_folder, 'train_data.txt'))
    f1 = open(os.path.join(conf2.train_folder, 'train_data.txt'), 'w')
    for line1, line2, line3  in zip(mixed_avg, clean_avg, snr2_list):
        f1.write("%s, %s, %s\n" % (line1, line2, line3))

    print(len(mixed_avg), ',', len(enh_avg))





    # generate test spectrograms]

    snr2_list = []
    mixed_avg = []
    clean_avg = []
    enh_avg = []

    for n in range(conf2.test_number):
        current_file = (random.choice(dnn2_data)).rstrip()
        dist = random.randint(1, 20)
        (clean, _) = pp.read_audio(current_file)

        mixed, noise_new, clean_new, s2nr = set_microphone_at_distance(clean, noise, conf2.fs, dist)

        (_, enh, _) = dnn1.predict_file(current_file, model1, scaler1)

        # s2nr = 1 / (1 + (1 / float(snr)))
        snr2_list.append(s2nr)

        mixed_avg.append(np.mean(mixed))
        clean_avg.append(np.mean(clean_new))
        enh_avg.append(np.mean(enh))

        sr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
        path_list = current_file.split(os.sep)
        mixed_name = "mix_%s_%s_%s" % (path_list[2], sr, os.path.basename(current_file))
        clean_name = "clean_%s_%s_%s" % (path_list[2], sr, os.path.basename(current_file))
        enh_name = "enh_%s_%s_%s" % (path_list[2], sr, os.path.basename(current_file))

        if n % 10 == 0:
            print(n)

        if conf2.save_single_files and n < conf1.n_files_to_save:

            mixed_path = os.path.join(conf2.train_folder, mixed_name)
            clean_path = os.path.join(conf2.train_folder, clean_name)
            enh_path = os.path.join(conf2.train_folder, enh_name)
            pp.write_audio(mixed_path, mixed, conf2.fs)
            pp.write_audio(clean_path, clean_new, conf2.fs)
            pp.write_audio(enh_path, enh, conf2.fs)

    print(len(mixed_avg), ',', len(enh_avg))

    if len(mixed_avg) != len(enh_avg):
        raise Exception('Number of mixed and enhanced audio must be the same')

    num_te = len(mixed_avg)

    if os.path.exists(os.path.join(conf2.test_folder, 'test_data.txt')):
        os.remove(os.path.join(conf2.test_folder, 'test_data.txt'))
    f1 = open(os.path.join(conf2.test_folder, 'test_data.txt'), 'w')
    for line1, line2, line3  in zip(mixed_avg, clean_avg, snr2_list):
        f1.write("%s, %s, %s\n" % (line1, line2, line3))


    return num_tr, num_te


########################################################################################################################
########################################################################################################################
# TRAIN
########################################################################################################################
########################################################################################################################
# get_gpu()

pp.create_folder(conf2.train_folder)
pp.create_folder(conf2.test_folder)
pp.create_folder(conf2.packed_feature_dir)
pp.create_folder(conf2.logs)
pp.create_folder(conf2.model_dir)

t1 = time.time()

if conf2.create_new_database:
    num_tr, num_te = prepare_database()


tr_x = []
tr_y_s2nr = []
te_x = []
te_y_s2nr = []



with open(os.path.join(conf2.train_folder, 'train_data.txt'), "r+") as train_file:
    data = train_file.readlines()
    for line in data:
        a = line.strip().split(",")
        tr_x.append((a[0], a[1]))
        tr_y_s2nr.append(a[2])

with open(os.path.join(conf2.test_folder, 'test_data.txt'), "r+") as test_file:
    data = test_file.readlines()
    for line in data:
        a = line.strip().split(",")
        te_x.append((a[0], a[1]))
        te_y_s2nr.append(a[2])



tr_x = np.asarray(tr_x)
te_x = np.asarray(te_x)
tr_y_s2nr = (np.asarray(tr_y_s2nr)).transpose()
te_y_s2nr = (np.asarray(te_y_s2nr)).transpose()




# Build model

n_hid = 1024

model = Sequential()
model.add(Dense(n_hid, input_dim=2, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(n_hid, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
# model.summary()


tsbd = TensorBoard(log_dir=conf2.logs)

model.compile(loss='mean_absolute_error',
              optimizer=SGD(lr=conf2.lr, momentum=0.9))#, decay=0.0015))

model.fit(tr_x, tr_y_s2nr, epochs=conf2.epochs, batch_size=conf2.batch_size,
          validation_data=(te_x, te_y_s2nr), callbacks=[tsbd])


test_loss = model.evaluate(te_x, te_y_s2nr)
print("\n Test_loss: %s \n" % test_loss)


print("Training time: %s s" % (time.time() - t1,))


model_path = os.path.join(conf2.model_dir, "md_%epochs.h5" % conf2.epochs)
model.save(model_path)
print("Saved model to %s" % model_path)

pred = model.predict(tr_x)
print(pred)




