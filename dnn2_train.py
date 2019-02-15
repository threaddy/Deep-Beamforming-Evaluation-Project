import numpy as np
import os
import pickle
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

from data_generator import DataGenerator
import prepare_data as pp
import config_dnn2 as conf2
import dnn1_eval as dnn1


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


########################################################################################################################
# CREATE DATASET
########################################################################################################################
work_dir = "data_train/dnn1_train"
dnn1.predict(work_dir, work_dir)

work_dir = "data_train/dnn2_train"
dnn1.predict(work_dir, work_dir)

########################################################################################################################
########################################################################################################################
# PREPARE DATASET
########################################################################################################################
########################################################################################################################
# def train():


# Load data.
t1 = time.time()
train_x_noisy = pp.load_data(conf2.train_folder, "mix")
train_x_enh = pp.load_data(conf2.train_folder, "enh")


dnn1_test_noisy = pp.load_data(conf2.test_folder, "mix")
dnn1_test_enh = pp.load_data(conf2.test_folder, "enh")

train2_snr = pickle.load(open(os.path.join(conf2.train_folder, "snr_list.p"), "rb"))
test1_snr = pickle.load(open(os.path.join(conf2.test_folder, "snr_list.p"), "rb"))

tr_x_noisy = []
for na in train_x_noisy:
    (a, _) = pp.read_audio(na)
    tr_x_noisy.append(np.mean(a))

tr_x_enh = []
for na in train_x_enh:
    (a, _) = pp.read_audio(na)
    tr_x_enh.append(np.mean(a))

te_x_noisy = []
for na in dnn1_test_noisy:
    (a, _) = pp.read_audio(na)
    te_x_noisy.append(np.mean(a))

te_x_enh = []
for na in dnn1_test_enh:
    (a, _) = pp.read_audio(na)
    te_x_enh.append(np.mean(a))

tr_y_s2nr = []
for l1 in train2_snr:
        a = 1 / (1 + (1 / float(l1)))
        tr_y_s2nr.append(a)

te_y_s2nr = []
for l1 in test1_snr:
        a = 1 / (1 + (1 / float(l1)))
        te_y_s2nr.append(a)

# print(len(tr_x_noisy), len(tr_y_s2nr))
# print(len(te_x_noisy), len(te_y_s2nr))


tr_x_noisy = np.asarray(tr_x_noisy)
tr_x_enh = np.asarray(tr_x_enh)
tr_x = (np.append([tr_x_noisy], [tr_x_enh], axis=0)).transpose()

te_x_noisy = np.asarray(te_x_noisy)
te_x_enh = np.asarray(te_x_enh)
te_x = (np.append([te_x_noisy], [te_x_enh], axis=0)).transpose()

tr_y_s2nr = np.asarray(tr_y_s2nr)
te_y_s2nr = np.asarray(te_y_s2nr)

# inserisci salvataggio in .h5 se necessario, quindi aggiungere codice ci caricamento dataset nel "TRAIN"
print(tr_x.shape, tr_y_s2nr.shape)
print(te_x.shape, te_y_s2nr.shape)
########################################################################################################################
########################################################################################################################
# TRAIN
########################################################################################################################
########################################################################################################################
# Build model
# (_, n_concat, n_freq) = tr_x.shape
n_hid = 1024

model = Sequential()
model.add(Dense(n_hid, input_dim=2, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(n_hid, activation='sigmoid'))
model.add(Dropout(0.2))
# model.add(Dense(n_hid, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
# model.summary()

model.compile(loss='mean_absolute_error',
              optimizer=SGD(lr=conf2.lr, momentum=0.0, decay=0.0015))

# model.fit(tr_x, tr_y_s2nr, epochs=5, batch_size=512)


tr_gen = DataGenerator(batch_size=conf2.batch_size, gtype='train')
eval_te_gen = DataGenerator(batch_size=conf2.batch_size, gtype='test', te_max_iter=100)
eval_tr_gen = DataGenerator(batch_size=conf2.batch_size, gtype='test', te_max_iter=100)

# Directories for saving models and training stats
model_dir = os.path.join("data_train", "dnn2_packed_features", "models")
pp.create_folder(model_dir)

stats_dir = os.path.join("data_train", "dnn2_packed_features", "training_stats")
pp.create_folder(stats_dir)

# Print loss before training.
iter = 0
tr_loss = eval(model, eval_tr_gen, tr_x, tr_y_s2nr)
te_loss = eval(model, eval_te_gen, te_x, te_y_s2nr)
print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))

# Save out training stats.
stat_dict = {'iter': iter,
             'tr_loss': tr_loss,
             'te_loss': te_loss, }
stat_path = os.path.join(stats_dir, "%diters.p" % iter)
pickle.dump(stat_dict, open(stat_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

iter = 0
tr_loss = eval(model, eval_tr_gen, tr_x, tr_y_s2nr)
te_loss = eval(model, eval_te_gen, te_x, te_y_s2nr)
print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))

# Save out training stats.
stat_dict = {'iter': iter,
             'tr_loss': tr_loss,
             'te_loss': te_loss, }
stat_path = os.path.join(stats_dir, "%diters.p" % iter)
pickle.dump(stat_dict, open(stat_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

# Train.
t1 = time.time()
for (batch_x, batch_y) in tr_gen.generate(xs=[tr_x], ys=[tr_y_s2nr]):
    loss = model.train_on_batch(batch_x, batch_y)
    iter += 1

    # Validate and save training stats.
    if iter % 10 == 0:
        tr_loss = eval(model, eval_tr_gen, tr_x, tr_y_s2nr)
        te_loss = eval(model, eval_te_gen, te_x, te_y_s2nr)
        print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))

        # Save out training stats.
        stat_dict = {'iter': iter,
                     'tr_loss': tr_loss,
                     'te_loss': te_loss, }
        stat_path = os.path.join(stats_dir, "%diters.p" % iter)
        pickle.dump(stat_dict, open(stat_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    # Save model.
    if iter % conf2.iterations == 0:
        model_path = os.path.join(model_dir, "md_%diters.h5" % iter)
        model.save(model_path)
        print("Saved model to %s" % model_path)

    if iter == (conf2.iterations + 1):
        break

print("Training time: %s s" % (time.time() - t1,))
