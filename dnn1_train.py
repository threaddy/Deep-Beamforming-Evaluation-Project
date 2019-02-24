import numpy as np
import os
import pickle
import time
import h5py

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD


from data_generator import DataGenerator
import prepare_data as pp
import config_dnn1 as conf1
from sklearn import preprocessing

from keras.callbacks import TensorBoard
import tensorflow as tf
import sys


########################################################################################################################################
# FUNCTIONS
########################################################################################################################################
def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


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


def compute_scaler(data_type):
    """Compute and write out scaler of data.
    """

    # Load data.
    t1 = time.time()
    hdf5_path = os.path.join("data_train", "dnn1_packed_features", data_type, "data.h5")
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        x = np.array(x)  # (n_segs, n_concat, n_freq)

    # Compute scaler.
    (n_segs, n_concat, n_freq) = x.shape
    x2d = x.reshape((n_segs * n_concat, n_freq))
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(x2d)
    # print(scaler.mean_)
    # print(scaler.scale_)

    # Write out scaler.
    out_path = os.path.join("data_train", "dnn1_packed_features", data_type, "scaler.p")
    pp.create_folder(os.path.dirname(out_path))
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
########################################################################################################################
########################################################################################################################
# PREPARE DATASET
########################################################################################################################
########################################################################################################################
# def train():


# Load data.
def prepare_database():
    t1 = time.time()
    train_x_files = np.asarray(pp.load_data(conf1.train_folder, "mix"))
    train_y_files = np.asarray(pp.load_data(conf1.train_folder, "clean"))

    dnn2_train_x_files = np.asarray(pp.load_data(conf1.dnn2_train_folder, "mix"))
    dnn2_train_y_files = np.asarray(pp.load_data(conf1.dnn2_train_folder, "clean"))

    test_x_files = np.append(train_x_files, dnn2_train_x_files, axis=0)
    test_y_files = np.append(train_y_files, dnn2_train_y_files, axis=0)

    tr_x_spec = []
    for na in train_x_files:
        (a, _) = pp.read_audio(na)
        b = pp.calc_sp(a, mode='complex')
        tr_x_spec.append(b)

    tr_y_spec = []
    for na in train_y_files:
        (c, _) = pp.read_audio(na)
        t = pp.calc_sp(c, mode='magnitude')
        tr_y_spec.append(t)




    te_x_spec = []
    for na in test_x_files:
        (a, _) = pp.read_audio(na)
        b = pp.calc_sp(a, mode='complex')
        te_x_spec.append(b)

    te_y_spec = []
    for na in test_y_files:
            (c, _) = pp.read_audio(na)
            t = pp.calc_sp(c, mode='magnitude')
            te_y_spec.append(t)

    print(len(tr_x_spec))
    print(len(tr_y_spec))
    num_tr = pp.pack_features(tr_x_spec, tr_y_spec, 'train')
    num_te = pp.pack_features(te_x_spec, te_y_spec, 'test')

    # compute_scaler("train")
    # compute_scaler("test")
    return num_tr, num_te






########################################################################################################################
########################################################################################################################
# TRAIN
########################################################################################################################
########################################################################################################################

# Load data.
t1 = time.time()

# COMMENT IF DATABASE ALREADY CREATED
if conf1.use_previous_files:
    num_tr, num_te = prepare_database()
else:
    num_tr = len([f for f in os.listdir(os.path.join("data_train", "dnn1_packed_features", "train"))
                                       if f.endswith('.h5')])
    num_te = len([f for f in os.listdir(os.path.join("data_train", "dnn2_packed_features", "train"))
                                       if f.endswith('.h5')])

tr_x = []
tr_y = []
te_x = []
te_y = []

for i in range(num_tr):
    tr_x_t, tr_y_t = pp.load_hdf5(os.path.join("data_train", "dnn1_packed_features", "train", "tf_data_%s.h5" % str(i+1)))
    tr_x.append(tr_x_t)
    tr_y.append(tr_y_t)

for i in range(num_te):
    te_x_t, te_y_t = pp.load_hdf5(os.path.join("data_train", "dnn1_packed_features", "test", "tf_data_%s.h5" % str(i+1)))
    te_x.append(te_x_t)
    te_y.append(te_y_t)


tr_x = np.concatenate(tr_x, axis=0)
tr_y = np.concatenate(tr_y, axis=0)
te_x = np.concatenate(te_x, axis=0)
te_y = np.concatenate(te_y, axis=0)

print(tr_x.shape, tr_y.shape)
print(te_x.shape, te_y.shape)
print("Load data time: %s s" % (time.time() - t1,))

# conf.batch_size = 512
print("%d iterations / epoch" % int(tr_x.shape[0] / conf1.batch_size))

# Scale data.
if True:
    t1 = time.time()
    scaler_path = os.path.join("data_train", "dnn1_packed_features", "train", "scaler.p")
    scaler = pickle.load(open(scaler_path, 'rb'))
    tr_x = pp.scale_on_3d(tr_x, scaler)
    tr_y = pp.scale_on_2d(tr_y, scaler)
    te_x = pp.scale_on_3d(te_x, scaler)
    te_y = pp.scale_on_2d(te_y, scaler)
    print("Scale data time: %s s" % (time.time() - t1,))

# Debug plot.
# if False:
#     plt.matshow(tr_x[0: 1000, 0, :].T, origin='lower', aspect='auto', cmap='jet')
#     plt.show()
#     pause

# Build model

# Build model
(_, n_concat, n_freq) = tr_x.shape
n_hid = 1024

model = Sequential()
model.add(Flatten(input_shape=(n_concat, n_freq)))
model.add(Dense(n_hid, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(n_hid, activation='sigmoid'))
model.add(Dropout(0.2))
# model.add(Dense(n_hid, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(n_freq, activation='linear'))
model.summary()

model.compile(loss='mean_absolute_error',
              optimizer=SGD(lr=conf1.lr, momentum=0.0, decay=0.0015))

callback = TensorBoard(log_dir='logs')
callback.set_model(model)
train_names = ['train_loss', 'train_mae']
val_names = ['val_loss', 'val_mae']

# Data generator.
tr_gen = DataGenerator(batch_size=conf1.batch_size, gtype='train')
eval_te_gen = DataGenerator(batch_size=conf1.batch_size, gtype='test', te_max_iter=100)
eval_tr_gen = DataGenerator(batch_size=conf1.batch_size, gtype='test', te_max_iter=100)

# Directories for saving models and training stats
model_dir = os.path.join("data_train", "dnn1_packed_features", "models")
pp.create_folder(model_dir)

stats_dir = os.path.join("data_train", "dnn1_packed_features", "training_stats")
pp.create_folder(stats_dir)

# Print loss before training.
iter = 0
tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
te_loss = eval(model, eval_te_gen, te_x, te_y)
print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))

# Save out training stats.
stat_dict = {'iter': iter,
             'tr_loss': tr_loss,
             'te_loss': te_loss, }
stat_path = os.path.join(stats_dir, "%diters.p" % iter)
pickle.dump(stat_dict, open(stat_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)




# Train.
t1 = time.time()
for (batch_x, batch_y) in tr_gen.generate(xs=[tr_x], ys=[tr_y]):
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
        stat_path = os.path.join(stats_dir, "%diters.p" % iter)
        pickle.dump(stat_dict, open(stat_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    # Save model.
    if iter % conf1.iterations == 0:
        model_path = os.path.join(model_dir, "md_%diters.h5" % iter)
        model.save(model_path)
        print("Saved model to %s" % model_path)

    if iter == (conf1.iterations+1):
        break


# model.fit(tr_x, tr_y, epochs= 5)
#
# test_loss, test_acc = model.evaluate(tr_x, tr_y)



print("Training time: %s s" % (time.time() - t1,))


