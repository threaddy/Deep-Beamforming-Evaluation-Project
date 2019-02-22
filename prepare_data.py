import numpy as np
import os
import h5py
import time
import soundfile
import tables

from scipy import signal

import config_dnn1 as conf


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs


def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)


def load_data(data_directory, beginning):  # load the dataset, return a list of all found file
    files = [f for f in sorted(os.listdir(data_directory))
             if f.startswith(beginning)]
    allfiles = []
    for f in files:
        allfiles.append(os.path.join(data_directory, f))
    return allfiles


def load_with_labels(data_directory, beginning):  # load the dataset, return a list of all found file and folder labels
    directories = [d for d in sorted(os.listdir(data_directory))
                   if os.path.isdir(os.path.join(data_directory, d))]
    allfiles = []
    labels = []
    for d in directories:
        speaker_directory = os.path.join(data_directory, d)
        utterance_names = [os.path.join(speaker_directory, f)
                           for f in sorted(os.listdir(speaker_directory))
                           if f.startswith(beginning)]

        label_name = [float(d) for f in os.listdir(speaker_directory)
                      if f.startswith(beginning)]
        allfiles.append(utterance_names)
        labels.append(label_name)
    return allfiles, labels


def calc_sp(audio, mode):
    """Calculate spectrogram.

    Args:
      audio: 1darray.
      mode: string, 'magnitude' | 'complex'

    Returns:
      spectrogram: 2darray, (n_time, n_freq).
    """
    ham_win = np.hamming(conf.n_window)
    [f, t, x] = signal.spectral.spectrogram(
        audio,
        window=ham_win,
        nperseg=conf.n_window,
        noverlap=conf.n_overlap,
        detrend=False,
        return_onesided=True,
        mode=mode)
    x = x.T
    if mode == 'magnitude':
        x = x.astype(np.float32)
    elif mode == 'complex':
        x = x.astype(np.complex64)
    else:
        raise Exception("Incorrect mode!")
    return x


def pack_features(in_x, in_y, data_type):
    """Load all features, apply log and conver to 3D tensor, write out to .h5 file.
      data_type: str, 'train' | 'test'.
      n_concat: int, number of frames to be concatenated.
      n_hop: int, hop frames.
    """

    x_all = []  # (n_segs, n_concat, n_freq)
    y_all = []  # (n_segs, n_freq)

    cnt = 0
    t1 = time.time()

    # Load all features.

    if len(in_x) != len(in_y):
        raise Exception("Error! Training input and output with different size")

    out_path = os.path.join(conf.packed_feature_dir, data_type, "data.h5")
    create_folder(os.path.dirname(out_path))
    i = 0

    for na in range(len(in_x)):

        in_x[na] = np.abs(in_x[na])

        # Pad start and finish of the spectrogram with boarder values.
        n_pad = (conf.n_concat - 1) / 2
        in_x[na] = pad_with_border(in_x[na], n_pad)
        in_y[na] = pad_with_border(in_y[na], n_pad)

        # Cut input spectrogram to 3D segments with n_concat.
        mixed_x_3d = mat_2d_to_3d(in_x[na], agg_num=conf.n_concat, hop=conf.n_hop)
        mixed_x_3d= log_sp(mixed_x_3d).astype(np.float32)
        x_all.append(mixed_x_3d)

        # Cut target spectrogram and take the center frame of each 3D segment.
        speech_x_3d = mat_2d_to_3d(in_y[na], agg_num=conf.n_concat, hop=conf.n_hop)
        y = speech_x_3d[:, int((conf.n_concat - 1) / 2), :]
        y = log_sp(y).astype(np.float32)
        y_all.append(y)


        x_all_new = np.concatenate(x_all, axis=0)  # (n_segs, n_concat, n_freq)
        y_all_new = np.concatenate(y_all, axis=0)  # (n_segs, n_freq)


        # Print.
        if cnt % 100 == 0:
            print(cnt)

        # if cnt == 3: break
        cnt += 1

        if (na+1) % 250 == 0:
            i += 1
            # Write out data to .h5 file.
            out_path = os.path.join(conf.packed_feature_dir, data_type, "data_%s.h5" % str(i))
            create_folder(os.path.dirname(out_path))

            with h5py.File(out_path, 'w') as hf:
                hf.create_dataset('x', data=x_all_new)
                hf.create_dataset('y', data=y_all_new)

            y_all = []
            x_all = []

            print("Write out to %s" % out_path)


    print("Pack features finished! %s s" % (time.time() - t1,))

    return i


def log_sp(x):
    return np.log(x + 1e-08)


def mat_2d_to_3d(x, agg_num, hop):
    """Segment 2D array to 3D segments.
    """
    # Pad to at least one block.
    len_x, n_in = x.shape
    if (len_x < agg_num):
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))

    # Segment 2d to 3d.
    len_x = len(x)
    i1 = 0
    x3d = []
    while (i1 + agg_num <= len_x):
        x3d.append(x[i1: i1 + agg_num])
        i1 += hop
    return np.array(x3d)


def pad_with_border(x, n_pad):
    """Pad the begin and finish of spectrogram with border frame value.
    """
    x_pad_list = [x[0:1]] * int(n_pad) + [x] + [x[-1:]] * int(n_pad)
    return np.concatenate(x_pad_list, axis=0)


def np_mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def load_hdf5(hdf5_path):
    """Load hdf5 data.
    """
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        y = hf.get('y')
        x = np.array(x)  # (n_segs, n_concat, n_freq)
        y = np.array(y)  # (n_segs, n_freq)
    return x, y


def scale_on_2d(x2d, scaler):
    """Scale 2D array data.
    """
    return scaler.transform(x2d)


def scale_on_3d(x3d, scaler):
    """Scale 3D array data.
    """
    (n_segs, n_concat, n_freq) = x3d.shape
    x2d = x3d.reshape((n_segs * n_concat, n_freq))
    x2d = scaler.transform(x2d)
    x3d = x2d.reshape((n_segs, n_concat, n_freq))
    return x3d


def inverse_scale_on_2d(x2d, scaler):
    """Inverse scale 2D array data.
    """
    return x2d * scaler.scale_[None, :] + scaler.mean_[None, :]
