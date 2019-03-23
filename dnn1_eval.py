
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from spectrogram_to_wave import recover_wav
from spectrogram_to_wave import real_to_complex
from keras.models import load_model

import prepare_data as pp
import dnn1_config as conf1





# scale = True
visualize_plot = False

def visualize(mixed_x, pred):
    fig, axs = plt.subplots(2, 1, sharex=False)
    axs[0].matshow(mixed_x.T, origin='lower', aspect='auto', cmap='jet')
    axs[1].matshow(pred.T, origin='lower', aspect='auto', cmap='jet')
    axs[1].set_title("mixed speech log spectrogram")
    axs[1].set_title("Enhanced speech log spectrogram")
    for j1 in range(2):
        axs[j1].xaxis.tick_bottom()
    plt.tight_layout()
    plt.show()



def predict_folder(input_file_folder: object, output_file_folder: object) -> object:
    # Load model.
    data_type = "test"
    model_path = os.path.join(conf1.model_dir, "md_%diters.h5" % conf1.iterations)
    model = load_model(model_path)

    # Load scaler.
    # if scale:
    scaler_path = os.path.join(conf1.packed_feature_dir, data_type, "scaler.p")
    scaler = pickle.load(open(scaler_path, 'rb'))

    # Load test data.
    # names = os.listdir(input_file_folder)

    names = [f for f in sorted(os.listdir(input_file_folder)) if f.startswith("mix")]

    mixed_all = []
    pred_all = []
    for (cnt, na) in enumerate(names):
        # Load feature.
        file_path = os.path.join(input_file_folder, na)
        (a, _) = pp.read_audio(file_path)
        mixed_complex = pp.calc_sp(a, 'complex')


        mixed_x = np.abs(mixed_complex)

        # Process data.
        n_pad = (conf1.n_concat - 1) / 2
        mixed_x = pp.pad_with_border(mixed_x, n_pad)
        mixed_x = pp.log_sp(mixed_x)
        # speech_x = dnn1_train.log_sp(speech_x)

        # Scale data.
        # if scale:
        mixed_x = pp.scale_on_2d(mixed_x, scaler)

        # Cut input spectrogram to 3D segments with n_concat.
        mixed_x_3d = pp.mat_2d_to_3d(mixed_x, agg_num=conf1.n_concat, hop=1)


        # Predict.
        pred = model.predict(mixed_x_3d)
        print(cnt, na)

        # Inverse scale.
        #if scale:
        mixed_x = pp.inverse_scale_on_2d(mixed_x, scaler)
        # speech_x = dnn1_train.inverse_scale_on_2d(speech_x, scaler)
        pred = pp.inverse_scale_on_2d(pred, scaler)

        # Debug plot.
        if visualize_plot:
            visualize(mixed_x, pred)

        mixed_all.append(mixed_complex)
        pred_all.append(real_to_complex(pred, mixed_complex))


        # Recover enhanced wav.
        pred_sp = np.exp(pred)
        s = recover_wav(pred_sp, mixed_complex, conf1.n_overlap, np.hamming)
        s *= np.sqrt((np.hamming(conf1.n_window) ** 2).sum())  # Scaler for compensate the amplitude
        # change after spectrogram and IFFT.

        # Write out enhanced wav.

        pp.create_folder(output_file_folder)
        audio_path = os.path.join(output_file_folder, "enh_%s" % na)
        pp.write_audio(audio_path, s, conf1.sample_rate)

    return mixed_all, pred_all



def predict_file(file_path, model, scaler):

    (a, _) = pp.read_audio(file_path)
    mixed_complex = pp.calc_sp(a, 'complex')

    mixed_x = np.abs(mixed_complex)

    # Process data.
    n_pad = (conf1.n_concat - 1) / 2
    mixed_x = pp.pad_with_border(mixed_x, n_pad)
    mixed_x = pp.log_sp(mixed_x)
    # speech_x = dnn1_train.log_sp(speech_x)


    # Scale data.
    # if scale:
    mixed_x = pp.scale_on_2d(mixed_x, scaler)
    # speech_x = pp.scale_on_2d(speech_x, scaler)

    # Cut input spectrogram to 3D segments with n_concat.
    mixed_x_3d = pp.mat_2d_to_3d(mixed_x, agg_num=conf1.n_concat, hop=1)

    # Predict.
    pred = model.predict(mixed_x_3d)

    if visualize_plot:
        visualize(mixed_x, pred)
    # Inverse scale.
    # if scale:
    mixed_x = pp.inverse_scale_on_2d(mixed_x, scaler)
    # speech_x = dnn1_train.inverse_scale_on_2d(speech_x, scaler)
    pred = pp.inverse_scale_on_2d(pred, scaler)


    # Debug plot.

    # Recover enhanced wav.
    pred_sp = np.exp(pred)
    s = recover_wav(pred_sp, mixed_complex, conf1.n_overlap, np.hamming)
    s *= np.sqrt((np.hamming(conf1.n_window) ** 2).sum())  # Scaler for compensate the amplitude
    # change after spectrogram and IFFT.

    # Write out enhanced wav.

    # audio_path = os.path.dirname(file_path)
    # pp.write_audio(audio_path, s, conf1.sample_rate)

    return mixed_complex, pred, s



def load_dnn():
    # Load model.
    data_type = "test"
    model_path = os.path.join(conf1.model_dir, "md_%diters.h5" % conf1.iterations)
    model = load_model(model_path)

    # Load scaler.
    #if scale:
    scaler_path = os.path.join(conf1.packed_feature_dir, data_type, "scaler.p")
    scaler = pickle.load(open(scaler_path, 'rb'))

    return model, scaler


def plot_training_stat(stats_dir, bgn_iter, fin_iter, interval_iter):
    """Plot training and testing loss.

    Args:
      stats_dir: str, path of training stats.
      bgn_iter: int, plot from bgn_iter
      fin_iter: int, plot finish at fin_iter
      interval_iter: int, interval of files.
    """

    tr_losses, te_losses, iters = [], [], []

    # Load stats.

    for iter in range(bgn_iter, fin_iter, interval_iter):
        stats_path = os.path.join(stats_dir, "%diters.p" % iter)
        dict = pickle.load(open(stats_path, 'rb'))
        tr_losses.append(dict['tr_loss'])
        te_losses.append(dict['te_loss'])
        iters.append(dict['iter'])

    # Plot
    line_tr, = plt.plot(tr_losses, c='b', label="Train")
    line_te, = plt.plot(te_losses, c='r', label="Test")
    plt.axis([0, len(iters), 0, max(tr_losses)])
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(handles=[line_tr, line_te])
    plt.xticks(np.arange(len(iters)), iters)
    plt.show()


# dnn1_inputs, dnn1_outputs = predict_folder("data_eval/dnn1_in", "data_eval/dnn1_out")
# model, scaler = load_dnn()
# predict_file('data_eval/sa1.wav', model, scaler)