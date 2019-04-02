import numpy as np
import os
import prepare_data as pp
import dnn1_eval as dnn1
import dnn1_config as conf1
import dnn2_eval as dnn2
from spectrogram_to_wave import recover_wav_complex
import matplotlib.pyplot as plt


visualize_on = False


def visualize(mat1, mat2, title1='title', title2='title'):
    if visualize_on:
        fig, axs = plt.subplots(2, 1, sharex=False)
        axs[0].matshow(mat1.T, origin='lower', aspect='auto', cmap='jet')
        axs[1].matshow(mat2.T, origin='lower', aspect='auto', cmap='jet')
        axs[0].set_title(title1)
        axs[1].set_title(title2)

        for j1 in range(2):
            axs[j1].xaxis.tick_bottom()
        plt.tight_layout()
        plt.show()

def dnn1_colors(input):
    scaler_path = os.path.join(conf1.packed_feature_dir, "test", "scaler.p")
    scaler = dnn1.pickle.load(open(scaler_path, 'rb'))

    # n_pad = (conf1.n_concat - 1) / 2
    # enh_pad[0] = pp.pad_with_border(enh_pad[0], n_pad)
    prova = pp.log_sp(input)

    prova = pp.scale_on_2d(np.abs(prova), scaler)
    prova = pp.inverse_scale_on_2d(prova, scaler)
    return -prova


def channel_weights(input_s2nrs):
    b = []
    qx = max(input_s2nrs)
    gamma = 0.5  # tunable threshold

    for qi in input_s2nrs:
        thresh = (float(qi) * (1 - float(qx))) / (float(qx) * (1 - float(qi)))

        if thresh > gamma:
            bi = 1
        else:
            bi = 0
        b.append(bi)

    ch_weights = np.multiply(input_s2nrs, np.array(b))

    return ch_weights


def mvdr(mix_audios, reweighted_audios):

    channel_num = len(mix_audios)

    # get maximum t_f shapes
    x_max = 0
    y_max = 0
    for f in reweighted_audios:
        if f.shape[0] > x_max:
            x_max = f.shape[0]
        if f.shape[1] > y_max:
            y_max = f.shape[1]

    # add zeros until all masks have the save dimensions
    pad_lenght_x = []
    pad_lenght_y = []

    rw_pad = []
    for c in reweighted_audios:
        pad_x = x_max - c.shape[0]
        pad_y = y_max - c.shape[1]
        pad_lenght_x.append(pad_x)
        pad_lenght_y.append(pad_y)
        t = np.pad(c, [(0, x_max - c.shape[0]), (0, y_max - c.shape[1])], mode='constant', constant_values=0)
        rw_pad.append(t)
    rw_pad = np.asarray(rw_pad)

    mix_pad = []
    for c in mix_audios:
        t = np.pad(c, [(0, x_max - c.shape[0]), (0, y_max - c.shape[1])], mode='constant', constant_values=0)
        mix_pad.append(t)
    mix_pad = np.asarray(mix_pad)

    # calculate noise
    noise_pad = []
    for c, d in zip(rw_pad, mix_pad):
        noise_pad.append(d - c)
    noise_pad = np.asarray(noise_pad)

    # calculate noise covariance matrix
    phinn = np.ones((channel_num, channel_num, rw_pad.shape[2]), dtype=complex)
    for a in range(channel_num):
        for b in range(channel_num):
            temp = np.multiply(noise_pad[a], noise_pad[b].conj())
            t2 = np.average(temp, axis=0)
            phinn[a, b] = t2

    # calculate re-weigheted audio covariance matrix
    phixx = np.ones((channel_num, channel_num, rw_pad.shape[2]), dtype=complex)
    for a in range(channel_num):
        for b in range(channel_num):
            temp = np.multiply(rw_pad[a], rw_pad[b].conj())
            phixx[a, b] = np.average(temp, axis=0)

    # calculate new MVDR weights
    w_opt = []
    for f in range(y_max):
        phinn_f = phinn[:, :, f]
        phixx_f = phixx[:, :, f]
        inv_phinn_f = np.linalg.inv(phinn_f)
        v, V = np.linalg.eig(phixx_f.T)
        c_phixx_f = V[:, 0].T
        w_num_f = np.dot(inv_phinn_f, c_phixx_f.T)
        w_den_f = np.dot((c_phixx_f.conj()).transpose(), w_num_f)
        w_opt_f = np.divide(w_num_f, w_den_f)
        w_opt.append(w_opt_f)

    w_opt = np.asarray(w_opt)

    # apply weights to each channel
    final_audios = np.zeros(rw_pad.shape, dtype=complex)
    for i in range(channel_num):
        for j in range(x_max):
            final_audios[i][j] = np.multiply(w_opt[:, i], rw_pad[i][j, :])
    # combine channels
    final = np.sum(final_audios, axis=0)

    # cut off padded values
    final_cut = final[0:(final.shape[0] - max(pad_lenght_x)), 0:(final.shape[1] - max(pad_lenght_y))]



    visualize(np.abs(rw_pad[0]), np.abs(mix_pad[0]), "reweighted amplitude", "mixed amplitude")
    # visualize(np.abs(rw_pad[0]), np.abs(rw_pad[0]))
    # visualize(np.abs(rw_pad[0]), np.abs(final_cut))
    #
    visualize(np.imag(rw_pad[0]), np.imag(final_cut), "enh imaginary", "final imaginary")
    visualize(dnn1_colors(np.abs(rw_pad[0])), dnn1_colors(np.abs(final_cut)), "reweighted amplitude", "final amplitude")

    return np.asarray(final_cut)




########################################################################################################################
# DAB
########################################################################################################################
def dab_run(snr_list, file_name="dab_out", mode='dab'):

    output_file_folder = os.path.join("data_eval", mode)

    # removing previous enhancements
    for file in os.listdir(os.path.join("data_eval", "dnn1_out")):
        file_path = os.path.join("data_eval", "dnn1_out", file)
        os.remove(file_path)

    dnn1_inputs, dnn1_outputs = dnn1.predict_folder(os.path.join("data_eval", "dnn1_in"), os.path.join("data_eval", "dnn1_out"))

    names = [f for f in sorted(os.listdir(os.path.join("data_eval", "dnn1_out"))) if f.startswith("enh")]
    dnn1_outputs = []
    for (cnt, na) in enumerate(names):
        # Load feature.
        file_path = os.path.join("data_eval", "dnn1_out", na)
        (a, _) = pp.read_audio(file_path)
        enh_complex = pp.calc_sp(a, 'complex')
        dnn1_outputs.append(enh_complex)


    # s2nrs = dnn2.predict("data_eval/dnn1_in", "data_eval/dnn1_out")

    # snr = np.array([5.62, 1.405, 0.703, 0.281])
    # snr = np.array([5.62, 2.81, 1.875, 1.406])
    s2nrs = snr_list * 1
    for i in range(len(snr_list)):
        s2nrs[i] = 1/(1+1/snr_list[i])

    ch_rw_outputs = []
    # calculate channel weights
    if mode == 'dab':
        new_weights = channel_weights(s2nrs)
        print(["{0:0.3f}".format(i) for i in new_weights])
        # multiply enhanced audio for the corresponding weight
        for i, p in zip(dnn1_outputs, new_weights):
            ch_rw_outputs.append(p * i)


    # cancel reweighting if db mode
    if mode == 'db':
        new_weights = s2nrs
        print(["{0:0.3f}".format(i) for i in new_weights])
        ch_rw_outputs = dnn1_outputs

    # execute mvdr
    final = mvdr(dnn1_inputs, ch_rw_outputs)

    (init, _) = pp.read_audio(os.path.join('data_eval', 'test_speech', file_name))
    init_sp = pp.calc_sp(init, mode='complex')

    visualize(dnn1_colors(np.abs(init_sp)), dnn1_colors(np.abs(final)), "source amplitude", "final amplitude")

    # Recover and save enhanced wav
    pp.create_folder(output_file_folder)
    s = recover_wav_complex(final, conf1.n_overlap, np.hamming)
    s *= np.sqrt((np.hamming(conf1.n_window) ** 2).sum())  # Scaler for compensate the amplitude
    audio_path = os.path.join(output_file_folder, file_name)
    pp.write_audio(audio_path, s, conf1.sample_rate)

    print('%s done' % mode)


########################################################################################################################
# DB
########################################################################################################################

