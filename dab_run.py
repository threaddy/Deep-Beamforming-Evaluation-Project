import numpy as np
import os
import prepare_data as pp
import dnn1_eval as dnn1
import config_dnn1 as conf1
import dnn2_eval as dnn2


output_file_folder = "data_eval/dab"

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


def mvdr(enh_audios, reweighted_audios):

    x_max = 0
    y_max = 0

    # get maximum t_f shapes
    for f in enh_audios:
        if f.shape[0] > x_max:
            x_max = f.shape[0]
        if f.shape[1] > y_max:
            y_max = f.shape[1]

    # add zeros until all masks have the save dimensions
    enh_masks_pad = []
    for c in enh_audios:
        t = np.pad(c, [(0, x_max - c.shape[0]), (0, y_max - c.shape[1])], mode='linear_ramp', end_values=0)
        enh_masks_pad.append(t)

    rw_masks_pad = []
    for c in reweighted_audios:
        t = np.pad(c, [(0, x_max - c.shape[0]), (0, y_max - c.shape[1])], mode='linear_ramp', end_values=0)
        rw_masks_pad.append(t)
    rw_masks_pad = np.asarray(rw_masks_pad)

    # calculate weights for noise covariance matrix
    eta = np.ones((x_max, y_max))
    for c in enh_masks_pad:
        t = np.ones(eta.shape) - c
        eta = np.multiply(eta, t)
    # print(eta)

    # calculate weights for estimated speech covariance matrix
    epsilon = np.ones((x_max, y_max))
    for c in enh_masks_pad:
        epsilon = np.multiply(epsilon, c)

    # print(epsilon)



    # estimated covariance matrix for speech

    temp = np.ones([rw_masks_pad.shape[0], rw_masks_pad.shape[0], rw_masks_pad.shape[1], rw_masks_pad.shape[2]])
    temp2 = np.ones([rw_masks_pad.shape[0], rw_masks_pad.shape[0], rw_masks_pad.shape[2]])
    phixx = temp2
    for i in range(channel_num):
        for j in range(channel_num):
            temp[i, j] = np.multiply(rw_masks_pad[i], rw_masks_pad[j].conj())
            temp2[i, j] = np.sum(np.multiply(temp[i, j], epsilon), axis=0)
            phixx[i, j] = np.divide(temp2[i, j], np.sum(epsilon, axis=0))


    # estimated covariance matrix for noise
    temp = np.ones([rw_masks_pad.shape[0], rw_masks_pad.shape[0], rw_masks_pad.shape[1], rw_masks_pad.shape[2]])
    temp2 = np.ones([rw_masks_pad.shape[0], rw_masks_pad.shape[0], rw_masks_pad.shape[2]])
    phinn = temp2
    for i in range(channel_num):
        for j in range(channel_num):
            temp[i, j] = np.multiply(rw_masks_pad[i], rw_masks_pad[j].conj())
            temp2[i, j] = np.sum(np.multiply(temp[i, j], eta), axis=0)
            phinn[i, j] = np.divide(temp2[i, j], np.sum(eta, axis=0))



    # print(w_opt)
    w_opt = []

    for freq in range(y_max):
        phinn_f = phinn[:, :, freq]
        inv_phinn_f = np.linalg.inv(phinn_f)
        v, V = np.linalg.eig(phinn_f.T)
        c_phixx_f = V[:, 0].T
        w_num_f = np.dot(inv_phinn_f, c_phixx_f.T)
        w_den_f = np.dot((c_phixx_f.conj()).transpose(), w_num_f)
        w_opt_f = np.divide(w_num_f, w_den_f)
        w_opt.append(w_opt_f)

    w_opt = np.asarray(w_opt)
    print(w_opt)




    # w_opt = np.ones((channel_num, y_max))
    final_audios = enh_masks_pad
    for i in range(channel_num):
        for j in range(x_max):
            final_audios[i][j] = np.multiply(w_opt[:, i], enh_masks_pad[i][j, :])



    return np.asarray(enh_masks_pad), np.asarray(final_audios)


dnn1_inputs, dnn1_outputs = dnn1.predict("data_eval/dnn1_in", "data_eval/dnn1_out")
# dnn1_outputs = []
# names = [f for f in sorted(os.listdir("data_eval/dnn1_out")) if f.startswith("enh")]
# for (cnt, na) in enumerate(names):
#     # Load feature.
#     file_path = os.path.join("data_eval/dnn1_out", na)
#     (a, _) = pp.read_audio(file_path)
#     dnn1_output = pp.calc_sp(a, 'complex')
#     dnn1_outputs.append(dnn1_output)


s2nrs = dnn2.predict("data_eval/dnn1_in", "data_eval/dnn1_out")
# calculate channel weights
new_weights = channel_weights(s2nrs)
channel_num = len(dnn1_outputs)

# multiply enhanced audio for the corresponding weight
ch_rw_outputs = []
for i, p in zip(dnn1_outputs, new_weights):
    ch_rw_outputs.append(p * i)

# execute mvdr
mvdr_inputs, mvdr_outputs = mvdr(dnn1_outputs, ch_rw_outputs)

# Recover and save enhanced wav
pp.create_folder(output_file_folder)
# dab_outputs_sp = np.exp(dab_outputs)
cnt = 0
for i in range(channel_num):
    dab_outputs_sp = np.exp(mvdr_outputs[i])
    cnt += 1
    s = dnn1.recover_wav(dab_outputs_sp, mvdr_inputs[i], conf1.n_overlap, np.hamming)
    s *= np.sqrt((np.hamming(conf1.n_window) ** 2).sum())  # Scaler for compensate the amplitude
    audio_path = os.path.join(output_file_folder, "dab_%s.wav" % cnt)
    pp.write_audio(audio_path, s, conf1.sample_rate)



