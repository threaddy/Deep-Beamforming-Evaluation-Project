import numpy as np
import os
import prepare_data as pp
import dnn1_eval as dnn1
import config_dnn1 as conf1
import dnn2_eval as dnn2
from spectrogram_to_wave import recover_wav_complex
import matplotlib.pyplot as plt

#

output_file_folder = "data_eval/dab"

def visualize(mixed_x, pred):
    fig, axs = plt.subplots(2, 1, sharex=False)
    axs[0].matshow(mixed_x.T, origin='lower', aspect='auto', cmap='jet')
    axs[1].matshow(pred.T, origin='lower', aspect='auto', cmap='jet')
    axs[1].set_title("Enhanced speech log spectrogram")
    for j1 in range(2):
        axs[j1].xaxis.tick_bottom()
    plt.tight_layout()
    plt.show()


def channel_weights(input_s2nrs):
    b = []
    qx = max(input_s2nrs)
    gamma = 0.1  # tunable threshold

    for qi in input_s2nrs:
        thresh = (float(qi) * (1 - float(qx))) / (float(qx) * (1 - float(qi)))

        if thresh > gamma:
            bi = 1
        else:
            bi = 0
        b.append(bi)

    ch_weights = np.multiply(input_s2nrs, np.array(b))

    return ch_weights


def mvdr(mix_audios, enh_audios, reweighted_audios):

    x_max = 0
    y_max = 0


    # get maximum t_f shapes
    for f in enh_audios:
        if f.shape[0] > x_max:
            x_max = f.shape[0]
        if f.shape[1] > y_max:
            y_max = f.shape[1]

    # add zeros until all masks have the save dimensions
    enh_pad = []
    pad_lenght_x = []
    pad_lenght_y = []

    for c in enh_audios:
        pad_x = x_max - c.shape[0]
        pad_y = y_max - c.shape[1]
        pad_lenght_x.append(pad_x)
        pad_lenght_y.append(pad_y)
        t = np.pad(c, [(0, pad_x), (0, pad_y)], mode='constant', constant_values=0)
        enh_pad.append(t)
    enh_pad = np.asarray(enh_pad)

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




    # calculate enhanced mask and noise mask
    epsilon = np.ones((x_max, y_max))
    eta = np.ones((x_max, y_max))
    noise_pad = []

    for c, d in zip(rw_pad, mix_pad):
        alpha = np.divide(c, d, out=np.zeros_like(c), where=d != 0)
        epsilon = np.multiply(epsilon, alpha)   # enh mask
        beta = np.ones(eta.shape) - alpha
        eta = np.multiply(eta, beta)            # noise mask


        noise_pad.append(d - c)

    noise_pad = np.asarray(noise_pad)



    phinn = np.ones((channel_num, channel_num, rw_pad.shape[2]), dtype=complex)
    for a in range(channel_num):
        for b in range(channel_num):
            temp = np.multiply(noise_pad[a], noise_pad[b].conj())
            t2 = np.average(temp, axis=0)
            phinn[a, b] = t2

    phixx = np.ones((channel_num, channel_num, rw_pad.shape[2]), dtype=complex)
    for a in range(channel_num):
        for b in range(channel_num):
            temp = np.multiply(rw_pad[a], rw_pad[b].conj())
            phixx[a, b] = np.average(temp, axis=0)



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



    # estimated covariance matrix for speech

    w_opt = np.asarray(w_opt)
    print(w_opt)

    # w_opt = np.ones((channel_num, y_max))
    final_audios = np.zeros(enh_pad.shape, dtype=complex)
    for i in range(channel_num):
        for j in range(x_max):
            final_audios[i][j] = np.multiply(w_opt[:, i], rw_pad[i][j, :])

    final = np.sum(final_audios, axis=0)



    final_cut = final[0:(final.shape[0] - max(pad_lenght_x)), 0:(final.shape[1] - max(pad_lenght_y))]

    visualize(np.abs(rw_pad[0]), np.abs(mix_pad[0]))
    # visualize(np.abs(enh_pad[0]), np.abs(rw_pad[0]))
    # visualize(np.abs(enh_pad[0]), np.abs(final_cut))
    #
    visualize(np.imag(enh_pad[0]), np.imag(final_cut))
    visualize(np.abs(rw_pad[0]), np.abs(final_cut))

    return np.asarray(final_cut)


########################################################################################################################
# DAB
########################################################################################################################

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
snr = np.array([5.62, 2.81, 1.875, 1.406])

s2nrs = snr
for i in range(len(snr)):
    s2nrs[i] = 1/(1+1/snr[i])


# calculate channel weights
new_weights = channel_weights(s2nrs)

print(new_weights)
channel_num = len(dnn1_outputs)

# multiply enhanced audio for the corresponding weight
ch_rw_outputs = []
# for i in range(len(dnn1_outputs)):
#     if new_weights[i] != 0:
#         ch_rw_outputs.append(new_weights[i] * dnn1_outputs[i])
#     else:
#         dnn1_inputs = np.delete(dnn1_inputs, i)
#         dnn1_inputs = np.delete(dnn1_inputs, i)


for i, p in zip(dnn1_outputs, new_weights):
    ch_rw_outputs.append(p * i)


(init, _) = pp.read_audio('data_eval/sa1.wav')
init_sp = pp.calc_sp(init, mode='complex')



# execute mvdr
final = mvdr(dnn1_inputs, dnn1_outputs, ch_rw_outputs)

visualize(np.abs(init_sp), np.abs(final))

# Recover and save enhanced wav
pp.create_folder(output_file_folder)

final_sp = np.exp(np.negative(np.abs(final)))

s = recover_wav_complex(final, conf1.n_overlap, np.hamming)
s *= np.sqrt((np.hamming(conf1.n_window) ** 2).sum())  # Scaler for compensate the amplitude
s_sp = pp.calc_sp(s, mode='complex')




audio_path = os.path.join(output_file_folder, "dab_out.wav")
pp.write_audio(audio_path, s, conf1.sample_rate)

(output, _) = pp.read_audio(os.path.join(output_file_folder, "dab_out.wav"))

output_sp = pp.calc_sp(output, 'magnitude')



print('done DAB')


########################################################################################################################
# DB
########################################################################################################################

