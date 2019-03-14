from __future__ import print_function
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import prepare_data as pp
import os
import random
import math
import matplotlib.pyplot as plt



import dnn1_eval as dnn1
import dnn2_eval as dnn2


working_dir = "data_eval/"
noise_path = 'noise/babble.wav'
room_dimensions = [30, 30]
wall_absorption = 1.0
fs = 16000


def new_mvdr(mics, source, delay=0.03, epsilon=5e-3):

    '''
    Compute the time-domain filters of the minimum variance distortionless
    response beamformer.
    '''
    source_l = [source]

    H = pra.build_rir_matrix(mics.R, source_l, mics.Lg, mics.fs,
                             epsilon=epsilon, unit_damping=True)
    L = H.shape[1] // 2

    # the constraint vector
    kappa = int(delay*mics.fs)
    h = H[:, kappa]

    # We first assume the sample are uncorrelated
    R_xx = np.dot(H[:, :L], H[:, :L].T)
    K_nq = np.dot(H[:, L:], H[:, L:].T)

    # Compute the TD filters
    C = pra.la.cho_factor(R_xx + K_nq, check_finite=False)
    g_val = pra.la.cho_solve(C, h)

    g_val /= np.inner(h, g_val)
    mics.filters = g_val.reshape((mics.M, mics.Lg))

    # compute and return SNR
    num = np.inner(g_val.T, np.dot(R_xx, g_val))
    denom = np.inner(np.dot(g_val.T, K_nq), g_val)

    return num/denom

def channel_weights(input_s2nrs):
    b = []
    qx = max(input_s2nrs)
    gamma = 0.2  # tunable threshold

    for qi in input_s2nrs:
        thresh = (float(qi) * (1 - float(qx))) / (float(qx) * (1 - float(qi)))

        if thresh > gamma:
            bi = 1
        else:
            bi = 0
        b.append(bi)

    ch_weights = np.multiply(input_s2nrs, np.array(b))

    return ch_weights



def guess_microphone(source_position, mic_distance):

    mic_in_room = False
    while mic_in_room == False:
        theta = random.uniform(0, 2 * math.pi)
        mic_position = source_position - mic_distance * np.array([math.cos(theta), math.sin(theta)])
        print(mic_position)
        if (0 <= mic_position[0] <= room_dimensions[0]) and (0 <= mic_position[1] <= room_dimensions[1]):
            mic_in_room = True
    return mic_position


def add_noise(clean_data, att_data):
    (_, noise_data) = wavfile.read(noise_path)
    noise_data = np.asarray(noise_data, dtype=np.float64)
    noise_data = pra.utilities.normalize(noise_data, bits=16)

    # clean_data = np.asarray(clean_data, dtype=np.float64)
    # clean_data = pra.utilities.normalize(clean_data, bits=16)

    meter_snr = 10 ** (10 / 20)  # chosen snr at one meter distance (15 dB)
    clean_energy = 0
    noise_energy = 0

    shift = random.randint(0, abs(len(clean_data) - len(noise_data)))  # calculate random shift for noise in the possible range, to avoid "overflows"
    l = len(clean_data)
    cut_noise_data = noise_data[shift:(l + shift)]

    shift = random.randint(0, abs(len(att_data) - len(
        noise_data)))                     # calculate random shift for noise in the possible range, to avoid "overflows"
    l = len(att_data)
    new_noise_data = noise_data[shift:(l + shift)]


    for t in clean_data:
        clean_energy = clean_energy + abs(t)

    for t in new_noise_data:
        noise_energy = noise_energy + abs(t)


    first_snr = clean_energy / noise_energy
    snr_ratio = meter_snr / first_snr

    new_noise_data = new_noise_data / snr_ratio

    mixed_data = new_noise_data + att_data

    return mixed_data


def rephase(enh_signals, m_distances, framerate):

    sound_speed = 343

    for (s, d) in zip(enh_signals, m_distances):
        frame_delay = int(math.ceil(framerate * d / sound_speed))  # compute number of frame to delay
        delay_silence = np.zeros(frame_delay)
        s = np.append(delay_silence, s)

    return enh_signals








def DS_generate(source_audio, out_folder, name):

    # Create the shoebox
    shoebox = pra.ShoeBox(
        room_dimensions,
        absorption= wall_absorption,
        fs=fs,
        max_order=15,
    )
    mic_distance = random.randint(1, 20)                   # distance from source to microphone
    source_position = np.array([random.uniform(0, room_dimensions[0]), random.uniform(0, room_dimensions[1])])

    # random way: guess microphone position until it's in the room: very long time for small rooms
    mic_in_room = False
    while mic_in_room == False:
        theta = random.uniform(0, 2 * math.pi)
        mic_position = source_position - mic_distance * np.array([math.cos(theta), math.sin(theta)])
        print(mic_position)
        if (0 <= mic_position[0] <= room_dimensions[0]) and (0 <= mic_position[1] <= room_dimensions[1]):
            mic_in_room = True


    # source and mic locations
    shoebox.add_source(source_position, signal=source_audio)
    shoebox.add_microphone_array(
        pra.MicrophoneArray(
            np.array([mic_position]).T,
            shoebox.fs)
    )

    shoebox.simulate()


    signal = shoebox.mic_array.signals[0, :]
    mixed_signal = add_noise(source_audio, signal)
    mixed_signal = pra.utilities.normalize(mixed_signal, bits=16)
    mixed_signal = np.array(mixed_signal, dtype=np.int16)

    pp.write_audio(os.path.join(out_folder, 'mix_%s' % name), mixed_signal, fs)

    (model1, scaler1) = dnn1.load_dnn()



    # shoebox.mic_array.to_wav(os.path.join(out_folder + '_DS', 'mix_' + name), norm=True, bitdepth=np.int16)



def DB_generate(source_audio, out_folder, name):

    #source_audio = pra.normalize(source_audio, bits=16)

    mic_distance = 5 # random.randint(1, 20)                   # mean distance from source to microphones
    source_position = np.array([random.uniform(0, room_dimensions[0]), random.uniform(0, room_dimensions[1])])

    # random way: guess array center until it's in the room: very long time for small rooms
    mic_in_room = False
    while mic_in_room == False:
        theta = random.uniform(0, 2 * math.pi)
        mic_center = source_position - mic_distance * np.array([math.cos(theta), math.sin(theta)])
        print(mic_center)
        if (0 <= mic_center[0] <= room_dimensions[0]) and (0 <= mic_center[1] <= room_dimensions[1]):
            mic_in_room = True


    # number of lateral microphones
    M = 4
    # counterclockwise rotation of array:
    phi = 0
    # distance between microphones
    d = 0.4

    mic_pos = pra.beamforming.linear_2D_array(mic_center, M, phi, d)
    mic_pos = np.concatenate((mic_pos, np.array(mic_center, ndmin=2).T), axis=1)

    distances = []
    for m in range(M):
        d = math.sqrt((source_position[0] - mic_pos[0, m])**2 + (source_position[1] - mic_pos[1, m])**2)
        distances.append(d)


    # create room
    shoebox = pra.ShoeBox(
        room_dimensions,
        absorption=wall_absorption,
        fs=fs,
        max_order=15,
    )

    # shoebox.mic_array.to_wav(os.path.join(out_folder + '_DB', 'mix_' + name), norm=True, bitdepth=np.int16)

    Lg_t = 0.100  # filter size in seconds
    Lg = np.ceil(Lg_t * fs)  # in samples
    fft_len = 512


    mics = pra.Beamformer(mic_pos, shoebox.fs, N=fft_len, Lg=Lg)

    # (_, noise_data) = wavfile.read(noise_path)
    # noise_data = np.zeros((len(source_audio)))
    # interferer = np.array([random.uniform(0, room_dimensions[0]), random.uniform(0, room_dimensions[1])])

    shoebox.add_source(source_position, signal=source_audio)
    # shoebox.add_source(interferer, delay=0., signal=noise_data)
    shoebox.add_microphone_array(mics)
    # mics.rake_delay_and_sum_weights(shoebox.sources[0])
    # new_mvdr(mics, shoebox.sources[0])

    shoebox.compute_rir()
    shoebox.simulate()

    # ADDING NOISE AND ENHANCING USING DNN1

    (model1, scaler1) = dnn1.load_dnn()
    enh_all = []
    for n in range(M+1):
        signal = np.asarray(shoebox.mic_array.signals[n, :], dtype=float)
        signal = pra.utilities.normalize(signal, bits=16)

        mixed_signal = add_noise(source_audio, signal)

        mixed_signal = np.array(mixed_signal, dtype=np.int16)

        mixed_file = os.path.join(out_folder, 'mix%d_%s' % (n, name))
        pp.write_audio(mixed_file, mixed_signal, fs)

        (_, _, enh_signal) = dnn1.predict_file(mixed_file, model1, scaler1)
        enh_signal = pra.utilities.normalize(enh_signal, bits=16)
        enh_signal = np.array(enh_signal, dtype=np.int16)

        enh_file = os.path.join(out_folder, 'enh%d_%s' % (n, name))
        pp.write_audio(enh_file, enh_signal, fs)


        enh_signal = np.array(enh_signal, dtype=np.float64)

        enh_all.append(enh_signal)



    enh_all = rephase(enh_all, distances, fs)

    # update microphone signals with enhanced signals
    new_mics = pra.Beamformer(mic_pos, shoebox.fs)
    new_mics.record(np.asarray(enh_all), shoebox.fs)
    #shoebox.mic_array = new_mics

    #re-calculating beamforming
    # new_mics.rake_delay_and_sum_weights(shoebox.sources[0])
    new_mvdr(new_mics, shoebox.sources[0])

    # # re-simulating room
    # shoebox.compute_rir()
    # shoebox.simulate()

    #calculating beamformed signal
    beam_signal = new_mics.process(FD=False)

    # beam_signal = pra.normalize(pra.highpass(beam_signal, fs))

    # fig, ax = shoebox.plot(freq=[500, 1000, 2000, 4000], img_order=0)
    # ax.legend(['500', '1000', '2000', '4000'])
    # fig.set_size_inches(20, 8)

    # sp1 = pp.calc_sp(source_audio, 'magnitude')
    # sp2 = pp.calc_sp(signal_das, 'magnitude')

    plt.figure(1)  # the first figure
    plt.subplot(211)  # the first subplot in the first figure
    plt.plot(source_audio)
    plt.subplot(212)  # the second subplot in the first figure
    plt.plot(beam_signal)
    plt.show()

    final_file = os.path.join(out_folder, 'final_%s' % name)
    # signal_das = pra.normalize(signal_das, bits=16)
    wavfile.write(final_file, fs, beam_signal)



def DAB_generate(source_audio, out_folder, name):

    shoebox = pra.ShoeBox(
        room_dimensions,
        absorption=wall_absorption,
        fs=fs,
        max_order=15,
    )

    # number of microphones
    M = 4

    source_position = np.array([random.uniform(0, room_dimensions[0]), random.uniform(0, room_dimensions[1])])

    distances = np.random.randint(1, 20, M)

    mic_pos = []
    for m in range(M):
        mic_distance = distances[m]
        mic_m = guess_microphone(source_position, mic_distance)         # random way: guess microphone position until it's in the room: very long time for small rooms
        mic_pos.append(mic_m)


    out_mic_file = os.path.join(out_folder, 'log_%s.txt' % name)

    if os.path.exists(out_mic_file):
        os.remove(out_mic_file)
    f1 = open(out_mic_file, 'w')
    for l in range(M):
        f1.write("%s, %f\n" % (str(mic_pos[l]), distances[l]))

    Lg_t = 0.100  # filter size in seconds
    Lg = np.ceil(Lg_t * fs)  # in samples
    fft_len = 512
    mics = pra.Beamformer(np.asarray(mic_pos).T, shoebox.fs, N=fft_len, Lg=Lg)

    #(_, noise_data) = wavfile.read(noise_path)
    # interferer = np.array([random.uniform(0, room_dimensions[0]), random.uniform(0, room_dimensions[1])])

    shoebox.add_source(source_position, signal=source_audio)
    # shoebox.add_source(interferer, delay=0., signal=noise_data[:len(source_audio)])
    shoebox.add_microphone_array(mics)
    #mics.rake_delay_and_sum_weights(shoebox.sources[0])
    # new_mvdr(mics, shoebox.sources[0])

    shoebox.compute_rir()
    shoebox.simulate()


    # ADDING NOISE AND ENHANCING USING DNN1
    (model1, scaler1) = dnn1.load_dnn()
    enh_all = []
    for n in range(M):
        signal = np.asarray(shoebox.mic_array.signals[n, :], dtype=float)
        signal = pra.utilities.normalize(signal, bits=16)

        mixed_signal = add_noise(source_audio, signal)

        mixed_signal = np.array(mixed_signal, dtype=np.int16)

        mixed_file = os.path.join(out_folder, 'mix%d_%s' % (n, name))
        pp.write_audio(mixed_file, mixed_signal, fs)

        (_, _, enh_signal) = dnn1.predict_file(mixed_file, model1, scaler1)
        enh_signal = pra.utilities.normalize(enh_signal, bits=16)
        enh_signal = np.array(enh_signal, dtype=np.int16)

        enh_file = os.path.join(out_folder, 'enh%d_%s' % (n, name))
        pp.write_audio(enh_file, enh_signal, fs)

        enh_signal = np.array(enh_signal, dtype=np.float64)

        enh_all.append(enh_signal)

    # ESTIMATING S2NR USING DNN2
    s2nrs = dnn2.predict(out_folder, out_folder)


    # calculate channel weights
    new_weights = channel_weights(s2nrs)
    print(new_weights)

    # multiply enhanced audio for the corresponding weight
    ch_rw_outputs = []
    for i, p in zip(enh_all, new_weights):
        ch_rw_outputs.append(p * i)

    # update microphone signals with enhanced signals
    new_mics = pra.Beamformer(np.asarray(mic_pos).T, shoebox.fs)
    new_mics.record(np.asarray(ch_rw_outputs), shoebox.fs)
    shoebox.mic_array = new_mics

    # re-calculating beamforming
    #new_mics.rake_delay_and_sum_weights(shoebox.sources[0])
    new_mvdr(new_mics, shoebox.sources[0])

    # re-simulating room
    shoebox.compute_rir()
    shoebox.simulate()

    # calculating beamformed signal
    signal_das = new_mics.process(FD=False)

    # fig, ax = shoebox.plot(freq=[500, 1000, 2000, 4000], img_order=0)
    # ax.legend(['500', '1000', '2000', '4000'])
    # fig.set_size_inches(20, 8)


    sp1 = pp.calc_sp(source_audio, 'magnitude')
    sp2 = pp.calc_sp(signal_das, 'magnitude')

    plt.figure(1)  # the first figure
    plt.subplot(211)  # the first subplot in the first figure
    plt.plot(source_audio)
    plt.subplot(212)  # the second subplot in the first figure
    plt.plot(signal_das)
    plt.show()



    final_file = os.path.join(out_folder, 'final_%s' % name)
    wavfile.write(final_file, fs, signal_das)





def visualize(mixed_x, pred):
    fig, axs = plt.subplots(3, 1, sharex=False)
    axs[0].matshow(mixed_x.T, origin='lower', aspect='auto', cmap='jet')
    # axs[1].matshow(speech_x.T, origin='lower', aspect='auto', cmap='jet')
    axs[2].matshow(pred.T, origin='lower', aspect='auto', cmap='jet')
    # axs[0].set_title("%ddb mixture log spectrogram" % int(te_snr))
    # axs[1].set_title("Clean speech log spectrogram")
    axs[2].set_title("Enhanced speech log spectrogram")
    for j1 in range(3):
        axs[j1].xaxis.tick_bottom()
    plt.tight_layout()
    plt.show()


###################################################################################################
# RUN ON ALL FILE
###################################################################################################

# room parameters




sources = [f for f in sorted(os.listdir(working_dir)) if f.endswith("wav")]


for f in sources:
    file_path = os.path.join(working_dir, f)
    # audio_anechoic, fs = pp.read_audio(file_path)

    (fs, audio_anechoic) = wavfile.read(file_path)
    audio_anechoic = np.asarray(audio_anechoic, dtype=np.int16)

    out_folder = working_dir + os.path.splitext(f)[0]

    DS_folder = (out_folder + '/_DS')
    DB_folder = (out_folder + '/_DB')
    DAB_folder = (out_folder + '/_DAB')


    pp.create_folder(DS_folder)
    pp.create_folder(DB_folder)
    pp.create_folder(DAB_folder)

    #DS_generate(audio_anechoic, DS_folder, f)
    #DB_generate(audio_anechoic, DB_folder, f)
    DAB_generate(audio_anechoic, DAB_folder, f)
