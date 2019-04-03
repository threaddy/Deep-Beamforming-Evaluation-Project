from __future__ import print_function
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import prepare_data as pp
import os
import random
import math
import matplotlib.pyplot as plt
from dab import dab_run
import metrics as m
import sys
import csv
import time


working_dir = "data_eval/"
noise_path = 'noise/babble.wav'
room_dimensions = [30, 30]
wall_absorption = 0.6
fs = 16000


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")



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

    meter_snr = 10 ** (15 / 20)  # chosen snr at one meter distance (15 dB)
    clean_energy = 0
    noise_energy = 0

    # shift = random.randint(0, abs(len(clean_data) - len(noise_data)))  # calculate random shift for noise in the possible range, to avoid "overflows"
    # l = len(clean_data)
    # cut_noise_data = noise_data[shift:(l + shift)]

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




def DB_generate(source_audio, out_folder, name):

    #source_audio = pra.normalize(source_audio, bits=16)

    mic_distance = random.randint(1, 20)                   # mean distance from source to microphones
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

    print(mic_distance)
    print(["{0:0.3f}".format(i) for i in distances])

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

    shoebox.add_source(source_position, signal=source_audio)
    shoebox.add_microphone_array(mics)
    shoebox.compute_rir()
    shoebox.simulate()

    # ADDING NOISE

    for n in range(M+1):
        signal = np.asarray(shoebox.mic_array.signals[n, :], dtype=float)
        signal = pra.utilities.normalize(signal, bits=16)

        mixed_signal = add_noise(source_audio, signal)

        mixed_signal = np.array(mixed_signal, dtype=np.int16)

        mixed_file = os.path.join(out_folder, 'mix%d_%s' % (n, name))
        pp.write_audio(mixed_file, mixed_signal, fs)




def DAB_generate(source_audio, out_folder, name, dist='None'):

    shoebox = pra.ShoeBox(
        room_dimensions,
        absorption=wall_absorption,
        fs=fs,
        max_order=15,
    )

    # number of microphones

    source_position = np.array([random.uniform(0, room_dimensions[0]), random.uniform(0, room_dimensions[1])])

    if dist == 'None':
        M = 4
        distances = np.random.randint(1, 20, M)
    else:
        distances = dist

    M = len(distances)
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

    shoebox.add_source(source_position, signal=source_audio)
    shoebox.add_microphone_array(mics)

    shoebox.compute_rir()
    shoebox.simulate()


    # ADDING NOISE AND SAVING

    for n in range(M):
        signal = np.asarray(shoebox.mic_array.signals[n, :], dtype=float)
        signal = pra.utilities.normalize(signal, bits=16)
        mixed_signal = add_noise(source_audio, signal)
        mixed_signal = np.array(mixed_signal, dtype=np.int16)


        mixed_file = os.path.join(out_folder, 'mix%d_%s' % (n, name))

        pp.write_audio(mixed_file, mixed_signal, fs)





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




sources = [f for f in sorted(os.listdir(os.path.join(working_dir, 'test_speech'))) if f.endswith("wav")]

    # out_folder = working_dir + os.path.splitext(f)[0]+'_sim'
    #
    # DS_folder = os.path.join(out_folder, 'DS')
    # DB_folder = os.path.join(out_folder, 'DB')
    # DAB_folder = os.path.join(out_folder, 'DAB')
    #
    #
    # pp.create_folder(DS_folder)
    # pp.create_folder(DB_folder)
    # pp.create_folder(DAB_folder)

    # DS_generate(audio_anechoic, DS_folder, f)
    # DB_generate(audio_anechoic, DB_folder, f)
    # DAB_generate(audio_anechoic, DAB_folder, f)

go_on = True

index_file = open('index_file_%s.csv' % str(int(time.time())), mode='w')
index_writer = csv.writer(index_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
index_writer.writerow(['', 'STOI', 'std.', 'PESQ', 'std.', 'SDR', 'std.'])
while go_on == True:
    work_dir = 'data_eval'
    speech_dir = os.path.join('data_eval', 'test_speech')
    eval_dir = os.path.join('data_eval', 'dnn1_in')

    input_string = input('Insert microphones distances: ')
    dist = input_string.split()
    dist = list(map(float, dist))
    d_mean = math.ceil(sum(dist)/len(dist))

    snr = [5.62/d for d in dist]

    source_files = [f for f in os.listdir(speech_dir)
                    if f.endswith(".wav")]

    # executing for all source speeches in speech folder
    for f in source_files:
        for file in os.listdir(eval_dir):
            file_path = os.path.join("data_eval", "dnn1_in", file)
            os.remove(file_path)

        (fs, audio_anechoic) = wavfile.read(os.path.join(speech_dir, f))
        audio_anechoic = np.asarray(audio_anechoic, dtype=np.int16)
        DAB_generate(audio_anechoic, eval_dir, f, dist)
        dab_run(snr, f, mode='dab')
        dab_run(snr, f, mode='db')



    print("--------NOiSY----------------------------------\n -------------")
    # calculating speech-per-speech indexes
    m.calculate_pesq_couple(speech_dir, 'data_eval/dnn1_in')
    avg_pesqs_N, std_pesqs_N = m.get_pesq_stats()
    avg_stoi_N, std_stoi_N = m.calc_stoi_couple(speech_dir, 'data_eval/dnn1_in')
    avg_sdr_N, std_sdr_N = m.calc_sdr_couple(speech_dir, 'data_eval/dnn1_in')


    print("--------DS-------------------------------------\n -------------")
    # calculating speech-per-speech indexes
    m.calculate_pesq_couple(speech_dir, 'data_eval/dnn1_out')
    avg_pesqs_DS, std_pesqs_DS = m.get_pesq_stats()
    avg_stoi_DS, std_stoi_DS = m.calc_stoi_couple(speech_dir, 'data_eval/dnn1_out')
    avg_sdr_DS, std_sdr_DS = m.calc_sdr_couple(speech_dir, 'data_eval/dnn1_out')

    print("--------DB-------------------------------------\n -------------")
    m.calculate_pesq_couple(speech_dir, 'data_eval/db')
    avg_pesqs_DB, std_pesqs_DB = m.get_pesq_stats()
    avg_stoi_DB, std_stoi_DB = m.calc_stoi_couple(speech_dir, 'data_eval/db')
    avg_sdr_DB, std_sdr_DB = m.calc_sdr_couple(speech_dir, 'data_eval/db')

    print("--------DAB------------------------------------\n -------------")
    # calculating speech-per-speech indexes
    m.calculate_pesq_couple(speech_dir, 'data_eval/dab')
    avg_pesqs_DAB, std_pesqs_DAB = m.get_pesq_stats()
    avg_stoi_DAB, std_stoi_DAB = m.calc_stoi_couple(speech_dir, 'data_eval/dab')
    avg_sdr_DAB, std_sdr_DAB = m.calc_sdr_couple(speech_dir, 'data_eval/dab')

    print('-----------------------------------------------------------------------------------------------------------')
    print('INDEX:\t STOI\t PESQ\t SDR\t ---------------------------------------------------------------------------')
    print('Noisy \t %f(%f) \t %f(%f) \t %f(%f)' % (avg_stoi_N, std_stoi_N,
                                                    avg_pesqs_N, std_pesqs_N,
                                                    avg_sdr_N, std_sdr_N))

    print('DS   \t %f(%f) \t %f(%f) \t %f(%f)' % (avg_stoi_DS, std_stoi_DS,
                                                   avg_pesqs_DS, std_pesqs_DS,
                                                   avg_sdr_DS, std_sdr_DS))

    print('DB   \t %f(%f) \t %f(%f) \t %f(%f)' % (avg_stoi_DB, std_stoi_DB,
                                                   avg_pesqs_DB, std_pesqs_DB,
                                                   avg_sdr_DB, std_sdr_DB))

    print('DAB \t %f(%f) \t %f(%f) \t %f(%f)' % (avg_stoi_DAB, std_stoi_DAB,
                                                   avg_pesqs_DAB, std_pesqs_DAB,
                                                   avg_sdr_DAB, std_sdr_DAB))

    index_writer.writerow([(''.join(str(e) for e in dist))])
    index_writer.writerow(['Noisy', avg_stoi_N, std_stoi_N, avg_pesqs_N, std_pesqs_N, avg_sdr_N, std_sdr_N])
    index_writer.writerow(['DS', avg_stoi_DS, std_stoi_DS, avg_pesqs_DS, std_pesqs_DS, avg_sdr_DS, std_sdr_DS])
    index_writer.writerow(['DB', avg_stoi_DB, std_stoi_DB, avg_pesqs_DB, std_pesqs_DB, avg_sdr_DB, std_sdr_DB])
    index_writer.writerow(['DAB', avg_stoi_DAB, std_stoi_DAB, avg_pesqs_DAB, std_pesqs_DAB, avg_sdr_DAB, std_sdr_DAB])

    #
    # room_dims = [30, 30]
    # iterations = 10000
    # steps = 1000
    # n_mics = len(dist)

    # distrib = m.monte_carlo(room_dims, n_mics, iterations, steps, 'rect')


    go_on = query_yes_no('Simulate new room?')
