import math
import random
import numpy as np
import prepare_data as pp
import os
from dab import dab_run
import metrics as m
import sys

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


def create_room(source_file, noise_file, dist):

    (clean, fs) = pp.read_audio(source_file)
    (noise, _) = pp.read_audio(noise_file)

    for file in os.listdir(os.path.join("data_eval", "dnn1_in")):
        file_path = os.path.join("data_eval", "dnn1_in", file)
        os.remove(file_path)

    for n in range(len(dist)):

        mixed, noise_new, clean_new, s2nr = set_microphone_at_distance(clean, noise, fs, dist[n])

        # s2nr = 1 / (1 + (1 / float(snr)))

        mixed_name = "mix_%s_%s" % (str(dist[n]), os.path.basename(source_file))
        clean_name = "clean_%s_%s" % (str(dist[n]), os.path.basename(source_file))


        mixed_path = os.path.join('data_eval/dnn1_in', mixed_name)
        clean_path = os.path.join('data_eval/dnn1_in', clean_name)

        pp.write_audio(mixed_path, mixed, fs)
        #pp.write_audio(clean_path, clean_new, fs)

go_on = True




while go_on == True:
    work_dir = 'data_eval'
    speech_dir = os.path.join('data_eval', 'test_speech')


    input_string = input('Insert microphones distances: ')
    dist = input_string.split()
    dist = list(map(int, dist))
    d_mean = math.ceil(sum(dist)/len(dist))

    snr = [5.62/d for d in dist]

    source_files = [f for f in os.listdir(speech_dir)
                    if f.endswith(".wav")]

    # executing for all source speeches in speech folder
    for f in source_files:
        create_room(os.path.join(speech_dir, f), 'noise/babble.wav', dist)
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

    room_dims = [30, 30]
    iterations = 10000
    steps = 1000
    n_mics = len(dist)

    # distrib = m.monte_carlo(room_dims, n_mics, iterations, steps, 'rect')


    go_on = query_yes_no('Simulate new room?')
