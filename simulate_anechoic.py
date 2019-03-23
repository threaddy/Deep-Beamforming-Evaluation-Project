import math
import random
import numpy as np
import prepare_data as pp
import os


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


def create_room(source_file, noise_file):
    dist = [2, 4, 8, 20]
    (clean, fs) = pp.read_audio(source_file)
    (noise, _) = pp.read_audio(noise_file)

    for n in range(len(dist)):

        mixed, noise_new, clean_new, s2nr = set_microphone_at_distance(clean, noise, fs, dist[n])

        # s2nr = 1 / (1 + (1 / float(snr)))

        mixed_name = "mix_%s_%s" % (str(dist[n]), os.path.basename(source_file))
        clean_name = "clean_%s_%s" % (str(dist[n]), os.path.basename(source_file))


        mixed_path = os.path.join('data_eval/dnn1_in', mixed_name)
        clean_path = os.path.join('data_eval/dnn1_in', clean_name)

        pp.write_audio(mixed_path, mixed, fs)
        #pp.write_audio(clean_path, clean_new, fs)


create_room('data_eval/sa1.wav', 'noise/babble.wav')
