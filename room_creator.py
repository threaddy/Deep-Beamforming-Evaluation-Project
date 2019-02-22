from __future__ import print_function
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import prepare_data as pp
import os
import random
import math
import sympy

working_dir = "test_speech/"




def DS_generate(source_audio, room_dimensions, out_folder, name):

    # Create the shoebox
    shoebox = pra.ShoeBox(
        room_dimensions,
        absorption=1.0,
        fs=fs,
        max_order=15,
    )
    mic_distance = random.randint(1, 20)                   # distance from source to microphone
    source_position = np.array([random.uniform(0, room_dimensions[0]), random.uniform(0, room_dimensions[1])])

    # random way: guess microphone position until it's in the room: very long time for small rooms
    # mic_in_room = False
    # while mic_in_room == False:
    #     theta = random.uniform(0, 2 * math.pi)
    #     mic_position = source_position - mic_distance * np.array([math.cos(theta), math.sin(theta)])
    #     print(mic_position)
    #     if (0 <= mic_position[0] <= room_dimensions[0]) and (0 <= mic_position[1] <= room_dimensions[1]):
    #         mic_in_room = True
    #

    p1, p2, p3, p4 = map(sympy.Point, [(0, 0), (room_dimensions[0], 0), (0, room_dimensions[1]), (room_dimensions[0], room_dimensions[1])])
    rect = sympy.Polygon(p1, p2, p3, p4)
    circ = sympy.Circle(sympy.Point(room_dimensions[0]/2, room_dimensions[1]/2), 5)
    inters = rect.intersection(circ)
    print()









    # # source and mic locations
    # shoebox.add_source(source_position, signal=source_audio)
    # shoebox.add_microphone_array(
    #     pra.MicrophoneArray(
    #         np.array([mic_position]).T,
    #         shoebox.fs)
    # )
    #
    # shoebox.simulate()
    # shoebox.mic_array.to_wav(os.path.join(out_folder + '_DS', 'mix_' + name), norm=True, bitdepth=np.int16)



def DB_generate(source_audio, room_dimensions, out_folder, name):
    mic_distance = random.randint(1, 20)                   # mean distance from source to microphones
    source_position = np.array([random.uniform(0, room_dimensions[0]), random.uniform(0, room_dimensions[1])])
    theta = random.uniform(0, 2 * math.pi)

    # random way: guess array center until it's in the room: very long time for small rooms
    mic_in_room = False
    while mic_in_room == False:
        theta = random.uniform(0, 2 * math.pi)
        mic_center = source_position - mic_distance * np.array([math.cos(theta), math.sin(theta)])
        print(mic_center)
        if (0 <= mic_center[0] <= room_dimensions[0]) and (0 <= mic_center[1] <= room_dimensions[1]):
            mic_in_room = True

    # number of microphones
    M = 4
    # counterclockwise rotation of array:
    phi = 0
    # distance between microphones
    d = 0.4

    mics = pra.beamforming.linear_2D_array(mic_center, M, phi, d)

    # create room
    shoebox = pra.ShoeBox(
        room_dimensions,
        absorption=wall_absorption,
        fs=fs,
        max_order=15,
    )

    # source and mic locations
    shoebox.add_source(source_position, signal=source_audio)
    shoebox.add_microphone_array(
        pra.MicrophoneArray(
            mics,
            shoebox.fs)
    )

    shoebox.simulate()
    shoebox.mic_array.to_wav(os.path.join(out_folder + '_DB', 'mix_' + name), norm=True, bitdepth=np.int16)

def DAB_generate(source_audio, room_dimensions, out_folder, name):

    source_position = np.array([random.uniform(0, room_dimensions[0]), random.uniform(0, room_dimensions[1])])




    shoebox = pra.ShoeBox(
        room_dimensions,
        absorption=wall_absorption,
        fs=fs,
        max_order=15,
    )

    # source and mic locations
    shoebox.add_source(source_position, signal=source_audio)
    shoebox.add_microphone_array(
        pra.MicrophoneArray(
            mics,
            shoebox.fs)
    )

    shoebox.simulate()
    shoebox.mic_array.to_wav(os.path.join(out_folder + '_DB', 'mix_' + name), norm=True, bitdepth=np.int16)


###################################################################################################
# RUN ON ALL FILE
###################################################################################################

# room parameters
room_dim = [20, 20]
wall_absorption = 0.2


sources = [f for f in sorted(os.listdir(working_dir)) if f.endswith("wav")]


for f in sources:
    file_path = os.path.join(working_dir, f)
    fs, audio_anechoic = wavfile.read(file_path)

    out_folder = working_dir + os.path.splitext(f)[0]
    pp.create_folder(out_folder + '_DS')
    pp.create_folder(out_folder + '_DB')
    pp.create_folder(out_folder + '_DAB')

    DS_generate(audio_anechoic, room_dim, out_folder, f)
    # DB_generate(audio_anechoic, room_dim, out_folder, f)
    # DAB_generate(audio_anechoic, room_dim, out_folder, f)
