"""
Summary:  Calculate PESQ and overal stats of enhanced speech.
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified: -
"""

import os
import csv
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pystoi
import prepare_data as pp

training_stats1 = "data_train/dnn1_packed_features/training_stats/"
training_stats2 = "data_train/dnn1_packed_features/training_stats/"
input_dir = "data_eval/dnn1_in"
output_dir = "data_eval/dab"

debug = False


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


def calculate_pesq(in_speech_dir, out_speech_dir):
    """Calculate PESQ of all enhaced speech.

    Args:
      out_speech_dir: str, path of workspace.
      in_speech_dir: str, path of clean speech.
    """

    # Remove already existed file.
    os.system('rm _pesq_itu_results.txt')
    os.system('rm _pesq_results.txt')

    # Calculate PESQ of all enhaced speech.
    in_names = [os.path.join(in_speech_dir, na)
                for na in sorted(os.listdir(in_speech_dir))
                if na.endswith(".wav")]
    out_names = [os.path.join(out_speech_dir, na)
                 for na in sorted(os.listdir(out_speech_dir))
                 if na.endswith(".wav")]

    for (na_in, na_out) in zip(out_names, in_names):
        # Call executable PESQ tool.
        cmd = ' '.join(["./pesq", na_in, na_out, "+16000"])
        os.system(cmd)


def get_pesq_stats():
    """Calculate stats of PESQ.
    """
    pesq_path = "_pesq_results.txt"
    with open(pesq_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)

    pesq_dict = {}
    for i1 in range(1, len(lis) - 1):
        li = lis[i1]
        na = li[0]
        pesq = float(li[1])
        noise_type = na.split('.')[1]
        if noise_type not in pesq_dict.keys():
            pesq_dict[noise_type] = [pesq]
        else:
            pesq_dict[noise_type].append(pesq)

    avg_list, std_list = [], []
    f = "{0:<16} {1:<16}"
    print(f.format("\nNoise", "PESQ"))
    print("---------------------------------")
    for noise_type in pesq_dict.keys():
        pesqs = pesq_dict[noise_type]
        avg_pesq = np.mean(pesqs)
        std_pesq = np.std(pesqs)
        avg_list.append(avg_pesq)
        std_list.append(std_pesq)
        print(f.format(noise_type, "%.2f +- %.2f" % (avg_pesq, std_pesq)))
    print(f.format("Avg.", "%.2f +- %.2f" % (np.mean(avg_list), np.mean(std_list))), "\n")


def calc_stoi(in_speech_dir, out_speech_dir):
    in_names = [os.path.join(in_speech_dir, na)
                for na in sorted(os.listdir(in_speech_dir))
                if na.endswith(".wav")]
    out_names = [os.path.join(out_speech_dir, na)
                 for na in sorted(os.listdir(out_speech_dir))
                 if na.endswith(".wav")]
    stoi_list = []
    print("---------------------------------")
    print("\t", "STOI", "\n")
    for f, g in zip(in_names, out_names):
        (x, fs1) = pp.read_audio(f)
        (y, fs2) = pp.read_audio(g)

        if fs1 != fs2:
            print("Error: output and input files have different sampling rate")
        res = pystoi.stoi(x, y, fs1)
        stoi_list.append(res)
        # print(g, "\t",  res)

    avg_stoi = np.mean(stoi_list)
    std_stoi = np.std(stoi_list)
    print("AVG STOI", avg_stoi)
    print("ST DEV STOI", std_stoi)
    print("---------------------------------")
    return avg_stoi, std_stoi


def calc_sdr(in_speech_dir, out_speech_dir):
    in_names = [os.path.join(in_speech_dir, na)
                for na in sorted(os.listdir(in_speech_dir))
                if na.endswith(".wav")]
    out_names = [os.path.join(out_speech_dir, na)
                 for na in sorted(os.listdir(out_speech_dir))
                 if na.endswith(".wav")]
    sdr_list = []
    for f, g in zip(in_names, out_names):
        (x, fs1) = pp.read_audio(f)
        (y, fs2) = pp.read_audio(g)

        if fs1 != fs2:
            print("Error: output and input files have different sampling rate")

        top = 1 / x.shape[0] * (sum(x ** 2))
        bottom = 1 / y.shape[0] * (sum(y ** 2))

        sdr_list.append(10 * np.log10(top / bottom))

    avg_sdr = np.mean(sdr_list)
    std_sdr = np.std(sdr_list)
    print("AVG SDR", avg_sdr)
    print("ST DEV SDR", std_sdr)
    print("---------------------------------")
    return avg_sdr, std_sdr


if debug == True:
    plot_training_stat(training_stats1, 0, 100, 10)
    calculate_pesq(input_dir, output_dir)
    get_pesq_stats()
    stoi_res = calc_stoi(input_dir, output_dir)
    sdr_res = calc_sdr(input_dir, output_dir)
