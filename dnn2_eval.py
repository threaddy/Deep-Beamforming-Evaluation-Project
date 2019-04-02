import numpy as np
import os
from keras.models import load_model
import prepare_data as pp
import dnn2_config as conf2



def predict(mix_folder, enh_folder):
    # Load model.

    s2nrs = []
    model_path = os.path.join(conf2.model_dir, "md_%depochs.h5" % conf2.epochs)
    model = load_model(model_path)

    # Load test data.
    # names = os.listdir(input_file_folder)

    names_mix = [f for f in sorted(os.listdir(mix_folder)) if f.startswith("mix")]
    names_enh = [f for f in sorted(os.listdir(enh_folder)) if f.startswith("enh")]

    if len(names_mix) != len(names_enh):
        print("files or labels are not loaded properly")

    names_pair = (np.append([names_mix], [names_enh], axis=0)).transpose()
    print("List of audio, 1 for each channel:")

    for (cnt, na) in enumerate(names_pair):
        # Load feature.
        noisy_file_path = os.path.join(mix_folder, na[0])
        (a_noisy, _) = pp.read_audio(noisy_file_path)

        enh_file_path = os.path.join(enh_folder, na[1])
        (a_enh, _) = pp.read_audio(enh_file_path)

        dnn2_input = np.array([a_noisy.mean(), a_enh.mean()])
        dnn2_input = dnn2_input.reshape([1, 2])

        print(dnn2_input)

        pred = model.predict(dnn2_input)
        print(cnt, na)

        # Inverse scale.


        # Debug plot.
        s2nrs.append(pred[0, 0])

    return np.array(s2nrs)




def predict_file(noisy_file_path, enh_file_path):
    # Load model.
    data_type = "test"
    model_path = os.path.join(conf2.model_dir, "md_%diters.h5" % conf2.iterations)
    model = load_model(model_path)

    (a_noisy, _) = pp.read_audio(noisy_file_path)
    (a_enh, _) = pp.read_audio(enh_file_path)

    dnn2_input = np.array([a_noisy.mean(), a_enh.mean()])
    dnn2_input = dnn2_input.reshape([1, 2])

    print(dnn2_input)

    s2nr = model.predict(dnn2_input)

    return s2nr



# res2 = predict("data_eval/dnn1_in", "data_eval/dnn1_out")
# res1 = 1 / (1/res2 - 1)
# print(res2)