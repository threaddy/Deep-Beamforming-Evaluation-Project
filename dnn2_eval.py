
import numpy as np
import os
from keras.models import load_model
import prepare_data as pp
import config_dnn2 as conf2


# input_file_folder = "data_train/dnn2_train/1"
# output_file_folder = "data_train/dnn2_train/1"



def predict(mix_folder, enh_folder):
    # Load model.

    s2nrs = []
    model_path = os.path.join("data_train", "dnn2_packed_features", "models", "md_%diters.h5" % conf2.iterations)
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

        pred = model.predict(dnn2_input)
        print(cnt, na)

        # Inverse scale.


        # Debug plot.
        s2nrs.append(pred[0, 0])

    return np.array(s2nrs)



