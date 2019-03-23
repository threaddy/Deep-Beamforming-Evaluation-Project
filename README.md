# Deep-Beamforming-Evaluation-Project

Here is python code to implement a Deep Beamforming Algotihm as described by Xiao-Lei Zhang in his/her paper 
(https://arxiv.org/abs/1811.01233). Two neural networks are implemented using Keras, to obtain a single channel enhancement for
speech signals and to extimated SNR in noisy enviroment. DNNs are based on Sednn (https://github.com/yongxuUSTC/sednn), have 
a look if you need a more versatile implementation. 

DNNs can be trained using Timit dataset. MVDR is used in the DAB script, to generate weights for each channel

Execute dnn1_train.py and dnn2_train.py for training. In dnn1_config and dnn2_config are training parameters su as epochs and number of istances to generate. 

Execute simulate_anechoic.py to create a microphone setup, simulate acquisitions and running DAB or classical Deep Beamforming. 
Multiple speech file can be used as source. The script returns average and standard devation for PESQ, STOI AND SDR indexes, for every sources. 
Write "visualize_on = true" at the beginning of dab.py to visualize plots during dab execution.
