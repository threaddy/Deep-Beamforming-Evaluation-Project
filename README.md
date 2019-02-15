# Deep-Beamforming-Evaluation-Project

Here is python code to implement a Deep Beamforming Algotihm as described by Xiao-Lei Zhang in is paper 
(https://arxiv.org/abs/1811.01233). Two neural networks are implemented using Keras, to obtain a single channel enhancement for
speech signals and to extimated SNR in noisy enviroment. DNNs are based on Sednn (https://github.com/yongxuUSTC/sednn), have 
a look if you need a more versatile implementation. 

DNNs can be trained using Timit dataset. MVDR is used in the DAB script, to generate weights for each channel

