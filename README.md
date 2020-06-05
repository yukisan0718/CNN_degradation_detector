CNN degradation detector
====

## Overview
Implementation of a CNN-based degradation detector on audio signals [1].

The "CNN_degradation_detector.py" provides a degradation detector using a CNN-based approach inspired by VGGNet [2] architecture, on pathological voice and normal running speech signals with various types of degradations.

The "CNN_degradation_CAM.py" provides visual explaination how the CNN model makes decision in identifying different type of degradations in speech signals by using the score-CAM [3].


## Requirement
soundfile 0.10.3

matplotlib 3.1.0

h5py 2.8.0

numpy 1.18.1

scipy 1.4.1

tensorflow-gpu 2.1.0

scikit-learn 0.21.2

pillow 6.2.1

opencv-python 4.1.2.30


## Dataset preparation
Please create folders for each degradation class ("CLEAN", "NZ", "DT", "RV", "NR", "OTHERS") in "audio_data" directory, and put audio files (.wav format) in it.


## References
[1] Y. Saishu, A. H. Poorjam, and M. G. Christensen: 'Identification of Degradations in Speech Signals using a CNN-based Approach', in the editing status...

[2] K. Simonyan and A. Zisserman: 'Very Deep Convolutional Networks for Large-Scale Image Recognition', 3rd International Conference on Learning Representations (ICLR) - Conference Track Proceedings, Vol.1, No.14, (2015)

[3] H. Wang, et. al.: 'Score-CAM: Improved Visual Explanations via Score-Weighted Class Activation Mapping', The CVPR Workshop on Fair, Data Efficient and Trusted Computer Vision, (2019)