# source-discriminator
This repository contains a naive implementation of the Conditional Adverserial Architecture proposed by Zhao M et al in 2017 (see reference directory).

In a nutshell, the algorithm tries to remove dependency on the subject of study. the data is labeled by sleep stage and subject, and the model will try to minimze the loss of sleep stage prediction while maximizing the loss of subject prediction, Hence removing the effect of the particular subject.

## The data
Data is provided in the form of files, each containing a 64x8192 matrix of floating point numbers, preceeded by a line containing the label. there are 903 files, the names of which indicate with label and subject are assigned to that sample, in the form of `<label>_<subject>_<session>_<subsession>.txt`  

The model has three components:
1 -  The Encoder: enigneers features out of the input signal. In the original paper it is a CNN combined with RNN. here we implement just a CNN with 