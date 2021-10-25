# source-discriminator
This repository contains a naive TensorFlow 2.0 implementation of the Conditional Adverserial Architecture proposed by Zhao M, et al in 2017 (see reference directory).

In a nutshell, the algorithm tries to remove dependency on the subject of study. the data is labeled by sleep stage and subject, and the model will try to minimze the loss of sleep stage prediction while maximizing the loss of subject prediction, Hence removing the effect of the particular subject.

## The data
Data is provided in the form of files, each containing a 64x8192 matrix of floating point numbers, preceeded by a line containing the label. there are 903 files, the names of which indicate with label and subject are assigned to that sample, in the form of `<label>_<subject>_<session>_<subsession>.txt` there is a csv file with a binary label and a path for each file.

In processing the data, we apply a few quality acid tests, by reading through the csv file, ensuring the labels agree with filename and the first line of the file, all arrays are the same size and contain valid numbers, and normalize their values. we also categorize the subject, drop the extra information and save the results in a simple feature_store.


## The model

The model has three components:
* The Encoder: enigneers features out of the input signal.
* The Predictor: uses the engineering feature to predict the label. Zhao tries to minimze the loss here.
* The Discriminator: fuses the output of the Encoder and the Predictor and tries to maximize the loss in predicting the subject. This is achieved by retraining the discriminator at each batch until the loss exceeds the entropy of the subject variable.

The Encoder in the original paper it is a CNN combined with RNN. Here, we implement just a CNN with ResNet18 Architecture as explained in [this Github user](https://github.com/calmisential/TensorFlow2.0_ResNet). We first tried ResNet50 but ran out of resources while training the model.

The last two models where implemented as FNN with 2 Dense layers, separated by batch normalization, leaky Relu and a 0.3 dropout.

## Evaluation

The data is split into 0.8 training, 0.1 evaluation, and 0.1 testing. Loss and accuracy (Mean and BinaryAccuracy) are calculated at the end of each epoch, the the max accuracy across epochs is found and the model saved as checkpoint. Once all epochs are processed, the one with the maximum accuracy will be loaded and its loss and accuracy over all train, eval, and test data are recalculated and reported.

The output of the Encoder for the test dataset goes through a 2 way dimensionality reduction with t-SNE and a 2 way and a 3 way reduction with PCA. the plots are saved before the training is finished.

# Deployment and Usage

In the docker directory, there are files and instructions to deploy and run the code in a GPU based environment. The Readme.md file contains an example on how to train the model.

The configuration file is the key in providing arguments to the implementation. an example is provided in the config.ini file, and the two sections ([TRAIN] and [PREDICT]) correspond to arguments used in the trainig or prediction stages.



