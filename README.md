# Multimodal-Sequence-Classifications-with-Deep-Learning
This study investigates identification of best feature set combinations from multimodal time series classification of reaction force vectors and IMU sensor data collected from Force based intrumented hand manipulation motions used in preclinical digital manual therapy (instrument assisted physiotherapy based massage) and Quatifiable Soft Tissue Manipulation.

This study focuses on the time series classification of multivariate force-motion manipulation sequences using deep learning models especially (LSTM-RNN architectures).
Two Fundamental Manual therapy motion sequences (Linear "Strumming" motion and Curvilinear "J-Stroke" motion) are labelled and classified to develop an AI guided training tool for manual therapy practice. 

There are two folders:
1) Clinical Data folder - contains the DataSet (all the raw data files as collected from 5 experienced therapists).
2) Data Labeling Validation folder - contains MATLAB scripts to graphically visualize the data and validate with motion animation tools introduced in the scripts.

The Deep Learning models developed to train and test the Dataset is defined in the MATLAB scripts.
a) InterClinicanDataOrgRNNTrain.m  - Organizes and segregates the complete dataset of motion patterns collected from 5 experienced clinicians, investigates the combination of features to be trained and tested for achieving higher classification accuracy using a LSTM-RNN architecture.
b) IntraClinicanDataOrgRNNTrain.m - Organizes the dataset of one invidual clinician to test intratherapist performed motion sequence classification accuracy using the same LSTM-RNN architecture  
