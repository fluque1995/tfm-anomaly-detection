# Deep Learning for Crowd Behavior Analysis in Videosurveillance

Master's thesis in Data Sciences: Study on the use of Deep Learning
for Crowd Behavior Analysis from videosurveillance
sources.

## Theoretical study

Theoretical study consists of a proposal of taxonomy for crowd
behavior analysis, published on Information Fusion with the title
_Revisiting crowd behavior analysis through deep learning: Taxonomy,
anomaly detection, crowd emotions, datasets, opportunities and
prospects_, which can be found in
https://www.sciencedirect.com/science/article/pii/S1566253520303201.

## Experimental analysis

In the experimental analysis, we have studied the usage of
spatio-temporal features extracted by deep learning models for crowd
anomaly detection. Specifically, we have proposed an enhancement over
the model in _Real-world Anomaly Detection in Surveillance Videos_
(https://arxiv.org/abs/1801.04264). Instead of using 3D convolutional
features, we propose a model which employs convolutional analysis for
frames together with a recurrent network (specifically, an LSTM model)
to learn the temporal structure of the convolutional features.

Experiments show that our spatio-temporal extractor outperforms the
original proposal by a decent margin, even when is pretrained on a
smaller dataset for video classification.

### Baseline implementations

This implementation, specially the original model replica (which
can be found in `original_model` folder strongly relies in
these previous works:

- https://github.com/WaqasSultani/AnomalyDetectionCVPR2018: Original
  implementation of the model
- https://github.com/ptirupat/AnomalyDetection_CVPR18: Reimplementation
  of the original world using Keras
- https://github.com/adamcasson/c3d: Implementation of C3D feature
  extractor in Keras using Tensorflow as backend

The original model has been adapted in order to be self-contained in
this repo and fully executable in Python. Original proposals rely on
external resources and MATLAB for some of the executions, while our
implementation is completely designed in Python, which ease the
execution.
