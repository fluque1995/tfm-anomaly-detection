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
can be found in `original_model` folder) strongly relies in
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

### Software requirements

The project is completely written in Python 3, using the following
libraries:

- Keras 2.2.4 (TensorFlow GPU backend)
- numpy 1.16.2
- scipy 1.2.0
- opencv_contrib_python 4.0.0.21
- pandas 1.0.5
- matplotlib 3.0.2
- scikit_learn 0.23.2

A requirements file is provided for `pip` installation. In order to
install dependencies, navigate to the project root folder and execute:

``` shell
pip install -r requirements.txt
```

### Data folders structure and datasets

In order to properly execute the models, some folders must be created
in advance. Executing the script `create_data_folders.sh` at root
project level will create the required folders with their default
names. Also, datasets must be downloaded. In particular:

- UCF-101 Dataset (https://www.crcv.ucf.edu/data/UCF101.php) is used
  to pretrain our feature extractor proposal. You can download the
  dataset with the proper folder structure for our experiments from
  [here](https://drive.google.com/file/d/1R2E9WjQS8c48S2z7mNTT8Gc1H1z2mnqP/view?usp=sharing)
  and place it into the root project folder
- UCF-Crime Dataset (https://www.crcv.ucf.edu/projects/real-world/) is
  used for evaluation. We provide a curated version of the dataset
  with the proper train-test splits for anomaly detection, as we have
  used it in our experiments. In order to use the dataset, you should
  download the following files:
  - TEST
  - TRAIN/NORMAL
  - TRAIN/ABNORMAL

**WARNING**: Datasets are heavy, and models are resource-consuming.
We strongly recommend using dedicated GPUs and computing nodes to
replicate the experiments, since usual PCs are not capable of handling
such volumes of data.

### Pretrained models

We provide several pretrained models used in our experiments:

- Models from the original proposal: These models represent the
  original feature extractor based on C3D and the two sets of weights
  for the classifier; the original trained model by the authors
  (`weights_L1L2.mat`) and the replica trained by us
  (`weights_own.mat`). These models can be downloaded from
  [here](https://drive.google.com/file/d/1s3qBXLZzMGAsmG8U0YTJJ4NOOK3KBakl/view?usp=sharing).
  The uncompressed folder must be placed in
  `original_model/trained_models` folder
- Models from our proposal: These models represent our proposed
  extractor based on a spatio-temporal network and the classifier
  model trained by us. These models can be downloaded from
  [here](https://drive.google.com/file/d/1XJ8DLRSHowEA3JB2xAUQGOzTo1y0ofQj/view?usp=sharing).
  The uncompressed folder must be placed in `proposal/trained_models`
  folder

### Code structure

Developed code is placed in two main folders, together with some
scripts to calculate results:

- `calculate_video_level_scores.py`: It calculates the percentage of
  normal and abnormal videos in which an alarm has been triggered. For
  normal videos, a lesser percentage means lesser false alarms, and
  thus a better model. For abnormal videos, a greater percentage means
  better capability of detection anomalies.
- `overlay_curves.py`: This script computes the ROC and PR curves
  given the predictions of both models, and represents them in two
  different graphs (one for ROCs and one for PRs).
- `original_model` folder: The code in this folder is prepared to
  replicate the original experiments, from feature extraction with C3D
  to training and evaluation of the anomaly classifier.
- `proposal` folder: The code in this folder is prepared to replicate
  our experiments. There are scripts to train the feature extractor
  over UCF-101, extract features from UCF-Crime dataset using the
  pretrained extractor, train and evaluate the anomaly classifier.

There is more information on how to reproduce the experiments in the
README files inside each folder.
