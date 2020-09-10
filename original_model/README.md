# Original experiments replication

In this folder you can find the code to replicate the original
experimentation from the article "Real-World Anomaly Detection in
Surveillance Videos". The provided code strongly relies on the
following sources:

- https://github.com/WaqasSultani/AnomalyDetectionCVPR2018: Original
  implementation of the model
- https://github.com/ptirupat/AnomalyDetection_CVPR18: Reimplementation
  of the original world using Keras
- https://github.com/adamcasson/c3d: Implementation of C3D feature
  extractor in Keras using Tensorflow as backend

## Experimentation replication

The folder is self-contained and fully written in Python. The
experiments can be completely performed by executing code inside this
folder, without depending on external resources. Code files inside
this folder can be divided into two groups; resource files and scripts.
In resource files, auxiliary utilities and models are defined. Scripts
are provided to replicate the experiments.

### Resource files

The resource files are listed and explained below, in alphabetical
order:

- `c3d.py`: Definition of C3D feature extractor and preprocessing
  functions for the input data
- `classifier.py`: Definition of the classifier model, together
  with functions to save and load the model to disk
- `configuration.py`: Configuration information for the experiments
  (data paths, output paths, annotation files, etc)
- `parameters.py`: Information about model structure
- `utils` folder: This folder contains utilities to process arrays
  and video files

### Scripts

The developed scripts are listed in the order that should be followed
to replicate the experiments.

1. `extract_features.py`: This script computes the features from the
videos composing the dataset (videos contained in `dataset` folder at
root project level), and stores them inside the folder
`raw_c3d_features` (if default configuration has been kept). In order
to work properly, the destination folder must exist. The folder
structure can be created with the bash script provided at root project
level.
2. `preprocess_features.py`: This script takes the previously extracted
features, whose number can vary depending on the original video length,
and computes a fized-size representation for each video. The new features
are stored inside the folder `processed_c3d_features`
3. `train_classifier.py`: This script trains the final classifier
model using the preprocessed features extracted before. After
training, it stores the resulting model inside the folder `trained_models`.
4. `predict_test_set.py`: After training, this script takes the trained
model and uses it to predict the test set (test features are calculated
in the first two steps).
5. `calculate_metrics.py`: When the predictions have been made, this
script calculates several performance metrics to validate the model.
