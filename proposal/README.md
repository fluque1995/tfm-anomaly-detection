# Proposal experiments replication

In this folder you can find the code to replicate the experimentation
of out proposal, using a spatio-temporal feature extractor instead of
the C3D convolutional model. We have tested different models in our
report, specifically extractors that provide descriptors of size 512,
768 and 1024 for each clip of 16 frames from the video. However, we
only provide the model of size 1024, since it has provided the best
results and the experiments are similar for all the models.

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

- `classifier.py`: Definition of the classifier model, together
  with functions to save and load the model to disk.
- `configuration.py`: Configuration information for the experiments
  (data paths, output paths, annotation files, etc).
- `models.py`: Definition of the feature extractor model.
- `parameters.py`: Information about model structure.
- `video_data_generator.py`: Adaptation of Keras datasets for video
  data handling. This code has been adapted from the video frame
  generator developed by [Patrice
  Ferlet](https://gist.github.com/metal3d) and the original can be
  downloaded from
  [here](https://gist.github.com/metal3d/0fe5539abfc534855ddfd351d06cfa06)
- `utils` folder: This folder contains utilities to process arrays
  and video files.

### Scripts

The developed scripts are listed in the order that should be followed
to replicate the experiments.

1. `train_feature_extractor.py`: This script trains the feature
extractor model, solving the video classification task over
UCF-101 dataset (must be downloaded). Afterwards, the model is
saved in `trained_models`.
1. `extract_temporal_features.py`: This script computes the features
from the videos composing the dataset (videos contained in `dataset`
folder at root project level), and stores them inside the folder.
`raw_lstm_features` (if default configuration has been kept). In order
to work properly, the destination folder must exist. The folder
structure can be created with the bash script provided at root project
level.
2. `preprocess_features.py`: This script takes the previously extracted
features, whose number can vary depending on the original video length,
and computes a fized-size representation for each video. The new features
are stored inside the folder `processed_lstm_features`.
3. `train_classifier.py`: This script trains the final classifier
model using the preprocessed features extracted before. After
training, it stores the resulting model inside the folder `trained_models`.
4. `predict_test_set.py`: After training, this script takes the trained
model and uses it to predict the test set (test features are calculated
in the first two steps).
5. `calculate_metrics.py`: When the predictions have been made, this
script calculates several performance metrics to validate the model.
