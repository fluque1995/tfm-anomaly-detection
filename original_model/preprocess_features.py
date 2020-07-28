import numpy as np
import os
import sklearn.preprocessing
import configuration as cfg

def transform_into_segments(features, n_segments=32):
    if features.shape[0] < n_segments:
        raise RuntimeError(
            "Number of prev segments lesser than expected output size"
        )

    cuts = np.linspace(0, features.shape[0], n_segments,
                       dtype=int, endpoint=False)

    new_feats = []
    for i, j in zip(cuts[:-1], cuts[1:]):
        new_feats.append(np.mean(features[i:j,:], axis=0))

    new_feats.append(np.mean(features[cuts[-1]:,:], axis=0))

    new_feats = np.array(new_feats)
    new_feats = sklearn.preprocessing.normalize(new_feats, axis=1)
    return new_feats

for filename in os.listdir(cfg.raw_normal_train_features):
    print("Processing {}".format(filename))
    raw_file_path = os.path.join(
        cfg.raw_normal_train_features, filename
    )
    processed_file_path = os.path.join(
        cfg.processed_normal_train_features, filename
    )

    with open(raw_file_path, "rb") as f:
        feats = np.load(f, allow_pickle=True)

    try:
        new_feats = transform_into_segments(feats)
        with open(processed_file_path, "wb") as f:
            np.save(f, new_feats, allow_pickle=True)
    except RuntimeError:
        print("Video {} too short".format(filename))

for filename in os.listdir(cfg.raw_abnormal_train_features):
    print("Processing {}".format(filename))
    raw_file_path = os.path.join(
        cfg.raw_abnormal_train_features, filename
    )
    processed_file_path = os.path.join(
        cfg.processed_abnormal_train_features, filename
    )
    with open(raw_file_path, "rb") as f:
        feats = np.load(f, allow_pickle=True)

    try:
        new_feats = transform_into_segments(feats)
        with open(processed_file_path, "wb") as f:
            np.save(f, new_feats, allow_pickle=True)
    except RuntimeError:
        print("Video {} too short".format(filename))

for filename in os.listdir(cfg.raw_test_features):
    print("Processing {}".format(filename))
    raw_file_path = os.path.join(
        cfg.raw_test_features, filename
    )
    processed_file_path = os.path.join(
        cfg.processed_test_features, filename
    )
    with open(raw_file_path, "rb") as f:
        feats = np.load(f, allow_pickle=True)

    try:
        new_feats = transform_into_segments(feats)
        with open(processed_file_path, "wb") as f:
            np.save(f, new_feats, allow_pickle=True)
    except RuntimeError:
        print("Video {} too short".format(filename))
