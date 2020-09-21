import os
import keras
import models
from utils import video_util
import configuration as cfg
import numpy as np
import sklearn.preprocessing

original_model = keras.models.load_model(cfg.extractor_model_weights)
spatiotemporal_extractor = keras.models.Model(
    inputs = original_model.input,
    outputs = original_model.get_layer("lstm_1").output
)

normal_videos = os.listdir(cfg.normal_videos_path)
normal_videos.sort()
for i, vid_name in enumerate(normal_videos):
    print("Processing {} ({}/{})".format(vid_name, i+1, len(normal_videos)))
    vid_path = os.path.join(cfg.normal_videos_path, vid_name)
    feats_path = os.path.join(cfg.raw_normal_train_features, vid_name[:-9] + ".npy")
        
    clips, frames = video_util.get_video_clips(vid_path)

    # Remove last clip if number of frames is not equal to 16
    if frames % 16 != 0:
        clips = clips[:-1]

    prep_clips = [keras.applications.xception.preprocess_input(np.array(clip))
                  for clip in clips]
    prep_clips = np.stack(prep_clips, axis=0)

    features = spatiotemporal_extractor.predict(prep_clips)
    features = sklearn.preprocessing.normalize(features, axis=1)

    with open(feats_path, "wb") as f:
        np.save(f, features)

abnormal_videos = os.listdir(cfg.abnormal_videos_path)
abnormal_videos.sort()
print("Processing abnormal videos...")
for i, vid_name in enumerate(abnormal_videos):
    print("Processing {} ({}/{})".format(vid_name, i+1, len(abnormal_videos)))
    vid_path = os.path.join(cfg.abnormal_videos_path, vid_name)
    feats_path = os.path.join(cfg.raw_abnormal_train_features, vid_name[:-9] + ".npy")
    
    clips, frames = video_util.get_video_clips(vid_path)

    # Remove last clip if number of frames is not equal to 16
    if frames % 16 != 0:
        clips = clips[:-1]

    prep_clips = [keras.applications.xception.preprocess_input(np.array(clip))
                  for clip in clips]
    prep_clips = np.stack(prep_clips, axis=0)

    features = spatiotemporal_extractor.predict(prep_clips)
    features = sklearn.preprocessing.normalize(features, axis=1)

    with open(feats_path, "wb") as f:
        np.save(f, features)


test_videos = os.listdir(cfg.test_set)
test_videos.sort()
print("Processing test videos...")
for i, vid_name in enumerate(test_videos):
    print("Processing {} ({}/{})".format(vid_name, i+1, len(test_videos)))
    vid_path = os.path.join(cfg.test_set, vid_name)
    feats_path = os.path.join(cfg.raw_test_features, vid_name[:-9] + ".npy")
    
    clips, frames = video_util.get_video_clips(vid_path)

    # Remove last clip if number of frames is not equal to 16
    if frames % 16 != 0:
        clips = clips[:-1]

    prep_clips = [keras.applications.xception.preprocess_input(np.array(clip))
                  for clip in clips]
    prep_clips = np.stack(prep_clips, axis=0)

    features = spatiotemporal_extractor.predict(prep_clips)
    features = sklearn.preprocessing.normalize(features, axis=1)

    with open(feats_path, "wb") as f:
        np.save(f, features)
