import c3d
import os
import configuration as cfg
import numpy as np
import sklearn.preprocessing

from utils import video_util

feature_extractor = c3d.c3d_feature_extractor()
#normal_videos = os.listdir(cfg.normal_videos_path)
#normal_videos.sort()

with open("remaining.txt", "r") as f:
    normal_videos = f.read().splitlines()

print("Processing normal videos...")
for vid_name in normal_videos:
    print("Processing {}".format(vid_name))
    vid_path = os.path.join(cfg.normal_videos_path, vid_name)
    feats_path = os.path.join(cfg.raw_normal_train_features, vid_name[:-9] + ".npy")
    
    clips, frames = video_util.get_video_clips(vid_path)

    # Remove last clip if number of frames is not equal to 16
    if frames % 16 != 0:
        clips = clips[:-1]

    prep_clips = [c3d.preprocess_input(np.array(clip)) for clip in clips]
    prep_clips = np.vstack(prep_clips)

    features = feature_extractor.predict(prep_clips)
    features = sklearn.preprocessing.normalize(features, axis=1)

    with open(feats_path, "wb") as f:
        np.save(f, features)

abnormal_videos = os.listdir(cfg.abnormal_videos_path)
abnormal_videos.sort()
print("Processing abnormal videos...")
for vid_name in abnormal_videos:
    print("Processing {}".format(vid_name))
    vid_path = os.path.join(cfg.abnormal_videos_path, vid_name)
    feats_path = os.path.join(cfg.raw_abnormal_train_features, vid_name[:-9] + ".npy")
    
    clips, frames = video_util.get_video_clips(vid_path)

    # Remove last clip if number of frames is not equal to 16
    if frames % 16 != 0:
        clips = clips[:-1]

    prep_clips = [c3d.preprocess_input(np.array(clip)) for clip in clips]
    prep_clips = np.vstack(prep_clips)

    features = feature_extractor.predict(prep_clips)
    features = sklearn.preprocessing.normalize(features, axis=1)

    with open(feats_path, "wb") as f:
        np.save(f, features)


test_videos = os.listdir(cfg.test_set)
test_videos.sort()
print("Processing test videos...")
for vid_name in test_videos:
    print("Processing {}".format(vid_name))
    vid_path = os.path.join(cfg.test_set, vid_name)
    feats_path = os.path.join(cfg.raw_test_features, vid_name[:-9] + ".npy")
    
    clips, frames = video_util.get_video_clips(vid_path)

    # Remove last clip if number of frames is not equal to 16
    if frames % 16 != 0:
        clips = clips[:-1]

    prep_clips = [c3d.preprocess_input(np.array(clip)) for clip in clips]
    prep_clips = np.vstack(prep_clips)

    features = feature_extractor.predict(prep_clips)
    features = sklearn.preprocessing.normalize(features, axis=1)

    with open(feats_path, "wb") as f:
        np.save(f, features)
