import pandas as pd
import numpy as np
import utils.video_util
import configuration as cfg
import os

ground_truth = pd.read_csv(cfg.test_temporal_annotations, header=None, sep="\s+", index_col=0, names=['Type', 'Start1', 'End1', 'Start2', 'End2'])
print(ground_truth)

frames_list = []
for idx, row in ground_truth.iterrows():
    video_file_path = os.path.join(cfg.test_set, idx[:-4] + "_x264.mp4")
    print(video_file_path)
    _, frames = utils.video_util.get_video_clips(video_file_path)
    print(frames)
    frames_list.append(frames)

ground_truth['Frames'] = frames_list

print(ground_truth)

ground_truth.to_csv("trial.csv", header=False)
