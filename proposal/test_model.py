import classifier
import configuration as cfg
import numpy as np
import os

def load_test_set(videos_path, videos_list):
    feats = []
    
    for vid in videos_list:
        vid_path = os.path.join(videos_path, vid)
        with open(vid_path, "rb") as f:
            feat = np.load(f)
        feats.append(feat)

    feats = np.array(feats)
    return feats

classifier_model = classifier.build_classifier_model()

vid_list = os.listdir(cfg.processed_test_features)
vid_list.sort()

test_set = load_test_set(cfg.processed_test_features, vid_list)

for filename, example in zip(vid_list, test_set):
    predictions_file = filename[:-4] + '.npy'
    pred_path = os.path.join(cfg.preds_folder, predictions_file)
    pred = classifier_model.predict_on_batch(example)
    with open(pred_path, "wb") as f:
        np.save(pred_path, pred, allow_pickle=True)
