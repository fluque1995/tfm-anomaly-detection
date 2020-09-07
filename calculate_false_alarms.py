import glob
import numpy as np

models = ["original", "replica", "1024", "768", "512"]

for model in models:
    normal_predictions_regex = "predictions_{}/Normal*".format(model)
    abnormal_predictions_regex = "predictions_{}/[!Normal]*".format(model)

    normal_predictions = glob.glob(normal_predictions_regex)
    abnormal_predictions = glob.glob(abnormal_predictions_regex)

    normal_pos_preds = 0
    normal_videos = 0
    for vid in normal_predictions:
        preds = np.load(vid)
        normal_videos += 1
        normal_pos_preds += np.max(np.round(preds))

    abnormal_pos_preds = 0
    abnormal_videos = 0
    for vid in abnormal_predictions:
        preds = np.load(vid)
        abnormal_videos += 1
        abnormal_pos_preds += np.max(np.round(preds))

    print("MODEL: {}".format(model))
    print("Normal videos with positive labels: {} %".format(
        100*normal_pos_preds/normal_videos))

    print("Abnormal videos with positive labels: {} %".format(
        100*abnormal_pos_preds/abnormal_videos))
