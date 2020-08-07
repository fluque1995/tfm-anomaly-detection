import sklearn.metrics
import numpy as np
import pandas as pd
import configuration as cfg
import os
import utils.video_util
import utils.array_util
import matplotlib.pyplot as plt

ground_truth = pd.read_csv(cfg.test_temporal_annotations, header=None, index_col=0)
print(ground_truth)

preds = []
gts = []

for idx, row in ground_truth.iterrows():
    preds_file_path = os.path.join(cfg.preds_folder, idx)
    frames = row[6]
    try:
        with open(preds_file_path, "rb") as f:
            curr_preds = np.load(f)

        padded_preds = utils.array_util.extrapolate(curr_preds, frames)
    except FileNotFoundError:
        padded_preds = np.zeros((frames,1))
        print("No predictions generated for {}".format(idx))

    curr_gts = np.zeros(frames)
    anomaly_start_1 =  row[2]
    anomaly_end_1 = row[3]

    anomaly_start_2 =  row[4]
    anomaly_end_2 = row[5]

    if anomaly_start_1 != -1 and anomaly_end_1 != -1:
        curr_gts[anomaly_start_1:anomaly_end_1+1] = 1

    if anomaly_start_2 != -1 and anomaly_end_2 != -1:
        curr_gts[anomaly_start_2:anomaly_end_2+1] = 1

    preds.append(padded_preds)
    gts.append(curr_gts)

gts = np.concatenate(gts)
preds = np.concatenate(preds)
preds_labels = np.round(preds)

acc = sklearn.metrics.accuracy_score(gts, preds_labels)
fpr, tpr, thr = sklearn.metrics.roc_curve(gts, preds)
conf_mat = sklearn.metrics.confusion_matrix(gts, preds_labels)
auc = sklearn.metrics.auc(fpr, tpr)

plt.title("Curva ROC")
plt.plot(fpr, tpr, 'b', label = "AUC: {}".format(auc))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(os.path.join(cfg.output_folder, "roc_lstm.png"))

print("Accuracy: {}, AUC: {}".format(acc, auc))
print("Confusion matrix")
print(conf_mat)
