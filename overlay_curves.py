import sklearn.metrics
import scipy.optimize, scipy.interpolate
import numpy as np
import pandas as pd
import os
import proposal.utils.array_util as array_util
import matplotlib.pyplot as plt

def calculate_information_for_curves(gts, preds):
    fpr, tpr, _ = sklearn.metrics.roc_curve(gts, preds)
    auc = sklearn.metrics.auc(fpr, tpr)
    prec, rec, _ = sklearn.metrics.precision_recall_curve(gts, preds)
    ap = sklearn.metrics.average_precision_score(gts, preds)

    return fpr, tpr, auc, prec, rec, ap
    

ground_truth = pd.read_csv("./dataset/test/temporal-annotation.txt", header=None, index_col=0)

preds_c3d = []
preds_lstm = []
gts = []

for idx, row in ground_truth.iterrows():
    c3d_preds_file_path = os.path.join("predictions_c3d", idx)
    lstm_preds_file_path = os.path.join("predictions_lstm", idx)
    frames = row[6]

    try:
        with open(c3d_preds_file_path, "rb") as f:
            curr_c3d_preds = np.load(f)
        with open(lstm_preds_file_path, "rb") as f:
            curr_lstm_preds = np.load(f)

        c3d_padded_preds = array_util.extrapolate(curr_c3d_preds, frames)
        lstm_padded_preds = array_util.extrapolate(curr_lstm_preds, frames)

    except FileNotFoundError:
        c3d_padded_preds = np.zeros((frames,1))
        lstm_padded_preds = np.zeros((frames,1))

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

    preds_c3d.append(c3d_padded_preds)
    preds_lstm.append(lstm_padded_preds)
    gts.append(curr_gts)

gts = np.concatenate(gts)

preds_c3d = np.concatenate(preds_c3d)
preds_lstm = np.concatenate(preds_lstm)

(
    fpr_c3d, tpr_c3d, auc_c3d,
    prec_c3d, rec_c3d, ap_c3d
) = calculate_information_for_curves(gts, preds_c3d)

(
    fpr_lstm, tpr_lstm, auc_lstm,
    prec_lstm, rec_lstm, ap_lstm
) = calculate_information_for_curves(gts, preds_lstm)


plt.title("Curva ROC")
plt.plot(fpr_c3d, tpr_c3d, 'b', label = "C3d - AUC: {:.5f}".format(auc_c3d))
plt.plot(fpr_lstm, tpr_lstm, 'g', label = "Lstm - AUC: {:.5f}".format(auc_lstm))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'k--')
plt.plot([1, 0], [0, 1],'k:')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("roc_overlay.pdf")

plt.clf()

plt.title("Curva PR")
plt.plot(rec_c3d, prec_c3d, 'b', label = "C3d - AP: {:.5f}".format(ap_c3d))
plt.plot(rec_lstm, prec_lstm, 'g', label = "Lstm - AP: {:.5f}".format(ap_lstm))
plt.legend(loc = 'upper right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Precison')
plt.xlabel('Recall')
plt.savefig("pr_overlay.pdf")
