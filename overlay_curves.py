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

preds_original = []
preds_replica = []

preds_1024 = []
preds_768 = []
preds_512 = []

gts = []

for idx, row in ground_truth.iterrows():
    original_preds_file_path = os.path.join("predictions_original", idx)
    replica_preds_file_path = os.path.join("predictions_replica", idx)
    lstm_1024_preds_file_path = os.path.join("predictions_1024", idx)
    lstm_768_preds_file_path = os.path.join("predictions_768", idx)
    lstm_512_preds_file_path = os.path.join("predictions_512", idx)
    frames = row[6]

    try:
        with open(original_preds_file_path, "rb") as f:
            curr_original_preds = np.load(f)
        with open(replica_preds_file_path, "rb") as f:
            curr_replica_preds = np.load(f)
        with open(lstm_1024_preds_file_path, "rb") as f:
            curr_lstm_1024_preds = np.load(f)
        with open(lstm_768_preds_file_path, "rb") as f:
            curr_lstm_768_preds = np.load(f)
        with open(lstm_512_preds_file_path, "rb") as f:
            curr_lstm_512_preds = np.load(f)

        original_padded_preds = array_util.extrapolate(curr_original_preds, frames)
        replica_padded_preds = array_util.extrapolate(curr_replica_preds, frames)
        lstm_1024_padded_preds = array_util.extrapolate(curr_lstm_1024_preds, frames)
        lstm_768_padded_preds = array_util.extrapolate(curr_lstm_768_preds, frames)
        lstm_512_padded_preds = array_util.extrapolate(curr_lstm_512_preds, frames)

    except FileNotFoundError:
        original_padded_preds = np.zeros((frames,1))
        replica_padded_preds = np.zeros((frames,1))
        lstm_1024_padded_preds = np.zeros((frames,1))
        lstm_768_padded_preds = np.zeros((frames,1))
        lstm_512_padded_preds = np.zeros((frames,1))

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

    preds_original.append(original_padded_preds)
    preds_replica.append(replica_padded_preds)
    preds_1024.append(lstm_1024_padded_preds)
    preds_768.append(lstm_768_padded_preds)
    preds_512.append(lstm_512_padded_preds)
    gts.append(curr_gts)

gts = np.concatenate(gts)

preds_original = np.concatenate(preds_original)
preds_replica = np.concatenate(preds_replica)
preds_1024 = np.concatenate(preds_1024)
preds_768 = np.concatenate(preds_768)
preds_512 = np.concatenate(preds_512)

(
    fpr_original, tpr_original, auc_original,
    prec_original, rec_original, ap_original
) = calculate_information_for_curves(gts, preds_original)

(
    fpr_replica, tpr_replica, auc_replica,
    prec_replica, rec_replica, ap_replica
) = calculate_information_for_curves(gts, preds_replica)

(
    fpr_1024, tpr_1024, auc_1024,
    prec_1024, rec_1024, ap_1024
) = calculate_information_for_curves(gts, preds_1024)

(
    fpr_768, tpr_768, auc_768,
    prec_768, rec_768, ap_768
) = calculate_information_for_curves(gts, preds_768)

(
    fpr_512, tpr_512, auc_512,
    prec_512, rec_512, ap_512
) = calculate_information_for_curves(gts, preds_512)

plt.title("Curva ROC")
plt.plot(fpr_original, tpr_original, 'b', label = "Original - AUC: {:.5f}".format(auc_original))
plt.plot(fpr_replica, tpr_replica, 'g', label = "Replica - AUC: {:.5f}".format(auc_replica))
plt.plot(fpr_1024, tpr_1024, 'r', label = "LSTM 1024 - AUC: {:.5f}".format(auc_1024))
plt.plot(fpr_768, tpr_768, 'm', label = "LSTM 768 - AUC: {:.5f}".format(auc_768))
plt.plot(fpr_512, tpr_512, 'y', label = "LSTM 512 - AUC: {:.5f}".format(auc_512))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'k--')
plt.plot([1, 0], [0, 1],'k:')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(os.path.join("/mnt/sdd/pacoluque/output", "roc_overlay.pdf"))

plt.clf()

plt.title("Curva PR")
plt.plot(rec_original, prec_original, 'b', label = "Original - AP: {:.5f}".format(ap_original))
plt.plot(rec_replica, prec_replica, 'g', label = "Replica - AP: {:.5f}".format(ap_replica))
plt.plot(rec_1024, prec_1024, 'r', label = "LSTM 1024 - AP: {:.5f}".format(ap_1024))
plt.plot(rec_768, prec_768, 'm', label = "LSTM 768 - AP: {:.5f}".format(ap_768))
plt.plot(rec_512, prec_512, 'y', label = "LSTM 512 - AP: {:.5f}".format(ap_512))

plt.legend(loc = 'upper right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Precison')
plt.xlabel('Recall')
plt.savefig(os.path.join("/mnt/sdd/pacoluque/output", "pr_overlay.pdf"))
