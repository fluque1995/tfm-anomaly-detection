import os

c3d_model_weights = './trained_models/c3d_sports1m.h5'
classifier_model_weigts = './trained_models/weights.mat'
classifier_model_json = './trained_models/model.json'

input_folder  = './input'
output_folder = '/mnt/sdd/pacoluque/output'

sample_video_path = '../dataset/train/abnormal/Arrest018_x264.mp4'

raw_dataset_folder = '../dataset/'
raw_features_folder = "../raw_c3d_features"
processed_features_folder = "../processed_c3d_features"

train_set = os.path.join(raw_dataset_folder, 'train')
normal_videos_path = os.path.join(train_set, "normal")
abnormal_videos_path = os.path.join(train_set, "abnormal")

raw_features_train_set = os.path.join(raw_features_folder, 'train')
raw_normal_train_features = os.path.join(raw_features_train_set, "normal")
raw_abnormal_train_features = os.path.join(raw_features_train_set, "abnormal")

processed_features_train_set = os.path.join(processed_features_folder, 'train')
processed_normal_train_features = os.path.join(processed_features_train_set, "normal")
processed_abnormal_train_features = os.path.join(processed_features_train_set, "abnormal")

test_set = os.path.join(raw_dataset_folder, 'test')
raw_test_features = os.path.join(raw_features_folder, 'test')
processed_test_features = os.path.join(processed_features_folder, 'test')

preds_folder = '../predictions'

test_temporal_annotations = os.path.join(test_set, "temporal-annotation.txt")
