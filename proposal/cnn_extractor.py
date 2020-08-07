import keras
import numpy as np
import os
import glob
import cv2

class Extractor():
    
    def __init__(self):
        xception = keras.applications.Xception(include_top=True, weights='imagenet')

        self.model = keras.models.Model(
            inputs=xception.input,
            outputs=xception.get_layer('avg_pool').output
        )

    def extract_features(self, images):
        return self.model.predict(images)


def main():
    feature_extractor = Extractor()

    train_dir = "../ucf101/train"
    feats_train_dir = "../img_features/train"

    classes = glob.glob(os.path.join(train_dir, '*'))
    classes = [os.path.basename(c) for c in classes]
    filecount = len(glob.glob(os.path.join(train_dir, '*/*')))
        
    i = 1
    print("Processing train videos...")
    for classname in classes:
        files = glob.glob(os.path.join(train_dir, classname, '*'))
        for file in files:
            print('\rProcessing file %d/%d' % (i, filecount), end='')
            i+=1
            frames = []
            vid = cv2.VideoCapture(file)
            while True:
                grabbed, frame = vid.read()
                if not grabbed:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (299,299))
                frames.append(frame)
            
            frames = np.array(frames)
            frames = keras.applications.xception.preprocess_input(frames)
            feats = feature_extractor.extract_features(frames)
            np.save(
                os.path.join(feats_train_dir, classname,
                             os.path.basename(file)[:-4] + '.npy'),
                feats
            )

    test_dir = "../ucf101/test"
    feats_test_dir = "../img_features/test"

    classes = glob.glob(os.path.join(test_dir, '*'))
    classes = [os.path.basename(c) for c in classes]
    filecount = len(glob.glob(os.path.join(test_dir, '*/*')))
        
    i = 1
    print("Processing test videos...")
    for classname in classes:
        files = glob.glob(os.path.join(test_dir, classname, '*'))
        for file in files:
            print('\rProcessing file %d/%d' % (i, filecount), end='')
            i+=1
            frames = []
            vid = cv2.VideoCapture(file)
            while True:
                grabbed, frame = vid.read()
                if not grabbed:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (299,299))    
                frames.append(frame)
            
            frames = np.array(frames)
            frames = keras.applications.xception.preprocess_input(frames)
            feats = feature_extractor.extract_features(frames)
            np.save(
                os.path.join(feats_test_dir, classname,
                             os.path.basename(file)[:-4] + '.npy'),
                feats
            )

if __name__ == '__main__':
    main()
